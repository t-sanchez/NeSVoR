from argparse import Namespace
from typing import Dict, List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from ..utils import MovingAverage, gaussian_blur, log_params, TrainLogger
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG
from ..transform import RigidTransform, transform_points
from ..image import Volume, Slice


class Dataset(object):
    def __init__(self, slices: List[Slice]) -> None:
        xyz_all = []
        v_all = []
        slice_idx_all = []
        transformation_all = []
        resolution_all = []

        for i, slice in enumerate(slices):
            xyz = slice.xyz_masked_untransformed
            v = slice.v_masked
            slice_idx = torch.full(v.shape, i, device=v.device)
            xyz_all.append(xyz)
            v_all.append(v)
            slice_idx_all.append(slice_idx)
            transformation_all.append(slice.transformation)
            resolution_all.append(slice.resolution_xyz)

        self.xyz = torch.cat(xyz_all)
        self.v = torch.cat(v_all)
        self.slice_idx = torch.cat(slice_idx_all)
        self.transformation = RigidTransform.cat(transformation_all)
        self.resolution = torch.stack(resolution_all, 0)
        self.count = self.v.shape[0]
        self.epoch = 0

    @property
    def bounding_box(self) -> torch.Tensor:
        max_r = self.resolution.max()
        xyz_transformed = self.xyz_transformed
        xyz_min = xyz_transformed.amin(0) - 2 * max_r
        xyz_max = xyz_transformed.amax(0) + 2 * max_r
        bounding_box = torch.stack([xyz_min, xyz_max], 0)
        return bounding_box

    @property
    def mean(self) -> float:
        q1, q2 = torch.quantile(
            self.v,
            torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
        )
        return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

    def get_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        if self.count + batch_size > self.xyz.shape[0]:  # new epoch, shuffle data
            self.count = 0
            self.epoch += 1
            idx = torch.randperm(self.xyz.shape[0], device=device)
            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]
        # fetch a batch of data
        batch = {
            "xyz": self.xyz[self.count : self.count + batch_size],
            "v": self.v[self.count : self.count + batch_size],
            "slice_idx": self.slice_idx[self.count : self.count + batch_size],
        }
        self.count += batch_size
        return batch

    @property
    def xyz_transformed(self) -> torch.Tensor:
        return transform_points(self.transformation[self.slice_idx], self.xyz)

    @property
    def mask(self) -> Volume:
        with torch.no_grad():
            resolution_min = self.resolution.min()
            resolution_max = self.resolution.max()
            xyz = self.xyz_transformed
            xyz_min = xyz.amin(0) - resolution_max * 10
            xyz_max = xyz.amax(0) + resolution_max * 10
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
            shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
            kji = ((xyz - xyz_min) / resolution_min).round().long()

            mask = torch.bincount(
                kji[..., 0]
                + shape[2] * kji[..., 1]
                + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            )
            mask = mask.view((1, 1) + shape).float()
            mask_threshold = (
                1.0 * resolution_min**3 / self.resolution.log().mean().exp() ** 3
            )
            mask_threshold *= mask.sum() / (mask > 0).sum()
            mask = (
                gaussian_blur(mask, (resolution_max / resolution_min).item(), 3)
                > mask_threshold
            )[0, 0]

            xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min
            return Volume(
                mask.float(),
                mask,
                RigidTransform(torch.cat([0 * xyz_c, xyz_c])[None], True),
                resolution_min,
                resolution_min,
                resolution_min,
            )


def train(slices: List[Slice], args: Namespace) -> Tuple[INR, List[Slice], Volume]:
    # create training dataset
    dataset = Dataset(slices)
    model = NeSVoR(
        dataset.transformation,
        dataset.resolution,
        dataset.mean,
        dataset.bounding_box,
        args,
    )
    # setup optimizer
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    # logging
    logging.debug(log_params(model))
    optimizer = torch.optim.Adam(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-6},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    # setup scheduler for lr decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    # setup grad scalar for mixed precision training
    fp16 = True
    scaler = torch.cuda.amp.GradScaler(1.0, enabled=fp16)
    # training
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        T_REG: args.weight_transformation,
        B_REG: args.weight_bias,
        I_REG: args.weight_image,
    }
    average = MovingAverage(1 - 0.001)
    t_start = time.time()
    # logging
    logging.info("NeSVoR training starts.")
    train_logger = TrainLogger(
        "time", "epoch", "iter", D_LOSS, S_LOSS, DS_LOSS, T_REG, B_REG, I_REG, "lr"
    )
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)
            loss = 0
            for k in losses:
                if k in loss_weights:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            # logging
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                average[D_LOSS],
                average[S_LOSS],
                average[DS_LOSS],
                average[T_REG],
                average[B_REG],
                average[I_REG],
                optimizer.param_groups[0]["lr"],
            )
            # train_logging(i, dataset.epoch, t_start, average, optimizer)
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
        """
        if train_time > tmp_counter:
            from .sample import sample_volume

            model.eval()
            # dataset.transformation = model.transformation
            vv = sample_volume(model.inr, dataset.mask, args)
            vv.save(
                "/home/junshen/SVoRT/github/NeSVoR/output_slices/%d.nii.gz"
                % tmp_counter
            )
            model.train()
            tmp_counter += 1
        """

    # outputs
    transformation = model.transformation
    dataset.transformation = transformation
    mask = dataset.mask
    output_slices = []
    for i in range(len(slices)):
        output_slice = slices[i].clone()
        output_slice.transformation = transformation[i]
        output_slices.append(output_slice)
    # Returning not just the inr, but the nesvor model
    ds = {
        "transformation":dataset.transformation,
        "resolution": dataset.resolution,
        "mean": dataset.mean,
        "bounding_box":dataset.bounding_box,
    }
    return model, output_slices, mask, ds
