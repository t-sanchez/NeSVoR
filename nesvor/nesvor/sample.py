from argparse import Namespace
from typing import List
import torch
from ..transform import transform_points
from ..image import Slice, Volume
from .models import INR, NeSVoR
from ..utils import resolution2sigma, meshgrid


def sample_volume(model: INR, mask: Volume, args: Namespace) -> Volume:
    model.eval()
    img = mask.resample(args.output_resolution, None)
    img.image[img.mask] = sample_points(model, img.xyz_masked, args)
    return img


def sample_points(model: INR, xyz: torch.Tensor, args: Namespace) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=args.device)
    batch_size = args.inference_batch_size
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(args.output_resolution, isotropic=True),
                args.n_inference_samples if args.output_psf else 0,
            )
            v_b = model(xyz_batch, False).mean(-1)
            v[i : i + batch_size] = v_b
    return v.view(shape)


def sample_slice(model: INR, slice: Slice, mask: Volume, args: Namespace) -> Slice:
    # clone the slice
    slice_sampled = slice.clone()
    slice_sampled.image = torch.zeros_like(slice_sampled.image)
    slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
            args.n_inference_samples if args.output_psf else 0,
        )

        v = model(xyz_masked, False).mean(-1)
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slices(model: INR, slices: List[Slice], mask: Volume, args: Namespace) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            slices_sampled.append(sample_slice(model, slice, mask, args))
    return slices_sampled


def sample_sigma(
    model: NeSVoR, slice_idx: int, slice: Slice, mask: Volume, args: Namespace
) -> Slice:
    # clone the slice
    slice_sampled = slice.clone()
    slice_sampled.image = torch.zeros_like(slice_sampled.image)
    slice_sampled.image_var = torch.zeros_like(slice_sampled.image)
    slice_sampled.sigma = torch.zeros_like(slice_sampled.image)
    slice_sampled.sigma_var = torch.zeros_like(slice_sampled.image)
    slice_sampled.mask = torch.zeros_like(slice_sampled.mask)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0

    if m.any():
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(slice_sampled.resolution_xyz, isotropic=False),
            args.n_inference_samples if args.output_psf else 0,
        )
        # v = model(xyz_masked, False).mean(-1)
        slice_idx = torch.tensor(slice_idx, dtype=torch.long, device=args.device).repeat(
            xyz_masked.shape[0]
        )
        se = model.slice_embedding(slice_idx)[:, None].expand(
            -1, args.n_inference_samples if args.output_psf else -1, -1
        )
        results = model.net_forward(xyz_masked, se)

        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        # Output is nmask x 512 -> Because we have multiple samples from the PSF. I'm not exactly sure
        # what it means in practice, what it implies for the final reconstruction.
        slice_sampled.image[slice_sampled.mask] = (
            results["density"].mean(-1).to(slice_sampled.image.dtype)
        )
        slice_sampled.sigma[slice_sampled.mask] = (
            results["log_var"].mean(-1).exp().to(slice_sampled.image.dtype)
        )
        slice_sampled.image_var[slice_sampled.mask] = (
            results["density"].var(-1).to(slice_sampled.image.dtype)
        )
        slice_sampled.sigma_var[slice_sampled.mask] = (
            results["log_var"].var(-1).exp().to(slice_sampled.image.dtype)
        )
    return slice_sampled


def sample_sigmas(model: NeSVoR, slices: List[Slice], mask: Volume, args: Namespace) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            slices_sampled.append(sample_sigma(model, i, slice, mask, args))
    return slices_sampled
