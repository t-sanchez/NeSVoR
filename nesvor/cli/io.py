import torch
from typing import Dict, Tuple, Any
from argparse import Namespace
from ..image import Volume, save_slices, load_slices, load_stack
from ..nesvor.models import INR, NeSVoR
from ..utils import merge_args


def inputs(args: Namespace) -> Tuple[Dict, Namespace]:
    input_dict: Dict[str, Any] = dict()
    if getattr(args, "input_stacks", None) is not None:
        input_dict["input_stacks"] = []
        for i, f in enumerate(args.input_stacks):
            stack = load_stack(
                f,
                args.stack_masks[i] if args.stack_masks is not None else None,
                device=args.device,
            )

            if args.thicknesses is not None:
                stack.thickness = args.thicknesses[i]
            input_dict["input_stacks"].append(stack)
    if getattr(args, "input_slices", None) is not None:
        input_dict["input_slices"] = load_slices(args.input_slices, args.device)
    if getattr(args, "input_model", None) is not None:
        # Saving and loading a NeSVoR model rather than just the INR.
        cp = torch.load(args.input_model, map_location=args.device)
        nesvor = NeSVoR(
            transformation=cp["dataset_info"]["transformation"],
            resolution=cp["dataset_info"]["resolution"],
            v_mean=cp["dataset_info"]["mean"],
            bounding_box=cp["dataset_info"]["bounding_box"],
            args=cp["args"],
        )
        nesvor.load_state_dict(cp["model"])
        input_dict["model"] = nesvor  # INR(cp["model"]["bounding_box"], cp["args"])
        input_dict["mask"] = cp["mask"]
        args = merge_args(cp["args"], args)
    return input_dict, args


def outputs(data: Dict, args: Namespace) -> None:
    if getattr(args, "output_volume", None) and "output_volume" in data:
        if args.output_intensity_mean:
            data["output_volume"].rescale(args.output_intensity_mean)
        data["output_volume"].save(args.output_volume)
    if getattr(args, "output_model", None) and "output_model" in data:
        torch.save(
            {
                "model": data["output_model"].state_dict(),
                "mask": data["mask"],
                "args": args,
                "dataset_info": data["dataset_info"],
            },
            args.output_model,
        )
    if getattr(args, "output_slices", None) and "output_slices" in data:
        save_slices(args.output_slices, data["output_slices"])
    if getattr(args, "simulated_slices", None) and "simulated_slices" in data:
        save_slices(args.simulated_slices, data["simulated_slices"])
    # Added on my side: saving the invidiual slices, their sigma, variance and variance of sigma.
    if getattr(args, "simulated_sigmas", None) and "simulated_sigmas" in data:
        import os

        os.makedirs(args.simulated_sigmas, exist_ok=True)
        slice_num = 0
        name_curr = None
        for i, image in enumerate(data["simulated_sigmas"]):
            name = image.name.split("_T2w")[0]
            if name_curr is None or name != name_curr:
                slice_num = 0
                name_curr = name
            else:
                slice_num += 1
            image.save(
                os.path.join(args.simulated_sigmas, f"{name}_desc-slice{slice_num}_T2w.nii.gz"),
                True,
            )
            image.save_select(
                os.path.join(args.simulated_sigmas, f"{name}_desc-slice{slice_num}_sigma.nii.gz"),
                True,
                "sigma",
            )
            image.save_select(
                os.path.join(args.simulated_sigmas, f"{name}_desc-slice{slice_num}_var.nii.gz"),
                True,
                "image_var",
            )
            image.save_select(
                os.path.join(
                    args.simulated_sigmas, f"{name}_desc-slice{slice_num}_sigma_var.nii.gz"
                ),
                True,
                "sigma_var",
            )


def load_model(args: Namespace) -> Tuple[INR, Volume, Namespace]:
    # Load NeSVoR model instead of an INR
    cp = torch.load(args.input_model, map_location=args.device)
    nesvor = NeSVoR(
        transformation=cp["dataset_info"]["transformation"],
        resolution=cp["dataset_info"]["resolution"],
        v_mean=cp["dataset_info"]["mean"],
        bounding_box=cp["dataset_info"]["bounding_box"],
        args=cp["args"],
    )
    nesvor.load_state_dict(cp["model"])
    # inr = INR(cp["model"]["bounding_box"], cp["args"])
    # inr.load_state_dict(cp["model"])
    mask = cp["mask"]
    args = merge_args(cp["args"], args)
    return nesvor.inr, mask, args
