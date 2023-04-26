import torch
from typing import Dict, Tuple, Any, Optional
from argparse import Namespace
from ..image import Volume, save_slices, load_slices, load_stack, load_volume
from ..nesvor.models import INR
from ..utils import merge_args
from ..preprocessing.masking.intersection import stack_intersect


def inputs(args: Namespace) -> Tuple[Dict, Namespace]:
    input_dict: Dict[str, Any] = dict()
    if getattr(args, "input_stacks", None) is not None:
        input_stacks = []
        for i, f in enumerate(args.input_stacks):
            stack = load_stack(
                f,
                args.stack_masks[i]
                if getattr(args, "stack_masks", None) is not None
                else None,
                device=args.device,
            )
            if getattr(args, "thicknesses", None) is not None:
                stack.thickness = args.thicknesses[i]
            input_stacks.append(stack)
        volume_mask: Optional[Volume]
        if getattr(args, "volume_mask", None):
            volume_mask = load_volume(args.volume_mask, device=args.device)
        elif getattr(args, "stacks_intersection", False):
            volume_mask = stack_intersect(input_stacks, box=True)
        else:
            volume_mask = None
        if volume_mask is not None:
            for stack in input_stacks:
                stack.apply_volume_mask(volume_mask)
        input_dict["input_stacks"] = input_stacks
        input_dict["volume_mask"] = volume_mask
    if getattr(args, "input_slices", None) is not None:
        input_dict["input_slices"] = load_slices(args.input_slices, args.device)
    if getattr(args, "input_model", None) is not None:
        cp = torch.load(args.input_model, map_location=args.device)
        input_dict["model"] = INR(cp["model"]["bounding_box"], cp["args"])
        input_dict["model"].load_state_dict(cp["model"])
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
            },
            args.output_model,
        )
    if getattr(args, "output_slices", None) and "output_slices" in data:
        save_slices(args.output_slices, data["output_slices"])
    if getattr(args, "simulated_slices", None) and "simulated_slices" in data:
        save_slices(args.simulated_slices, data["simulated_slices"])
    for k in ["output_stack_masks", "output_corrected_stacks"]:
        if getattr(args, k, None) and k in data:
            for m, p in zip(data[k], getattr(args, k)):
                m.save(p)


def load_model(args: Namespace) -> Tuple[INR, Volume, Namespace]:
    cp = torch.load(args.input_model, map_location=args.device)
    inr = INR(cp["model"]["bounding_box"], cp["args"])
    inr.load_state_dict(cp["model"])
    mask = cp["mask"]
    args = merge_args(cp["args"], args)
    return inr, mask, args
