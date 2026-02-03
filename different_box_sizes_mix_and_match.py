#!/usr/bin/env python3
"""
Create the final inference directory structure.

Step 1:
- define machine_dir, expert_dir, final_dir
- create `data_pkl/` and `iou_dict/` inside final_dir
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any
from statistics import mean
import torch
import numpy as np
import pandas as pd


DEFAULT_MACHINE_DIR = Path(
    #"/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus/box_14_12_k10_vtus_all"
    "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_sun/box_14_12_k10_sun_all"
)
DEFAULT_EXPERT_DIR = Path(
    #"/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus/box_10_14_k10_vtus_all"
    "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_sun/box_10_14_k10_sun_all"
)
DEFAULT_FINAL_DIR = Path(
    #"/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_vtus/box_14_14_k10_vtus_all"
    "/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/miccai_data_pkl_sun/box_14_14_k10_sun_all"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Set up the final inference directory structure."
    )
    p.add_argument(
        "--machine-dir",
        type=Path,
        default=DEFAULT_MACHINE_DIR,
        help=f"Path to machine inference dir (default: {DEFAULT_MACHINE_DIR})",
    )
    p.add_argument(
        "--expert-dir",
        type=Path,
        default=DEFAULT_EXPERT_DIR,
        help=f"Path to expert inference dir (default: {DEFAULT_EXPERT_DIR})",
    )
    p.add_argument(
        "--final-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help=f"Path to final inference dir (default: {DEFAULT_FINAL_DIR})",
    )
    return p.parse_args()


def _load_data_pkl(path: Path) -> Any:
    """
    Load a *_data.pkl file. These pkls often contain torch tensors, so torch must
    be available in the Python environment.
    """
    # try:
    #     import torch as _torch  # noqa: F401
    # except ModuleNotFoundError as e:
    #     raise ModuleNotFoundError(
    #         "Missing dependency 'torch' needed to unpickle these files. "
    #         "Run inside the medsam2 conda env (or any env where torch is installed)."
    #     ) from e

    with path.open("rb") as f:
        return pickle.load(f)


def _extract_video_masks_lnodefer(obj: Any) -> tuple[Any, Any, Any]:
    """
    Extract (video_name, Masks, L_no_defer) from either:
    - dict-like pickle: obj['video_name'], obj['Masks'], obj['L_no_defer']
    - attribute-like pickle: obj.video_name, obj.Masks, obj.L_no_defer
    """
    if isinstance(obj, dict):
        return obj.get("video_name"), obj.get("Masks"), obj.get("L_no_defer") , obj.get("L_post_defer_list")

    return (
        getattr(obj, "video_name", None),
        getattr(obj, "Masks", None),
        getattr(obj, "L_no_defer", None),
        getattr(obj, "L_post_defer_list", None),
    )


def _load_iou_pkl(path: Path) -> dict:
    """Load an *_iou_dict.pkl expected to be a dict."""
    try:
        import torch as _torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'torch' needed to unpickle these files. "
            "Run inside the medsam2 conda env (or any env where torch is installed)."
        ) from e

    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def _as_list(x: Any) -> list:
    """Best-effort conversion to a Python list for slicing/concatenation."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    # torch / numpy-ish
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x)


def _key_suffix_num(k: str) -> int | None:
    """Parse '0_<n>' -> n, else None."""
    if not k.startswith("0_"):
        return None
    try:
        return int(k.split("_", 1)[1])
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    machine_dir: Path = args.machine_dir
    expert_dir: Path = args.expert_dir
    final_dir: Path = args.final_dir

    # Create final output structure
    final_data_dir = final_dir / "data_pkl"
    final_iou_dir = final_dir / "iou_dict"
    final_data_dir.mkdir(parents=True, exist_ok=True)
    final_iou_dir.mkdir(parents=True, exist_ok=True)

    # Read machine_dir file names we need for processing:
    # - machine_dir/data_pkl/*_data.pkl
    # - corresponding machine_dir/iou_dict/*_iou_dict.pkl
    machine_data_dir = machine_dir / "data_pkl"
    machine_iou_dir = machine_dir / "iou_dict"
    expert_iou_dir = expert_dir / "iou_dict"

    data_files = sorted(machine_data_dir.glob("*_data.pkl"))
    print(f"found_data_pkls: {len(data_files)} in {machine_data_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_iou: list[Path] = []

    for data_pkl in data_files:
        iou_name = data_pkl.name.replace("_data.pkl", "_iou_dict.pkl")
        machine_iou_pkl = machine_iou_dir / iou_name
        expert_iou_pkl = expert_iou_dir / iou_name
        if not machine_iou_pkl.exists() or not expert_iou_pkl.exists():
            if not machine_iou_pkl.exists():
                missing_iou.append(machine_iou_pkl)
            if not expert_iou_pkl.exists():
                missing_iou.append(expert_iou_pkl)
            continue
        pairs.append((data_pkl, machine_iou_pkl))

    print(f"found_pairs: {len(pairs)}")
    if missing_iou:
        print(f"missing_iou_dict: {len(missing_iou)} (showing up to 10)")
        for p in missing_iou[:10]:
            print("  missing:", p)

    # Load each data_pkl and extract: video_name, Masks, L_no_defer
    for data_pkl, _iou_pkl in pairs:
        data_obj = _load_data_pkl(data_pkl)
        video_name, Masks, L_no_defer, L_post_defer_list = _extract_video_masks_lnodefer(data_obj)

        new_video_name = video_name
        new_Mask = Masks
        new_L_no_defer = L_no_defer

        # Build new_iou_dict using BOTH machine and expert iou_dict pkls
        iou_name = data_pkl.name.replace("_data.pkl", "_iou_dict.pkl")
        machine_iou_pkl = machine_iou_dir / iou_name
        expert_iou_pkl = expert_iou_dir / iou_name

        machine_iou_dict = _load_iou_pkl(machine_iou_pkl)
        expert_iou_dict = _load_iou_pkl(expert_iou_pkl)

        new_iou_dict: dict[str, Any] = {}
        new_iou_dict["0"] = machine_iou_dict.get("0")
        new_iou_dict["0_0"] = expert_iou_dict.get("0_0")

        # for all other expert keys except "0" and "0_0" (expected "0_<number>")
        other_keys = [k for k in expert_iou_dict.keys() if k not in {"0", "0_0"}]
        for k in other_keys:
            n = _key_suffix_num(str(k))
            if n is None:
                continue
            m_list = _as_list(machine_iou_dict.get(k))
            e_list = _as_list(expert_iou_dict.get(k))
            new_iou_dict[str(k)] = m_list[:n] + e_list[n:]

        # Compute new_L_post_defer_list:
        # start from "0_0", then "0_1", "0_2", ... in ascending order (present keys only)
        numeric_keys = []
        for k in new_iou_dict.keys():
            if k == "0_0":
                continue
            n = _key_suffix_num(k)
            if n is not None:
                numeric_keys.append((n, k))
        numeric_keys.sort(key=lambda t: t[0])

        ordered_keys = ["0_0"] + [k for _n, k in numeric_keys]
        means: list[float] = []
        for k in ordered_keys:
            vals = _as_list(new_iou_dict.get(k))
            means.append(float(mean(vals)) if vals else float("nan"))

        #new_L_post_defer_list = ",".join(str(x) for x in means)

        print("loaded:", data_pkl.name, "| video_name:", new_video_name)
        print("new_L_post_defer_list:", means)

        # Save outputs to final_dir
        out_data_pkl = final_data_dir / data_pkl.name
        out_iou_pkl = final_iou_dir / iou_name

        out_payload = {
            "video_name": new_video_name,
            "Masks": new_Mask,
            "L_no_defer": new_L_no_defer,
            "L_post_defer_list": means,
        }

        with out_data_pkl.open("wb") as f:
            pickle.dump(out_payload, f)
        with out_iou_pkl.open("wb") as f:
            pickle.dump(new_iou_dict, f)

        print("saved_data_pkl:", out_data_pkl)
        print("saved_iou_pkl :", out_iou_pkl)

    


if __name__ == "__main__":
    main()


