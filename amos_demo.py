#!/usr/bin/env python3
"""Download three AMOS22 CT scans, prepare them for nnU-Net, or predict."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

REPOSITORY = "MedOtter/amos22-ct-dataset"
REVISION = "main"
CASES = ("amos_0001", "amos_0004", "amos_0005")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
NNUNET_DATASET = ROOT / "data" / "nnUNet_raw" / "Dataset901_AMOS22_Small"


def download() -> None:
    """Fetch only three CT volumes and their matching label volumes."""
    from huggingface_hub import hf_hub_download

    for case in CASES:
        for split in ("imagesTr", "labelsTr"):
            remote_name = f"train/{split}/{case}.nii.gz"
            local_dir = RAW / split
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {remote_name}")
            hf_hub_download(
                repo_id=REPOSITORY,
                repo_type="dataset",
                revision=REVISION,
                filename=remote_name,
                local_dir=RAW,
            )


def prepare() -> None:
    """Copy files into the single-channel naming layout required by nnU-Net."""
    images_out = NNUNET_DATASET / "imagesTr"
    labels_out = NNUNET_DATASET / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        image = RAW / "train" / "imagesTr" / f"{case}.nii.gz"
        label = RAW / "train" / "labelsTr" / f"{case}.nii.gz"
        if not image.exists() or not label.exists():
            raise FileNotFoundError("Run `python amos_demo.py download` first.")
        # _0000 means CT is the first and only input channel.
        shutil.copy2(image, images_out / f"{case}_0000.nii.gz")
        shutil.copy2(label, labels_out / f"{case}.nii.gz")

    metadata = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0, "spleen": 1, "right_kidney": 2,
            "left_kidney": 3, "gallbladder": 4, "esophagus": 5,
            "liver": 6, "stomach": 7, "aorta": 8, "inferior_vena_cava": 9,
            "pancreas": 10, "right_adrenal_gland": 11,
            "left_adrenal_gland": 12, "duodenum": 13, "bladder": 14,
            "prostate_or_uterus": 15,
        },
        "numTraining": len(CASES),
        "file_ending": ".nii.gz",
        "name": "AMOS22_Small",
        "description": "Three AMOS22 CT cases - pipeline smoke test only",
    }
    (NNUNET_DATASET / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(CASES)} cases in {NNUNET_DATASET}")


def preview(input_path: Path, mask_path: Path, output_path: Path) -> None:
    """Save a side-by-side CT input and segmentation-overlay preview."""
    # This script writes a PNG; it does not need a graphical desktop window.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = nib.load(str(input_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    if ct.shape != mask.shape:
        raise ValueError(f"Different shapes: CT {ct.shape}, mask {mask.shape}")
    slice_index = ct.shape[2] // 2
    image_slice = np.rot90(ct[:, :, slice_index])
    mask_slice = np.rot90(mask[:, :, slice_index])
    labels_in_slice = np.unique(mask_slice.astype(np.int16))
    foreground_labels = labels_in_slice[labels_in_slice != 0].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image_slice, cmap="gray", vmin=-150, vmax=250)
    axes[0].set_title("Input CT image")
    axes[0].axis("off")

    axes[1].imshow(image_slice, cmap="gray", vmin=-150, vmax=250)
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[1].imshow(overlay, cmap="turbo", alpha=0.55, vmin=1, vmax=15)
    axes[1].set_title("Segmentation output")
    axes[1].axis("off")
    figure.tight_layout(pad=0.5)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    print("Preview completed")
    print(f"  Input CT: {input_path}")
    print(f"  Segmentation mask: {mask_path}")
    print(f"  CT volume size: {ct.shape} (width, height, slices)")
    print(f"  Displayed slice: {slice_index + 1} of {ct.shape[2]}")
    print(f"  Organ label IDs visible in this slice: {foreground_labels or 'none'}")
    print(f"  Side-by-side image saved to: {output_path}")


def predict(input_path: Path, model_folder: Path, output_path: Path) -> None:
    """Run the nnU-Net prediction command supplied by the trained model."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not model_folder.exists():
        raise FileNotFoundError(model_folder)

    input_dir = output_path / "input"
    result_dir = output_path / "masks"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    # nnU-Net requires the single input channel suffix.
    nnunet_input = input_dir / f"{input_path.name.removesuffix('.nii.gz')}_0000.nii.gz"
    shutil.copy2(input_path, nnunet_input)

    command = [
        "nnUNetv2_predict_from_modelfolder", "-i", str(input_dir),
        "-o", str(result_dir), "-m", str(model_folder), "-f", "0",
        "-chk", "checkpoint_final.pth", "--disable_tta",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Prediction written to {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download exactly three images and labels")
    commands.add_parser("prepare", help="make an nnU-Net dataset folder")
    preview_parser = commands.add_parser("preview", help="save one CT+mask PNG")
    preview_parser.add_argument("--input", type=Path, required=True)
    preview_parser.add_argument("--mask", type=Path, required=True)
    preview_parser.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict", help="predict with a fine-tuned model")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--model-folder", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "download":
        download()
    elif args.command == "prepare":
        prepare()
    elif args.command == "preview":
        preview(args.input, args.mask, args.output)
    else:
        predict(args.input, args.model_folder, args.output)


if __name__ == "__main__":
    main()
