#!/usr/bin/env python3
"""Download AMOS22 examples, export viewable images, preview masks, or predict."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

REPOSITORY = "MedOtter/amos22-ct-dataset"
REVISION = "main"
CASES = ("amos_0001", "amos_0004", "amos_0005")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
NNUNET_DATASET = ROOT / "data" / "nnUNet_raw" / "Dataset901_AMOS22_Small"
DEFAULT_INPUT_DIR = RAW / "train" / "imagesTr"
DEFAULT_MASK_DIR = RAW / "train" / "labelsTr"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


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


def is_nifti(path: Path) -> bool:
    """Return true for .nii and .nii.gz files."""
    return path.suffix == ".nii" or path.name.endswith(".nii.gz")


def normalize_ct_slice(image_slice: np.ndarray) -> np.ndarray:
    """Convert CT values into an 8-bit image that normal viewers can open."""
    clipped = np.clip(image_slice, -150, 250)
    normalized = (clipped + 150) / 400
    return (normalized * 255).astype(np.uint8)


def load_nifti_slice(input_path: Path, slice_index: int | None = None) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Load one 2D slice from a 3D NIfTI CT volume."""
    volume = nib.load(str(input_path)).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape {volume.shape}")
    selected_slice = volume.shape[2] // 2 if slice_index is None else slice_index
    if selected_slice < 0 or selected_slice >= volume.shape[2]:
        raise ValueError(f"Slice must be between 0 and {volume.shape[2] - 1}")
    image_slice = np.rot90(volume[:, :, selected_slice])
    return image_slice, volume.shape, selected_slice


def load_viewable_image(input_path: Path) -> np.ndarray:
    """Load a normal PNG/JPG image as a grayscale array."""
    with Image.open(input_path) as image:
        return np.asarray(image.convert("L"))


def export_image(input_path: Path, output_path: Path, slice_index: int | None = None) -> None:
    """Convert one CT slice from .nii.gz into a normal PNG or JPG image."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not is_nifti(input_path):
        raise ValueError("export-image expects a .nii or .nii.gz input file.")
    if output_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Use a .png, .jpg, or .jpeg output file.")

    image_slice, volume_shape, selected_slice = load_nifti_slice(input_path, slice_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_ct_slice(image_slice)).save(output_path)
    print("Image export completed")
    print(f"  Input CT volume: {input_path}")
    print(f"  CT volume size: {volume_shape} (width, height, slices)")
    print(f"  Exported slice: {selected_slice + 1} of {volume_shape[2]}")
    print(f"  Viewable image saved to: {output_path}")


def find_nifti_files(input_dir: Path) -> list[Path]:
    """Find .nii and .nii.gz files in one folder."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and is_nifti(path))


def preview(input_path: Path, mask_path: Path, output_path: Path, quiet: bool = False) -> None:
    """Save a side-by-side CT input and segmentation-overlay preview."""
    # This script writes a PNG; it does not need a graphical desktop window.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    if is_nifti(input_path):
        image_slice, input_shape, slice_index = load_nifti_slice(input_path)
        display_slice = normalize_ct_slice(image_slice)
        if input_shape != mask.shape:
            raise ValueError(f"Different shapes: CT {input_shape}, mask {mask.shape}")
    elif input_path.suffix.lower() in IMAGE_SUFFIXES:
        display_slice = load_viewable_image(input_path)
        input_shape = display_slice.shape
        slice_index = mask.shape[2] // 2
    else:
        raise ValueError("Input must be .nii, .nii.gz, .png, .jpg, or .jpeg")

    mask_slice = np.rot90(mask[:, :, slice_index])
    labels_in_slice = np.unique(mask_slice.astype(np.int16))
    foreground_labels = labels_in_slice[labels_in_slice != 0].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(display_slice, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input CT image")
    axes[0].axis("off")

    axes[1].imshow(display_slice, cmap="gray", vmin=0, vmax=255)
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[1].imshow(overlay, cmap="turbo", alpha=0.55, vmin=1, vmax=15)
    axes[1].set_title("Segmentation output")
    axes[1].axis("off")
    figure.tight_layout(pad=0.5)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    if not quiet:
        print("Preview completed")
        print(f"  Input CT: {input_path}")
        print(f"  Segmentation mask: {mask_path}")
        print(f"  Input size: {input_shape}")
        print(f"  Displayed mask slice: {slice_index + 1} of {mask.shape[2]}")
        print(f"  Organ label IDs visible in this slice: {foreground_labels or 'none'}")
        print(f"  Side-by-side image saved to: {output_path}")


def run_demo(
    input_dir: Path = DEFAULT_INPUT_DIR,
    mask_dir: Path = DEFAULT_MASK_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    slice_index: int | None = None,
) -> None:
    """Convert every input CT in a folder to PNG and create side-by-side outputs."""
    input_files = find_nifti_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {input_dir}")

    converted_dir = output_dir / "converted_png"
    preview_dir = output_dir / "side_by_side"
    converted_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("Running full demo")
    print(f"  Input folder: {input_dir}")
    print(f"  Mask folder: {mask_dir}")
    print(f"  Output folder: {output_dir}")
    print(f"  CT files found: {len(input_files)}")

    completed = 0
    for input_path in input_files:
        case_name = input_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        mask_path = mask_dir / f"{case_name}.nii.gz"
        png_path = converted_dir / f"{case_name}.png"
        preview_path = preview_dir / f"{case_name}_input_vs_output.png"

        print(f"\nProcessing {case_name}")
        export_image(input_path, png_path, slice_index)
        if mask_path.exists():
            preview(png_path, mask_path, preview_path, quiet=True)
            print(f"  Side-by-side output saved to: {preview_path}")
        else:
            print(f"  No matching mask found, skipped side-by-side output: {mask_path}")
        completed += 1

    print("\nDemo completed")
    print(f"  Converted PNG files: {converted_dir}")
    print(f"  Side-by-side outputs: {preview_dir}")
    print(f"  Files processed: {completed}")


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
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run-demo", help="convert every CT input and make side-by-side outputs")
    run_parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    run_parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--slice", type=int, default=None, help="0-based slice number; default is the middle slice")
    commands.add_parser("download", help="download exactly three images and labels")
    commands.add_parser("prepare", help="make an nnU-Net dataset folder")
    export_parser = commands.add_parser("export-image", help="convert one CT slice to PNG/JPG")
    export_parser.add_argument("--input", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--slice", type=int, default=None, help="0-based slice number; default is the middle slice")
    preview_parser = commands.add_parser("preview", help="save one CT+mask PNG")
    preview_parser.add_argument("--input", type=Path, required=True)
    preview_parser.add_argument("--mask", type=Path, required=True)
    preview_parser.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict", help="predict with a fine-tuned model")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--model-folder", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command is None or args.command == "run-demo":
        run_demo(
            input_dir=getattr(args, "input_dir", DEFAULT_INPUT_DIR),
            mask_dir=getattr(args, "mask_dir", DEFAULT_MASK_DIR),
            output_dir=getattr(args, "output_dir", DEFAULT_OUTPUT_DIR),
            slice_index=getattr(args, "slice", None),
        )
    elif args.command == "download":
        download()
    elif args.command == "prepare":
        prepare()
    elif args.command == "export-image":
        export_image(args.input, args.output, args.slice)
    elif args.command == "preview":
        preview(args.input, args.mask, args.output)
    else:
        predict(args.input, args.model_folder, args.output)


if __name__ == "__main__":
    main()
