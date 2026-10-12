
from pathlib import Path
import sys

import cv2
import pandas as pd
from PIL import Image


# --------------------------------------------------
# 1. Locate project and import existing segmenter
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make project imports work when this script is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tissue import TissueSegmenter


# --------------------------------------------------
# 2. Select the exact preflight image
# --------------------------------------------------

dataset_name = "duodenum_first"

image_name = "BC11007_I-05-16_D101 - 40x.jpg"

image_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / dataset_name
    / image_name
)

calibration_csv = (
    PROJECT_ROOT
    / "outputs"
    / "calibration.csv"
)

output_dir = (
    PROJECT_ROOT
    / "outputs"
    / "tissue_roi_inspection"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

if not image_path.is_file():
    raise FileNotFoundError(
        f"Selected image not found: {image_path}"
    )

if not calibration_csv.is_file():
    raise FileNotFoundError(
        f"Calibration CSV not found: {calibration_csv}"
    )


# --------------------------------------------------
# 3. Read original dimensions from image metadata
#    without decoding the full pixel array
# --------------------------------------------------

previous_limit = Image.MAX_IMAGE_PIXELS

try:
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as image:
        original_width, original_height = image.size

finally:
    Image.MAX_IMAGE_PIXELS = previous_limit

# --------------------------------------------------
# 4. Verify calibration record
# --------------------------------------------------

calibration = pd.read_csv(calibration_csv)

required_columns = {
    "dataset",
    "filename",
    "scale_bar_um",
    "scale_bar_pixels",
    "um_per_pixel",
    "pixels_per_um",
}

missing = required_columns - set(calibration.columns)

if missing:
    raise ValueError(
        f"Calibration CSV is missing columns: {sorted(missing)}"
    )

matches = calibration[
    (calibration["dataset"].astype(str).str.strip()
     == dataset_name)
    &
    (calibration["filename"].astype(str).str.strip()
     == image_name)
]

if len(matches) != 1:
    raise ValueError(
        "Expected exactly one calibration record for this image; "
        f"found {len(matches)}. Check calibration.csv."
    )

record = matches.iloc[0]

um_per_pixel = float(record["um_per_pixel"])
pixels_per_um = float(record["pixels_per_um"])

if (
    not (um_per_pixel > 0)
    or not (pixels_per_um > 0)
):
    raise ValueError(
        "Calibration values must be positive."
    )

print("\n--- CALIBRATION ---")
print(f"Scale bar: {record['scale_bar_um']} µm")
print(f"Measured bar: {record['scale_bar_pixels']} pixels")
print(f"µm per pixel: {um_per_pixel:.6f}")
print(f"Pixels per µm: {pixels_per_um:.5f}")


# --------------------------------------------------
# 5. Decode JPEG at reduced resolution
# --------------------------------------------------

print("\n--- REDUCED-RESOLUTION READ ---")

reduced_bgr = cv2.imread(
    str(image_path),
    cv2.IMREAD_REDUCED_COLOR_8
)

if reduced_bgr is None:
    raise RuntimeError(
        "OpenCV could not decode the image at reduced resolution."
    )

reduced_height, reduced_width = reduced_bgr.shape[:2]

reduced_pixels = reduced_width * reduced_height

print(
    f"Reduced dimensions: "
    f"{reduced_width:,} x {reduced_height:,}"
)

print(f"Reduced pixel count: {reduced_pixels:,}")

print(
    "Approximate RGB array size: "
    f"{reduced_pixels * 3 / (1024 ** 2):.1f} MiB"
)

print(
    "Width reduction factor: "
    f"{original_width / reduced_width:.2f}x"
)

print(
    "Height reduction factor: "
    f"{original_height / reduced_height:.2f}x"
)

reduced_rgb = cv2.cvtColor(
    reduced_bgr,
    cv2.COLOR_BGR2RGB
)

del reduced_bgr


# --------------------------------------------------
# 6. Run the existing tissue ROI segmenter
# --------------------------------------------------

print("\n--- TISSUE SEGMENTATION ---")

segmenter = TissueSegmenter(
    max_processing_size=2500,
    blur_size=31,
    close_kernel_size=101,
    min_component_fraction=0.002,
)

working_image, tissue_mask = segmenter.segment(
    reduced_rgb,
    return_working_mask=True
)

if working_image.shape[:2] != tissue_mask.shape[:2]:
    raise RuntimeError(
        "Working image and tissue mask dimensions do not match."
    )

tissue_fraction = (
    (tissue_mask > 0).sum()
    / tissue_mask.size
)

print(
    f"Working dimensions: "
    f"{working_image.shape[1]:,} x "
    f"{working_image.shape[0]:,}"
)

print(f"Tissue-mask coverage: {tissue_fraction:.1%}")


# --------------------------------------------------
# 7. Save inspection images
# --------------------------------------------------

mask_path = output_dir / "tissue_mask.png"
overlay_path = output_dir / "tissue_overlay.png"
preview_path = output_dir / "reduced_preview.jpg"

overlay_rgb = segmenter.create_overlay(
    working_image,
    tissue_mask,
    alpha=0.30,
)

# OpenCV writes color images in BGR order.
if not cv2.imwrite(
    str(mask_path),
    tissue_mask
):
    raise RuntimeError(f"Could not save: {mask_path}")

if not cv2.imwrite(
    str(overlay_path),
    cv2.cvtColor(
        overlay_rgb,
        cv2.COLOR_RGB2BGR
    )
):
    raise RuntimeError(f"Could not save: {overlay_path}")

if not cv2.imwrite(
    str(preview_path),
    cv2.cvtColor(
        working_image,
        cv2.COLOR_RGB2BGR
    )
):
    raise RuntimeError(f"Could not save: {preview_path}")


# --------------------------------------------------
# 8. Report
# --------------------------------------------------

print("\n--- INSPECTION COMPLETE ---")
print(f"Preview: {preview_path}")
print(f"Mask:    {mask_path}")
print(f"Overlay: {overlay_path}")

print(
    "\nThis experiment inspected a reduced-resolution image."
)

print(
    "The mask is a preliminary tissue ROI, not a villus/crypt "
    "segmentation or a validated anatomical measurement."
)

print(
    "Do not use these reduced-resolution images to calculate "
    "villus height, crypt depth, or VH:CD."
)
