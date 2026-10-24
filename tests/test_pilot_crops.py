
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pyvips


# --------------------------------------------------
# 1. Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET = "duodenum_first"
IMAGE_NAME = "BC11007_I-05-16_D101 - 40x.jpg"

IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / DATASET / IMAGE_NAME
)

PREVIEW_PATH = (
    PROJECT_ROOT / "outputs"
    / "tissue_roi_inspection"
    / "reduced_preview.jpg"
)

CALIBRATION_PATH = PROJECT_ROOT / "outputs" / "calibration.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pilot_crops"

CROP_SIZE = 2048


# --------------------------------------------------
# 2. Candidate centers on the 6 x 6 preview grid
#
# A: upper mucosal edge, near cells 1.2-1.3
# B: central tissue, near cells 2.3-3.4
# C: separate tissue region, near cells 4.4-5.4
#
# Fractions refer to the centers of grid cells.
# These are approximate exploratory locations.
# --------------------------------------------------

CANDIDATES = {
    "A_upper_mucosa": (2.5 / 6, 0.5 / 6),
    "B_central_tissue": (3.5 / 6, 2.5 / 6),
    "C_separate_region": (4.5 / 6, 4.5 / 6),
}


def main():
    for path in (IMAGE_PATH, PREVIEW_PATH, CALIBRATION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 3. Read image calibration
    # --------------------------------------------------

    calibration = pd.read_csv(CALIBRATION_PATH)

    matches = calibration[
        (calibration["dataset"].astype(str).str.strip() == DATASET)
        & (calibration["filename"].astype(str).str.strip() == IMAGE_NAME)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one calibration record; found {len(matches)}"
        )

    um_per_pixel = float(matches.iloc[0]["um_per_pixel"])

    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError("Calibration must be positive and finite.")

    # --------------------------------------------------
    # 4. Read preview dimensions
    # --------------------------------------------------

    preview = cv2.imread(str(PREVIEW_PATH), cv2.IMREAD_COLOR)

    if preview is None:
        raise RuntimeError(f"Could not read preview: {PREVIEW_PATH}")

    preview_height, preview_width = preview.shape[:2]

    # --------------------------------------------------
    # 5. Open original image lazily
    # --------------------------------------------------

    image = pyvips.Image.new_from_file(
        str(IMAGE_PATH),
        access="random",
    )

    original_width = image.width
    original_height = image.height

    print(f"Original image: {original_width} x {original_height}")
    print(f"Preview: {preview_width} x {preview_height}")
    print(f"Calibration: {um_per_pixel:.6f} µm/pixel")
    print(
        f"Crop size: {CROP_SIZE} x {CROP_SIZE} pixels "
        f"({CROP_SIZE * um_per_pixel:.1f} µm per side)"
    )

    # --------------------------------------------------
    # 6. Mark crop positions on preview
    # --------------------------------------------------

    preview_marked = preview.copy()
    records = []
    crop_images = []

    for label, (x_fraction, y_fraction) in CANDIDATES.items():

        # Candidate center in preview coordinates.
        preview_cx = x_fraction * preview_width
        preview_cy = y_fraction * preview_height

        # Map each axis independently to original coordinates.
        original_cx = round(
            preview_cx * original_width / preview_width
        )
        original_cy = round(
            preview_cy * original_height / preview_height
        )

        # Keep the entire crop inside the original image.
        crop_width = min(CROP_SIZE, original_width)
        crop_height = min(CROP_SIZE, original_height)

        left = max(
            0,
            min(original_width - crop_width,
                original_cx - crop_width // 2),
        )
        top = max(
            0,
            min(original_height - crop_height,
                original_cy - crop_height // 2),
        )

        # Extract from the original JPEG, not the preview.
        crop = image.crop(left, top, crop_width, crop_height)

        crop_path = OUTPUT_DIR / f"{label}.jpg"
        crop.jpegsave(str(crop_path), Q=95)

        # Mark the approximate crop footprint on the preview.
        x1 = round(left * preview_width / original_width)
        y1 = round(top * preview_height / original_height)
        x2 = round(
            (left + crop_width) * preview_width / original_width
        )
        y2 = round(
            (top + crop_height) * preview_height / original_height
        )

        cv2.rectangle(
            preview_marked,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            preview_marked,
            label.split("_")[0],
            (x1 + 5, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        records.append({
            "candidate": label,
            "original_x": left,
            "original_y": top,
            "width_px": crop.width,
            "height_px": crop.height,
            "um_per_pixel": um_per_pixel,
            "width_um": crop.width * um_per_pixel,
            "height_um": crop.height * um_per_pixel,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "original_width": original_width,
            "original_height": original_height,
        })

        # Contact sheet thumbnails are for inspection only.
        crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)

        if crop_bgr is None:
            raise RuntimeError(f"Could not reopen crop: {crop_path}")

        thumbnail = cv2.resize(
            crop_bgr,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )

        cv2.putText(
            thumbnail,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        crop_images.append(thumbnail)

        print(
            f"{label}: origin=({left}, {top}), "
            f"size={crop.width}x{crop.height}"
        )

    # --------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------

    preview_output = OUTPUT_DIR / "crop_locations_on_preview.jpg"

    if not cv2.imwrite(str(preview_output), preview_marked):
        raise RuntimeError(f"Could not save: {preview_output}")

    manifest_path = OUTPUT_DIR / "crop_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    contact_sheet = np.vstack([
        np.hstack(crop_images[:2]),
        np.hstack([crop_images[2], np.full_like(crop_images[2], 245)]),
    ])

    contact_path = OUTPUT_DIR / "pilot_contact_sheet.jpg"

    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError(f"Could not save: {contact_path}")

    print("\nPilot extraction complete.")
    print(f"Marked preview: {preview_output}")
    print(f"Contact sheet: {contact_path}")
    print(f"Coordinate manifest: {manifest_path}")
    print("These crops are exploratory; anatomical suitability is not yet established.")


if __name__ == "__main__":
    main()
