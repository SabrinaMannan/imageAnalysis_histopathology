
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pyvips


# --------------------------------------------------
# 1. Project paths and existing tissue segmenter
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tissue import TissueSegmenter


DATASET = "duodenum_first"
IMAGE_NAME = "BC11007_I-05-16_D101 - 40x.jpg"
PATCH_SIZE = 512
CANDIDATE_SIZE = 128  # Candidate window on the tissue-mask grid
MAX_PATCHES = 12
MIN_MASK_COVERAGE = 0.70

IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / DATASET / IMAGE_NAME
)
CALIBRATION_PATH = PROJECT_ROOT / "outputs" / "calibration.csv"
MASK_PATH = (
    PROJECT_ROOT / "outputs" / "tissue_roi_inspection"
    / "tissue_mask.png"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tissue_guided_patches"


def main():
    for path in (IMAGE_PATH, CALIBRATION_PATH, MASK_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Read and validate calibration
    # --------------------------------------------------

    calibration = pd.read_csv(CALIBRATION_PATH)

    matches = calibration[
        (calibration["dataset"].astype(str).str.strip() == DATASET)
        & (calibration["filename"].astype(str).str.strip() == IMAGE_NAME)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one calibration record; found {len(matches)}"
        )

    record = matches.iloc[0]
    um_per_pixel = float(record["um_per_pixel"])

    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError("Invalid um_per_pixel calibration value.")

    # --------------------------------------------------
    # 3. Open original JPEG lazily
    # --------------------------------------------------

    image = pyvips.Image.new_from_file(
        str(IMAGE_PATH),
        access="random",
    )

    original_width = image.width
    original_height = image.height

    print(f"Original image: {original_width} x {original_height}")
    print(f"Calibration: {um_per_pixel:.6f} µm/pixel")
    print(
        f"A {PATCH_SIZE}-pixel patch spans "
        f"{PATCH_SIZE * um_per_pixel:.2f} µm per side"
    )

    # --------------------------------------------------
    # 4. Load the existing tissue mask
    # --------------------------------------------------

    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"Could not read tissue mask: {MASK_PATH}")

    mask_height, mask_width = mask.shape
    tissue = (mask > 0).astype(np.uint8)

    print(f"Tissue-mask grid: {mask_width} x {mask_height}")
    print(f"Mask coverage: {tissue.mean():.1%}")

    # --------------------------------------------------
    # 5. Find candidate windows on the mask grid
    # --------------------------------------------------

    # Use integral image to calculate coverage efficiently.
    integral = cv2.integral(tissue)

    window_w = min(CANDIDATE_SIZE, mask_width)
    window_h = min(CANDIDATE_SIZE, mask_height)
    window_area = window_w * window_h

    candidates = []

    # Stride avoids testing every possible pixel location.
    stride_x = max(1, window_w // 2)
    stride_y = max(1, window_h // 2)

    for y in range(0, mask_height - window_h + 1, stride_y):
        for x in range(0, mask_width - window_w + 1, stride_x):
            tissue_pixels = (
                integral[y + window_h, x + window_w]
                - integral[y, x + window_w]
                - integral[y + window_h, x]
                + integral[y, x]
            )

            coverage = float(tissue_pixels) / window_area

            if coverage >= MIN_MASK_COVERAGE:
                candidates.append((coverage, x, y))

    candidates.sort(reverse=True)

    if not candidates:
        raise RuntimeError(
            "No candidate windows met the tissue-coverage threshold. "
            "Inspect the existing mask before changing the threshold."
        )

    # --------------------------------------------------
    # 6. Choose spatially separated candidates
    # --------------------------------------------------

    selected = []

    # Minimum separation is measured on the mask grid.
    min_distance = min(window_w, window_h) * 0.75

    for coverage, x, y in candidates:
        cx = x + window_w / 2
        cy = y + window_h / 2

        if all(
            (cx - sx) ** 2 + (cy - sy) ** 2 >= min_distance ** 2
            for _, sx, sy, _, _ in selected
        ):
            selected.append((coverage, cx, cy, x, y))

        if len(selected) >= MAX_PATCHES:
            break

    # --------------------------------------------------
    # 7. Map mask coordinates to original coordinates
    #    and extract original-resolution patches
    # --------------------------------------------------

    patch_paths = []
    manifest = []

    for index, (coverage, cx, cy, _, _) in enumerate(selected, start=1):
        # Scale each axis independently using actual dimensions.
        original_cx = int(round(cx * original_width / mask_width))
        original_cy = int(round(cy * original_height / mask_height))

        # Keep the complete patch inside the image.
        left = max(0, min(original_width - PATCH_SIZE,
                          original_cx - PATCH_SIZE // 2))
        top = max(0, min(original_height - PATCH_SIZE,
                         original_cy - PATCH_SIZE // 2))

        patch = image.crop(left, top, PATCH_SIZE, PATCH_SIZE)

        patch_path = OUTPUT_DIR / f"patch_{index:02d}.jpg"
        patch.jpegsave(str(patch_path), Q=95)
        patch_paths.append(patch_path)

        manifest.append({
            "patch": patch_path.name,
            "mask_coverage": coverage,
            "original_x": left,
            "original_y": top,
            "patch_width_px": patch.width,
            "patch_height_px": patch.height,
            "um_per_pixel": um_per_pixel,
            "patch_width_um": patch.width * um_per_pixel,
            "patch_height_um": patch.height * um_per_pixel,
        })

        print(
            f"{patch_path.name}: coverage={coverage:.1%}, "
            f"origin=({left}, {top}), "
            f"size={patch.width}x{patch.height}"
        )

    # Save candidate coordinates and calibration for traceability.
    manifest_path = OUTPUT_DIR / "patch_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    # --------------------------------------------------
    # 8. Create a contact sheet for visual quality control
    # --------------------------------------------------

    tiles = []

    for index, patch_path in enumerate(patch_paths, start=1):
        patch = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)

        if patch is None:
            continue

        # Resize only for the contact sheet, never for measurement.
        tile = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"Patch {index:02d}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if tiles:
        columns = 3
        rows = (len(tiles) + columns - 1) // columns
        sheet = np.full(
            (rows * 256, columns * 256, 3),
            245,
            dtype=np.uint8,
        )

        for i, tile in enumerate(tiles):
            row, col = divmod(i, columns)
            sheet[
                row * 256:(row + 1) * 256,
                col * 256:(col + 1) * 256,
            ] = tile

        sheet_path = OUTPUT_DIR / "contact_sheet.jpg"

        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"Could not save contact sheet: {sheet_path}")

        print(f"Contact sheet: {sheet_path}")

    print(f"Patch manifest: {manifest_path}")
    print(f"Saved {len(patch_paths)} original-resolution patches.")
    print(
        "These are ROI-guided candidate patches only, "
        "not validated anatomical measurement regions."
    )


if __name__ == "__main__":
    main()
