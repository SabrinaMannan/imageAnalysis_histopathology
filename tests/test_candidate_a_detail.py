
from pathlib import Path

import cv2
import pyvips


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "duodenum_first"
    / "BC11007_I-05-16_D101 - 40x.jpg"
)

PREVIEW_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tissue_roi_inspection"
    / "reduced_preview.jpg"
)

# NEW folder: leaves all previous results untouched.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "candidate_a_detail_check2"

CROP_SIZE = 2048
UM_PER_PIXEL = 0.180327


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    preview = cv2.imread(str(PREVIEW_PATH))
    if preview is None:
        raise FileNotFoundError(PREVIEW_PATH)

    preview_h, preview_w = preview.shape[:2]

    # Center of grid cell row 1, column 2 in a 6 x 6 grid.
    preview_cx = (1.5 / 6) * preview_w
    preview_cy = (0.5 / 6) * preview_h

    image = pyvips.Image.new_from_file(
        str(IMAGE_PATH),
        access="random",
    )

    original_w, original_h = image.width, image.height

    center_x = round(preview_cx * original_w / preview_w)
    center_y = round(preview_cy * original_h / preview_h)

    left = max(0, min(original_w - CROP_SIZE, center_x - CROP_SIZE // 2))
    top = max(0, min(original_h - CROP_SIZE, center_y - CROP_SIZE // 2))

    crop = image.crop(left, top, CROP_SIZE, CROP_SIZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crop_path = OUTPUT_DIR / "candidate_a_detail_check2.jpg"
    crop.jpegsave(str(crop_path), Q=98)

    marked = preview.copy()

    x1 = round(left * preview_w / original_w)
    y1 = round(top * preview_h / original_h)
    x2 = round((left + CROP_SIZE) * preview_w / original_w)
    y2 = round((top + CROP_SIZE) * preview_h / original_h)

    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 3)

    location_path = OUTPUT_DIR / "candidate_a_location_check2.jpg"
    cv2.imwrite(str(location_path), marked)

    print(f"Original image dimensions: {original_w} x {original_h}")
    print(f"Crop origin: x={left}, y={top}")
    print(f"Crop size: {CROP_SIZE} x {CROP_SIZE} pixels")
    print(f"Approximate physical span: {CROP_SIZE * UM_PER_PIXEL:.1f} µm per side")
    print(f"Detail crop: {crop_path}")
    print(f"Location preview: {location_path}")


if __name__ == "__main__":
    main()
