
from pathlib import Path

import pandas as pd
from PIL import Image
import yaml


# --------------------------------------------------
# 1. Locate the project and read configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

config_path = PROJECT_ROOT / "config.yaml"

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
image_dir = raw_dir / "duodenum_first"
calibration_csv = PROJECT_ROOT / config["data"]["output_dir"] / "calibration.csv"


# --------------------------------------------------
# 2. Validate paths
# --------------------------------------------------

if not image_dir.is_dir():
    raise FileNotFoundError(
        f"Image folder not found: {image_dir}"
    )

if not calibration_csv.is_file():
    raise FileNotFoundError(
        f"Calibration CSV not found: {calibration_csv}"
    )


# --------------------------------------------------
# 3. Select one image only
# --------------------------------------------------

supported_formats = {".jpg", ".jpeg", ".png"}

images = sorted(
    path
    for path in image_dir.iterdir()
    if path.is_file()
    and path.suffix.lower() in supported_formats
)

if not images:
    raise FileNotFoundError(
        f"No supported images found in {image_dir}"
    )

image_path = images[0]


# --------------------------------------------------
# 4. Read original dimensions without decoding pixels
# --------------------------------------------------

previous_limit = Image.MAX_IMAGE_PIXELS

try:
    # This trusted local image is inspected for metadata only.
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format

finally:
    # Restore Pillow's normal safety limit.
    Image.MAX_IMAGE_PIXELS = previous_limit


pixels = width * height
estimated_gib = pixels * 3 / (1024 ** 3)


# --------------------------------------------------
# 5. Load calibration records
# --------------------------------------------------

calibration = pd.read_csv(calibration_csv)

print("\n--- ONE-IMAGE PREFLIGHT ---")
print(f"Image: {image_path}")
print(f"Format: {image_format}")
print(f"Original dimensions: {width:,} x {height:,} pixels")
print(f"Pixel count: {pixels:,}")
print(
    "Estimated memory for one RGB image: "
    f"{estimated_gib:.2f} GiB"
)
print(
    "Note: processing may require substantially more memory."
)


# --------------------------------------------------
# 6. Match the image to its calibration record
# --------------------------------------------------

print("\n--- CALIBRATION CHECK ---")
print(f"Calibration CSV: {calibration_csv}")
print(f"Calibration columns: {list(calibration.columns)}")

required_columns = {
    "dataset",
    "filename",
    "scale_bar_um",
    "scale_bar_pixels",
    "um_per_pixel",
    "pixels_per_um",
}

missing_columns = required_columns - set(calibration.columns)

if missing_columns:
    print(
        "WARNING: Missing expected columns: "
        f"{sorted(missing_columns)}"
    )

elif "filename" in calibration.columns:
    matches = calibration[
        calibration["filename"].astype(str).str.strip()
        == image_path.name
    ]

    # Also confirm the dataset folder where possible.
    if "dataset" in calibration.columns:
        matches = matches[
            matches["dataset"].astype(str).str.strip()
            == image_dir.name
        ]

    if matches.empty:
        print(
            f"NO MATCH for {image_path.name} "
            f"in dataset {image_dir.name}"
        )
    else:
        print("Matching calibration record:")
        print(matches.to_string(index=False))

        if len(matches) > 1:
            print(
                "WARNING: Multiple matching records found. "
                "Resolve duplicates before using calibration."
            )


# --------------------------------------------------
# 7. Finish
# --------------------------------------------------

print("\nPreflight complete.")
print("No image pixels were decoded, processed, or modified.")
