
from pathlib import Path

import pandas as pd
from PIL import Image
import yaml


# Project root = parent of the experiments folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
calibration_csv = PROJECT_ROOT / "outputs" / "calibration.csv"

# Inspect only the first image in duodenum_first.
image_dir = raw_dir / "duodenum_first"

if not image_dir.is_dir():
    raise FileNotFoundError(f"Image folder not found: {image_dir}")

if not calibration_csv.is_file():
    raise FileNotFoundError(
        f"Calibration CSV not found: {calibration_csv}"
    )

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise FileNotFoundError(f"No supported images found in {image_dir}")

image_path = images[0]

# PIL reads image dimensions without loading all pixel data.
import cv2

image_info = cv2.imread(
    str(image_path),
    cv2.IMREAD_REDUCED_COLOR_8
)

if image_info is None:
    raise ValueError(f"Could not read image: {image_path}")

height, width = image_info.shape[:2]
image_format = image_path.suffix.lower().lstrip(".")

del image_info

pixels = width * height
estimated_gib = pixels * 3 / (1024 ** 3)

calibration = pd.read_csv(calibration_csv)

print("\n--- ONE-IMAGE PREFLIGHT ---")
print(f"Image: {image_path}")
print(f"Format: {image_format}")
print(f"Dimensions: {width:,} x {height:,} pixels")
print(f"Pixel count: {pixels:,}")
print(f"Approx. memory for one 3-channel image: {estimated_gib:.2f} GiB")
print("\nThis estimate excludes additional image copies and processing arrays.")

print("\n--- CALIBRATION CHECK ---")
print(f"Calibration CSV: {calibration_csv}")
print(f"Calibration columns: {list(calibration.columns)}")

if "filename" not in calibration.columns:
    print("No 'filename' column found; check the CSV schema.")
else:
    matches = calibration[
        calibration["filename"].astype(str) == image_path.name
    ]

    if matches.empty:
        print(f"No calibration row found for {image_path.name}")
    else:
        print(matches.to_string(index=False))

print("\nPreflight complete. No image pixels were processed or modified.")
