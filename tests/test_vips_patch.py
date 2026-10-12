
from pathlib import Path
import pyvips

# Project root: F:\imageAnalysis_histopathology
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "duodenum_first"
    / "BC11007_I-05-16_D101 - 40x.jpg"
)

PATCH_SIZE = 512

def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Open lazily: do not convert the entire slide into a NumPy array.
    image = pyvips.Image.new_from_file(
        str(IMAGE_PATH),
        access="random",
    )

    print(f"Image: {IMAGE_PATH.name}")
    print(f"Original dimensions: {image.width} x {image.height}")
    print(f"Bands: {image.bands}")
    print(f"Format: {image.format}")

    # Test a patch from the top-left corner.
    # This is a technical test, not a selected tissue measurement region.
    width = min(PATCH_SIZE, image.width)
    height = min(PATCH_SIZE, image.height)

    patch = image.crop(0, 0, width, height)

    output_dir = PROJECT_ROOT / "outputs" / "patch_reading_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "patch_0000.jpg"
    patch.jpegsave(str(output_path), Q=95)

    print(f"Patch dimensions: {patch.width} x {patch.height}")
    print(f"Saved patch: {output_path}")
    print("Patch-reading test completed.")

if __name__ == "__main__":
    main()
