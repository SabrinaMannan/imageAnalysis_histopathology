from pathlib import Path
import matplotlib.pyplot as plt

from src.tissue import TissueSegmenter


# =========================================================
# FIND PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILENAME = "BC11007_I-05-16_D201 - 40x.jpg"


# =========================================================
# SEARCH FOR IMAGE
# =========================================================

matches = list(
    PROJECT_ROOT.rglob(FILENAME)
)

if not matches:
    raise FileNotFoundError(
        f"\nCould not find:\n{FILENAME}\n\n"
        f"Searched inside:\n{PROJECT_ROOT}\n\n"
        "Check that the image is inside the project folder."
    )

IMAGE_PATH = matches[0]

print("\nImage found:")
print(IMAGE_PATH)


# =========================================================
# CREATE SEGMENTER
# =========================================================

segmenter = TissueSegmenter(
    max_processing_size=2500,
    blur_size=31,
    close_kernel_size=101,
    min_component_fraction=0.002
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = segmenter.load_image(
    IMAGE_PATH
)

print(
    "Original image size:",
    image.shape[1],
    "x",
    image.shape[0]
)


# =========================================================
# SEGMENT
# =========================================================

working_image, mask = segmenter.segment(
    image,
    return_working_mask=True
)


print(
    "Working image size:",
    working_image.shape[1],
    "x",
    working_image.shape[0]
)


# =========================================================
# CREATE OVERLAY
# =========================================================

overlay = segmenter.create_overlay(
    working_image,
    mask
)


# =========================================================
# DISPLAY
# =========================================================

plt.figure(
    figsize=(15, 5)
)


plt.subplot(1, 3, 1)

plt.imshow(
    working_image
)

plt.title(
    "Original"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    mask,
    cmap="gray"
)

plt.title(
    "Tissue Mask"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    overlay
)

plt.title(
    "Tissue Overlay"
)

plt.axis("off")


plt.tight_layout()

plt.show()