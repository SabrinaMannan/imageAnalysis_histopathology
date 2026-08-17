"""
tests/test_villus.py

Test script for villus candidate detection.

Run from project root:

    python -m tests.test_villus
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.villus import VillusDetector


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = Path(
    r"F:\imageAnalysis_histopathology\data\raw\duodenum_first\BC11007_I-05-16_D101 - 40x.jpg"
)

MAX_DIMENSION = 2500


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path: Path) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"\nImage does not exist:\n{path}\n"
        )

    print("Loading image...")
    print(f"Path: {path}")

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read image:\n{path}"
        )

    # OpenCV BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    print(
        f"Original image shape: {image.shape}"
    )

    return image


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_for_processing(
    image: np.ndarray,
    max_dimension: int = 2500,
):

    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / max(height, width),
    )

    if scale == 1.0:
        return image.copy()

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


# ============================================================
# CREATE TISSUE MASK
# ============================================================

def create_tissue_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Create a simple tissue ROI mask.

    This is only for testing the villus detector.

    It identifies the pink tissue against the pale/white
    background using color saturation.

    Later this can be replaced by the finalized tissue
    segmentation pipeline.
    """

    print("Creating tissue mask...")

    # --------------------------------------------------------
    # Convert RGB → HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[:, :, 1]

    # --------------------------------------------------------
    # Threshold pink tissue
    # --------------------------------------------------------

    # Histology tissue has higher saturation than the
    # pale background.
    _, mask = cv2.threshold(
        saturation,
        20,
        255,
        cv2.THRESH_BINARY,
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Keep large connected tissue regions
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(mask)

    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        # Keep components larger than 0.1% of image.
        minimum_area = (
            image.shape[0]
            * image.shape[1]
            * 0.001
        )

        for label_id in range(
            1,
            num_labels,
        ):

            area = stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]

            if area >= minimum_area:

                cleaned[
                    labels == label_id
                ] = 255

    # --------------------------------------------------------
    # Fill small holes
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    final_mask = np.zeros_like(
        cleaned
    )

    cv2.drawContours(
        final_mask,
        contours,
        -1,
        255,
        thickness=-1,
    )

    print(
        f"Tissue mask shape: "
        f"{final_mask.shape}"
    )

    print(
        "Tissue pixels:",
        int(np.count_nonzero(final_mask)),
    )

    return final_mask


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VILLUS CANDIDATE DETECTION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = load_image(
        IMAGE_PATH
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    print("Resizing image for processing...")

    image_small = resize_for_processing(
        image,
        MAX_DIMENSION,
    )

    print(
        f"Processing image shape: "
        f"{image_small.shape}"
    )

    # --------------------------------------------------------
    # Tissue mask
    # --------------------------------------------------------

    tissue_mask = create_tissue_mask(
        image_small
    )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    print(
        "Running villus candidate detection..."
    )

    detector = VillusDetector(
        min_area=500,
        max_area=500_000,
        min_aspect_ratio=1.5,
        min_solidity=0.20,
        max_candidates=100,
    )

    result = detector.detect(
        image_small,
        tissue_mask,
    )

    candidates = result[
        "candidates"
    ]

    candidate_mask = result[
        "candidate_mask"
    ]

    print(
        f"Candidates detected: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Overlays
    # --------------------------------------------------------

    tissue_overlay = (
        detector.create_tissue_overlay(
            image_small,
            tissue_mask,
        )
    )

    candidate_overlay = (
        detector.create_overlay(
            image_small,
            candidates,
        )
    )

    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    if candidates:

        print("\nCandidate measurements:")
        print("-" * 70)

        for candidate in candidates:

            print(
                f"#{candidate.candidate_id:03d} "
                f"| area={candidate.area_px:.0f} px "
                f"| width={candidate.width} "
                f"| height={candidate.height} "
                f"| aspect={candidate.aspect_ratio:.2f} "
                f"| solidity={candidate.solidity:.2f}"
            )

    else:

        print(
            "\nNo candidates detected."
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    # Original
    axes[0].imshow(
        image_small
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis("off")

    # Tissue ROI
    axes[1].imshow(
        tissue_mask,
        cmap="gray",
    )

    axes[1].set_title(
        "Tissue ROI"
    )

    axes[1].axis("off")

    # Candidate mask
    axes[2].imshow(
        candidate_mask,
        cmap="gray",
    )

    axes[2].set_title(
        "Villus Candidates"
    )

    axes[2].axis("off")

    # Overlay
    axes[3].imshow(
        candidate_overlay
    )

    axes[3].set_title(
        f"Candidate Overlay "
        f"({len(candidates)})"
    )

    axes[3].axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()