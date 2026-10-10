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

# Detector settings. Kept here (not buried in main()) since these
# are the knobs that most directly determine what counts as a
# villus candidate and are the first things to retune against a
# new image scale or magnification.
MIN_AREA = 500
MAX_AREA = 500_000
MIN_ASPECT_RATIO = 1.5
MIN_SOLIDITY = 0.20
MAX_CANDIDATES = 100

# How far (in resized-image pixels) a region must protrude beyond
# the smoothed tissue baseline to count as a villus.
MIN_PROTRUSION_DEPTH = 5.0

# Radius (in resized-image pixels) of the structuring element used
# to compute the smoothed baseline. Roughly: anything narrower than
# this gets treated as "not part of the smooth surface." This scales
# with MAX_DIMENSION -- if you change the resize target or process a
# different magnification, re-check this against the image's
# µm/pixel calibration and expected villus width. Use the radius
# sweep plot in main() to pick a value: this is the value actually
# used by the detector below.
STRUCTURAL_RADIUS = 80

# Radii (in resized-image pixels) to compare side by side when
# tuning STRUCTURAL_RADIUS -- see the sweep plot in main().
RADIUS_SWEEP = [15, 40, 80, 150, 250]

# Optional (x, y, width, height) region, in resized-image pixel
# coordinates, to inspect closely -- e.g. a densely villous area
# where detection looks suspiciously sparse. Leave as None to skip.
# To find coordinates: run the script once, hover the mouse over
# the region of interest in any of the plot windows (the pixel
# coordinates are shown live in the toolbar), and note a bounding
# box around it.
CROP_REGION = None  # e.g. (50, 50, 400, 400)


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
    # Radius sweep: compare candidate structural_radius values
    # before committing to one for full detection. Pick the
    # smallest radius where the opened mask visibly rounds off
    # the villi (loses the fine serration) while still tracking
    # the gross tissue outline.
    # --------------------------------------------------------

    sweep_fig, sweep_axes = plt.subplots(
        1,
        len(RADIUS_SWEEP),
        figsize=(5 * len(RADIUS_SWEEP), 5),
    )

    for ax, radius in zip(sweep_axes, RADIUS_SWEEP):
        opened = VillusDetector.preview_opening(tissue_mask, radius)
        ax.imshow(opened, cmap="gray")
        ax.set_title(f"radius={radius}")
        ax.axis("off")

    sweep_fig.suptitle("STRUCTURAL_RADIUS sweep (opened mask)")
    plt.tight_layout()

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    print(
        "Running villus candidate detection..."
    )

    detector = VillusDetector(
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        min_aspect_ratio=MIN_ASPECT_RATIO,
        min_solidity=MIN_SOLIDITY,
        min_protrusion_depth=MIN_PROTRUSION_DEPTH,
        structural_radius=STRUCTURAL_RADIUS,
        max_candidates=MAX_CANDIDATES,
    )

    result = detector.detect(
        image_small,
        tissue_mask,
        debug=True,
    )

    candidates = result[
        "candidates"
    ]

    candidate_mask = result[
        "candidate_mask"
    ]

    print(
        f"Raw protrusion regions before filtering: "
        f"{result['raw_candidate_count']}"
    )

    print(
        f"Candidates detected (after filtering): "
        f"{len(candidates)}"
    )

    if result["raw_candidate_count"] > 0 and len(candidates) == 0:
        print(
            "All raw protrusions were filtered out -- check "
            "MIN_AREA / MIN_ASPECT_RATIO / MIN_SOLIDITY / "
            "MIN_PROTRUSION_DEPTH against the debug plot below."
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
                f"| solidity={candidate.solidity:.2f} "
                f"| depth={candidate.protrusion_depth_px:.1f} px"
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

    # --------------------------------------------------------
    # Debug plot: inspect the intermediate protrusion-detection
    # steps directly. Use this to tune STRUCTURAL_RADIUS and
    # MIN_PROTRUSION_DEPTH -- if "Opened (smoothed) mask" still
    # closely follows the ragged tissue boundary instead of
    # smoothing over it, STRUCTURAL_RADIUS is too small. If "Raw
    # protrusions" already outlines the villous boundary well but
    # "Villus Candidates" (above) is nearly empty, the problem is
    # the area/aspect/solidity/depth filter thresholds, not the
    # radius.
    # --------------------------------------------------------

    fig2, debug_axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    debug_axes[0].imshow(tissue_mask, cmap="gray")
    debug_axes[0].set_title("Tissue mask (input)")
    debug_axes[0].axis("off")

    debug_axes[1].imshow(result["opened_mask"], cmap="gray")
    debug_axes[1].set_title(
        f"Opened (smoothed) mask\n"
        f"structural_radius={STRUCTURAL_RADIUS}"
    )
    debug_axes[1].axis("off")

    debug_axes[2].imshow(result["raw_protrusion_mask"], cmap="gray")
    debug_axes[2].set_title(
        f"Raw protrusions\n(before area/aspect/depth filters, "
        f"n={result['raw_candidate_count']})"
    )
    debug_axes[2].axis("off")

    depth_display = debug_axes[3].imshow(
        result["depth_map"], cmap="magma"
    )
    debug_axes[3].set_title(
        f"Protrusion depth (px)\nmin_protrusion_depth="
        f"{MIN_PROTRUSION_DEPTH}"
    )
    debug_axes[3].axis("off")
    fig2.colorbar(
        depth_display, ax=debug_axes[3], fraction=0.046, pad=0.04
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Region zoom: closely compare the original image against
    # the tissue mask (and opened/protrusion masks) in one small
    # area. Use this to check whether the tissue mask is already
    # smoothing away real villus structure in a densely folded
    # region -- if the mask boundary there looks noticeably
    # coarser/smoother than the actual tissue in "Original crop",
    # the bottleneck is create_tissue_mask, not the detector's
    # structural_radius/depth settings.
    # --------------------------------------------------------

    if CROP_REGION is not None:

        cx, cy, cw, ch = CROP_REGION

        def crop(arr):
            return arr[cy:cy + ch, cx:cx + cw]

        zoom_fig, zoom_axes = plt.subplots(1, 4, figsize=(20, 6))

        zoom_axes[0].imshow(crop(image_small))
        zoom_axes[0].set_title("Original crop")
        zoom_axes[0].axis("off")

        zoom_axes[1].imshow(crop(tissue_mask), cmap="gray")
        zoom_axes[1].set_title("Tissue mask crop")
        zoom_axes[1].axis("off")

        zoom_axes[2].imshow(crop(result["opened_mask"]), cmap="gray")
        zoom_axes[2].set_title(
            f"Opened mask crop\nstructural_radius={STRUCTURAL_RADIUS}"
        )
        zoom_axes[2].axis("off")

        zoom_axes[3].imshow(crop(candidate_mask), cmap="gray")
        zoom_axes[3].set_title("Villus candidates crop")
        zoom_axes[3].axis("off")

        zoom_fig.suptitle(f"Region zoom: x={cx}, y={cy}, w={cw}, h={ch}")
        plt.tight_layout()

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()