import matplotlib.pyplot as plt

from src.tissue import TissueSegmenter


# =========================================================
# IMAGE PATH
# =========================================================

IMAGE_PATH = (
    "data/raw/duodenum_first/"
    "BC11007_I-05-16_D101 - 40x.jpg"
)


# =========================================================
# MAIN
# =========================================================

def main():

    segmenter = TissueSegmenter()

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = segmenter.load_image(
        IMAGE_PATH
    )

    print("Image loaded.")
    print(
        "Original size:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    # -----------------------------------------------------
    # Segment tissue
    # -----------------------------------------------------

    mask = segmenter.segment(
        image
    )

    print("Tissue segmentation completed.")

    # -----------------------------------------------------
    # Create overlay
    # -----------------------------------------------------

    overlay = segmenter.create_overlay(
        image,
        mask
    )

    # -----------------------------------------------------
    # Resize ONLY for display
    # -----------------------------------------------------

    display_original = (
        segmenter.resize_for_display(
            image
        )
    )

    display_mask = (
        segmenter.resize_for_display(
            mask
        )
    )

    display_overlay = (
        segmenter.resize_for_display(
            overlay
        )
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7)
    )

    axes[0].imshow(
        display_original
    )

    axes[0].set_title(
        "Original"
    )

    axes[1].imshow(
        display_mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Tissue Mask"
    )

    axes[2].imshow(
        display_overlay
    )

    axes[2].set_title(
        "Tissue Overlay"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()