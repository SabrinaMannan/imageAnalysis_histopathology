
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class PatchExtractor:
    """
    Extract patches from large histopathology images.

    The original image is cropped without resizing.
    A downsampled tissue mask is used to estimate how much
    tissue each patch contains.

    This class does NOT identify villi or crypts.
    """

    def __init__(
        self,
        patch_size=512,
        stride=512,
        min_tissue_fraction=0.60,
    ):
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive.")

        if not 0 <= min_tissue_fraction <= 1:
            raise ValueError("min_tissue_fraction must be between 0 and 1.")

        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_fraction = min_tissue_fraction

    def extract_from_image(
        self,
        image_path,
        tissue_mask,
        output_dir,
        dataset_name,
    ):
        """
        Extract original-resolution patches using a tissue mask.

        tissue_mask may be smaller than the original image.
        Coordinates are mapped between the two resolutions.
        """

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_h, image_w = image.shape[:2]
        mask_h, mask_w = tissue_mask.shape[:2]

        if mask_h == 0 or mask_w == 0:
            raise ValueError("The tissue mask is empty.")

        if not np.any(tissue_mask > 0):
            raise ValueError("The tissue mask contains no tissue pixels.")

        records = []
        patch_number = 0

        for y in range(0, image_h - self.patch_size + 1, self.stride):
            for x in range(0, image_w - self.patch_size + 1, self.stride):

                # Map original-image coordinates to mask coordinates.
                mx1 = round(x * mask_w / image_w)
                my1 = round(y * mask_h / image_h)

                mx2 = round(
                    (x + self.patch_size) * mask_w / image_w
                )
                my2 = round(
                    (y + self.patch_size) * mask_h / image_h
                )

                mx1 = max(0, min(mx1, mask_w))
                my1 = max(0, min(my1, mask_h))
                mx2 = max(mx1 + 1, min(mx2, mask_w))
                my2 = max(my1 + 1, min(my2, mask_h))

                mask_crop = tissue_mask[my1:my2, mx1:mx2]

                if mask_crop.size == 0:
                    continue

                tissue_fraction = float(
                    np.count_nonzero(mask_crop > 0) / mask_crop.size
                )

                if tissue_fraction < self.min_tissue_fraction:
                    continue

                # Crop the original image; do not resize it.
                patch = image[
                    y:y + self.patch_size,
                    x:x + self.patch_size,
                ]

                if (
                    patch.shape[0] != self.patch_size
                    or patch.shape[1] != self.patch_size
                ):
                    continue

                patch_number += 1

                filename = (
                    f"{dataset_name}_{image_path.stem}_"
                    f"x{x}_y{y}.png"
                )
                output_path = output_dir / filename

                # OpenCV expects BGR when writing.
                patch_bgr = cv2.cvtColor(
                    patch, cv2.COLOR_RGB2BGR
                )

                if not cv2.imwrite(str(output_path), patch_bgr):
                    raise IOError(
                        f"Could not save patch: {output_path}"
                    )

                records.append({
                    "dataset": dataset_name,
                    "source_image": image_path.name,
                    "patch_filename": filename,
                    "x_px": x,
                    "y_px": y,
                    "patch_size_px": self.patch_size,
                    "tissue_fraction": tissue_fraction,
                })

        return records


def save_patch_metadata(records, output_csv):
    """Save patch coordinates and QC metadata to a CSV."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset",
        "source_image",
        "patch_filename",
        "x_px",
        "y_px",
        "patch_size_px",
        "tissue_fraction",
    ]

    pd.DataFrame(records, columns=columns).to_csv(
        output_csv, index=False
    )
