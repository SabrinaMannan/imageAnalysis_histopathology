from pathlib import Path

import cv2
import pandas as pd


class DatasetSummary:
    """
    Create a summary of all images in the dataset.
    """

    def __init__(self):
        self.records = []

    def inspect_image(self, image_path: Path, dataset_name: str):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read {image_path}")
            return

        height, width, channels = image.shape

        self.records.append(
            {
                "dataset": dataset_name,
                "filename": image_path.name,
                "width": width,
                "height": height,
                "channels": channels,
                "filesize_mb": round(image_path.stat().st_size / (1024**2), 2),
            }
        )

    def to_dataframe(self):

        return pd.DataFrame(self.records)