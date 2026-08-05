"""
Image loading utilities.
"""

from pathlib import Path

import cv2


class ImageLoader:
    """
    Handles image discovery and loading.
    """

    def __init__(self, raw_directory, supported_formats):

        self.raw_directory = Path(raw_directory)

        self.supported_formats = supported_formats

    def discover_datasets(self):
        """
        Discover all dataset folders.
        """

        return sorted(
            [
                folder
                for folder in self.raw_directory.iterdir()
                if folder.is_dir()
            ]
        )

    def get_images(self, dataset_folder):
        """
        Return every image inside a dataset folder.
        """

        images = []

        dataset_folder = Path(dataset_folder)

        for extension in self.supported_formats:

            images.extend(
                dataset_folder.glob(f"*{extension}")
            )

        return sorted(images)

    def load_image(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:

            raise ValueError(
                f"Could not read {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image