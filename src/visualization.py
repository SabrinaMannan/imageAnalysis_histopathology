from pathlib import Path

import cv2
import numpy as np


class ImageVisualizer:
    """
    Utilities for displaying and saving histology images
    at a manageable size.
    """

    def __init__(
        self,
        max_width=1200,
        max_height=900,
    ):
        self.max_width = max_width
        self.max_height = max_height

    def resize_to_fit(self, image):
        """
        Resize image while preserving aspect ratio.
        """

        height, width = image.shape[:2]

        scale = min(
            self.max_width / width,
            self.max_height / height,
            1.0,
        )

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        return resized

    def show_image(
        self,
        image,
        window_name="Histology Image",
    ):
        """
        Display an image at a manageable size.
        """

        display = self.resize_to_fit(
            image
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.imshow(
            window_name,
            display,
        )

        print()
        print("IMAGE VIEWER")
        print("=" * 50)
        print("Press ESC or Q to close.")
        print("=" * 50)

        while True:

            key = cv2.waitKey(50) & 0xFF

            if key == 27:
                break

            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    def save_image(
        self,
        image,
        output_path,
    ):
        """
        Save an image to disk.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:

            raise IOError(
                f"Could not save image: "
                f"{output_path}"
            )

        print(
            f"Saved image: {output_path}"
        )