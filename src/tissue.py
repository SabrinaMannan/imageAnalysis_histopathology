"""
Robust tissue segmentation using HSV saturation.
"""

from pathlib import Path
import cv2
import numpy as np


class TissueSegmenter:

    def __init__(self,
                 saturation_threshold=20,
                 kernel_size=11,
                 minimum_area=100000):

        self.saturation_threshold = saturation_threshold
        self.kernel_size = kernel_size
        self.minimum_area = minimum_area

    def create_mask(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        saturation = hsv[:, :, 1]

        _, mask = cv2.threshold(
            saturation,
            self.saturation_threshold,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones(
            (self.kernel_size, self.kernel_size),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        return mask

    def find_tissue(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        large_contours = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > self.minimum_area:
                large_contours.append(contour)

        return large_contours

    def draw_contours(self, image, contours):

        output = image.copy()

        cv2.drawContours(
            output,
            contours,
            -1,
            (0,255,0),
            20
        )

        return output

    def create_final_mask(self, image, contours):

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            mask,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

        return mask

    def save_image(self, image, filename):

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(filename),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

    def save_mask(self, mask, filename):

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(filename), mask)