from pathlib import Path

import cv2
import numpy as np


class VillusDetector:
    """
    First-pass villus candidate detector.

    This is a candidate-generation method, NOT the final
    villus measurement algorithm.

    The purpose is to identify elongated/protruding structures
    for visual inspection before developing the final method.
    """

    def __init__(
        self,
        min_area=10000,
        max_area=5000000,
        min_aspect_ratio=1.5,
        min_solidity=0.30,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_solidity = min_solidity

    # =========================================================
    # FIND CONTOURS
    # =========================================================

    def find_contours(self, tissue_mask):
        """
        Find connected tissue contours.
        """

        contours, _ = cv2.findContours(
            tissue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours

    # =========================================================
    # CONTOUR FEATURES
    # =========================================================

    @staticmethod
    def contour_features(contour):
        """
        Calculate geometric properties of a contour.
        """

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if height == 0:
            aspect_ratio = 0
        else:
            aspect_ratio = max(
                width / height,
                height / width,
            )

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area == 0:
            solidity = 0
        else:
            solidity = (
                area / hull_area
            )

        return {
            "area": area,
            "perimeter": perimeter,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
        }

    # =========================================================
    # CANDIDATE FILTER
    # =========================================================

    def is_candidate(self, features):
        """
        Determine whether a contour should be retained
        as a potential villus candidate.

        NOTE:
        This is intentionally conservative and provisional.
        """

        if features["area"] < self.min_area:
            return False

        if features["area"] > self.max_area:
            return False

        if (
            features["aspect_ratio"]
            < self.min_aspect_ratio
        ):
            return False

        if (
            features["solidity"]
            < self.min_solidity
        ):
            return False

        return True

    # =========================================================
    # DETECT CANDIDATES
    # =========================================================

    def detect(
        self,
        tissue_mask,
    ):
        """
        Generate candidate regions from the tissue mask.

        Returns
        -------
        candidates : list of dict
        """

        contours = self.find_contours(
            tissue_mask
        )

        candidates = []

        for contour_id, contour in enumerate(
            contours
        ):

            features = self.contour_features(
                contour
            )

            if self.is_candidate(
                features
            ):

                features[
                    "contour_id"
                ] = contour_id

                features[
                    "contour"
                ] = contour

                candidates.append(
                    features
                )

        return candidates

    # =========================================================
    # OVERLAY
    # =========================================================

    def create_overlay(
        self,
        image,
        candidates,
    ):
        """
        Draw candidate regions on the image.
        """

        overlay = image.copy()

        for number, candidate in enumerate(
            candidates,
            start=1,
        ):

            contour = candidate[
                "contour"
            ]

            cv2.drawContours(
                overlay,
                [contour],
                -1,
                (255, 0, 0),
                8,
            )

            x = candidate["x"]
            y = candidate["y"]

            cv2.putText(
                overlay,
                str(number),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 0, 0),
                5,
                cv2.LINE_AA,
            )

        return overlay