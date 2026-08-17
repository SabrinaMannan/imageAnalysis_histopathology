"""
src/villus.py

Villus candidate detection inside a tissue ROI.

Important:
------------
This module performs FIRST-PASS candidate detection.

It does NOT claim to perfectly identify individual villi.
The purpose is to generate biologically plausible candidate
regions that can later be refined/validated, including with
manual annotation or a U-Net-based segmentation model.

Input:
    - RGB image
    - binary tissue mask

Output:
    - candidate mask
    - candidate contours
    - candidate measurements
    - overlay image
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VillusCandidate:
    """
    Information describing one candidate region.
    """

    candidate_id: int

    x: int
    y: int
    width: int
    height: int

    area_px: float
    perimeter_px: float

    aspect_ratio: float
    solidity: float

    contour: Any


class VillusDetector:
    """
    First-pass detector for villus-like protrusions.

    The detector works from the boundary of the tissue ROI.

    It identifies portions of the tissue boundary that form
    elongated/protruding structures.

    This is candidate generation, not final segmentation.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 500_000,
        min_aspect_ratio: float = 1.5,
        min_solidity: float = 0.20,
        min_protrusion_depth: float = 5.0,
        max_candidates: int = 100,
    ):

        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_solidity = min_solidity
        self.min_protrusion_depth = min_protrusion_depth
        self.max_candidates = max_candidates

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> np.ndarray:
        """
        Validate and convert tissue mask to binary uint8.
        """

        if mask is None:
            raise ValueError("Tissue mask is None.")

        if not isinstance(mask, np.ndarray):
            raise TypeError("Tissue mask must be a NumPy array.")

        if mask.ndim != 2:
            raise ValueError(
                f"Tissue mask must be 2D. Got shape {mask.shape}"
            )

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Convert any non-zero value to 255.
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        return mask

    @staticmethod
    def _prepare_image(
        image: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize image to the mask dimensions when necessary.

        target_shape = (height, width)
        """

        if image is None:
            raise ValueError("Image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"Image must be 2D or 3D. Got shape {image.shape}"
            )

        target_h, target_w = target_shape

        if image.shape[:2] == (target_h, target_w):
            return image.copy()

        return cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    # ============================================================
    # CONTOURS
    # ============================================================

    @staticmethod
    def find_tissue_contours(
        tissue_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Find external tissue contours.
        """

        contours, _ = cv2.findContours(
            tissue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return list(contours)

    # ============================================================
    # GEOMETRIC FEATURES
    # ============================================================

    @staticmethod
    def contour_features(
        contour: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate geometric properties of a contour.
        """

        area = float(cv2.contourArea(contour))

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if min(width, height) == 0:
            aspect_ratio = 0.0
        else:
            aspect_ratio = max(
                width / height,
                height / width,
            )

        hull = cv2.convexHull(contour)

        hull_area = float(
            cv2.contourArea(hull)
        )

        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0.0

        return {
            "area_px": area,
            "perimeter_px": perimeter,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
        }

    # ============================================================
    # CONTOUR FILTER
    # ============================================================

    def is_candidate(
        self,
        features: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a region is geometrically plausible.
        """

        area = features["area_px"]
        aspect_ratio = features["aspect_ratio"]
        solidity = features["solidity"]

        if area < self.min_area:
            return False

        if area > self.max_area:
            return False

        if aspect_ratio < self.min_aspect_ratio:
            return False

        if solidity < self.min_solidity:
            return False

        return True

    # ============================================================
    # SIMPLE BOUNDARY CANDIDATES
    # ============================================================

    def _boundary_candidates(
        self,
        tissue_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Generate candidate regions from the tissue boundary.

        This is intentionally conservative.

        The idea is:

            tissue ROI
                ↓
            boundary
                ↓
            identify outward protruding regions
                ↓
            candidate regions

        It is NOT individual-villus segmentation.
        """

        contours = self.find_tissue_contours(
            tissue_mask
        )

        if not contours:
            return []

        candidates = []

        for contour in contours:

            if len(contour) < 5:
                continue

            # Convex hull gives a reference envelope.
            hull = cv2.convexHull(
                contour
            )

            hull_mask = np.zeros_like(
                tissue_mask
            )

            cv2.drawContours(
                hull_mask,
                [hull],
                -1,
                255,
                thickness=-1,
            )

            contour_mask = np.zeros_like(
                tissue_mask
            )

            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                255,
                thickness=-1,
            )

            # Regions that are inside the hull but outside
            # the actual tissue contour.
            defect_region = cv2.subtract(
                hull_mask,
                contour_mask,
            )

            # Clean tiny artifacts.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            )

            defect_region = cv2.morphologyEx(
                defect_region,
                cv2.MORPH_OPEN,
                kernel,
            )

            defect_region = cv2.morphologyEx(
                defect_region,
                cv2.MORPH_CLOSE,
                kernel,
            )

            defect_contours, _ = cv2.findContours(
                defect_region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for defect in defect_contours:

                if cv2.contourArea(defect) <= 0:
                    continue

                x, y, w, h = cv2.boundingRect(
                    defect
                )

                if w == 0 or h == 0:
                    continue

                area = float(
                    cv2.contourArea(defect)
                )

                aspect_ratio = max(
                    w / h,
                    h / w,
                )

                if (
                    area >= self.min_area
                    and area <= self.max_area
                    and aspect_ratio >= self.min_aspect_ratio
                ):
                    candidates.append(
                        defect
                    )

        return candidates

    # ============================================================
    # CANDIDATE GENERATION
    # ============================================================

    def detect(
        self,
        image: np.ndarray,
        tissue_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect villus candidate regions.

        Parameters
        ----------
        image:
            Original RGB/BGR image.

        tissue_mask:
            Binary tissue ROI mask.

        Returns
        -------
        dict
            {
                "image": resized image,
                "tissue_mask": binary mask,
                "candidate_mask": binary mask,
                "candidates": list,
                "count": integer
            }
        """

        tissue_mask = self._validate_mask(
            tissue_mask
        )

        image = self._prepare_image(
            image,
            tissue_mask.shape,
        )

        # --------------------------------------------------------
        # Generate geometric candidates
        # --------------------------------------------------------

        contours = self._boundary_candidates(
            tissue_mask
        )

        candidates: List[VillusCandidate] = []

        for contour in contours:

            features = self.contour_features(
                contour
            )

            if not self.is_candidate(
                features
            ):
                continue

            candidate = VillusCandidate(
                candidate_id=len(candidates) + 1,
                x=features["x"],
                y=features["y"],
                width=features["width"],
                height=features["height"],
                area_px=features["area_px"],
                perimeter_px=features[
                    "perimeter_px"
                ],
                aspect_ratio=features[
                    "aspect_ratio"
                ],
                solidity=features[
                    "solidity"
                ],
                contour=contour,
            )

            candidates.append(
                candidate
            )

            if len(candidates) >= self.max_candidates:
                break

        # --------------------------------------------------------
        # Candidate mask
        # --------------------------------------------------------

        candidate_mask = np.zeros_like(
            tissue_mask
        )

        for candidate in candidates:

            cv2.drawContours(
                candidate_mask,
                [candidate.contour],
                -1,
                255,
                thickness=-1,
            )

        return {
            "image": image,
            "tissue_mask": tissue_mask,
            "candidate_mask": candidate_mask,
            "candidates": candidates,
            "count": len(candidates),
        }

    # ============================================================
    # OVERLAY
    # ============================================================

    def create_overlay(
        self,
        image: np.ndarray,
        candidates: List[VillusCandidate],
    ) -> np.ndarray:
        """
        Draw candidate contours and IDs over the image.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        overlay = image.copy()

        if overlay.ndim == 2:
            overlay = cv2.cvtColor(
                overlay,
                cv2.COLOR_GRAY2BGR,
            )

        for candidate in candidates:

            contour = candidate.contour

            cv2.drawContours(
                overlay,
                [contour],
                -1,
                (0, 0, 255),
                3,
            )

            label_x = candidate.x
            label_y = max(
                20,
                candidate.y,
            )

            cv2.putText(
                overlay,
                str(candidate.candidate_id),
                (
                    label_x,
                    label_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    # ============================================================
    # DRAW TISSUE ROI
    # ============================================================

    @staticmethod
    def create_tissue_overlay(
        image: np.ndarray,
        tissue_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Draw tissue ROI boundary over the original image.
        """

        mask = VillusDetector._validate_mask(
            tissue_mask
        )

        image = VillusDetector._prepare_image(
            image,
            mask.shape,
        )

        if image.ndim == 2:
            overlay = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            overlay = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 0),
            3,
        )

        return overlay

    # ============================================================
    # CONVERT RESULTS TO TABLE-FRIENDLY DATA
    # ============================================================

    @staticmethod
    def candidates_to_dicts(
        candidates: List[VillusCandidate],
    ) -> List[Dict[str, Any]]:
        """
        Convert candidate objects into dictionaries.

        Contours are removed because they are not convenient
        for CSV/DataFrame storage.
        """

        rows = []

        for candidate in candidates:

            row = {
                "candidate_id":
                    candidate.candidate_id,

                "x":
                    candidate.x,

                "y":
                    candidate.y,

                "width":
                    candidate.width,

                "height":
                    candidate.height,

                "area_px":
                    candidate.area_px,

                "perimeter_px":
                    candidate.perimeter_px,

                "aspect_ratio":
                    candidate.aspect_ratio,

                "solidity":
                    candidate.solidity,
            }

            rows.append(row)

        return rows