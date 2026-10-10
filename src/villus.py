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

How candidates are found
-------------------------
Villi are finger-like protrusions of tissue extending outward
from the mucosal surface. To find them, the tissue mask is
compared against a *morphologically opened* (smoothed) version
of itself:

    tissue mask
        ↓
    smoothed baseline = opening(tissue mask, structural_radius)
        ↓
    protrusions = tissue mask MINUS smoothed baseline
        ↓
    candidate regions

Opening removes anything narrower/shallower than
`structural_radius`, so subtracting it from the original mask
leaves exactly the parts of the tissue that stick out past that
smoothing scale -- i.e. villus-like protrusions.

(An earlier version of this module computed
`convex_hull(contour) - contour`, which isolates the
*concavities* along the boundary -- the crypts/valleys between
villi -- not the villi themselves. That was backwards for this
use case and has been replaced by the opening-based approach
above.)

Input:
    - RGB image
    - binary tissue mask

Output:
    - candidate mask
    - candidate contours
    - candidate measurements (including estimated protrusion depth)
    - overlay image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

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

    protrusion_depth_px: float

    contour: Any


class VillusDetector:
    """
    First-pass detector for villus-like protrusions.

    The detector compares the tissue mask against a smoothed
    (opened) version of itself and keeps the parts of the tissue
    that protrude beyond that smoothed baseline by at least
    `min_protrusion_depth` pixels.

    This is candidate generation, not final segmentation.
    """

    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 500_000,
        min_aspect_ratio: float = 1.5,
        min_solidity: float = 0.20,
        min_protrusion_depth: float = 5.0,
        structural_radius: int = 15,
        max_candidates: int = 100,
    ):
        """
        Parameters
        ----------
        min_area, max_area:
            Pixel-area bounds for a candidate region.

        min_aspect_ratio:
            Minimum elongation (long side / short side, from the
            minimum-area rotated rectangle) required to keep a
            candidate. Villi are elongated; near-circular blobs
            are filtered out.

        min_solidity:
            Minimum area / convex-hull-area ratio for a candidate.

        min_protrusion_depth:
            Minimum distance (in pixels) a region must extend
            beyond the smoothed baseline to count as a protrusion,
            measured via a distance transform. This is what
            separates real villi from shallow surface irregularity.

        structural_radius:
            Radius (in pixels) of the elliptical structuring
            element used for the morphological opening that
            defines the "smoothed baseline" tissue shape. This is
            a *shape-scale* parameter (how wide something has to
            be to survive smoothing), distinct from
            `min_protrusion_depth` (how far it has to stick out).
            Should be set relative to the image's µm/pixel
            calibration and expected villus width.

        max_candidates:
            Maximum number of candidates to keep, taken as the
            largest-area regions after filtering.
        """

        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_solidity = min_solidity
        self.min_protrusion_depth = min_protrusion_depth
        self.structural_radius = structural_radius
        self.max_candidates = max_candidates

    # ============================================================
    # PARAMETER TUNING HELPERS
    # ============================================================

    @staticmethod
    def preview_opening(tissue_mask: np.ndarray, radius: int) -> np.ndarray:
        """
        Compute just the morphological-opening smoothed baseline for
        a given structural radius, without running full candidate
        detection.

        Useful for quickly comparing several `structural_radius`
        values against a tissue mask: the opened mask should
        visibly round off/erase the villi while keeping the gross
        tissue outline. If it still closely follows the ragged
        boundary, the radius is too small.
        """

        mask = VillusDetector._validate_mask(tissue_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )

        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
        depth_map: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Calculate geometric properties of a contour.

        If `depth_map` is provided (a per-pixel distance-from-
        baseline map, same size as the source image), the maximum
        depth within the contour is also returned.
        """

        area = float(cv2.contourArea(contour))

        perimeter = float(cv2.arcLength(contour, True))

        x, y, width, height = cv2.boundingRect(contour)

        # Elongation from the minimum-area rotated rectangle, not
        # the axis-aligned bounding box -- a bounding box badly
        # underestimates elongation for a tilted/diagonal villus.
        if len(contour) >= 3:
            (_, (rect_w, rect_h), _) = cv2.minAreaRect(contour)
            short_side = min(rect_w, rect_h)
            long_side = max(rect_w, rect_h)
            aspect_ratio = (
                long_side / short_side if short_side > 0 else 0.0
            )
        else:
            aspect_ratio = 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        protrusion_depth_px = 0.0
        if depth_map is not None:
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            masked_depth = depth_map[mask > 0]
            if masked_depth.size > 0:
                protrusion_depth_px = float(masked_depth.max())

        return {
            "area_px": area,
            "perimeter_px": perimeter,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "protrusion_depth_px": protrusion_depth_px,
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

        if features["area_px"] < self.min_area:
            return False

        if features["area_px"] > self.max_area:
            return False

        if features["aspect_ratio"] < self.min_aspect_ratio:
            return False

        if features["solidity"] < self.min_solidity:
            return False

        if features["protrusion_depth_px"] < self.min_protrusion_depth:
            return False

        return True

    # ============================================================
    # PROTRUSION CANDIDATES
    # ============================================================

    def _boundary_candidates(
        self,
        tissue_mask: np.ndarray,
        return_debug: bool = False,
    ):
        """
        Generate candidate regions that protrude from the tissue
        boundary, along with a depth map (distance beyond the
        smoothed baseline, in pixels) for later filtering.

        This is intentionally conservative and is NOT
        individual-villus segmentation.

        If `return_debug` is True, also returns the opened
        (smoothed baseline) mask and the raw protrusion mask
        *before* the small-artifact cleanup step, for tuning
        `structural_radius`.
        """

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * self.structural_radius + 1,
                2 * self.structural_radius + 1,
            ),
        )

        # Smoothed baseline: opening removes anything narrower
        # than the structural element, i.e. removes villus-scale
        # protrusions and leaves the underlying mucosal surface.
        opened_mask = cv2.morphologyEx(
            tissue_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        # Everything that got removed by opening is a candidate
        # protrusion.
        protrusion_region = cv2.subtract(tissue_mask, opened_mask)
        raw_protrusion_mask = protrusion_region.copy()

        # Clean tiny artifacts.
        clean_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        protrusion_region = cv2.morphologyEx(
            protrusion_region, cv2.MORPH_OPEN, clean_kernel
        )
        protrusion_region = cv2.morphologyEx(
            protrusion_region, cv2.MORPH_CLOSE, clean_kernel
        )

        # Depth of every pixel below the smoothed baseline is its
        # distance to the nearest point still inside opened_mask.
        depth_map = cv2.distanceTransform(
            cv2.bitwise_not(opened_mask),
            cv2.DIST_L2,
            5,
        )
        # Only meaningful where there's actually tissue.
        depth_map = np.where(tissue_mask > 0, depth_map, 0).astype(
            np.float32
        )

        contours, _ = cv2.findContours(
            protrusion_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = [c for c in contours if cv2.contourArea(c) > 0]

        if return_debug:
            return candidates, depth_map, opened_mask, raw_protrusion_mask

        return candidates, depth_map

    # ============================================================
    # CANDIDATE GENERATION
    # ============================================================

    def detect(
        self,
        image: np.ndarray,
        tissue_mask: np.ndarray,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect villus candidate regions.

        Parameters
        ----------
        image:
            Original RGB/BGR image.

        tissue_mask:
            Binary tissue ROI mask.

        debug:
            If True, include the intermediate opened mask,
            *unfiltered* protrusion mask, and depth map in the
            result. Use this to check whether `structural_radius`
            and the filter thresholds (`min_area`,
            `min_aspect_ratio`, `min_solidity`,
            `min_protrusion_depth`) are appropriate for the image
            scale before trusting the filtered candidates --
            especially useful when the filtered candidate count
            looks implausibly low relative to how serrated the
            tissue boundary visibly is.

        Returns
        -------
        dict
            {
                "image": resized image,
                "tissue_mask": binary mask,
                "candidate_mask": binary mask,
                "candidates": list,
                "count": integer,
                # present only when debug=True:
                "opened_mask": ...,
                "raw_protrusion_mask": ...,
                "depth_map": ...,
                "raw_candidate_count": ...,
            }
        """

        tissue_mask = self._validate_mask(tissue_mask)
        image = self._prepare_image(image, tissue_mask.shape)

        (
            contours,
            depth_map,
            opened_mask,
            raw_protrusion_mask,
        ) = self._boundary_candidates(tissue_mask, return_debug=True)

        scored: List[Tuple[VillusCandidate, float]] = []

        for contour in contours:
            features = self.contour_features(contour, depth_map)

            if not self.is_candidate(features):
                continue

            candidate = VillusCandidate(
                candidate_id=0,  # assigned after sorting, below
                x=features["x"],
                y=features["y"],
                width=features["width"],
                height=features["height"],
                area_px=features["area_px"],
                perimeter_px=features["perimeter_px"],
                aspect_ratio=features["aspect_ratio"],
                solidity=features["solidity"],
                protrusion_depth_px=features["protrusion_depth_px"],
                contour=contour,
            )

            scored.append((candidate, features["area_px"]))

        # Keep the largest-area candidates rather than an arbitrary
        # discovery-order prefix, then assign stable IDs.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        scored = scored[: self.max_candidates]

        candidates: List[VillusCandidate] = []
        for i, (candidate, _) in enumerate(scored, start=1):
            candidate.candidate_id = i
            candidates.append(candidate)

        candidate_mask = np.zeros_like(tissue_mask)
        for candidate in candidates:
            cv2.drawContours(
                candidate_mask,
                [candidate.contour],
                -1,
                255,
                thickness=-1,
            )

        result = {
            "image": image,
            "tissue_mask": tissue_mask,
            "candidate_mask": candidate_mask,
            "candidates": candidates,
            "count": len(candidates),
        }

        if debug:
            result["opened_mask"] = opened_mask
            result["raw_protrusion_mask"] = raw_protrusion_mask
            result["depth_map"] = depth_map
            result["raw_candidate_count"] = len(contours)

        return result

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
            raise ValueError("Image cannot be None.")

        overlay = image.copy()

        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        for candidate in candidates:
            contour = candidate.contour

            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)

            label_x = candidate.x
            label_y = max(20, candidate.y)

            cv2.putText(
                overlay,
                str(candidate.candidate_id),
                (label_x, label_y),
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

        mask = VillusDetector._validate_mask(tissue_mask)
        image = VillusDetector._prepare_image(image, mask.shape)

        if image.ndim == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

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
                "candidate_id": candidate.candidate_id,
                "x": candidate.x,
                "y": candidate.y,
                "width": candidate.width,
                "height": candidate.height,
                "area_px": candidate.area_px,
                "perimeter_px": candidate.perimeter_px,
                "aspect_ratio": candidate.aspect_ratio,
                "solidity": candidate.solidity,
                "protrusion_depth_px": candidate.protrusion_depth_px,
            }
            rows.append(row)

        return rows