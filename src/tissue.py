import cv2
import numpy as np


class TissueSegmenter:
    """
    Tissue ROI segmentation for H&E histology images.

    Purpose
    -------
    Detect the overall tissue section and exclude the glass/background.

    Important:
    This is NOT intended to segment villi, crypts, cells, or lumens.

    Memory strategy
    ---------------
    Very large histology images are resized before segmentation.
    Expensive operations are performed only on the smaller working image.
    """

    def __init__(
        self,
        max_processing_size=2500,
        blur_size=31,
        close_kernel_size=101,
        min_component_fraction=0.002
    ):

        self.max_processing_size = max_processing_size
        self.blur_size = blur_size
        self.close_kernel_size = close_kernel_size
        self.min_component_fraction = min_component_fraction

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    def load_image(self, image_path):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # =========================================================
    # RESIZE FOR PROCESSING
    # =========================================================

    def resize_for_processing(self, image):

        h, w = image.shape[:2]

        largest_dimension = max(h, w)

        if largest_dimension <= self.max_processing_size:
            return image.copy()

        scale = (
            self.max_processing_size
            / largest_dimension
        )

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    # =========================================================
    # CREATE TISSUE SCORE
    # =========================================================

    def create_tissue_score(self, image):

        """
        Create a low-memory score representing how tissue-like
        each pixel is.

        H&E tissue is generally more colored/pink than the
        near-white background.

        We combine:

        1. HSV saturation
        2. Red-minus-blue signal

        This is more robust than using saturation alone.
        """

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        saturation = hsv[:, :, 1].astype(np.float32)

        red = image[:, :, 0].astype(np.float32)
        blue = image[:, :, 2].astype(np.float32)

        red_blue = red - blue

        # Normalize both signals approximately to 0-1
        saturation_score = saturation / 255.0

        red_blue_score = (
            np.clip(red_blue, 0, 80) / 80.0
        )

        # Combined tissue score
        score = (
            0.60 * saturation_score
            +
            0.40 * red_blue_score
        )

        # Convert back to uint8 immediately
        score = np.clip(
            score * 255,
            0,
            255
        ).astype(np.uint8)

        return score

    # =========================================================
    # INITIAL SEGMENTATION
    # =========================================================

    def create_initial_mask(self, image):

        score = self.create_tissue_score(image)

        # Smooth the score so individual cells/crypt details
        # do not dominate the tissue boundary.

        blur_size = self.blur_size

        if blur_size % 2 == 0:
            blur_size += 1

        blurred = cv2.GaussianBlur(
            score,
            (blur_size, blur_size),
            0
        )

        # Otsu threshold
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return mask

    # =========================================================
    # MORPHOLOGICAL CLEANING
    # =========================================================

    def clean_mask(self, mask):

        kernel_size = self.close_kernel_size

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        # Close gaps inside the tissue boundary.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Remove tiny isolated objects.
        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            small_kernel
        )

        return mask

    # =========================================================
    # FILL TISSUE REGIONS
    # =========================================================

    def fill_tissue_regions(self, mask):

        """
        Convert fragmented foreground into filled tissue regions.

        External contours are used so internal crypts/lumens
        do not become separate background regions.
        """

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        filled = np.zeros_like(mask)

        if not contours:
            return filled

        image_area = (
            mask.shape[0]
            *
            mask.shape[1]
        )

        min_area = (
            image_area
            *
            self.min_component_fraction
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area >= min_area:

                cv2.drawContours(
                    filled,
                    [contour],
                    -1,
                    255,
                    thickness=cv2.FILLED
                )

        return filled

    # =========================================================
    # KEEP MAIN TISSUE
    # =========================================================

    def keep_main_components(self, mask):

        """
        Keep substantial tissue components.

        This prevents tiny staining artifacts elsewhere
        in the slide from becoming tissue.
        """

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return mask

        image_area = (
            mask.shape[0]
            *
            mask.shape[1]
        )

        min_area = (
            image_area
            *
            self.min_component_fraction
        )

        result = np.zeros_like(mask)

        components = []

        for label in range(1, num_labels):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= min_area:

                components.append(
                    (label, area)
                )

        # Sort largest first
        components.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Keep substantial components.
        #
        # Usually the tissue section is one major component,
        # but fragmented tissue pieces may exist.
        for label, area in components:

            result[
                labels == label
            ] = 255

        return result

    # =========================================================
    # SEGMENT
    # =========================================================

    def segment(
        self,
        image,
        return_working_mask=False
    ):

        # ---------------------------------------------
        # Small working image
        # ---------------------------------------------

        working_image = (
            self.resize_for_processing(image)
        )

        # ---------------------------------------------
        # Initial tissue detection
        # ---------------------------------------------

        mask = self.create_initial_mask(
            working_image
        )

        # ---------------------------------------------
        # Smooth / connect tissue
        # ---------------------------------------------

        mask = self.clean_mask(mask)

        # ---------------------------------------------
        # Fill tissue region
        # ---------------------------------------------

        mask = self.fill_tissue_regions(mask)

        # ---------------------------------------------
        # Remove tiny components
        # ---------------------------------------------

        mask = self.keep_main_components(mask)

        if return_working_mask:

            return (
                working_image,
                mask
            )

        return mask

    # =========================================================
    # OVERLAY
    # =========================================================

    def create_overlay(
        self,
        image,
        mask,
        alpha=0.30
    ):

        overlay = image.copy()

        tissue = mask > 0

        # Green overlay
        green = np.zeros_like(image)
        green[:, :, 1] = 255

        overlay[tissue] = (
            (1 - alpha) * image[tissue]
            +
            alpha * green[tissue]
        ).astype(np.uint8)

        # Outer tissue boundary
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 0),
            2
        )

        return overlay

    # =========================================================
    # DISPLAY RESIZE
    # =========================================================

    @staticmethod
    def resize_for_display(
        image,
        max_width=700,
        max_height=700
    ):

        h, w = image.shape[:2]

        scale = min(
            max_width / w,
            max_height / h,
            1.0
        )

        new_w = max(
            1,
            int(w * scale)
        )

        new_h = max(
            1,
            int(h * scale)
        )

        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )