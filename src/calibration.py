from pathlib import Path

import cv2
import pandas as pd


class ScaleBarCalibrator:
    """
    Interactive scale-bar calibration for histology images.

    Shows:
    1. A small overview of the complete image.
    2. A zoomed bottom-right region for scale-bar selection.

    Controls:
        Left click = select scale-bar endpoint
        R          = reset selection
        ENTER      = accept selection
        ESC / Q    = cancel
    """

    def __init__(
        self,
        crop_fraction=0.25,
        overview_width=700,
        overview_height=500,
        zoom_width=700,
        zoom_height=500,
    ):
        self.crop_fraction = crop_fraction

        self.overview_width = overview_width
        self.overview_height = overview_height

        self.zoom_width = zoom_width
        self.zoom_height = zoom_height

    # =========================================================
    # IMAGE LOADING
    # =========================================================

    def load_image(self, image_path):
        """Load an image as RGB."""

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return image

    # =========================================================
    # RESIZE
    # =========================================================

    @staticmethod
    def resize_to_fit(
        image,
        max_width,
        max_height,
    ):
        """
        Resize an image while preserving aspect ratio.

        Returns
        -------
        resized : numpy.ndarray
        scale : float
        """

        height, width = image.shape[:2]

        scale = min(
            max_width / width,
            max_height / height,
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

        return resized, scale

    # =========================================================
    # WHOLE IMAGE OVERVIEW
    # =========================================================

    def create_overview(self, image):
        """Create a small overview of the complete image."""

        overview, scale = self.resize_to_fit(
            image,
            self.overview_width,
            self.overview_height,
        )

        return overview, scale

    # =========================================================
    # SCALE BAR REGION
    # =========================================================

    def create_zoom_region(self, image):
        """
        Extract the bottom-right portion of the image.

        Returns
        -------
        zoom :
            Resized zoom region.

        scale :
            Resize factor used for the zoom region.

        x_offset :
            Original-image x coordinate of crop start.

        y_offset :
            Original-image y coordinate of crop start.
        """

        height, width = image.shape[:2]

        x_offset = int(
            width * (1 - self.crop_fraction)
        )

        y_offset = int(
            height * (1 - self.crop_fraction)
        )

        crop = image[
            y_offset:height,
            x_offset:width,
        ]

        zoom, scale = self.resize_to_fit(
            crop,
            self.zoom_width,
            self.zoom_height,
        )

        return (
            zoom,
            scale,
            x_offset,
            y_offset,
        )

    # =========================================================
    # INTERACTIVE SCALE BAR SELECTION
    # =========================================================

    def select_scale_bar(
        self,
        image,
        image_path,
    ):
        """
        Display the image overview and scale-bar region.

        User selects two endpoints of the scale bar.
        """

        # -----------------------------------------------------
        # Create overview
        # -----------------------------------------------------

        overview, _ = self.create_overview(
            image
        )

        # -----------------------------------------------------
        # Create zoom region
        # -----------------------------------------------------

        (
            zoom,
            zoom_scale,
            x_offset,
            y_offset,
        ) = self.create_zoom_region(image)

        # Convert RGB -> BGR for OpenCV
        overview_bgr = cv2.cvtColor(
            overview,
            cv2.COLOR_RGB2BGR,
        )

        zoom_bgr = cv2.cvtColor(
            zoom,
            cv2.COLOR_RGB2BGR,
        )

        # Keep an untouched copy for reset
        original_zoom = zoom_bgr.copy()

        # -----------------------------------------------------
        # Window names
        # -----------------------------------------------------

        overview_window = "Image Overview"

        calibration_window = (
            "Scale Bar Calibration"
        )

        # -----------------------------------------------------
        # Selected points
        # -----------------------------------------------------

        points = []

        # -----------------------------------------------------
        # Redraw calibration window
        # -----------------------------------------------------

        def redraw():

            display = original_zoom.copy()

            # Draw selected points
            for point in points:

                cv2.circle(
                    display,
                    point,
                    7,
                    (0, 0, 255),
                    -1,
                )

            # Draw line between two points
            if len(points) == 2:

                cv2.line(
                    display,
                    points[0],
                    points[1],
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                calibration_window,
                display,
            )

        # -----------------------------------------------------
        # Mouse callback
        # -----------------------------------------------------

        def mouse_callback(
            event,
            x,
            y,
            flags,
            param,
        ):

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            # Do not allow more than two points
            if len(points) >= 2:
                return

            points.append(
                (x, y)
            )

            redraw()

        # -----------------------------------------------------
        # Create overview window
        # -----------------------------------------------------

        cv2.namedWindow(
            overview_window,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            overview_window,
            self.overview_width,
            self.overview_height,
        )

        cv2.imshow(
            overview_window,
            overview_bgr,
        )

        # -----------------------------------------------------
        # Create calibration window
        # -----------------------------------------------------

        cv2.namedWindow(
            calibration_window,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            calibration_window,
            self.zoom_width,
            self.zoom_height,
        )

        cv2.setMouseCallback(
            calibration_window,
            mouse_callback,
        )

        redraw()

        # -----------------------------------------------------
        # Instructions
        # -----------------------------------------------------

        print()
        print("=" * 60)
        print("SCALE BAR CALIBRATION")
        print("=" * 60)

        print(
            f"Image: {image_path}"
        )

        print()
        print(
            "LEFT CLICK = select scale-bar endpoint"
        )

        print(
            "R          = reset selected points"
        )

        print(
            "ENTER      = accept two points"
        )

        print(
            "ESC / Q    = cancel"
        )

        print()
        print(
            "Do NOT click the scale-bar text."
        )

        print("=" * 60)

        # -----------------------------------------------------
        # Main interaction loop
        # -----------------------------------------------------

        while True:

            key = cv2.waitKey(50) & 0xFF

            # ---------------------------------------------
            # ESC
            # ---------------------------------------------

            if key == 27:

                cv2.destroyAllWindows()

                raise RuntimeError(
                    "Calibration cancelled."
                )

            # ---------------------------------------------
            # Q
            # ---------------------------------------------

            if key == ord("q"):

                cv2.destroyAllWindows()

                raise RuntimeError(
                    "Calibration cancelled."
                )

            # ---------------------------------------------
            # R = reset
            # ---------------------------------------------

            if key == ord("r"):

                points.clear()

                redraw()

                print(
                    "Selection reset."
                )

                continue

            # ---------------------------------------------
            # ENTER = accept
            # ---------------------------------------------

            if key in (10, 13):

                if len(points) != 2:

                    print(
                        "Please select exactly "
                        "two points first."
                    )

                    continue

                break

        # -----------------------------------------------------
        # Close windows
        # -----------------------------------------------------

        cv2.destroyAllWindows()

        # -----------------------------------------------------
        # Convert zoom coordinates to original image
        # coordinates
        # -----------------------------------------------------

        x1_zoom, y1_zoom = points[0]

        x2_zoom, y2_zoom = points[1]

        # Coordinates inside original crop
        x1_crop = (
            x1_zoom / zoom_scale
        )

        y1_crop = (
            y1_zoom / zoom_scale
        )

        x2_crop = (
            x2_zoom / zoom_scale
        )

        y2_crop = (
            y2_zoom / zoom_scale
        )

        # Coordinates in full original image
        x1 = (
            x_offset + x1_crop
        )

        y1 = (
            y_offset + y1_crop
        )

        x2 = (
            x_offset + x2_crop
        )

        y2 = (
            y_offset + y2_crop
        )

        # -----------------------------------------------------
        # Calculate scale-bar pixel length
        # -----------------------------------------------------

        pixel_length = (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

        return {
            "pixel_length": pixel_length,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    # =========================================================
    # CALIBRATE ONE IMAGE
    # =========================================================

    def calibrate(
        self,
        image_path,
    ):
        """Calibrate one image."""

        image_path = Path(
            image_path
        )

        image = self.load_image(
            image_path
        )

        result = self.select_scale_bar(
            image,
            image_path,
        )

        pixel_length = result[
            "pixel_length"
        ]

        print()
        print(
            f"Measured scale-bar length: "
            f"{pixel_length:.2f} pixels"
        )

        # -----------------------------------------------------
        # Ask for scale-bar value
        # -----------------------------------------------------

        while True:

            try:

                scale_bar_um = float(
                    input(
                        "Enter scale-bar value "
                        "in micrometers "
                        "(e.g. 100, 200, 500): "
                    )
                )

                if scale_bar_um <= 0:

                    raise ValueError

                break

            except ValueError:

                print(
                    "Please enter a positive number."
                )

        # -----------------------------------------------------
        # Calculate calibration
        # -----------------------------------------------------

        um_per_pixel = (
            scale_bar_um
            / pixel_length
        )

        pixels_per_um = (
            pixel_length
            / scale_bar_um
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        print()
        print("-" * 60)
        print("CALIBRATION RESULT")
        print("-" * 60)

        print(
            f"Scale bar:       "
            f"{scale_bar_um:.2f} µm"
        )

        print(
            f"Pixel length:    "
            f"{pixel_length:.2f} pixels"
        )

        print(
            f"µm per pixel:    "
            f"{um_per_pixel:.8f}"
        )

        print(
            f"pixels per µm:   "
            f"{pixels_per_um:.4f}"
        )

        print("-" * 60)

        return {
            "dataset": image_path.parent.name,
            "filename": image_path.name,
            "scale_bar_um": scale_bar_um,
            "scale_bar_pixels": pixel_length,
            "um_per_pixel": um_per_pixel,
            "pixels_per_um": pixels_per_um,
            "x1": result["x1"],
            "y1": result["y1"],
            "x2": result["x2"],
            "y2": result["y2"],
        }


# =============================================================
# BATCH CALIBRATION
# =============================================================

def run_batch_calibration(
    raw_dir,
    output_path,
):
    """Calibrate all supported images."""

    raw_dir = Path(
        raw_dir
    )

    output_path = Path(
        output_path
    )

    supported_formats = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    image_paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_formats
        )
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No supported images found "
            f"in {raw_dir}"
        )

    print()
    print("=" * 60)
    print("BATCH SCALE CALIBRATION")
    print("=" * 60)

    print(
        f"Images found: "
        f"{len(image_paths)}"
    )

    calibrator = ScaleBarCalibrator()

    results = []

    for number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print("#" * 60)

        print(
            f"IMAGE {number} "
            f"OF {len(image_paths)}"
        )

        print("#" * 60)

        try:

            result = calibrator.calibrate(
                image_path
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"Skipped: {image_path}"
            )

            print(
                f"Reason: {error}"
            )

    if not results:

        raise RuntimeError(
            "No images were successfully "
            "calibrated."
        )

    dataframe = pd.DataFrame(
        results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("BATCH CALIBRATION COMPLETE")
    print("=" * 60)

    print(
        f"Successfully calibrated: "
        f"{len(results)}"
    )

    print(
        f"Saved to: {output_path}"
    )


# =============================================================
# SINGLE IMAGE RECALIBRATION
# =============================================================

def recalibrate_image(
    image_path,
):
    """
    Recalibrate a single image.

    Does not modify the existing CSV.
    """

    calibrator = ScaleBarCalibrator()

    return calibrator.calibrate(
        image_path
    )