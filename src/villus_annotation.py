from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath


class VillusAnnotator:
    """
    Interactive manual annotation tool for individual villi.

    The original high-resolution image is NOT converted into
    a full-resolution mask.

    Instead, polygon coordinates are stored in the coordinate
    system of the original image.

    This keeps memory usage low and allows masks to be generated
    later for individual training patches.
    """

    def __init__(
        self,
        image,
        tissue_mask=None,
        display_width=2500,
    ):
        self.image = image
        self.tissue_mask = tissue_mask

        self.original_height = image.shape[0]
        self.original_width = image.shape[1]

        self.display_width = display_width

        scale = min(
            1.0,
            display_width / self.original_width
        )

        self.scale = scale

        display_height = int(
            self.original_height * scale
        )

        self.display_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

        if tissue_mask is not None:

            self.display_mask = cv2.resize(
                tissue_mask,
                (display_width, display_height),
                interpolation=cv2.INTER_NEAREST,
            )

        else:
            self.display_mask = None

        self.polygons = []

        self.current_selector = None

        self.fig = None
        self.ax = None

    # =========================================================
    # COORDINATE CONVERSION
    # =========================================================

    def display_to_original(self, x, y):
        """
        Convert display coordinates to original image coordinates.
        """

        original_x = x / self.scale
        original_y = y / self.scale

        return [
            float(original_x),
            float(original_y),
        ]

    # =========================================================
    # ADD POLYGON
    # =========================================================

    def add_polygon(self, vertices):

        original_vertices = []

        for x, y in vertices:

            original_vertices.append(
                self.display_to_original(x, y)
            )

        self.polygons.append(
            original_vertices
        )

        print(
            f"Added villus #{len(self.polygons)} "
            f"with {len(vertices)} points."
        )

        self.redraw()

    # =========================================================
    # POLYGON CALLBACK
    # =========================================================

    def polygon_callback(self, vertices):

        self.add_polygon(vertices)

        # Create a new selector for next villus
        self.current_selector = PolygonSelector(
            self.ax,
            self.polygon_callback,
            useblit=True,
        )

    # =========================================================
    # REDRAW
    # =========================================================

    def redraw(self):

        self.ax.clear()

        self.ax.imshow(
            cv2.cvtColor(
                self.display_image,
                cv2.COLOR_BGR2RGB
            )
        )

        # Show existing annotations
        for i, polygon in enumerate(
            self.polygons,
            start=1,
        ):

            points = np.array(polygon)

            display_points = (
                points * self.scale
            )

            closed = np.vstack(
                [
                    display_points,
                    display_points[0],
                ]
            )

            self.ax.plot(
                closed[:, 0],
                closed[:, 1],
                linewidth=2,
            )

            centroid = display_points.mean(
                axis=0
            )

            self.ax.text(
                centroid[0],
                centroid[1],
                str(i),
                fontsize=12,
            )

        self.ax.set_title(
            f"Villus annotations: {len(self.polygons)}\n"
            "Draw polygon around each individual villus"
        )

        self.ax.axis("off")

        self.fig.canvas.draw_idle()

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "image_width": self.original_width,
            "image_height": self.original_height,
            "display_scale": self.scale,
            "num_villi": len(self.polygons),
            "polygons": self.polygons,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
            )

        print()
        print(
            f"Saved {len(self.polygons)} villus annotations."
        )
        print(
            f"File: {output_path}"
        )

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        self.fig, self.ax = plt.subplots(
            figsize=(14, 10)
        )

        self.redraw()

        self.current_selector = PolygonSelector(
            self.ax,
            self.polygon_callback,
            useblit=True,
        )

        print()
        print("=" * 60)
        print("VILLUS ANNOTATION")
        print("=" * 60)
        print()
        print("Draw a polygon around ONE villus.")
        print()
        print("Controls:")
        print("  Left click  -> create polygon points")
        print("  Close polygon -> finish villus")
        print("  Close window -> finish annotation")
        print()
        print("=" * 60)

        plt.show()


# =============================================================
# LOAD IMAGE
# =============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image does not exist:\n{image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    return image


# =============================================================
# LOAD MASK
# =============================================================

def load_mask(mask_path):

    mask_path = Path(mask_path)

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Mask does not exist:\n{mask_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:

        raise ValueError(
            f"Could not read tissue mask:\n{mask_path}"
        )

    return mask


# =============================================================
# CREATE ANNOTATION
# =============================================================

def create_annotation(
    image_path,
    output_path,
    tissue_mask_path=None,
):

    print("Loading image...")

    image = load_image(
        image_path
    )

    print(
        f"Image shape: {image.shape}"
    )

    tissue_mask = None

    if tissue_mask_path is not None:

        print(
            "Loading tissue mask..."
        )

        tissue_mask = load_mask(
            tissue_mask_path
        )

        print(
            f"Tissue mask shape: "
            f"{tissue_mask.shape}"
        )

    annotator = VillusAnnotator(
        image=image,
        tissue_mask=tissue_mask,
        display_width=2500,
    )

    annotator.run()

    annotator.save(
        output_path
    )


# =============================================================
# EXAMPLE
# =============================================================

if __name__ == "__main__":

    image_path = (
        r"F:\imageAnalysis_histopathology\data\raw"
        r"\duodenum_first"
        r"\BC11007_I-05-16_D101 - 40x.jpg"
    )

    output_path = (
        r"F:\imageAnalysis_histopathology\data\annotations"
        r"\BC11007_I-05-16_D101 - 40x.json"
    )

    create_annotation(
        image_path=image_path,
        output_path=output_path,
    )