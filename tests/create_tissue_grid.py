
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREVIEW_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tissue_roi_inspection"
    / "reduced_preview.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tissue_roi_inspection"
    / "numbered_tissue_grid.jpg"
)

GRID_ROWS = 6
GRID_COLS = 6


def main():
    image = cv2.imread(str(PREVIEW_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not read preview: {PREVIEW_PATH}"
        )

    height, width = image.shape[:2]

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = round(col * cell_width)
            y1 = round(row * cell_height)
            x2 = round((col + 1) * cell_width)
            y2 = round((row + 1) * cell_height)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"{row + 1},{col + 1}"

            cv2.putText(
                image,
                label,
                (x1 + 8, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                label,
                (x1 + 8, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError("Could not save grid image.")

    print(f"Grid dimensions: {GRID_ROWS} rows x {GRID_COLS} columns")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
