from pathlib import Path

from src.calibration import recalibrate_image


def main():

    image_path = Path(
        "data/raw/duodenum_first/"
        "BC11007_I-05-16_D101 - 40x.jpg"
    )

    result = recalibrate_image(
        image_path
    )

    print()
    print("RECALIBRATION COMPLETE")
    print(result)


if __name__ == "__main__":
    main()