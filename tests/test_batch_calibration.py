from pathlib import Path

from src.calibration import run_batch_calibration


def main():

    raw_dir = Path("data/raw")

    output_path = Path(
        "outputs/calibration.csv"
    )

    run_batch_calibration(
        raw_dir,
        output_path
    )


if __name__ == "__main__":
    main()