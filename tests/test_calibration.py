from pathlib import Path

from src.calibration import run_batch_calibration


def main():

    run_batch_calibration(
        raw_dir=Path("data/raw"),
        output_path=Path(
            "outputs/calibration.csv"
        ),
    )


if __name__ == "__main__":
    main()


