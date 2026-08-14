from pathlib import Path

import pandas as pd


def run_calibration_qc(
    calibration_file,
    output_file,
):
    """
    Perform quality control on image calibration results.

    Parameters
    ----------
    calibration_file : str or Path
        CSV containing calibration results.

    output_file : str or Path
        CSV where the QC results will be saved.
    """

    calibration_file = Path(calibration_file)
    output_file = Path(output_file)

    # ---------------------------------------------------------
    # Load calibration data
    # ---------------------------------------------------------

    df = pd.read_csv(calibration_file)

    required_columns = [
        "dataset",
        "filename",
        "scale_bar_um",
        "scale_bar_pixels",
        "um_per_pixel",
        "pixels_per_um",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    mean_um_per_pixel = (
        df["um_per_pixel"].mean()
    )

    median_um_per_pixel = (
        df["um_per_pixel"].median()
    )

    std_um_per_pixel = (
        df["um_per_pixel"].std()
    )

    min_um_per_pixel = (
        df["um_per_pixel"].min()
    )

    max_um_per_pixel = (
        df["um_per_pixel"].max()
    )

    print()
    print("=" * 60)
    print("CALIBRATION QUALITY CONTROL")
    print("=" * 60)

    print(
        f"Number of images: "
        f"{len(df)}"
    )

    print(
        f"Mean µm/pixel:    "
        f"{mean_um_per_pixel:.6f}"
    )

    print(
        f"Median µm/pixel:  "
        f"{median_um_per_pixel:.6f}"
    )

    print(
        f"SD:               "
        f"{std_um_per_pixel:.6f}"
    )

    print(
        f"Minimum:          "
        f"{min_um_per_pixel:.6f}"
    )

    print(
        f"Maximum:          "
        f"{max_um_per_pixel:.6f}"
    )

    # ---------------------------------------------------------
    # Calculate deviation from median
    # ---------------------------------------------------------

    df["deviation_from_median_percent"] = (
        (
            df["um_per_pixel"]
            - median_um_per_pixel
        )
        / median_um_per_pixel
        * 100
    )

    # ---------------------------------------------------------
    # Flag potential outliers
    #
    # We use ±10% as a REVIEW threshold.
    #
    # This does NOT automatically mean the calibration
    # is wrong. It means we should inspect it.
    # ---------------------------------------------------------

    df["qc_status"] = "OK"

    review_mask = (
        df[
            "deviation_from_median_percent"
        ].abs()
        > 10
    )

    df.loc[
        review_mask,
        "qc_status"
    ] = "REVIEW"

    # ---------------------------------------------------------
    # Print flagged images
    # ---------------------------------------------------------

    flagged = df[
        df["qc_status"] == "REVIEW"
    ]

    print()

    if len(flagged) == 0:

        print(
            "No calibration values exceeded "
            "the ±10% review threshold."
        )

    else:

        print(
            f"Images requiring review: "
            f"{len(flagged)}"
        )

        print()

        for _, row in flagged.iterrows():

            print(
                row["filename"]
            )

            print(
                f"  µm/pixel: "
                f"{row['um_per_pixel']:.6f}"
            )

            print(
                f"  deviation: "
                f"{row['deviation_from_median_percent']:.2f}%"
            )

            print()

    # ---------------------------------------------------------
    # Dataset-level summary
    # ---------------------------------------------------------

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    dataset_summary = (
        df.groupby("dataset")[
            "um_per_pixel"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        dataset_summary
    )

    # ---------------------------------------------------------
    # Save QC results
    # ---------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"QC results saved to:"
    )

    print(
        output_file
    )

    return df


if __name__ == "__main__":

    calibration_file = (
        "outputs/calibration.csv"
    )

    output_file = (
        "outputs/calibration_qc.csv"
    )

    run_calibration_qc(
        calibration_file,
        output_file,
    )