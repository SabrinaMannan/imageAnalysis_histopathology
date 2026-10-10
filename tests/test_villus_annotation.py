from pathlib import Path

from src.villus_annotation import create_annotation


IMAGE_PATH = Path(
    r"F:\imageAnalysis_histopathology\data\raw"
    r"\duodenum_first"
    r"\BC11007_I-05-16_D101 - 40x.jpg"
)

OUTPUT_PATH = Path(
    r"F:\imageAnalysis_histopathology\data\annotations"
    r"\BC11007_I-05-16_D101 - 40x.json"
)


if __name__ == "__main__":

    print("=" * 60)
    print("Starting villus annotation")
    print("=" * 60)

    create_annotation(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
    )

    print()
    print("Annotation complete.")