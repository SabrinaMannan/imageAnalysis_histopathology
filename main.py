from src.config import load_config
from src.image_io import ImageLoader
from src.dataset_summary import DatasetSummary
from pathlib import Path
from src.tissue import TissueSegmenter


def main():

    config = load_config()

    loader = ImageLoader(
        config["data"]["raw_dir"],
        config["image"]["supported_formats"]
    )

    datasets = loader.discover_datasets()

    print("=" * 60)
    print("Datasets discovered")
    print("=" * 60)

    summary = DatasetSummary()

    for dataset in datasets:

        print(dataset.name)

        images = loader.get_images(dataset)

        print(f"Images: {len(images)}")

        print("-" * 40)

        for image_path in images:
            summary.inspect_image(image_path, dataset.name)

    df = summary.to_dataframe()

    print(df.head())

    df.to_csv("outputs/dataset_summary.csv", index=False)

    print("\nDataset summary saved to outputs/dataset_summary.csv")

# Tissue Segmentation Test


    segmenter = TissueSegmenter()

    dataset = datasets[0]

    image_path = loader.get_images(dataset)[0]

    image = loader.load_image(image_path)

    mask = segmenter.create_mask(image)

    contours = segmenter.find_tissue(mask)

    overlay = segmenter.draw_contours(
        image,
        contours
    )

    final_mask = segmenter.create_final_mask(
        image,
        contours
    )

    segmenter.save_mask(
        final_mask,
        "outputs/masks/test_mask.jpg"
    )

    segmenter.save_image(
        overlay,
        "outputs/overlays/test_overlay.jpg"
    )

    print(f"Large tissue fragments detected: {len(contours)}")

if __name__ == "__main__":
    main()