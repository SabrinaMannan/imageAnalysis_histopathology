from src.config import load_config
from src.image_io import ImageLoader
from src.dataset_summary import DatasetSummary


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


if __name__ == "__main__":
    main()