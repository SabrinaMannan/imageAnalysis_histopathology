from pathlib import Path

from src.config import load_config
from src.image_io import ImageLoader


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

    for dataset in datasets:

        print(dataset.name)

        images = loader.get_images(dataset)

        print(f"Images: {len(images)}")

        print("-" * 40)


if __name__ == "__main__":

    main()