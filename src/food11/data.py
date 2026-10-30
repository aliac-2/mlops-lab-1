
import shutil
from pathlib import Path

from PIL import Image

CLASS_NAMES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

CLASSES = [CLASS_NAMES[i] for i in range(11)]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "food11_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "food11_processed"
MINI_DIR = PROJECT_ROOT / "data" / "food11_processed_mini"

SPLITS = ["training", "evaluation", "validation"]


def category_from_filename(filename: str) -> str:
    """Return the Food-11 category from a filename such as 3_120.jpg."""
    class_id = int(Path(filename).stem.split("_")[0])
    return CLASS_NAMES[class_id]


def resize_image(img: Image.Image, size: int = 128) -> Image.Image:
    """Convert image to RGB and resize it to size x size."""
    return img.convert("RGB").resize(
        (size, size),
        Image.Resampling.LANCZOS,
    )


def prepare_dataset():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if MINI_DIR.exists():
        shutil.rmtree(MINI_DIR)

    for split in SPLITS:
        raw_split = RAW_DIR / split

        for class_name in CLASS_NAMES.values():
            (PROCESSED_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )
            (MINI_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )

        if not raw_split.exists():
            continue

        mini_counts = {class_id: 0 for class_id in CLASS_NAMES}

        for image_path in raw_split.iterdir():
            if not image_path.is_file():
                continue

            class_id = int(image_path.stem.split("_")[0])
            class_name = category_from_filename(image_path.name)

            with Image.open(image_path) as image:
                image = resize_image(image)

                processed_path = (
                    PROCESSED_DIR / split / class_name / image_path.name
                )
                image.save(processed_path)

                if mini_counts[class_id] < 100:
                    mini_path = (
                        MINI_DIR / split / class_name / image_path.name
                    )
                    image.save(mini_path)
                    mini_counts[class_id] += 1


if __name__ == "__main__":
    prepare_dataset()
    print("Food-11 data preparation completed.")
