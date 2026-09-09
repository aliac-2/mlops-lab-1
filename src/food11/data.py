from pathlib import Path
from PIL import Image
import shutil

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "food11_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "food11_processed"
MINI_DIR = PROJECT_ROOT / "data" / "food11_processed_mini"

SPLITS = ["training", "evaluation", "validation"]


def prepare_dataset():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if MINI_DIR.exists():
        shutil.rmtree(MINI_DIR)

    for split in SPLITS:
        raw_split = RAW_DIR / split

        for class_name in CLASS_NAMES.values():
            (PROCESSED_DIR / split / class_name).mkdir(
                parents=True, exist_ok=True
            )
            (MINI_DIR / split / class_name).mkdir(
                parents=True, exist_ok=True
            )

        if not raw_split.exists():
            continue

        mini_counts = {class_id: 0 for class_id in CLASS_NAMES}

        for image_path in raw_split.iterdir():
            if not image_path.is_file():
                continue

            class_id = int(image_path.stem.split("_")[0])
            class_name = CLASS_NAMES[class_id]

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize((128, 128), Image.Resampling.LANCZOS)

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