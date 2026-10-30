
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.food11.data import CLASSES

SPLITS = ("training", "validation", "evaluation")


def validate(root: Path, min_per_class: int, tau: float) -> bool:
    errors = {
        "Class folders": [],
        "Image size and RGB": [],
        "Minimum images per class": [],
        "No duplicates across splits": [],
        "Class distribution": [],
    }
    counts = {}
    hashes = {}

    for split in SPLITS:
        split_dir = root / split
        actual = (
            {p.name for p in split_dir.iterdir() if p.is_dir()}
            if split_dir.is_dir()
            else set()
        )
        if actual != set(CLASSES):
            errors["Class folders"].append(
                f"{split}: expected 11 classes, found {len(actual)}"
            )

        counts[split] = Counter()
        for category in CLASSES:
            folder = split_dir / category
            if not folder.is_dir():
                continue

            for path in folder.iterdir():
                if not path.is_file():
                    continue

                counts[split][category] += 1
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != (128, 128):
                            errors["Image size and RGB"].append(str(path))
                except (OSError, ValueError, UnidentifiedImageError):
                    errors["Image size and RGB"].append(str(path))

                digest = hashlib.md5(path.read_bytes()).hexdigest()
                previous_split = hashes.get(digest)
                if previous_split and previous_split != split:
                    errors["No duplicates across splits"].append(
                        f"{path} duplicates an image from {previous_split}"
                    )
                hashes[digest] = split

            if counts[split][category] < min_per_class:
                errors["Minimum images per class"].append(
                    f"{split}/{category}: {counts[split][category]} "
                    f"< {min_per_class}"
                )

    total_train = sum(counts["training"].values())
    total_val = sum(counts["validation"].values())
    if not total_train or not total_val:
        errors["Class distribution"].append(
            "Training or validation split is empty"
        )
    else:
        for category in CLASSES:
            train_share = counts["training"][category] / total_train
            val_share = counts["validation"][category] / total_val
            if abs(train_share - val_share) > tau:
                errors["Class distribution"].append(category)

    for name, problems in errors.items():
        status = "PASS" if not problems else "FAIL"
        print(f"{status}: {name}")
        for problem in problems[:3]:
            print(f"  - {problem}")
        if len(problems) > 3:
            print(f"  ... and {len(problems) - 3} more")

    return all(not problems for problems in errors.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    success = validate(args.dataset, args.min_per_class, args.tau)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
