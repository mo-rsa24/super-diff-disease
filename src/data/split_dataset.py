import os
import shutil
import random
import argparse
from pathlib import Path


def is_split_already_done(split_dir, class_folders):
    """Check if all class folders exist and are non-empty inside a split directory."""
    for cls in class_folders:
        cls_path = split_dir / cls
        if not cls_path.exists():
            return False
        if not any(cls_path.glob("*.jpg")):
            return False
    return True

def split_dataset(
    input_dir,
    output_dir,
    classes=('Normal Chest X-rays', 'TB Chest X-rays'),
    split_ratio=(0.7, 0.15, 0.15),
    seed=42
):
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"

    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    split_names = ['train', 'val', 'test']

    for cls in classes:
        cls_path = input_dir / cls
        all_images = list(cls_path.glob("*.jpg"))
        random.shuffle(all_images)

        n_total = len(all_images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        n_test = n_total - n_train - n_val

        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:]
        }

        print(f"[{cls}] Total: {n_total} → Train: {n_train}, Val: {n_val}, Test: {n_test}")

        for split_name in split_names:
            split_path = output_dir / split_name
            if is_split_already_done(split_path, classes):
                print(f"✅ Skipping {split_name}/{cls} — already exists and populated.")
                continue

            class_split_path = split_path / cls
            class_split_path.mkdir(parents=True, exist_ok=True)
            for img_path in splits[split_name]:
                shutil.copy(img_path, class_split_path / img_path.name)
            print(f"📂 Created {split_name}/{cls} with {len(splits[split_name])} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Chest X-ray dataset.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to root dataset folder (with Normal/TB class folders)")
    parser.add_argument("--output_dir", type=str, default="output_split",
                        help="Path to save train/val/test folders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        classes=('Normal Chest X-rays', 'TB Chest X-rays'),
        split_ratio=(0.7, 0.15, 0.15),
        seed=args.seed
    )
