# utils/data_manifest.py

import os, json, hashlib
from pathlib import Path
from collections import defaultdict

def hash_file(filepath, block_size=65536):
    """Generate SHA1 hash of a file."""
    hasher = hashlib.sha1()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]

def generate_data_manifest(
    dataset_root,
    task,
    split_seed,
    preprocessing_script_path,
    run_dir,
    expected_splits=["train", "val", "test"]
):
    task_root = Path(dataset_root) / task.upper()
    class_distribution = defaultdict(int)
    sample_counts = {}

    for split in expected_splits:
        split_path = task_root / split
        total = 0
        for cls in os.listdir(split_path):
            cls_path = split_path / cls
            count = len(list(cls_path.glob("*.jpg")))
            class_distribution[cls] += count
            total += count
        sample_counts[f"num_{split}_samples"] = total

    preprocessing_hash = hash_file(preprocessing_script_path)

    manifest = {
        "source": f"{task}_Xray_dataset",
        "split_seed": split_seed,
        "preprocessing_hash": preprocessing_hash,
        **sample_counts,
        "class_distribution": dict(class_distribution)
    }

    out_path = Path(run_dir) / "data"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
