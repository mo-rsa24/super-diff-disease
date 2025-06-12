# src/factories/chestxray_dataset.py

from torch.utils.data import Dataset
import os

from src.data.dataset import ChestXrayDataset
from src.factories.registry import register_dataset
from src.transforms import safe_augmentation


@register_dataset("CHEST_XRAY")
class ChestXray_Wrapper(Dataset):
    def __init__(self, config: dict):
        # Validate that config["dataset"] is either "TB" or "PNEUMONIA"
        task_name = config["dataset"].strip().upper()
        if task_name not in ("TB", "PNEUMONIA"):
            raise ValueError(f"Unsupported task='{task_name}' for ChestXray. Must be 'TB' or 'PNEUMONIA'.")

        # Resolve root directory
        paths = config.get("paths", {})
        if "cluster_base" not in paths or "local_base" not in paths:
            raise KeyError("`paths.cluster_base` and `paths.local_base` must be defined in config.")

        base_dir = paths["cluster_base"] if paths.get("use_cluster", False) else paths["local_base"]
        dataset_sub = paths.get("dataset_subdir", "datasets/cleaned")
        dataset_dir = os.path.join(base_dir, dataset_sub)


        # Check directory exists
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Expected chest‐xray folder not found: {dataset_dir}")

        # Validate split
        split = config.get("training", {}).get("split", "train").strip().lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split='{split}'. Must be 'train', 'val', or 'test'.")

        # Build augmentations
        aug_risk = config.get("training", {}).get("augmentation", "low")
        aug_norm = config.get("training", {}).get("normalization", "minmax")
        aug_transform = safe_augmentation(risk=aug_risk, normalization=aug_norm)

        class_filt = config.get("training", {}).get("class_filter", None)

        # Instantiate the raw dataset
        self.inner = ChestXrayDataset(
            root_dir=dataset_dir,
            split=split,
            aug=aug_transform,
            class_filter=class_filt,
            task=task_name,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]
