# src/factories/mnist_dataset.py

from torch.utils.data import Dataset
import torch
from datasets import load_dataset  # HuggingFace datasets
from torchvision import transforms

@register_dataset("MNIST")
class HF_MNIST(Dataset):
    def __init__(self, config: dict):
        raw_split = config.get("training", {}).get("split", "train").strip().lower()
        if raw_split == "val":
            hf_split = "test"
        elif raw_split in ("train", "test"):
            hf_split = raw_split
        else:
            raise ValueError(f"Unsupported split='{raw_split}' for MNIST. Choose 'train', 'val', or 'test'.")

        try:
            self.raw = load_dataset("mnist", split=hf_split)
        except Exception as e:
            raise RuntimeError("Failed to load MNIST from HuggingFace. "
                               "Check your Internet connection or HF cache.") from e

        img_size = int(config.get("model", {}).get("image_size", 64))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        pil_img = item["image"]
        label = item["label"]
        img_tensor = self.transform(pil_img)
        return {"image": img_tensor, "label": label}
