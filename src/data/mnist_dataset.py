# data/mnist_factory.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset


def get_mnist_dataloaders(batch_size: int = 128,
                          img_size: int = 32,
                          num_workers: int = 2,
                          split_ratio: float = 0.9,
                          seed: int = 42):
    """
    Returns train_loader and val_loader for MNIST diffusion experiments.
    - Downloads HF "mnist" train split, resizes to [img_size, img_size].
    - Normalizes images to [-1, 1].
    - Splits original 60k train set into train/val using split_ratio.
    """

    # 1. Define transformations
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),                        # -> [0,1]
        transforms.Normalize((0.5,), (0.5,))          # -> [-1,1]
    ])

    # 2. Load HF MNIST train split
    raw_train = load_dataset("mnist", split="train")
    # Wrap as a simple torch Dataset
    class HF_MNIST_Dataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            img = item["image"]                     # PIL Image
            label = item["label"]
            img_t = self.transform(img)             # (1, img_size, img_size)
            return {"image": img_t, "label": label}

    full_dataset = HF_MNIST_Dataset(raw_train, transform)

    # 3. Train/Val split
    n_train = int(len(full_dataset) * split_ratio)
    n_val = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size: int = 128, img_size: int = 32, num_workers: int = 2):
    """
    Returns a test DataLoader for HF MNIST test split.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    raw_test = load_dataset("mnist", split="test")

    class HF_MNIST_Test(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            img = item["image"]
            label = item["label"]
            img_t = self.transform(img)
            return {"image": img_t, "label": label}

    test_ds = HF_MNIST_Test(raw_test, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return test_loader
