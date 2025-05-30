import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.utils import make_grid

from src.data.dataset import ChestXrayDataset
from src.transforms import build_preprocessing, safe_augmentation


def print_class_mapping():
    mapping = {0: "Normal", 1: "TB"}
    print(f"\n✅ Label Mapping:\n{mapping}")
    return mapping

def verify_dataset(dataset):
    labels = [sample['class'] for sample in dataset]
    counter = Counter(labels)
    total = len(labels)

    print(f"\n📊 Total Samples: {total}")
    for label, count in counter.items():
        percent = 100 * count / total
        print(f" - Class {label} ({'Normal' if label == 0 else 'TB'}): {count} samples ({percent:.2f}%)")

    return counter

def load_dataset(root_dir, normalization='minmax', resize_strategy='center_crop', hist_eq=False, aug_risk='low'):
    transform = build_preprocessing(normalization, resize_strategy, hist_eq) if aug_risk == "none" else None
    aug = safe_augmentation(aug_risk, normalization=normalization)
    return ChestXrayDataset(root_dir, transform=transform, aug=aug)

def show_image_pair(before, after, label, title="Transform Comparison"):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(before, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(after, cmap='gray')
    ax[1].set_title("Transformed")
    for a in ax: a.axis('off')
    fig.suptitle(f"{title} | Label: {'Normal' if label == 0 else 'TB'}")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_batch(
    dataset,
    num_samples=8,
    title="Transformed CXR Grid",
    show_labels=True,
    show_debug=False,
    rgb=False,
    ncols=4,
    padding=2,
    label_map={0: "Normal", 1: "TB"},
    num_attempts=1,
    seed=None,
):
    """
    Visualize a batch of chest X-ray images, with optional multiple augmentations per sample.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    nrows = int(np.ceil((num_samples * num_attempts) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten() if nrows > 1 else [axes]

    for ax in axes:
        ax.axis('off')

    idx = 0
    for base_i, sample_idx in enumerate(indices):
        for j in range(num_attempts):
            if idx >= len(axes):
                break
            sample = dataset[sample_idx]
            img = sample["image"]
            label = sample["class"]

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.shape[0] != 1 and img.shape[-1] == 1:
                img = img.permute(2, 0, 1)

            if rgb and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            img_np = img.numpy()
            img_disp = np.transpose(img_np, (1, 2, 0))  # C, H, W → H, W, C

            ax = axes[idx]
            ax.imshow(img_disp.squeeze(), cmap="gray" if not rgb else None)
            ax.axis("off")

            label_str = label_map.get(label, f"Class {label}")
            if show_debug:
                label_str += f"\n{tuple(img.shape)}"
                label_str += f"\n{img.min():.2f}-{img.max():.2f}"
            if num_attempts > 1:
                label_str += f"\nAug#{j+1}"

            ax.set_title(label_str, fontsize=9)
            idx += 1

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(pad=padding)
    plt.show()

def visualize_augmented_variants(dataset, image_idx=0, num_variants=4, label_map={0: "Normal", 1: "TB"}):
    """
    Show multiple augmented variants of the same image index.
    """
    fig, axes = plt.subplots(1, num_variants, figsize=(num_variants * 3, 3))
    if num_variants == 1:
        axes = [axes]

    for i in range(num_variants):
        sample = dataset[image_idx]
        img = sample["image"]
        label = sample["class"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.shape[0] != 1 and img.shape[-1] == 1:
            img = img.permute(2, 0, 1)

        img_np = img.numpy()
        img_disp = np.transpose(img_np, (1, 2, 0))  # [C, H, W] → [H, W, C]

        axes[i].imshow(img_disp.squeeze(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{label_map.get(label)}\nVariant #{i+1}", fontsize=10)

    plt.suptitle(f"Augmented Variants of Sample #{image_idx}", fontsize=13)
    plt.tight_layout()
    plt.show()




