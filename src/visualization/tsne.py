
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from umap import UMAP
from torch.utils.data import DataLoader, Subset

from src.models.feature_extractor import get_chexnet_densenet121_feature_extractor, get_resnet18_feature_extractor, extract_features
from src.visualization.images import show_image


def run_tsne(dataset, max_samples=300, batch_size=32, title="t-SNE of Chest X-ray Embeddings"):
    """
    Projects CNN features using t-SNE and visualizes the result.
    """
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # model = get_resnet18_feature_extractor()
    model = get_chexnet_densenet121_feature_extractor()
    features, labels = extract_features(model, dataloader, max_samples=max_samples)

    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, palette=["green", "red"], alpha=0.7)
    plt.title(title)
    plt.legend(title="Class (0 = Normal, 1 = TB)")
    plt.grid(True)
    plt.show()

def run_projection(dataset, method="tsne", max_samples=300, batch_size=32, title="Embedding Projection"):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    # model = get_resnet18_feature_extractor()
    model = get_chexnet_densenet121_feature_extractor()
    features, labels = extract_features(model, dataloader, max_samples=max_samples)

    if method == "tsne":
        projector = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        projector = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = projector.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette=["green", "red"], alpha=0.7)
    plt.title(f"{method.upper()} Projection of Chest X-ray Embeddings")
    plt.legend(title="Class (0 = Normal, 1 = TB)")
    plt.grid(True)
    plt.show()

def run_projection_with_thumbnails(
    dataset,
    method="tsne",
    max_samples=300,
    batch_size=32,
    image_size=(32, 32),
    title="t-SNE with Thumbnails",
    grayscale=True,
    label_map={0: "Normal", 1: "TB"},
    class_colors={0: "green", 1: "red"},
):
    from models.feature_extractor import get_resnet18_feature_extractor, extract_features
    from torch.utils.data import DataLoader, Subset
    import torch

    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # model = get_resnet18_feature_extractor()
    model = get_chexnet_densenet121_feature_extractor()
    features, labels = extract_features(model, loader, max_samples=max_samples)

    # Project features to 2D
    if method == "tsne":
        projector = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        projector = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    embedded = projector.fit_transform(features)

    # Normalize embeddings to [0, 1] for image positioning
    x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
    y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()
    norm_embedded = (embedded - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    for idx, (xy, i) in enumerate(zip(norm_embedded, indices)):
        sample = dataset[i]
        label = sample["class"]
        img = sample["image"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.shape[0] != 1 and img.shape[-1] == 1:
            img = img.permute(2, 0, 1)

        img_np = img.numpy().squeeze()
        img_disp = np.stack([img_np]*3, axis=-1) if grayscale else np.transpose(img.numpy(), (1, 2, 0))

        imagebox = OffsetImage(img_disp, zoom=image_size[0] / img_disp.shape[0], cmap="gray" if grayscale else None)
        ab = AnnotationBbox(imagebox, xy, frameon=True, bboxprops=dict(edgecolor=class_colors[label], linewidth=2))
        ax.add_artist(ab)

    plt.tight_layout()
    plt.show()


def compare_tsne_umap_thumbnails(
    dataset,
    max_samples=300,
    batch_size=32,
    image_size=(32, 32),
    label_map={0: "Normal", 1: "TB"},
    class_colors={0: "green", 1: "red"},
    grayscale=True,
    figsize=(18, 9),
    title="t-SNE vs UMAP with Thumbnails"
):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # model = get_resnet18_feature_extractor()
    model = get_chexnet_densenet121_feature_extractor()
    features, labels = extract_features(model, loader, max_samples=max_samples)

    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)
    umap = UMAP(n_components=2, random_state=42).fit_transform(features)

    projections = {
        "t-SNE": tsne,
        "UMAP": umap,
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    for ax, (method_name, embedded) in zip(axes, projections.items()):
        x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
        y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()
        norm_embedded = (embedded - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])

        for idx, (xy, i) in enumerate(zip(norm_embedded, indices)):
            sample = dataset[i]
            label = sample["class"]
            img = sample["image"]

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.shape[0] != 1 and img.shape[-1] == 1:
                img = img.permute(2, 0, 1)

            img_np = img.numpy().squeeze()
            img_disp = np.stack([img_np] * 3, axis=-1) if grayscale else np.transpose(img.numpy(), (1, 2, 0))

            imagebox = OffsetImage(img_disp, zoom=image_size[0] / img_disp.shape[0], cmap="gray" if grayscale else None)
            ab = AnnotationBbox(imagebox, xy, frameon=True, bboxprops=dict(edgecolor=class_colors[label], linewidth=2))
            ax.add_artist(ab)

        ax.set_title(method_name)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def run_projection_3d(
    dataset,
    method="tsne",
    max_samples=300,
    batch_size=32,
    label_map={0: "Normal", 1: "TB"},
    class_colors={0: "green", 1: "red"},
    title="3D Latent Projection of Chest X-rays"
):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # model = get_resnet18_feature_extractor()
    model = get_chexnet_densenet121_feature_extractor()
    features, labels = extract_features(model, loader, max_samples=max_samples)
    labels = np.array(labels)

    # 3D projection
    if method == "tsne":
        projector = TSNE(n_components=3, random_state=42)
    elif method == "umap":
        projector = UMAP(n_components=3, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedded = projector.fit_transform(features)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title)

    for cls in np.unique(labels):
        idxs = labels == cls
        ax.scatter(
            embedded[idxs, 0],
            embedded[idxs, 1],
            embedded[idxs, 2],
            label=label_map.get(cls, str(cls)),
            alpha=0.7,
            s=30,
            c=class_colors.get(cls, "gray")
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()
    plt.tight_layout()
    plt.show()


"""
tsne_umap.py

📌 Latent Space Visualization (TSNE, UMAP, PCA) for Diffusion Models.
Used to project intermediate U-Net features and assess cluster behavior.

Usage:
    from src.visuals.tsne_umap import plot_projection
    plot_projection(features, labels, "Epoch 5", "umap_epoch5.png", method="umap")
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

def plot_projection(features, labels, title, save_path, method="tsne", n_components=2):
    if method == "tsne":
        projector = TSNE(n_components=n_components)
    elif method == "umap":
        projector = UMAP(n_components=n_components)
    elif method == "pca":
        projector = PCA(n_components=n_components)
    else:
        raise ValueError("Unsupported method: " + method)

    proj = projector.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(f"{method.upper()} projection - {title}")
    plt.colorbar(scatter)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
