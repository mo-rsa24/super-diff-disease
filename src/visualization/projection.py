import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from models.feature_extractor import get_resnet18_feature_extractor, extract_features
from sklearn.manifold import TSNE
from umap import UMAP
from torch.utils.data import Subset, DataLoader


def run_projection_3d_with_thumbnails(
    dataset,
    method="tsne",
    max_samples=150,
    batch_size=32,
    image_size=(28, 28),
    label_map={0: "Normal", 1: "TB"},
    class_colors={0: "green", 1: "red"},
    grayscale=True,
    save_path=None,
    animate_rotation=False
):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model = get_resnet18_feature_extractor()
    features, labels = extract_features(model, loader, max_samples=max_samples)
    labels = np.array(labels)

    if method == "tsne":
        projector = TSNE(n_components=3, random_state=42)
    elif method == "umap":
        projector = UMAP(n_components=3, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedded = projector.fit_transform(features)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"3D {method.upper()} Projection with Thumbnails")

    for i, (x, y, z) in enumerate(embedded):
        sample = dataset[indices[i]]
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

        # Normalize image
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-6)

        ax.scatter(x, y, z, color=class_colors.get(label, "gray"), s=50, alpha=0.6)

        # Optional: thumbnails as tiny text overlays (simplified, better in 2D)
        ax.text(x, y, z, f"{label_map.get(label, label)}", fontsize=6, color=class_colors.get(label, "gray"))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=45)
    plt.tight_layout()

    # Optional animation
    # if animate_rotation:
    #     def update(angle):
    #         ax.view_init(elev=20, azim=angle)
    #         return fig,

    #     ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    #     if save_path:
    #         from matplotlib.animation import FFMpegWriter
    #         writer = FFMpegWriter(fps=20, metadata=dict(artist='Your Name'), bitrate=1800)
    #         ani.save(save_path, writer=writer, dpi=150)
    # else:
    #     plt.show()
