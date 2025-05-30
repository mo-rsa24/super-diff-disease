import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP
from torch.utils.data import Subset, DataLoader
from models.feature_extractor import get_resnet18_feature_extractor, extract_features
from PIL import Image
import torch
import base64
from io import BytesIO

def tensor_to_base64_image(tensor, size=(64, 64)):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    img_np = (tensor.numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # C, H, W → H, W, C
    pil_img = Image.fromarray(img_np).resize(size)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_str}">'

def run_plotly_projection_3d_with_thumbnails(
    dataset,
    method="tsne",
    max_samples=300,
    batch_size=32,
    label_map={0: "Normal", 1: "TB"},
    class_colors={0: "green", 1: "red"},
    save_path=None
):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model = get_resnet18_feature_extractor()
    features, labels = extract_features(model, loader, max_samples=max_samples)
    labels = np.array(labels)

    # Projection
    if method == "tsne":
        projector = TSNE(n_components=3, random_state=42)
    elif method == "umap":
        projector = UMAP(n_components=3, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedded = projector.fit_transform(features)

    fig = go.Figure()

    for cls in np.unique(labels):
        x, y, z, hover_images = [], [], [], []
        for i, point_idx in enumerate(np.where(labels == cls)[0]):
            x.append(embedded[point_idx, 0])
            y.append(embedded[point_idx, 1])
            z.append(embedded[point_idx, 2])

            sample = dataset[indices[point_idx]]
            tensor = sample["image"]
            html_thumb = tensor_to_base64_image(tensor)
            label_name = label_map.get(cls, f"Class {cls}")
            hover_images.append(f"{label_name}<br>{html_thumb}")

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=label_map.get(cls, str(cls)),
            marker=dict(size=6, color=class_colors.get(cls, 'gray'), opacity=0.85),
            hoverinfo='text',
            hovertext=hover_images
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"3D {method.upper()} Projection with Thumbnails"
    )

    if save_path:
        fig.write_html(save_path)
        print(f"📁 Interactive plot saved to {save_path}")
    else:
        fig.show()