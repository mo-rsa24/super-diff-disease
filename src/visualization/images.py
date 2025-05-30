import cv2, random, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as T
from typing import List

_TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])  # →[-1,1]

def sample_paths(root: Path, klass: str, k: int = 8) -> List[Path]:
    paths = list((root/klass).glob("*.png"))   # or .jpeg
    return random.sample(paths, k)

def load_img(path: Path, clahe: bool = False):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img) if clahe else img
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype("uint8")

def plot_image_grid(root, n_per_class: int = 8, clahe=False):
    figs = []
    grid_imgs = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        for p in sample_paths(Path(root)/"train", cls, n_per_class):
            grid_imgs.append(load_img(p, clahe))
    grid = make_grid([_TO_TENSOR(i) for i in grid_imgs],
                     nrow=n_per_class, padding=2, normalize=True)
    plt.figure(figsize=(12,6))
    plt.title(f"Balanced {n_per_class}×2 sample grid (CLAHE={clahe})")
    plt.imshow(grid.permute(1,2,0).cpu(), cmap="gray"); plt.axis("off")

def plot_histogram(path: Path):
    img = load_img(path)
    plt.figure(figsize=(4,3))
    plt.hist(img.ravel(), bins=64, color="steelblue")
    plt.title(path.stem); plt.xlabel("Pixel value"); plt.ylabel("count")


import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt

def show_image(img, title=None):
    """
    Display an image from various formats using matplotlib.pyplot.imshow().
    
    Args:
        img: A PyTorch tensor, NumPy array, PIL image, or torchvision grid output.
        title (str, optional): Optional title for the plot.
    """
    # Convert PIL to numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Convert torch tensor to numpy
    elif isinstance(img, torch.Tensor):
        if img.ndim == 4:
            # Assume batched tensor: take first image
            img = img[0]

        if img.ndim == 3:
            # CHW → HWC if channels-first
            if img.shape[0] in [1, 3]:  # Likely CHW
                img = img.permute(1, 2, 0)
            elif img.shape[-1] in [1, 3]:  # Possibly already HWC
                pass
            else:
                raise ValueError(f"Ambiguous 3D tensor shape: {img.shape}")
        
        elif img.ndim == 2:
            # Grayscale, already 2D
            pass

        elif isinstance(img, torch.Tensor):
            raise ValueError(f"Unsupported tensor shape: {img.shape}")
        
        # Convert to numpy
        img = img.detach().cpu().numpy()

    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # Likely CHW
            img = np.transpose(img, (1, 2, 0))

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Squeeze singleton channels (e.g., grayscale [H, W, 1] → [H, W])
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)

    # Normalize float images if needed (matplotlib expects 0–1 floats or 0–255 ints)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1)

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
