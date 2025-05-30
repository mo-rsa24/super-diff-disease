# main.py

import argparse
import yaml
from PIL import Image

from test.test_and_visualize_pipeline import (
    print_class_mapping, verify_dataset, load_dataset,
    show_image_pair, visualize_batch, visualize_augmented_variants
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config YAML")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() # P( x   | label = Normal, label = TB)
    config = load_config(args.config)

    DATASET_ROOT = config.get("dataset_root", "ChestXray_Split/train")
    normalization = config.get("normalization", "minmax")
    resize_strategy = config.get("resize_strategy", "center_crop")
    hist_eq = config.get("hist_eq", False)
    aug_risk = config.get("augmentation", "medium")

    print("📦 Loading Dataset...")
    dataset = load_dataset(
        root_dir=DATASET_ROOT,
        normalization=normalization,
        resize_strategy=resize_strategy,
        hist_eq=hist_eq,
        aug_risk="low"
    )

    if config.get("print_class_counts", True):
        print_class_mapping()
        verify_dataset(dataset)

    if config.get("visualize_grid", False):
        visualize_batch(
            dataset,
            num_samples=12,
            title="Augmented Chest X-rays",
            show_labels=True,
            show_debug=True,
            rgb=True,
            ncols=4,
            padding=2,
            label_map={0: "Normal", 1: "TB"}
        )
        visualize_augmented_variants(dataset, image_idx=3, num_variants=5)

    if config.get("visualize_aug", False):
        sample_path = dataset.samples[0]
        original_img = Image.open(sample_path).convert("L")
        transformed_img = dataset[0]["image"].squeeze(0).numpy()
        show_image_pair(original_img, transformed_img, dataset[0]["class"])

    if config.get("visualize_tsne", False):
        from visualization.tsne import run_projection

        run_projection(dataset, method="tsne")

    if config.get("visualize_tsne_thumbnails", False):
        from visualization.tsne import run_projection_with_thumbnails

        run_projection_with_thumbnails(
            dataset,
            method="tsne",
            max_samples=300,
            image_size=(32, 32),
            title="t-SNE with Thumbnails",
            grayscale=True,
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"}
        )

    if config.get("visualize_umap", False):
        from visualization.tsne import run_projection

        run_projection(dataset, method="umap")
        
    if config.get("visualize_umap_thumbnails", False):
        from visualization.tsne import run_projection_with_thumbnails

        run_projection_with_thumbnails(
            dataset,
            method="umap",
            max_samples=300,
            image_size=(32, 32),
            title="UMAP with Thumbnails",
            grayscale=True,
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"}
        )
    if config.get("compare_tsne_umap_thumbnails", False):
        from visualization.tsne import compare_tsne_umap_thumbnails

        compare_tsne_umap_thumbnails(
            dataset,
            max_samples=300,
            image_size=(32, 32),
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"},
            title="t-SNE vs UMAP with Chest X-ray Thumbnails"
        )
    
    if config.get("visualize_3d_projection", False):
        from visualization.tsne import run_projection_3d

        run_projection_3d(
            dataset,
            method="tsne",  # or "umap"
            max_samples=300,
            title="3D t-SNE Projection (Chest X-rays)",
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"}
        )

    if config.get("visualize_3d_projection_thumbnails", False):
        from visualization.projection import run_projection_3d_with_thumbnails

        run_projection_3d_with_thumbnails(
            dataset,
            method="umap",  # or "tsne"
            max_samples=150,
            image_size=(28, 28),
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"},
            grayscale=True,
            save_path="projection_rotation.mp4",  # Optional
            animate_rotation=True
        )

    if config.get("visualize_3d_projection_plotly", False):
        from visualization.plotly import run_plotly_projection_3d_with_thumbnails

        run_plotly_projection_3d_with_thumbnails(
            dataset,
            method="umap",  # or "tsne"
            max_samples=300,
            label_map={0: "Normal", 1: "TB"},
            class_colors={0: "green", 1: "red"},
            save_path="plotly_3d_projection.html"
        )


    if config.get("gradcam", False):
        from visualization.gradcam import run_gradcam

        run_gradcam(dataset)

