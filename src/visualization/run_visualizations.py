# run_visualizations.py
import os, argparse, yaml
from pathlib import Path
from src.visualization.tsne import run_tsne, run_projection_with_thumbnails, compare_tsne_umap_thumbnails, run_projection_3d
from src.visualization.plotly import run_plotly_projection_3d_with_thumbnails
from src.visualization.projection import run_projection_3d_with_thumbnails
from src.visualization.images import plot_image_grid

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_save_dir(base_dir, experiment_id, run_id, task):
    path = Path(base_dir) / task / f"experiment_{experiment_id}_run_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--dataset", required=True)
    
    # Visualization toggles
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne_thumbnails", action="store_true")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--compare_tsne_umap", action="store_true")
    parser.add_argument("--projection3d", action="store_true")
    parser.add_argument("--projection3d_plotly", action="store_true")
    parser.add_argument("--image_grid", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

    # Setup
    dataset_dir = Path(config["paths"]["local_base"]) / "datasets" / "cleaned" / args.dataset.upper()
    experiment = f"experiment_{args.experiment_id}"
    run = f"run_{args.run_id}"
    base_out_dir = Path(config["paths"]["local_base"]) / "outputs"

    # Dummy loader (replace with real dataset)
    from dataset import ChestXrayDataset
    from transforms import safe_augmentation

    dataset = ChestXrayDataset(
        root_dir=dataset_dir,
        aug=safe_augmentation("low", "minmax"),
        class_filter=None,
    )

    # Run selected visualizations
    if args.tsne:
        print("📌 Running t-SNE...")
        run_tsne(dataset, title=f"t-SNE {experiment} {run}")

    if args.tsne_thumbnails:
        print("🖼️ Running t-SNE with thumbnails...")
        run_projection_with_thumbnails(dataset, method="tsne", title="t-SNE with thumbnails")

    if args.umap:
        print("📌 Running UMAP projection...")
        run_projection_with_thumbnails(dataset, method="umap", title="UMAP with thumbnails")

    if args.compare_tsne_umap:
        print("🔁 Comparing t-SNE vs UMAP...")
        compare_tsne_umap_thumbnails(dataset)

    if args.projection3d:
        print("🧠 Running 3D latent projection...")
        run_projection_3d_with_thumbnails(dataset, method="tsne")

    if args.projection3d_plotly:
        print("📊 Running interactive Plotly 3D projection...")
        run_plotly_projection_3d_with_thumbnails(dataset, method="tsne", save_path=str(get_save_dir(base_out_dir, args.experiment_id, args.run_id, "plotly") / "projection3d.html"))

    if args.image_grid:
        print("🖼️ Generating image grid...")
        plot_image_grid(root=dataset_dir)

if __name__ == "__main__":
    main()
