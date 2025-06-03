# src/reporting/paper_report_generator.py

import os, yaml, json
from datetime import datetime

def generate(run_dir):
    # Load metrics
    metrics_csv = os.path.join(run_dir, "metrics", "training_log.csv")
    results_path = os.path.join(run_dir, "reports", "results_for_paper.md")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    final_metrics = {
        "FID": None,
        "SSIM": None,
        "LPIPS": None
    }

    # Grab last values from metrics (or best if sorted later)
    try:
        with open(metrics_csv, "r") as f:
            lines = f.read().splitlines()
            headers = lines[0].split(",")
            last_row = lines[-1].split(",")

            for i, key in enumerate(headers):
                if key in final_metrics:
                    final_metrics[key] = last_row[i]
    except Exception as e:
        print(f"⚠️ Could not read metrics: {e}")

    # Load config
    try:
        with open(os.path.join(run_dir, "metadata", "config.yaml")) as f:
            config = yaml.safe_load(f)
    except:
        config = {}

    # Report body
    with open(results_path, "w") as f:
        f.write(f"# 🧪 Results Report for `{run_dir}`\n")
        f.write(f"Generated on `{datetime.utcnow().isoformat()}`\n\n")

        f.write("## 📊 Final Metrics\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in final_metrics.items():
            f.write(f"| {k} | {v or 'N/A'} |\n")

        f.write("\n## 🖼️ Sample Outputs\n")
        for img_name in ["epoch_050_denoised.png", "epoch_050_interpolation.png"]:
            f.write(f"![{img_name}](../samples/{img_name})\n")

        f.write("\n## ⚙️ Configuration Summary\n\n")
        summary = {
            "model": config.get("model", "UNet"),
            "dataset": config.get("dataset", "Unknown"),
            "loss": config.get("training", {}).get("loss_type", "default"),
            "seed": config.get("training", {}).get("seed", 42),
            "timesteps": config.get("training", {}).get("num_timesteps"),
            "batch_size": config.get("training", {}).get("batch_size"),
        }
        f.write("```yaml\n" + yaml.dump(summary) + "```\n")

    print(f"📄 Paper-ready markdown report saved at: {results_path}")
