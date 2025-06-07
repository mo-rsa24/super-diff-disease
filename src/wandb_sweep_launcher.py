# wandb_sweep_launcher.py
import argparse
import os
import yaml
import subprocess

def run_wandb_sweep(config_path="sweep.yaml", project="super-diff-xray"):
    sweep_id = subprocess.check_output(["wandb", "sweep", config_path]).decode("utf-8").strip().split("/")[-1]
    print(f"🔁 Sweep launched: {sweep_id}")
    subprocess.call(["wandb", "agent", f"{project}/{sweep_id}"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sweep.yaml", help="Path to W&B sweep config file.")
    parser.add_argument("--project", type=str, default="super-diff-xray", help="W&B project name.")
    args = parser.parse_args()
    run_wandb_sweep(args.config, args.project)
