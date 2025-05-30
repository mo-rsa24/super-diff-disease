
# 🧬 SuperDiff: Diffusion Models for Disease Discovery

## 🎯 Project Goals
Use class-specific diffusion models for Tuberculosis and Pneumonia to uncover disease patterns in chest X-rays using Super-Diff (Itô Density Estimator).

## 🏗️ Architecture Overview
- Class-conditional UNet Diffusion Models
- Super-Diff Itô Density Estimator for feature disentanglement
- Comparison & visualization via UMAP/TSNE, Grad-CAM
- Deployment via SLURM (Wits HPC) and local testing on VSCode

## 🛠️ Tooling
- Python 3.10+, PyTorch, Hydra, WandB
- SLURM job submission (cluster-side)
- GitHub + SSH integration
- Optional: Databricks, Streamlit dashboard

## 🔧 Environment Setup

### Local (VSCode / Legion)
```bash
conda create -n superdiff python=3.10
pip install -r requirements.txt
>>>>>>> 502efa2 (🔧 Initial project scaffolding with SSH & branching support)
