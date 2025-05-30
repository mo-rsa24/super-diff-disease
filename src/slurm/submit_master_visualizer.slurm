#!/bin/bash
#SBATCH --job-name=viz_exp1
#SBATCH --partition=biggpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/viz_exp1.log
#SBATCH --error=logs/viz_exp1.err

source ~/.bashrc
conda activate cxr

srun bash scripts/run_visualizations.sh $1 $2 $3