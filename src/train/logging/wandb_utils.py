# src/utils/wandb_utils.py

import wandb
from torchvision.utils import make_grid

def log_images_wandb(wandb_tracker, real_batch, gen_batch, epoch):
    if wandb_tracker is None:
        return
    grid_real = make_grid(real_batch, nrow=4, normalize=True)
    grid_gen = make_grid(gen_batch, nrow=4, normalize=True)
    wandb_tracker.log({
        "Grid/Real": wandb.Image(grid_real, caption=f"Epoch {epoch} Real"),
        "Grid/Generated": wandb.Image(grid_gen, caption=f"Epoch {epoch} Generated")
    }, step=epoch)

def log_table_wandb(wandb_tracker, real_batch, gen_batch, epoch):
    if wandb_tracker is None:
        return
    data = []
    for i in range(min(8, real_batch.size(0))):
        real_img = wandb.Image(real_batch[i].cpu(), caption=f"Real {i}")
        gen_img = wandb.Image(gen_batch[i].cpu(), caption=f"Generated {i}")
        data.append([epoch, real_img, gen_img])
    table = wandb.Table(columns=["epoch", "real", "generated"], data=data)
    wandb_tracker.log({"Samples": table}, step=epoch)

def log_metrics_wandb(wandb_tracker, metrics: dict, step: int):
    if wandb_tracker is None:
        return
    wandb_tracker.log(metrics, step=step)
