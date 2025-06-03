from src.metrics.fid_lpips import compute_fid_chexnet, compute_lpips, compute_ssim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from datetime import timedelta
from src.utils.visualization import show_real_vs_generated
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
import os
import random


def log_training_start(logger, model_name, experiment_id, run_id, total_epochs, total_batches):
    logger.info(
        f"🚀 Training started\n"
        f"🔧 Model: {model_name}\n"
        f"🧪 Experiment: {experiment_id}\n"
        f"🏃‍♂️ Run ID: {run_id}\n"
        f"⏳ Total Epochs: {total_epochs}, Batches/Epoch: {total_batches}"
    )

def log_batch_progress(logger, epoch, total_epochs, step, total_steps, loss, lr, time_elapsed, gpu_mem=None):
    """
    Human‐readable progress log at batch granularity.
    """
    hrs, rem = divmod(int(time_elapsed), 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    msg = (
        f"[Epoch {epoch}/{total_epochs}] "
        f"Step {step}/{total_steps} | "
        f"Loss: {loss:.4f} | "
        f"LR: {lr:.2e} | "
        f"Elapsed: {elapsed_str}"
    )
    if gpu_mem is not None:
        msg += f" | GPU_mem: {gpu_mem:.0f} MB"
    logger.info(msg)

def log_epoch_summary(logger, epoch, total_epochs, train_loss, val_loss=None, val_acc=None,
                      epoch_time=None, checkpoint_path=None, best_acc=None, metrics=None, save_dir=None):
    msg = (
        f"✅ Epoch {epoch}/{total_epochs} completed\n"
        f"📉 Train Loss: {train_loss:.4f}"
    )
    if val_loss is not None:
        msg += f" | 🧪 Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        msg += f" | 🎯 Val Acc: {val_acc:.2f}%"
    if epoch_time:
        msg += f" | ⏱️ Time: {timedelta(seconds=int(epoch_time))}"
    if best_acc is not None:
        msg += f"\n🌟 Best Val Acc so far: {best_acc:.2f}%"
    if checkpoint_path:
        msg += f"\n💾 Checkpoint saved: {checkpoint_path}"
    if save_dir:
        log_epoch_metrics(epoch, train_loss, val_loss or None, metrics or {}, save_dir)
    logger.info(msg)

def log_training_end(logger, best_val_acc, training_time):
    logger.info(
        f"🏁 Training finished!\n"
        f"🔝 Best Val Accuracy: {best_val_acc:.2f}%\n"
        f"🕒 Total Duration: {timedelta(seconds=int(training_time))}"
    )

def log_exception(logger, epoch=None, step=None, exception=None):
    msg = "❌ Exception during training"
    if epoch is not None:
        msg += f" at Epoch {epoch}"
    if step is not None:
        msg += f", Step {step}"
    logger.exception(f"{msg}: {str(exception)}")


def log_training_stats(epoch, avg_loss, duration, optimizer, logger, writer=None, wandb_tracker=None):
    lr = optimizer.param_groups[0]["lr"]

    if writer:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Epoch Duration (sec)", duration, epoch)
        writer.add_scalar("Learning Rate", lr, epoch)

    if wandb_tracker:
        wandb_tracker.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "epoch_duration_sec": duration,
            "lr": lr
        })

    log_json(
        logger,
        "📊 Logged training stats",
        epoch=epoch,
        avg_loss=avg_loss,
        duration_sec=duration,
        learning_rate=lr
    )

    
def log_json(logger, message, **extra_data):
    """
    Structured logging helper. Accepts any key-value metadata
    and attaches it to a structured JSON log.
    """
    payload = {k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in extra_data.items()}
    logger.info(message, extra={"extra_data": payload})

def log_visualizations(epoch, real_batch, generated_batch, writer, wandb_tracker, config, vis_dir, logger):
    os.makedirs(vis_dir, exist_ok=True)
    first_real, first_gen = real_batch[0], generated_batch[0]
    
    image_path = os.path.join(vis_dir, f"epoch_{epoch+1:03d}.png")
    show_real_vs_generated(
        real=first_real.squeeze().cpu(),
        generated=first_gen.squeeze().cpu(),
        title=f"Epoch {epoch+1}",
        save_path=image_path
    )

    log_json(logger, "🖼️ Saved sample comparison image", epoch=epoch+1, path=image_path)
                
    if config["observability"].get("log_single_sample", True):
        if writer:
            writer.add_image("Sample/Real", first_real, epoch + 1)
            writer.add_image("Sample/Generated", first_gen, epoch + 1)
        if wandb_tracker:
            wandb_tracker.log({"sample_pair": wandb_tracker.Image(make_grid([first_real, first_gen], nrow=2, normalize=True))}, step=epoch + 1)
    
        log_json(logger, "☁️ Saved sample comparison image on Weights & Biases / Tensorboard", epoch=epoch+1, path=image_path)
    
    if config["observability"].get("log_batch_grid", True):
        grid_real = make_grid(real_batch, nrow=4, normalize=True)
        grid_gen = make_grid(generated_batch, nrow=4, normalize=True)
        if writer:
            writer.add_image("Grid/Real", grid_real, epoch + 1)
            writer.add_image("Grid/Generated", grid_gen, epoch + 1)
        if wandb_tracker:
            wandb_tracker.log({"real_grid": wandb_tracker.Image(grid_real), "gen_grid": wandb_tracker.Image(grid_gen)}, step=epoch + 1)
        log_json(logger, "🖼️ Grid visualizations saved", epoch=epoch+1, vis_dir=vis_dir)


    if config["observability"].get("wandb_sample_table", False):
        wandb_tracker.log({"Samples": wandb_tracker.Table(columns=["epoch", "real", "generated"],
            data=[[epoch + 1, wandb_tracker.Image(first_real), wandb_tracker.Image(first_gen)]])})
        log_json(logger, "☁️ Saved image comparison table on Weights & Biases / Tensorboard", epoch=epoch+1, path=image_path)
        
# src/utils/training_logger_utils.py

import os, json, csv
from datetime import datetime

def log_grad_norms(logger, model, epoch=None, step=None, writer=None, wandb_tracker=None):
    """
    Compute and log the global L2‐norm of gradients across all parameters.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item()
            total_norm_sq += param_norm ** 2
    total_norm = total_norm_sq ** 0.5

    tag = "train/grad_norm"
    if epoch is not None and step is None:
        logger.info(f"📏 [Epoch {epoch}] Grad Norm: {total_norm:.4f}")
    elif epoch is not None and step is not None:
        logger.info(f"📏 [Epoch {epoch} Step {step}] Grad Norm: {total_norm:.4f}")
    else:
        logger.info(f"📏 Grad Norm: {total_norm:.4f}")

    if writer and epoch is not None:
        writer.add_scalar(tag, total_norm, epoch if step is None else (epoch * 100000 + step))
    if wandb_tracker:
        wandb_tracker.log({tag: total_norm}, step=(epoch if step is None else (epoch * 100000 + step)))


def log_scheduler_step(logger, optimizer, epoch=None, step=None, writer=None, wandb_tracker=None):
    """
    Log the current LR whenever the scheduler steps.
    """
    lr_now = optimizer.param_groups[0]["lr"]
    if epoch is not None and step is None:
        logger.info(f"🔧 Scheduler step at epoch {epoch}: new LR = {lr_now:.2e}")
    elif epoch is not None and step is not None:
        logger.info(f"🔧 Scheduler step at epoch {epoch}, step {step}: new LR = {lr_now:.2e}")
    else:
        logger.info(f"🔧 Scheduler: new LR = {lr_now:.2e}")

    if writer and epoch is not None:
        writer.add_scalar("train/lr", lr_now, epoch if step is None else (epoch * 100000 + step))
    if wandb_tracker:
        wandb_tracker.log({"train/lr": lr_now}, step=(epoch if step is None else (epoch * 100000 + step)))


def log_validation_metrics(logger, metrics_dict, epoch=None, writer=None, wandb_tracker=None):
    """
    Given a dict of metrics (e.g. {'val_loss': 0.123, 'val_fid': 12.45}), log them.
    """
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    msg_parts = []
    for k, v in metrics_dict.items():
        msg_parts.append(f"{k}: {v:.4f}")
    logger.info(f"{prefix}Validation ‖ " + " | ".join(msg_parts))

    if writer and epoch is not None:
        for k, v in metrics_dict.items():
            writer.add_scalar(f"val/{k}", v, epoch)
    if wandb_tracker and epoch is not None:
        wandb_tracker.log({f"val/{k}": v for k, v in metrics_dict.items()}, step=epoch)

def log_epoch_metrics(epoch, train_loss, val_loss, metrics, save_dir):
    timestamp = datetime.utcnow().isoformat()
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "metrics": metrics,
        "timestamp": timestamp
    }

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"epoch_{epoch:03d}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    csv_path = os.path.join(save_dir, "training_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def evaluate_batch_metrics(real_batch, gen_batch, config, logger=None, wandb_tracker=None, epoch=None):
    results = {}

    if config["observability"].get("compute_fid", False):
        results["fid"] = compute_fid_chexnet(real_batch.cpu(), gen_batch.cpu())
        if logger: logger.info(f"[Epoch {epoch}] FID: {results['fid']:.4f}")
        if wandb_tracker: wandb_tracker.log({"FID": results["fid"]}, step=epoch)
        log_json(logger, f"📐 Computed FID", epoch=epoch, fid=results["fid"])
    if config["observability"].get("compute_lpips", False):
        results["lpips"] = compute_lpips(real_batch.cpu(), gen_batch.cpu())
        if logger: logger.info(f"[Epoch {epoch}] LPIPS: {results['lpips']:.4f}")
        if wandb_tracker: wandb_tracker.log({"LPIPS": results["lpips"]}, step=epoch)
        log_json(logger, f"🎯 Computed LPIPS", epoch=epoch, lpips=results["lpips"])

    if config["observability"].get("compute_ssim", False):
        results["ssim"] = compute_ssim(real_batch.cpu(), gen_batch.cpu())
        if logger: logger.info(f"[Epoch {epoch}] SSIM: {results['ssim']:.4f}")
        if wandb_tracker: wandb_tracker.log({"SSIM": results["ssim"]}, step=epoch)
        log_json(logger, f"🔍 Computed SSIM", epoch=epoch, ssim=results["ssim"])

    return results

def plot_projection(features, labels, title, save_path, logger, method="tsne"):
    """
    Projects high-dimensional features to 2D using TSNE or UMAP.

    Args:
        features (Tensor or ndarray): (B, D)
        labels (list or ndarray): (B,) class labels or dummy labels
        title (str): Plot title
        save_path (str): Output path for saving the plot
        method (str): "tsne" or "umap"
    """
    features = features.cpu().numpy() if hasattr(features, 'cpu') else features
    reducer = TSNE(n_components=2) if method == "tsne" else umap.UMAP(n_components=2)
    projection = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=labels, palette="tab10", s=20)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", fontsize='small')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    log_json(logger, f"📈 {method.upper()} projection saved", projection=method, title=title, path=save_path)

    plt.close()


@torch.no_grad()
def visualize_feature_maps(model, input_batch, layers_to_hook, device, save_dir, sample_id=0):
    """
    Hooks into specified layers and visualizes intermediate feature maps for a random sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    features = {}
    hooks = []

    def get_hook(name):
        def hook_fn(module, input, output):
            features[name] = output.detach().cpu()
        return hook_fn

    for name, layer in layers_to_hook.items():
        hooks.append(layer.register_forward_hook(get_hook(name)))

    idx = random.randint(0, input_batch.size(0) - 1)
    _ = model(input_batch.to(device))

    for name, feat in features.items():
        fmap = feat[idx]
        fig, axes = plt.subplots(1, min(4, fmap.size(0)), figsize=(12, 4))
        for i in range(min(4, fmap.size(0))):
            axes[i].imshow(fmap[i], cmap='viridis')
            axes[i].set_title(f"{name} | Ch {i}")
            axes[i].axis('off')
        fig.tight_layout()
        path = os.path.join(save_dir, f"featuremap_{name}_epoch_{sample_id}.png")
        plt.savefig(path)
        plt.close(fig)

    for h in hooks:
        h.remove()
