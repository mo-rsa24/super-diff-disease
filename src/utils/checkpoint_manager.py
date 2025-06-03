import os
import json
import glob
import torch
from datetime import datetime

class CheckpointManager:
    """
    Manages saving/loading of checkpoints, rotation of old files,
    and persistent metadata in checkpoints/log.json.
    """
    def __init__(self, save_dir: str, run_id: str, keep_last_k: int = 3):
        """
        Args:
            save_dir (str): Base directory where checkpoints/ and logs/ will live.
            run_id (str): Unique identifier for this training run (e.g., "superdiff_tb_v4").
            keep_last_k (int): How many most‐recent checkpoint files to keep on disk.
        """
        self.run_id = run_id
        self.keep_last_k = keep_last_k
        self.base_dir = os.path.join(save_dir, run_id)
        self.ckpt_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_path = os.path.join(self.ckpt_dir, "log.json")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Initialize (or load) the log.json
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({"history": []}, f, indent=2)

    def _write_log(self, entry: dict):
        # Append a new entry to checkpoints/log.json
        with open(self.log_path, "r+") as f:
            data = json.load(f)
            data["history"].append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def _list_ckpts(self):
        # Return a sorted list of all .ckpt filenames (full paths)
        pattern = os.path.join(self.ckpt_dir, "epoch_*.ckpt")
        return sorted(glob.glob(pattern))

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        ema: object,
        epoch: int,
        global_step: int,
        extra: dict = None
    ):
        """
        Save a unified checkpoint with model, optimizer, scheduler, EMA, epoch, step, plus any extra metadata.
        Enforces rotation: keeps only the last `keep_last_k` files on disk.
        """
        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.ema_model.state_dict() if ema else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        if extra:
            checkpoint["extra"] = extra

        # Filename: epoch_XXX.ckpt
        filename = f"epoch_{epoch:03d}.ckpt"
        full_path = os.path.join(self.ckpt_dir, filename)
        torch.save(checkpoint, full_path)

        # Write metadata to log.json
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "path": full_path,
            "saved_at": datetime.utcnow().isoformat(),
        }
        if extra:
            log_entry["extra"] = extra
        self._write_log(log_entry)

        # Rotate old checkpoints
        all_ckpts = self._list_ckpts()
        if len(all_ckpts) > self.keep_last_k:
            # delete the oldest (len - keep_last_k) files
            to_delete = all_ckpts[: len(all_ckpts) - self.keep_last_k]
            for old_path in to_delete:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        ema: object = None,
        device: str = "cpu"
    ):
        """
        Loads weights/states from the most recent checkpoint file found in ckpt_dir.
        Returns: (epoch, global_step)
        Raises:
            FileNotFoundError if no .ckpt exists.
        """
        all_ckpts = self._list_ckpts()
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoints found in {self.ckpt_dir}")

        latest_path = all_ckpts[-1]
        checkpoint = torch.load(latest_path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if ema and checkpoint.get("ema") is not None:
            ema.ema_model.load_state_dict(checkpoint["ema"])

        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        return epoch, global_step

    def load_any(
        self,
        target_epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        ema: object = None,
        device: str = "cpu"
    ):
        """
        Load a specific epoch_{target_epoch:03d}.ckpt if it exists.
        """
        filename = f"epoch_{target_epoch:03d}.ckpt"
        path = os.path.join(self.ckpt_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if ema and checkpoint.get("ema") is not None:
            ema.ema_model.load_state_dict(checkpoint["ema"])

        return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)
