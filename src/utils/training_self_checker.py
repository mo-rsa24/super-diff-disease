import os
import json
import torch
import traceback
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

from src.monitoring.email_alert_mailtrap import alert_on_failure
from src.utils.crash_dump_manager import CrashDumpManager


class TrainingSelfChecker:
    """
    Performs health checks during training:
      - Gradient health (no NaN / Inf, optional clipping)
      - Loss sanity (finite, within thresholds)
      - Checkpoint integrity (required keys exist)
      - On failure: invokes CrashDumpManager and sends email alert.
    """

    def __init__(self,
                 run_id: str,
                 save_dir: str,
                 checkpoint_keys: list = None,
                 grad_clip: float = 1.0,
                 loss_threshold: float = None):
        """
        Args:
            run_id (str): Unique identifier for this run (e.g. "run_2025_06_03_tb_v2").
            save_dir (str): Base directory where run folders (and crash_dumps/) reside.
            checkpoint_keys (list): Keys that must appear in every checkpoint dict.
                                   Defaults to ["epoch","global_step","model","optimizer","ema","scheduler"].
            grad_clip (float): If > 0, clip gradients to this max‐norm after computing losses.
            loss_threshold (float | None): If set, raise if loss > threshold.
        """
        self.run_id = run_id
        self.checkpoint_keys = checkpoint_keys or [
            "epoch", "global_step", "model", "optimizer", "ema", "scheduler"
        ]
        self.grad_clip = grad_clip
        self.loss_threshold = loss_threshold

        # Prepare CrashDumpManager
        self.crash_manager = CrashDumpManager(save_dir=save_dir, run_id=run_id)

    def check_loss(self, loss: torch.Tensor, epoch: int, batch_idx: int, logger=None):
        """
        Verify that loss is finite and below threshold. On failure, dump and alert.
        """
        if not torch.isfinite(loss):
            msg = f"Invalid loss (NaN/Inf) at epoch {epoch}, batch {batch_idx}"
            self._handle_failure(epoch, batch_idx, msg, logger)
        elif self.loss_threshold is not None and loss.item() > self.loss_threshold:
            msg = f"Loss {loss.item():.4f} exceeds threshold {self.loss_threshold:.4f} at epoch {epoch}, batch {batch_idx}"
            self._handle_failure(epoch, batch_idx, msg, logger)

    def check_gradients(self, model: torch.nn.Module, epoch: int, batch_idx: int, logger=None):
        """
        Ensure no NaN/Inf in any gradient. Optionally clip gradients.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    msg = f"🚨 Invalid gradient detected in '{name}' at epoch {epoch}, batch {batch_idx}"
                    self._handle_failure(epoch, batch_idx, msg, logger)
        # Optionally clip to avoid exploding gradients
        if self.grad_clip and self.grad_clip > 0:
            clip_grad_norm_(model.parameters(), self.grad_clip)

    def verify_checkpoint_dict(self, checkpoint: dict, path: str, logger=None):
        """
        Ensure that every required key is present in a checkpoint dict before saving/loading.
        """
        missing = [k for k in self.checkpoint_keys if k not in checkpoint]
        if missing:
            msg = f"Checkpoint at {path} is missing keys: {missing}"
            self._handle_failure(checkpoint.get("epoch", -1),
                                 checkpoint.get("global_step", -1),
                                 msg, logger)

    def verify_checkpoint_file(self, path: str, logger=None):
        """
        Load a checkpoint file from `path`, verify it contains all required keys.
        If file does not exist or is corrupted, invoke crash dump and alert.
        """
        if not os.path.exists(path):
            msg = f"Checkpoint file not found: {path}"
            self._handle_failure(-1, -1, msg, logger)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            msg = f"Failed to load checkpoint '{path}': {str(e)}"
            self._handle_failure(-1, -1, msg, logger)
            return
        self.verify_checkpoint_dict(checkpoint, path, logger)

    def _handle_failure(self, epoch: int, batch_idx: int, error_msg: str, logger=None):
        """
        Central failure handler:
          - Instruct CrashDumpManager to write partial checkpoint + JSON log
          - Send email alert via alert_on_failure
          - Raise RuntimeError to abort training
        """
        # 1. Crash dump with current states
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        dump_name = f"failure_epoch_{epoch:03d}_batch_{batch_idx:03d}_{timestamp}"
        try:
            self.crash_manager.create_crash_dump(dump_name, logger, error_msg)
        except Exception as e_dump:
            if logger:
                logger.error(f"❌ CrashDumpManager failed: {e_dump}")

        # 2. Send email alert
        try:
            alert_on_failure(
                run_id=self.run_id,
                last_epoch=epoch,
                error_msg=error_msg
            )
        except Exception as e_email:
            if logger:
                logger.error(f"❌ Could not send failure email: {e_email}")

        # 3. Raise to halt training
        raise RuntimeError(error_msg)
