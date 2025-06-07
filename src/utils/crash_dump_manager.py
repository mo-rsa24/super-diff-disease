import os
import json
import torch
import traceback
from datetime import datetime

class CrashDumpManager:
    """
    Responsible for writing out:
      - A partial checkpoint (model, optimizer, scheduler, EMA, epoch, step)
      - A structured JSON error log
    into runs/<run_id>/crash_dumps/.
    """

    def __init__(self, save_dir: str, run_id: str):
        """
        Args:
            save_dir (str): Base directory where run folders live.
            run_id (str): Unique run identifier, e.g. "run_2025_06_03_tb_v2".
        """
        self.run_id = run_id
        self.base_dir = os.path.join(save_dir, run_id)
        self.crash_dir = os.path.join(self.base_dir, "crash_dumps")
        os.makedirs(self.crash_dir, exist_ok=True)

    def create_crash_dump(self, dump_name: str, logger=None, error_msg: str = None,
                          model: torch.nn.Module = None,
                          optimizer: torch.optim.Optimizer = None,
                          scheduler=None,
                          ema=None,
                          epoch: int = None,
                          global_step: int = None,
                          extra: dict = None):
        """
        Writes out two files under crash_dumps/:
          1. <dump_name>.pt  --> partial checkpoint (model, optimizer, scheduler, EMA, epoch, step)
          2. <dump_name>.json --> JSON error log with traceback and metadata

        Args:
            dump_name (str): Unique name, e.g. "failure_epoch_012_batch_005_2025-06-03T14-22-10".
            logger: Optional logger to write debug/info messages.
            error_msg (str): The exception message that caused the dump.
            model, optimizer, scheduler, ema: Current training objects (can be None).
            epoch (int): Current epoch at failure.
            global_step (int): Current global step at failure.
            extra (dict): Any additional key-value pairs to log.
        """
        timestamp = datetime.utcnow().isoformat()
        # 1. Partial checkpoint
        if model is not None and optimizer is not None:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.ema_model.state_dict() if ema is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "error_msg": error_msg,
                "timestamp": timestamp,
                "extra": extra or {}
            }
            ckpt_path = os.path.join(self.crash_dir, f"{dump_name}.pt")
            torch.save(checkpoint, ckpt_path)
            if logger:
                logger.info(f"💾 Crash dump checkpoint saved: {ckpt_path}")
        else:
            ckpt_path = None
            if logger:
                logger.info("⚠️ Partial checkpoint skipped (model/optimizer was None).")

        # 2. JSON error log (traceback + metadata)
        trace_str = traceback.format_exc()
        log_entry = {
            "dump_name": dump_name,
            "error_message": error_msg,
            "traceback": trace_str,
            "epoch": epoch,
            "global_step": global_step,
            "timestamp": timestamp,
            "checkpoint_path": ckpt_path,
            "extra": extra or {}
        }
        json_path = os.path.join(self.crash_dir, f"{dump_name}.json")
        with open(json_path, "w") as f:
            json.dump(log_entry, f, indent=2)
        if logger:
            logger.info(f"📜 Crash dump JSON saved: {json_path}")
