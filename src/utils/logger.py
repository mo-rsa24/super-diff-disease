# src/utils/logger.py
import logging, os
from datetime import datetime

def init_logger(log_dir, filename="training.log", log_to_stdout=False):
    os.makedirs(log_dir, exist_ok=True)
    full_path = os.path.join(log_dir, filename)

    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Reset

    fh = logging.FileHandler(full_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_stdout:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("🧠 Logger initialized")
    return logger
