# src/factories/registry.py

import copy
from typing import Callable, Dict
from datasets import Dataset

_DATASET_REGISTRY: Dict[str, Callable[[dict], Dataset]] = {}
_MODEL_DIFFUSION_REGISTRY: Dict[str, Callable[[dict], tuple]] = {}


def register_dataset(name: str):
    def decorator(fn: Callable[[dict], Dataset]):
        key = name.strip().upper()
        if key in _DATASET_REGISTRY:
            raise KeyError(f"Dataset '{key}' is already registered.")
        _DATASET_REGISTRY[key] = fn
        return fn
    return decorator


def get_dataset(name: str, config: dict, split_override: str = None) -> Dataset:
    key = name.strip().upper()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise KeyError(f"Dataset '{name}' not registered. Available: {available}")

    # Create a deep‐copy so nested dicts aren’t shared
    cfg_copy = copy.deepcopy(config)
    if split_override is not None:
        so = split_override.strip().lower()
        if so not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split_override='{split_override}'. Must be 'train', 'val', or 'test'.")
        cfg_copy.setdefault("training", {})["split"] = so

    return _DATASET_REGISTRY[key](cfg_copy)


def register_model_diffusion(name: str):
    def decorator(fn: Callable[[dict], tuple]):
        key = name.strip().upper()
        if key in _MODEL_DIFFUSION_REGISTRY:
            raise KeyError(f"Model/Diffusion '{key}' is already registered.")
        _MODEL_DIFFUSION_REGISTRY[key] = fn
        return fn
    return decorator


def get_model_diffusion(name: str, config: dict) -> tuple:
    key = name.strip().upper()
    if key not in _MODEL_DIFFUSION_REGISTRY:
        available = ", ".join(_MODEL_DIFFUSION_REGISTRY.keys())
        raise KeyError(f"Model/Diffusion '{name}' not registered. Available: {available}")
    # We do not need to split-override here, so pass config directly (which is already a deep copy if needed)
    return _MODEL_DIFFUSION_REGISTRY[key](config)
