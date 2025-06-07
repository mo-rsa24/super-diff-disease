# schedulers/beta_schedule.py

import torch
import numpy as np


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """
    Generates a linear schedule of betas from beta_start → beta_end over `timesteps` steps.

    Returns:
      betas: torch.Tensor of shape (timesteps,)
      alphas: torch.Tensor of shape (timesteps,) = 1 - betas
      alpha_cumprod: torch.Tensor of shape (timesteps,) = cumulative product of alphas
    """
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod


def get_noise_schedule(timesteps: int):
    """
    Convenience wrapper to return a dict of all schedules for DDPM:
      {
        "betas": torch.Tensor([...]),
        "alphas": torch.Tensor([...]),
        "alpha_cumprod": torch.Tensor([...]),
        "sqrt_alpha_cumprod": torch.Tensor([...]),
        "sqrt_one_minus_alpha_cumprod": torch.Tensor([...])
      }
    """
    betas, alphas, alpha_cumprod = linear_beta_schedule(timesteps)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": sqrt_alpha_cumprod,
        "sqrt_one_minus_alpha_cumprod": sqrt_one_minus_alpha_cumprod
    }
