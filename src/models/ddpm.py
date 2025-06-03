# models/ddpm.py
import torch
import torch.nn.functional as F
import numpy as np

class DDPM:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, self.T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def denoise_step(self, model, x_t, t):
        """
        Perform one reverse diffusion step: p_theta(x_{t-1} | x_t)
        Args:
            model: the trained UNet
            x_t: current noisy sample (B, C, H, W)
            t: int or tensor (scalar timestep)
        Returns:
            x_{t-1}
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])  # [B]

        
        t = t.to(device)  # 🩹 Move index tensor to the same device

        # 🛠 Ensure betas/alphas are on the same device too
        betas = self.betas.to(device)[t].view(-1, 1, 1, 1)
        alphas = self.alphas.to(device)[t].view(-1, 1, 1, 1)
        alpha_bars = self.alpha_bars.to(device)[t].view(-1, 1, 1, 1)

        # 🧠 Predict noise using model
        pred_noise = model(x_t, t)


        mean = (1. / torch.sqrt(alphas)) * (x_t - ((1 - alphas) / torch.sqrt(1 - alpha_bars)) * pred_noise)

        if t[0].item() == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(betas) * noise



    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_bar = self.alpha_bars.to(x_start.device)[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise


    def p_losses(self, denoise_model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    def training_step(self, model, x):
        bsz = x.size(0)
        t = torch.randint(0, self.T, (bsz,), device=x.device).long()
        return self.p_losses(model, x, t)

    def sample(self, model, image_shape, device):
        model.eval()
        x = torch.randn(image_shape).to(device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            alpha = self.alphas[t].to(device)
            alpha_bar = self.alpha_bars[t].to(device)
            beta = self.betas[t].to(device)

            pred_noise = model(x, t_tensor)
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        return x


# ddpm.py
def reconstruct(model, image, timesteps=[0, 100, 500, 999]):
    x_t = image
    visuals = [x_t]
    for t in reversed(timesteps):
        noise = torch.randn_like(x_t)
        x_t = model.denoise_step(x_t, torch.tensor([t]))
        visuals.append(x_t)
    return visuals
