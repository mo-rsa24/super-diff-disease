# unet.py
import torch
import torch.nn as nn
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=min(4, in_ch), num_channels=in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=min(4, out_ch), num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.time_emb = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.block(x)
        t_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1)
        return h + t_emb


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.downs = nn.ModuleList([
            ResidualBlock(in_channels, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        ])
        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.ups = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim),
            ResidualBlock(base_channels, out_channels, time_emb_dim)
        ])

    def extract_bottleneck(self, x, t):
        t_emb = self.time_mlp(t)
        for down in self.downs:
            x = down(x, t_emb)
        return x


    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = x
        for down in self.downs:
            h = down(h, t_emb)
        h = self.mid(h, t_emb)
        for up in self.ups:
            h = up(h, t_emb)
        return h
