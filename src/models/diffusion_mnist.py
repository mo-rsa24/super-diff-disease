# models/diffusion_mnist.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """
    Creates sinusoidal positional embeddings for timestep t.
    Output dimension = embed_dim.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor):
        # t: (batch_size,)
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (batch_size, embed_dim)


class ResidualBlock(nn.Module):
    """
    A ResNet‐style block that accepts a time embedding and conditions convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        # If channels differ, use a 1×1 conv to match dimensions
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        x: (B, C_in, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.block1(x)
        # Add time embedding (after projecting to out_channels)
        t_out = self.time_mlp(t_emb)  # (B, out_channels)
        h = h + t_out[..., None, None]
        h = self.block2(h)
        return h + self.res_conv(x)  # Residual add


class SelfAttention(nn.Module):
    """
    A simple self‐attention block at a given resolution.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each (B, C, H, W)

        # reshape for multihead: (B, heads, C/head, H*W)
        def reshape_to_heads(tensor):
            return tensor.view(B, self.num_heads, C // self.num_heads, H * W)

        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        # q: (B, heads, C/head, HW), k: (B, heads, C/head, HW)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, C/head)
        k = k.permute(0, 1, 3, 2)  # (B, heads, HW, C/head)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, C/head)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, HW, C/head)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return self.proj_out(out) + x  # residual


class UNet(nn.Module):
    """
    UNet for 1‐channel 32×32 images (e.g. MNIST). Base channels = 128, multipliers = [1,2,2,2].
    Adds an attention block at the bottleneck.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 128, time_emb_dim: int = 512):
        super().__init__()
        self.time_embed = SinusoidalPosEmb(time_emb_dim)
        # MLP to expand time embedding to a hidden dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Channel multipliers for each downsample step
        ch = [base_channels * m for m in [1, 2, 2, 2]]  # [128, 256, 256, 256]
        # Input conv
        self.conv_in = nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1)

        # Down blocks
        self.down1 = ResidualBlock(ch[0], ch[0], time_emb_dim)
        self.down2 = ResidualBlock(ch[0], ch[1], time_emb_dim)
        self.down3 = ResidualBlock(ch[1], ch[2], time_emb_dim)

        # Bottleneck
        self.bot1 = ResidualBlock(ch[2], ch[3], time_emb_dim)
        self.attn = SelfAttention(ch[3])
        self.bot2 = ResidualBlock(ch[3], ch[2], time_emb_dim)

        # Up blocks
        self.up3 = ResidualBlock(ch[2] * 2, ch[1], time_emb_dim)
        self.up2 = ResidualBlock(ch[1] * 2, ch[0], time_emb_dim)
        self.up1 = ResidualBlock(ch[0] * 2, ch[0], time_emb_dim)

        # Output conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch[0]),
            nn.GELU(),
            nn.Conv2d(ch[0], in_channels, kernel_size=1)
        )

        # Downsample / Upsample
        self.pool = nn.MaxPool2d(2)       # halving spatial dims
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (B, 1, 32, 32)
        t: (B,) integer timesteps in [0, T-1]
        """
        # 1. Time embedding
        t_emb = self.time_embed(t)                  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)                # (B, time_emb_dim)

        # 2. Down / encode
        h = self.conv_in(x)                         # (B, ch0, 32, 32)
        h1 = self.down1(h, t_emb)                   # (B, ch0, 32, 32)
        h2 = self.down2(self.pool(h1), t_emb)        # (B, ch1, 16, 16)
        h3 = self.down3(self.pool(h2), t_emb)        # (B, ch2, 8, 8)

        # 3. Bottleneck
        hb = self.bot1(self.pool(h3), t_emb)        # (B, ch3, 4, 4)
        hb = self.attn(hb)                          # (B, ch3, 4, 4)
        hb = self.bot2(hb, t_emb)                   # (B, ch2, 4, 4)

        # 4. Up / decode
        hu3 = self.upsample(hb)                     # (B, ch2, 8, 8)
        hu3 = torch.cat([hu3, h3], dim=1)           # (B, ch2*2, 8, 8)
        hu3 = self.up3(hu3, t_emb)                  # (B, ch1, 8, 8)

        hu2 = self.upsample(hu3)                    # (B, ch1, 16, 16)
        hu2 = torch.cat([hu2, h2], dim=1)           # (B, ch1*2, 16, 16)
        hu2 = self.up2(hu2, t_emb)                  # (B, ch0, 16, 16)

        hu1 = self.upsample(hu2)                    # (B, ch0, 32, 32)
        hu1 = torch.cat([hu1, h1], dim=1)           # (B, ch0*2, 32, 32)
        hu1 = self.up1(hu1, t_emb)                  # (B, ch0, 32, 32)

        return self.conv_out(hu1)                   # (B, 1, 32, 32)
