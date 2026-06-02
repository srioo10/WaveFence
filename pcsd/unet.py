"""
unet.py — PCSD: Profile-Conditioned Spectral Denoiser.

A lightweight U-Net conditioned on the model's Lipschitz profile via FiLM.
The Lipschitz profile tells the denoiser WHERE (in frequency) the model
is most sensitive, so it can apply targeted denoising.

Architecture:
  - Input: adversarial image (3, 32, 32) or (3, 224, 224)
  - Conditioning: Lipschitz profile vector (n_bins,)
  - FiLM: profile → (gamma, beta) per decoder layer
  - Output: denoised image (same size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (Perez et al. 2018)."""

    def __init__(self, n_channels, profile_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(profile_dim, n_channels)
        self.beta_fc = nn.Linear(profile_dim, n_channels)

        # Initialize to identity mapping
        nn.init.ones_(self.gamma_fc.weight.data[:, :1])
        nn.init.zeros_(self.gamma_fc.weight.data[:, 1:])
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(self, x, profile):
        """
        Args:
            x: (B, C, H, W) feature map
            profile: (B, profile_dim) conditioning vector
        Returns:
            (B, C, H, W) modulated feature map
        """
        gamma = self.gamma_fc(profile).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(profile).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PCSD(nn.Module):
    """
    Profile-Conditioned Spectral Denoiser.

    Lightweight U-Net with FiLM conditioning from Lipschitz profile.

    Args:
        in_channels: input channels (3 for RGB)
        profile_dim: length of Lipschitz profile vector (n_bins)
        base_ch: base channel count (scaled up in encoder)
    """

    def __init__(self, in_channels=3, profile_dim=32, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)         # 32
        self.enc2 = ConvBlock(base_ch, base_ch * 2)          # 64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)      # 128

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)  # 128

        # Decoder with FiLM conditioning
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)      # 128+128 → 64
        self.film3 = FiLMLayer(base_ch * 2, profile_dim)

        self.dec2 = ConvBlock(base_ch * 4, base_ch)           # 64+64 → 32
        self.film2 = FiLMLayer(base_ch, profile_dim)

        self.dec1 = ConvBlock(base_ch * 2, base_ch)           # 32+32 → 32
        self.film1 = FiLMLayer(base_ch, profile_dim)

        # Output: residual prediction
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

        # Pooling / upsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, profile):
        """
        Args:
            x: (B, 3, H, W) potentially adversarial image
            profile: (B, profile_dim) Lipschitz profile vector

        Returns:
            (B, 3, H, W) denoised image (residual learning: output = x - noise)
        """
        # Encoder
        e1 = self.enc1(x)                    # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))        # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))        # (B, 128, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))   # (B, 128, H/8, W/8)

        # Decoder + skip connections + FiLM
        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.film3(d3, profile)         # FiLM conditioning

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.film2(d2, profile)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.film1(d1, profile)

        # Residual learning: predict the noise, subtract it
        noise = self.out_conv(d1)
        denoised = x - noise

        return denoised


class DnCNN(nn.Module):
    """
    Baseline DnCNN denoiser (Zhang et al. 2017).
    No conditioning — just blind denoising.
    For comparison with PCSD.
    """

    def __init__(self, in_channels=3, num_layers=8, num_features=64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, num_features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, profile=None):
        """profile is ignored — this is the unconditional baseline."""
        noise = self.net(x)
        return x - noise
