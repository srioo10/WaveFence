"""
film.py — Feature-wise Linear Modulation (FiLM) layer.

Perez et al. 2018: "FiLM: Visual Reasoning with a General Conditioning Layer"

FiLM modulates intermediate features based on a conditioning signal.
In PCSD, the conditioning signal is the PGS vulnerability profile.

    output = γ(v) * features + β(v)

This lets the denoiser adapt its behavior per-model:
different models have different vulnerability profiles → FiLM
makes the denoiser target different frequency bands.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on vulnerability profile.

    Args:
        n_channels: number of feature channels to modulate
        v_dim: dimension of the conditioning vector (= n_bins, typically 16)
    """
    def __init__(self, n_channels: int, v_dim: int = 16):
        super().__init__()
        self.gamma_fc = nn.Linear(v_dim, n_channels)
        self.beta_fc = nn.Linear(v_dim, n_channels)

        # Initialize gamma near 1 and beta near 0 (identity init)
        nn.init.ones_(self.gamma_fc.weight.data[:, 0])
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(self, features: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) feature maps
            v: (B, v_dim) vulnerability profile

        Returns:
            (B, C, H, W) modulated features
        """
        gamma = self.gamma_fc(v).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(v).unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        return gamma * features + beta
