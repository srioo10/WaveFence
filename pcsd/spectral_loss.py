"""
spectral_loss.py — Vulnerability-weighted spectral band loss for PCSD.

Novel loss (part of Contribution C4): Penalizes residual perturbation
energy MORE in frequency bands where the model is vulnerable.

    L_spectral = Σ_b energy_residual(b) × vulnerability(b)

Generic denoisers (DnCNN, U-Net) use L1/L2 loss which treats all
frequencies equally. This loss says: "I don't care if you're slightly
off in safe bands, but you MUST be clean in vulnerable bands."
"""

import torch
import torch.nn as nn

from caat.spectral import radial_bin_energy_torch


class SpectralBandLoss(nn.Module):
    """
    Vulnerability-weighted spectral band loss.

    Penalizes residual energy in vulnerable frequency bands more heavily
    than in safe bands.

    Args:
        n_bins: number of radial frequency bins
    """
    def __init__(self, n_bins: int = 16):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, x_purified: torch.Tensor, x_clean: torch.Tensor,
                vulnerability_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_purified: (B, 3, H, W) denoiser output
            x_clean: (B, 3, H, W) clean target
            vulnerability_profile: (B, n_bins) or (n_bins,) vulnerability weights

        Returns:
            scalar loss
        """
        B = x_purified.size(0)

        # Expand vulnerability profile if needed
        if vulnerability_profile.dim() == 1:
            v = vulnerability_profile.unsqueeze(0).expand(B, -1)  # (B, n_bins)
        else:
            v = vulnerability_profile

        residual = x_purified - x_clean  # (B, 3, H, W)
        loss = torch.tensor(0.0, device=x_purified.device)

        for c in range(3):
            # FFT of residual per channel
            F_res = torch.fft.fftshift(
                torch.fft.fft2(residual[:, c])
            )
            power = F_res.abs() ** 2  # (B, H, W)

            # Energy per radial bin
            energy = radial_bin_energy_torch(power, self.n_bins)  # (B, n_bins)

            # Weight by vulnerability
            weighted = energy * v  # (B, n_bins)
            loss = loss + weighted.sum(dim=1).mean()

        return loss / 3.0
