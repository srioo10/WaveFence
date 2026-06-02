"""
pgs_mask.py — PGS vulnerability mask for CAAT inner loop.

Computes which frequency bands have the largest human-model gap,
then creates a 2D mask for spectral projection of adversarial gradients.
"""

import numpy as np
import torch

from .config import EPS
from .spectral import compute_s_model, get_csf_profile, expand_radial_to_2d


def compute_pgs_mask(model, loader, device, n_bins, px_per_degree,
                     max_images=200, use_hanning=False):
    """
    Compute the PGS vulnerability mask.

    This mask is used in CAAT's inner loop to project adversarial gradients
    into the frequency bands where the model is most blind.

    Args:
        model: nn.Module
        loader: DataLoader
        device: torch device
        n_bins: radial bins
        px_per_degree: viewing geometry
        max_images: images for S_model computation
        use_hanning: for ViTs

    Returns:
        mask_1d: (n_bins,) normalized vulnerability weights [0, 1]
    """
    # Compute model sensitivity
    s_model_raw = compute_s_model(
        model, loader, device, n_bins,
        use_hanning=use_hanning,
        max_images=max_images,
        desc="PGS mask",
    )

    # Get CSF
    _, csf_raw = get_csf_profile(n_bins, px_per_degree)

    # Normalize to PDFs
    s_model_pdf = s_model_raw / (s_model_raw.sum() + EPS)
    csf_pdf = csf_raw / (csf_raw.sum() + EPS)

    # Gap: where model is less sensitive than humans
    gap = np.maximum(0, csf_pdf - s_model_pdf)

    # Normalize to [0, 1] — higher = more vulnerable = more weighting
    mask_1d = gap / (gap.max() + EPS)

    return mask_1d


def get_2d_mask(mask_1d, H, W, device):
    """
    Expand 1D radial mask to 2D frequency domain for gradient projection.

    Args:
        mask_1d: (n_bins,) vulnerability mask
        H, W: spatial dimensions
        device: torch device

    Returns:
        (H, W) tensor on device
    """
    mask_2d = expand_radial_to_2d(mask_1d, H, W)
    return mask_2d.to(device)


def spectral_project(grad: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Project adversarial gradient into vulnerable frequency bands.

    This is the key operation in CAAT's inner loop:
    instead of using the full gradient (like PGD), we multiply
    the gradient's FFT by the vulnerability mask so perturbations
    concentrate in the human-model blind spot.

    Args:
        grad: (B, C, H, W) gradient tensor
        mask_2d: (H, W) frequency-domain vulnerability mask

    Returns:
        (B, C, H, W) projected gradient in spatial domain
    """
    B, C, H, W = grad.shape
    device = grad.device
    mask = mask_2d.to(device)

    projected = torch.zeros_like(grad)
    for b in range(B):
        for c in range(C):
            # Forward FFT
            G = torch.fft.fftshift(torch.fft.fft2(grad[b, c]))
            # Apply vulnerability mask
            G_masked = G * mask
            # Inverse FFT
            projected[b, c] = torch.fft.ifft2(
                torch.fft.ifftshift(G_masked)
            ).real

    return projected
