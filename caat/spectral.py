"""
spectral.py — Core spectral analysis utilities.

Extracted and cleaned from spectral_vulnerability_map.py.
Provides: CSF computation, S_model gradient sensitivity, radial binning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import EPS


# ── CSF: Mannos-Sakrison 1974 ─────────────────────────────────────────────────
def csf_mannos_sakrison(freqs_cpd: np.ndarray) -> np.ndarray:
    """
    Human Contrast Sensitivity Function.
    Mannos & Sakrison (1974): a*(c + b*f)*exp(-(b*f)^d)
    Peaks around 3–5 cycles/degree, falls off at high frequencies.

    Args:
        freqs_cpd: spatial frequencies in cycles per degree

    Returns:
        CSF values (non-negative)
    """
    a, b, c = 2.6, 0.114, 1.1
    csf = a * (0.0192 + b * freqs_cpd) * np.exp(-(b * freqs_cpd) ** c)
    return np.maximum(csf, 0.0)


def get_csf_profile(n_bins: int, px_per_degree: float) -> tuple:
    """
    Build CSF values at radial frequency bin centers.

    Args:
        n_bins: number of radial frequency bins
        px_per_degree: pixels per degree (viewing geometry dependent)
            CIFAR-10/GTSRB 32×32: ~4.57 px/degree
            ImageNet 224×224: 32.0 px/degree

    Returns:
        (bin_centers_cpd, csf_values) — both shape (n_bins,)
    """
    f_max_cpd = 0.5 * px_per_degree       # Nyquist frequency in cpd
    bin_edges = np.linspace(0, f_max_cpd, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    csf_vals = csf_mannos_sakrison(bin_centers)
    return bin_centers, csf_vals


# ── 2D Hanning window (suppresses ViT patch-grid artifacts) ───────────────────
def hanning_2d(H: int, W: int, device) -> torch.Tensor:
    """Hanning window to suppress ViT patch-grid spectral harmonics."""
    return torch.outer(
        torch.hann_window(H, periodic=False, device=device),
        torch.hann_window(W, periodic=False, device=device),
    )


# ── Vectorized radial binning (GPU-ready) ─────────────────────────────────────
def radial_profile_gpu(magnitude: torch.Tensor, n_bins: int) -> np.ndarray:
    """
    Radially-averaged power spectrum using vectorized bincount.

    Args:
        magnitude: (H, W) FFT magnitude, already fftshifted
        n_bins: number of radial bins

    Returns:
        (n_bins,) radial mean profile
    """
    device = magnitude.device
    H, W = magnitude.shape
    cy, cx = H // 2, W // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(torch.sqrt(torch.tensor(cx**2 + cy**2, dtype=torch.float32)))

    idx = (r / r_max * (n_bins - 1)).clamp(0, n_bins - 1).long().view(-1)
    mag_flat = magnitude.view(-1).float()

    sums = torch.bincount(idx, weights=mag_flat, minlength=n_bins)
    counts = torch.bincount(idx, minlength=n_bins).clamp(min=1)
    return (sums / counts).cpu().numpy()


def radial_bin_energy_torch(power_2d: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Differentiable radial binning for spectral band loss.
    Returns per-bin SUMMED energy (not averaged), suitable for loss computation.

    Args:
        power_2d: (B, H, W) batch of |FFT|^2 maps (fftshifted)
        n_bins: number of radial bins

    Returns:
        (B, n_bins) energy per bin
    """
    B, H, W = power_2d.shape
    device = power_2d.device
    cy, cx = H // 2, W // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(torch.sqrt(torch.tensor(cx**2 + cy**2, dtype=torch.float32)))
    idx = (r / r_max * (n_bins - 1)).clamp(0, n_bins - 1).long()  # (H, W)

    # One-hot bin membership → sum per bin
    idx_flat = idx.view(-1)  # (H*W,)
    one_hot = F.one_hot(idx_flat, n_bins).float()  # (H*W, n_bins)

    power_flat = power_2d.view(B, -1)              # (B, H*W)
    energy = torch.matmul(power_flat, one_hot)      # (B, n_bins)
    return energy


# ── S_model(f): gradient sensitivity spectrum ─────────────────────────────────
def compute_s_model(model, loader, device, n_bins, use_hanning=False,
                    max_images=None, desc="S_model(f)"):
    """
    Compute the model's gradient sensitivity spectrum.

    For each image:
      1. Forward pass → get logits
      2. Backward w.r.t. 2nd-most-likely class (targeted gradient)
      3. Take |gradient|, average across channels → (H, W)
      4. FFT → radial binning → profile

    Args:
        model: nn.Module
        loader: DataLoader
        device: torch device
        n_bins: number of radial frequency bins
        use_hanning: apply Hanning window (recommended for ViTs)
        max_images: limit number of images (for speed)
        desc: progress bar description

    Returns:
        profile_avg: (n_bins,) mean radial gradient PSD
    """
    model.eval()
    profiles = []
    n_processed = 0

    win = None  # lazily initialized

    for imgs, labels in tqdm(loader, desc=f"  {desc}"):
        imgs = imgs.to(device)

        for i in range(imgs.shape[0]):
            if max_images and n_processed >= max_images:
                break

            img = imgs[i:i + 1]
            H, W = img.shape[2], img.shape[3]

            # Initialize Hanning window on first image
            if use_hanning and win is None:
                win = hanning_2d(H, W, device)

            # Targeted gradient toward 2nd-most-likely class
            img_req = img.detach().clone().requires_grad_(True)
            logits = model(img_req)
            sorted_cls = logits[0].argsort(descending=True)
            target_cls = sorted_cls[1]       # 2nd most likely
            loss = logits[0][target_cls]
            loss.backward()

            grad = img_req.grad[0].abs().mean(dim=0)  # (H, W)

            if win is not None:
                grad = grad * win

            F_shift = torch.fft.fftshift(torch.fft.fft2(grad))
            profile = radial_profile_gpu(F_shift.abs(), n_bins)
            profiles.append(profile)
            n_processed += 1

        if max_images and n_processed >= max_images:
            break

    return np.stack(profiles).mean(axis=0)


# ── Build 2D radial frequency mask from 1D profile ───────────────────────────
def expand_radial_to_2d(profile_1d: np.ndarray, H: int, W: int) -> torch.Tensor:
    """
    Expand a radial (1D) frequency profile to a 2D frequency-domain mask.

    Args:
        profile_1d: (n_bins,) values per radial bin
        H, W: spatial dimensions

    Returns:
        (H, W) tensor where each pixel contains the profile value
        corresponding to its radial frequency bin
    """
    n_bins = len(profile_1d)
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[:H, :W].astype(np.float32)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = np.sqrt(cx**2 + cy**2)
    idx = np.clip((r / r_max * (n_bins - 1)).astype(int), 0, n_bins - 1)
    mask_2d = profile_1d[idx]
    return torch.from_numpy(mask_2d).float()
