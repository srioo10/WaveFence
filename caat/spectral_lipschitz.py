"""
spectral_lipschitz.py — Novel robustness metrics (replacing PRI/VCI).

Two metrics that DON'T use CSF:

1. GSE (Gradient Spectral Entropy)
   - Shannon entropy of the model's gradient PSD distribution
   - Measures how spread/concentrated gradient sensitivity is across frequencies
   - High GSE → attention spread evenly → harder to target-attack
   - Low GSE → concentrated in few bands → easy to exploit

2. SLI (Spectral Lipschitz Index)  
   - Per-frequency-band Lipschitz estimation using band-limited perturbations
   - SLI = max(band_lipschitz) / mean(band_lipschitz)
   - Measures: "is there one frequency band WAY more sensitive than average?"
   - High SLI → one exploitable band → vulnerable
   - SLI ≈ 1 → all bands equally sensitive → hard to target specifically

Together: GSE captures structural diversity, SLI captures exploitability.

Author: Sooraj S (IIITDM Kancheepuram)
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .spectral import radial_profile_gpu, hanning_2d


# ═══════════════════════════════════════════════════════════
#  GSE: Gradient Spectral Entropy
# ═══════════════════════════════════════════════════════════

def compute_gse(s_model_raw: np.ndarray) -> dict:
    """
    Gradient Spectral Entropy.

    Args:
        s_model_raw: (n_bins,) raw gradient PSD from compute_s_model()

    Returns:
        dict with 'gse', 'gse_raw', 'entropy_max', 'spectral_pdf'
    """
    n_bins = len(s_model_raw)
    eps = 1e-12

    # Normalize to probability distribution (no CSF comparison!)
    pdf = s_model_raw / (s_model_raw.sum() + eps)

    # Shannon entropy
    entropy = -np.sum(pdf * np.log2(pdf + eps))
    entropy_max = np.log2(n_bins)

    # Normalize to [0, 1]
    gse = float(entropy / entropy_max)
    gse = max(0.0, min(1.0, gse))

    return {
        "gse": gse,
        "gse_raw": float(entropy),
        "entropy_max": float(entropy_max),
        "spectral_pdf": pdf,
    }


# ═══════════════════════════════════════════════════════════
#  SLI: Spectral Lipschitz Index
# ═══════════════════════════════════════════════════════════

def _make_bandpass_mask(H: int, W: int, bin_idx: int, n_bins: int,
                        device) -> torch.Tensor:
    """Create a 2D frequency-domain mask that passes only one radial bin."""
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(np.sqrt(cx**2 + cy**2))
    idx = (r / r_max * (n_bins - 1)).clamp(0, n_bins - 1).long()

    mask = (idx == bin_idx).float()
    return mask


def _bandpass_noise(shape, bin_idx: int, n_bins: int, device,
                    epsilon: float = 0.01) -> torch.Tensor:
    """
    Generate random noise that is band-limited to a specific frequency bin.

    Args:
        shape: (B, C, H, W) — output shape
        bin_idx: which frequency bin to restrict to
        n_bins: total bins
        device: torch device
        epsilon: L2 norm of the perturbation

    Returns:
        (B, C, H, W) band-limited noise
    """
    B, C, H, W = shape

    mask = _make_bandpass_mask(H, W, bin_idx, n_bins, device)

    noise = torch.zeros(shape, device=device)
    for b in range(B):
        for c in range(C):
            # Random noise in frequency domain
            rand_phase = torch.randn(H, W, device=device)
            rand_mag = torch.randn(H, W, device=device).abs()

            # Create complex noise
            F_noise = torch.fft.fftshift(
                torch.fft.fft2(rand_phase * rand_mag)
            )

            # Apply bandpass
            F_filtered = F_noise * mask

            # Back to spatial
            spatial = torch.fft.ifft2(torch.fft.ifftshift(F_filtered)).real
            noise[b, c] = spatial

    # Normalize to target L2 norm
    for b in range(B):
        norm = noise[b].norm()
        if norm > 1e-8:
            noise[b] = noise[b] / norm * epsilon

    return noise


def compute_sli(model, loader, device, n_bins, epsilon=0.01,
                max_images=100, desc="SLI"):
    """
    Spectral Lipschitz Index.

    For each frequency band, estimates the local Lipschitz constant:
      L_b = E[||f(x + δ_b) - f(x)||₂ / ||δ_b||₂]
    where δ_b is band-limited noise in band b.

    SLI = max(L_b) / mean(L_b)

    Args:
        model: classifier in eval mode
        loader: DataLoader
        device: torch device
        n_bins: number of frequency bins
        epsilon: perturbation magnitude for Lipschitz estimation
        max_images: limit for speed
        desc: progress bar label

    Returns:
        dict with 'sli', 'band_lipschitz' (per-band L values),
        'max_band', 'mean_lipschitz', 'lipschitz_profile'
    """
    model.eval()
    eps_val = 1e-12

    # Accumulate per-band Lipschitz estimates
    band_sums = np.zeros(n_bins)
    band_counts = np.zeros(n_bins)
    n_processed = 0

    for images, labels in tqdm(loader, desc=f"  {desc}"):
        images = images.to(device)
        B = images.size(0)

        if n_processed >= max_images:
            break

        # Get clean logits
        with torch.no_grad():
            logits_clean = model(images)

        # For each frequency band, perturb and measure output change
        for b in range(n_bins):
            delta_b = _bandpass_noise(
                images.shape, b, n_bins, device, epsilon=epsilon
            )

            with torch.no_grad():
                logits_pert = model(images + delta_b)

            # Output change (L2 norm of logit difference)
            output_diff = (logits_pert - logits_clean).norm(dim=1)  # (B,)

            # Input change (L2 norm of perturbation)
            input_diff = delta_b.view(B, -1).norm(dim=1)  # (B,)

            # Lipschitz ratio
            ratios = output_diff / (input_diff + eps_val)

            band_sums[b] += ratios.sum().item()
            band_counts[b] += B

        n_processed += B

    # Average per-band Lipschitz
    band_lipschitz = band_sums / (band_counts + eps_val)

    # SLI = max / mean (how concentrated is the worst-case sensitivity)
    max_lip = band_lipschitz.max()
    mean_lip = band_lipschitz.mean()
    sli = float(max_lip / (mean_lip + eps_val))

    # Also compute total Lipschitz (average across all bands)
    total_lip = float(mean_lip)

    max_band = int(band_lipschitz.argmax())

    return {
        "sli": sli,
        "band_lipschitz": band_lipschitz,
        "max_band": max_band,
        "max_lipschitz": float(max_lip),
        "mean_lipschitz": total_lip,
        "lipschitz_profile": band_lipschitz / (max_lip + eps_val),  # normalized
    }


# ═══════════════════════════════════════════════════════════
#  Combined Grading
# ═══════════════════════════════════════════════════════════

def get_robustness_grade(gse: float, sli: float) -> dict:
    """
    Assign robustness grade based on GSE and SLI.

    GSE high + SLI low → robust (spectral attention spread, no exploitable spike)
    GSE low + SLI high → vulnerable (concentrated attention + exploitable spike)
    """
    if gse > 0.85 and sli < 2.0:
        grade, color, zone = "A", "green", "SAFE"
        rec = "Model has diverse spectral sensitivity and no exploitable band. Safe for deployment."
    elif gse > 0.75 and sli < 3.0:
        grade, color, zone = "B", "green", "MODERATE"
        rec = "Model has reasonable spectral spread. Consider adversarial training for critical applications."
    elif gse > 0.65 or sli < 4.0:
        grade, color, zone = "C", "yellow", "RISKY"
        rec = "Model has concentrated spectral sensitivity. Adversarial training recommended."
    elif gse > 0.50:
        grade, color, zone = "D", "orange", "DANGEROUS"
        rec = "Model has significant spectral concentration. Do not deploy without hardening."
    else:
        grade, color, zone = "F", "red", "CRITICAL"
        rec = "Model is severely vulnerable. Immediate remediation required."

    return {
        "grade": grade,
        "color": color,
        "zone": zone,
        "recommendation": rec,
    }


def format_report(gse: float, sli: float, mean_lip: float,
                  model_name: str = "Unknown") -> str:
    """Generate human-readable robustness report."""
    grade = get_robustness_grade(gse, sli)

    report = f"""
========================================================
  RobuScan v2 -- Spectral Robustness Report
========================================================
  Model:     {model_name}

  GSE:       {gse:.4f}   (Gradient Spectral Entropy)
  SLI:       {sli:.4f}   (Spectral Lipschitz Index)
  Mean Lip:  {mean_lip:.4f}   (Average Lipschitz constant)

  Grade:     {grade['grade']}  ({grade['zone']})
  {grade['recommendation']}
========================================================"""
    return report.strip()
