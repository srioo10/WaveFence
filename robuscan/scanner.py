"""
scanner.py — RobuScan core backend (v2).

Uses GSE + MeanLip (our validated metrics) instead of old PRI/VCI.

Pipeline:
  model → S_model → GSE
  model → Lipschitz profile → MeanLip → Grade
"""

import numpy as np
import torch

from caat.spectral import compute_s_model
from caat.spectral_lipschitz import compute_gse, compute_sli, get_robustness_grade


def scan_model(model, loader, device, n_bins, max_images=200):
    """
    Full RobuScan v2 pipeline: model → GSE + MeanLip + grade.

    Args:
        model:      nn.Module in eval mode
        loader:     DataLoader with test images
        device:     torch device
        n_bins:     radial frequency bins
        max_images: images for metric computation

    Returns:
        dict with all scan results
    """
    # ── Step 1: GSE ──────────────────────────────────────
    s_model = compute_s_model(
        model, loader, device, n_bins,
        max_images=max_images, desc="GSE",
    )
    gse_result = compute_gse(s_model)

    # ── Step 2: MeanLip + band profile ───────────────────
    sli_result = compute_sli(
        model, loader, device, n_bins,
        epsilon=0.01,
        max_images=min(50, max_images),
        desc="MeanLip",
    )

    # ── Step 3: Grade ─────────────────────────────────────
    grade_info = get_robustness_grade(gse_result["gse"], sli_result["sli"])
    if isinstance(grade_info, dict):
        grade       = grade_info["grade"]
        zone        = grade_info.get("zone", "")
        recommend   = grade_info.get("recommendation", "")
    else:
        grade = str(grade_info)
        zone = ""
        recommend = ""

    band_lip = sli_result["band_lipschitz"]
    max_band = int(sli_result["max_band"])

    return {
        # GSE
        "gse":          gse_result["gse"],
        "spectral_pdf": gse_result["spectral_pdf"],
        # MeanLip
        "mean_lip":     sli_result["mean_lipschitz"],
        "band_lipschitz": band_lip,
        "max_band":     max_band,
        # Grade
        "grade":        grade,
        "zone":         zone,
        "recommendation": recommend,
    }
