"""
config.py — Central configuration for the entire project.
Lightning / local compatible version
"""

from pathlib import Path
import torch

# ── Paths (FIXED for Lightning) ───────────────────────────────────────────────
PROJECT_ROOT = Path("/teamspace/studios/this_studio")

DATA_DIR       = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"
FIGURES_DIR    = PROJECT_ROOT / "figures"
PROJECT_DATA   = PROJECT_ROOT / "project_data"

# Create directories (safe)
for d in [
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    RESULTS_DIR / "training_logs",
    RESULTS_DIR / "evaluation",
    RESULTS_DIR / "scans",
    FIGURES_DIR / "paper",
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Spectral Analysis ────────────────────────────────────────────────────────
N_BINS_SMALL = 16
N_BINS_LARGE = 64
EPS = 1e-12

PX_PER_DEGREE_32  = 32.0 / 7.0
PX_PER_DEGREE_224 = 224.0 / 7.0

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "clean": {
        "epochs": 20,   # 🔥 start with 20 (can increase later)
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 128,
        "scheduler": "cosine",
    },

    "pgd_at": {
        "epochs": 20,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 128,
        "scheduler": "cosine",
        "epsilon": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "pgd_steps": 10,
    },

    "caat": {
        "epochs": 20,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 128,
        "scheduler": "cosine",
        "epsilon": 8.0 / 255.0,
        "alpha": 2.0 / 255.0,
        "pgd_steps": 10,
        "lambda_pri": 0.1,
        "mask_update_freq": 10,
    },
}

# ── Attacks ──────────────────────────────────────────────────────────────────
ATTACK_CONFIG = {
    "fgsm": {"epsilon": 8.0 / 255.0},
    "pgd": {"epsilon": 8.0 / 255.0, "alpha": 2.0 / 255.0, "steps": 20},
}

# ── Pretrained (optional) ─────────────────────────────────────────────────────
PRETRAINED = {
    "resnet18_cifar10": CHECKPOINT_DIR / "resnet18_cifar10_clean.pth",
}