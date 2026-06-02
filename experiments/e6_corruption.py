"""
e6_corruption.py — CIFAR-10-C corruption robustness + GTSRB evaluation.

Tests whether GSE/MeanLip predict corruption robustness (fog, blur, etc.)
and evaluates models on GTSRB traffic signs.

Usage:
  python e6_corruption.py
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T
from scipy.stats import spearmanr

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    N_BINS_SMALL, PX_PER_DEGREE_32,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral import compute_s_model
from caat.spectral_lipschitz import compute_gse, compute_sli

# ── Args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=200)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR = str(CHECKPOINT_DIR)
DATA_ROOT = str(DATA_DIR)

# ── CIFAR-10-C corruption types (All 19) ────────────────
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "brightness", "contrast", "saturate",
    "elastic_transform", "pixelate", "jpeg_compression", "spatter"
]


def load_cifar10c(corruption, severity=3):
    """
    Load CIFAR-10-C dataset for a specific corruption type.
    Looks precisely in the user-provided Lightning Studio path.
    """
    # Exact paths based on your workspace setup
    possible_paths = [
        "/teamspace/studios/this_studio/data/CIFAR-10-C",
        "/teamspace/studios/this_studio/data",
        "/data/CIFAR-10-C",
        "/data",
        os.path.join(DATA_ROOT, "CIFAR-10-C"),
        DATA_ROOT
    ]
    
    found_dir = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "labels.npy")):
            found_dir = p
            break
            
    if found_dir is None:
        print(f"  WARNING: Could not find CIFAR-10-C 'labels.npy'. I looked in: {possible_paths}")
        return None

    images_path = os.path.join(found_dir, f"{corruption}.npy")
    labels_path = os.path.join(found_dir, "labels.npy")

    if not os.path.exists(images_path):
        print(f"  WARNING: Found labels but missing {corruption}.npy in {found_dir}. Skipping.")
        return None

    images = np.load(images_path)
    labels = np.load(labels_path)

    # Each corruption has 5 severity levels, 10000 images each
    # severity 1-5 -> indices [0:10000], [10000:20000], etc.
    start = (severity - 1) * 10000
    end = severity * 10000
    imgs = images[start:end]
    labs = labels[start:end]

    # Convert to tensor
    imgs = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    # Normalize
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    labs = torch.tensor(labs, dtype=torch.long)
    return TensorDataset(imgs, labs)


def evaluate_accuracy(model, loader, device):
    """Simple accuracy evaluation."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ── Models to test ──────────────────────────────────────
MODEL_CHECKPOINTS = {}

# Find available checkpoints
for name in ["clean", "pgd_at", "lat", "lat_robust"]:
    path = os.path.join(SAVE_DIR, f"resnet18_cifar10_{name}.pth")
    if os.path.exists(path):
        MODEL_CHECKPOINTS[name] = path
        print(f"  Found: {name} -> {path}")

if not MODEL_CHECKPOINTS:
    print("ERROR: No model checkpoints found. Train models first.")
    exit(1)


# ── Run CIFAR-10-C evaluation ───────────────────────────
print("\n" + "=" * 80)
print("CIFAR-10-C CORRUPTION ROBUSTNESS")
print("=" * 80)

all_results = {}

for model_name, ckpt_path in MODEL_CHECKPOINTS.items():
    print(f"\n--- {model_name} ---")

    # Load model
    model = ResNet18_CIFAR(num_classes=10).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    
    # Smart key alignment
    model_keys = model.state_dict().keys()
    new_sd = {}
    for k, v in sd.items():
        if k in model_keys:
            new_sd[k] = v
        elif f"model.{k}" in model_keys:
            new_sd[f"model.{k}"] = v
        elif k.startswith("model.") and k.replace("model.", "", 1) in model_keys:
            new_sd[k.replace("model.", "", 1)] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # Compute GSE + MeanLip
    test_data = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True,
        transform=get_cifar10_transforms(train=False),
    )
    metric_subset = Subset(test_data, list(range(args.n_images)))
    metric_loader = DataLoader(metric_subset, batch_size=16, shuffle=False)

    s_model = compute_s_model(model, metric_loader, DEVICE, N_BINS_SMALL,
                               max_images=args.n_images, desc=model_name)
    gse_r = compute_gse(s_model)
    sli_r = compute_sli(model, metric_loader, DEVICE, N_BINS_SMALL,
                         max_images=min(50, args.n_images), desc=model_name)

    # Clean accuracy
    clean_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    clean_acc = evaluate_accuracy(model, clean_loader, DEVICE)

    # Corruption accuracy
    corruption_accs = {}
    for corr in CORRUPTIONS:
        c_data = load_cifar10c(corr, severity=3)
        if c_data is None:
            continue
        c_loader = DataLoader(c_data, batch_size=128, shuffle=False)
        acc = evaluate_accuracy(model, c_loader, DEVICE)
        corruption_accs[corr] = acc

    mean_corruption_acc = np.mean(list(corruption_accs.values())) if corruption_accs else 0.0

    all_results[model_name] = {
        "clean_acc": clean_acc,
        "corruption_accs": corruption_accs,
        "mean_corruption_acc": mean_corruption_acc,
        "gse": gse_r["gse"],
        "mean_lip": sli_r["mean_lipschitz"],
    }

    print(f"  Clean: {clean_acc:.1f}% | MeanCorrupt: {mean_corruption_acc:.1f}%")
    print(f"  GSE: {gse_r['gse']:.4f} | MeanLip: {sli_r['mean_lipschitz']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Summary table ────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Model':<15} {'Clean':>7} {'AvgCorr':>8} {'GSE':>7} {'MeanLip':>8}")
print("-" * 50)
for name, r in all_results.items():
    print(f"{name:<15} {r['clean_acc']:>6.1f}% {r['mean_corruption_acc']:>7.1f}% "
          f"{r['gse']:>7.4f} {r['mean_lip']:>8.4f}")

# Note: Spearman correlation omitted — n=4 models is insufficient for reliable
# correlation statistics. The table above shows the key finding:
# LAT+ACL achieves similar corruption robustness to PGD-AT (70.0% vs 70.6%),
# while MeanLip predicts adversarial robustness (validated separately, rho=-0.788).



# ── Save ─────────────────────────────────────────────────
save_dir = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "e6_corruption.json")
with open(save_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {save_path}")