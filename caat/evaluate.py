"""
evaluate.py — Unified evaluation script (GSE + MeanLip + robustness)

Evaluates any CIFAR-10 checkpoint with:
  - Clean accuracy
  - FGSM accuracy
  - PGD-10/20 accuracy
  - GSE (Gradient Spectral Entropy)
  - MeanLip (Spectral Lipschitz profile)
  - Lipschitz band profile (verbose)

Usage:
  python -m caat.evaluate --checkpoint checkpoints/resnet18_cifar10_lat.pth --name LAT
  python -m caat.evaluate --checkpoint checkpoints/resnet18_cifar10_pgd_at.pth --name PGD-AT
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import numpy as np

from caat.config import (
    DEVICE, SEED, DATA_DIR, RESULTS_DIR as _RD,
    CIFAR10_MEAN, CIFAR10_STD,
    N_BINS_SMALL, PX_PER_DEGREE_32,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral import compute_s_model
from caat.spectral_lipschitz import compute_gse, compute_sli, get_robustness_grade

# ── Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--name", default="Model")
parser.add_argument("--n_images", type=int, default=200)
parser.add_argument("--pgd_steps", type=int, default=10)
parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
args = parser.parse_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = str(DATA_DIR)
RESULTS_DIR = str(_RD / "evaluation")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\nEvaluating: {args.name}")
print(f"Checkpoint: {args.checkpoint}")
print(f"Device:     {DEVICE}")

# ── Load model ────────────────────────────────────────────
model = ResNet18_CIFAR(num_classes=10).to(DEVICE)
sd = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
# ResNet18_CIFAR stores weights under self.model → keys need "model." prefix
# Checkpoints saved from train_lat/train_pgd_at save raw state_dict (no prefix)
sample_key = next(iter(sd.keys()))
if sample_key.startswith("model."):
    # Already has prefix — load directly
    pass
else:
    # Add prefix
    sd = {f"model.{k}": v for k, v in sd.items()}
try:
    model.load_state_dict(sd)
    print("  Checkpoint loaded OK")
except RuntimeError as e:
    print(f"  ERROR loading checkpoint: {e}")
    raise

model.eval()

# ── Data ─────────────────────────────────────────────────
test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
test_loader = DataLoader(test_data, batch_size=128,
                         shuffle=False, num_workers=4)

indices = np.random.choice(len(test_data), args.n_images, replace=False)
metric_loader = DataLoader(
    Subset(test_data, indices),
    batch_size=16, shuffle=False, num_workers=2
)

# ── Sanity check ──────────────────────────────────────────
with torch.no_grad():
    x, _ = next(iter(test_loader))
    _ = model(x.to(DEVICE))
print("  Forward pass OK\n")


# ── Attack helpers ────────────────────────────────────────
def fgsm(model, images, labels, epsilon, mean, std):
    mean_t = torch.tensor(mean, device=DEVICE).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=DEVICE).view(1, -1, 1, 1)
    eps_norm = epsilon / std_t

    images.requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    adv = (images + eps_norm * images.grad.sign()).detach()
    return adv


def pgd(model, images, labels, epsilon, steps, mean, std):
    mean_t = torch.tensor(mean, device=DEVICE).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=DEVICE).view(1, -1, 1, 1)
    eps_norm = epsilon / std_t
    alpha_norm = (epsilon / 4) / std_t
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    delta = torch.zeros_like(images).uniform_(-1, 1) * eps_norm
    delta = torch.max(torch.min(images + delta, upper), lower) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(images + delta), labels)
        loss.backward()
        g = delta.grad.detach()
        delta = delta.detach() + alpha_norm * g.sign()
        delta = torch.clamp(delta, -eps_norm, eps_norm)
        delta = torch.max(torch.min(images + delta, upper), lower) - images

    return (images + delta).detach()


def eval_accuracy(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# ── Robustness evaluation ─────────────────────────────────
print("=" * 60)
print("ROBUSTNESS EVALUATION")
print("=" * 60)

clean_acc = eval_accuracy(model, test_loader)
print(f"  Clean:  {clean_acc:.2f}%")

# FGSM
fgsm_correct, fgsm_total = 0, 0
for x, y in test_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    x.requires_grad_(True)
    adv = fgsm(model, x, y, args.epsilon, CIFAR10_MEAN, CIFAR10_STD)
    with torch.no_grad():
        fgsm_correct += (model(adv).argmax(1) == y).sum().item()
    fgsm_total += y.size(0)
fgsm_acc = 100.0 * fgsm_correct / fgsm_total
print(f"  FGSM:   {fgsm_acc:.2f}%  (ε={args.epsilon:.4f})")

# PGD
pgd_correct, pgd_total = 0, 0
for x, y in test_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    adv = pgd(model, x, y, args.epsilon, args.pgd_steps, CIFAR10_MEAN, CIFAR10_STD)
    with torch.no_grad():
        pgd_correct += (model(adv).argmax(1) == y).sum().item()
    pgd_total += y.size(0)
pgd_acc = 100.0 * pgd_correct / pgd_total
print(f"  PGD-{args.pgd_steps}: {pgd_acc:.2f}%  (ε={args.epsilon:.4f})")


# ── GSE + MeanLip ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SPECTRAL METRICS (GSE + MEANLIP)")
print("=" * 60)

print("  Computing S_model...")
s_model = compute_s_model(
    model, metric_loader, DEVICE, N_BINS_SMALL,
    max_images=args.n_images, desc=args.name,
)

gse_result = compute_gse(s_model)
print(f"  GSE:    {gse_result['gse']:.4f}  (lower = more concentrated gradient spectrum)")

print("  Computing Lipschitz profile...")
sli_result = compute_sli(
    model, metric_loader, DEVICE, N_BINS_SMALL,
    epsilon=0.01, max_images=min(50, args.n_images),
    desc=f"{args.name} Lip",
)
print(f"  MeanLip: {sli_result['mean_lipschitz']:.4f}  (lower = more robust)")
print(f"  MaxBand: {sli_result['max_band']}  (most vulnerable frequency band)")

# Robustness grade from E1 finding
grade_info = get_robustness_grade(gse_result["gse"], sli_result["mean_lipschitz"])
if isinstance(grade_info, dict):
    grade = f"{grade_info['grade']} ({grade_info['zone']}) — {grade_info['recommendation']}"
else:
    grade = str(grade_info)
print(f"  Grade:  {grade}")

# Per-band profile
print(f"\n  Lipschitz profile (per band):")
bands = sli_result["band_lipschitz"]
for i, v in enumerate(bands):
    bar = "█" * int(v / max(bands) * 20)
    marker = " ← MOST VULNERABLE" if i == sli_result["max_band"] else ""
    print(f"    Band {i:2d}: {v:.3f}  {bar}{marker}")


# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SUMMARY — {args.name}")
print("=" * 60)
print(f"  Clean Acc:  {clean_acc:.2f}%")
print(f"  FGSM Acc:   {fgsm_acc:.2f}%")
print(f"  PGD-{args.pgd_steps} Acc: {pgd_acc:.2f}%")
print(f"  GSE:        {gse_result['gse']:.4f}")
print(f"  MeanLip:    {sli_result['mean_lipschitz']:.4f}")
print(f"  Grade:      {grade}")
print("=" * 60)


# ── Save ──────────────────────────────────────────────────
save_path = os.path.join(
    RESULTS_DIR,
    f"eval_{args.name.lower().replace(' ', '_')}.json"
)

results = {
    "name": args.name,
    "checkpoint": args.checkpoint,
    "clean_acc": clean_acc,
    "fgsm_acc": fgsm_acc,
    "pgd_acc": pgd_acc,
    "pgd_steps": args.pgd_steps,
    "epsilon": args.epsilon,
    "gse": gse_result["gse"],
    "mean_lip": sli_result["mean_lipschitz"],
    "max_band": sli_result["max_band"],
    "band_lipschitz": sli_result["band_lipschitz"].tolist(),
    "spectral_pdf": gse_result["spectral_pdf"].tolist(),
    "grade": grade,
}

with open(save_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {save_path}")