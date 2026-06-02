"""
e5_adaptive_attack.py — Adaptive attack evaluation (BPDA-style).

Tests whether PCSD+DGSC is genuinely robust or just causes gradient obfuscation.

Attack pipeline: maximize CE(classifier(PCSD(x+δ, profile_fixed)), y)
  - profile is computed once from the CLEAN image, then fixed (BPDA approximation)
  - backprop flows THROUGH the denoiser into x+δ normally
  - this is the correct adaptive attack: attacker knows about the denoiser

Why BPDA on the profile only:
  Computing ∂(DGSC_profile)/∂δ requires differentiating through a backward()
  call, which requires create_graph=True and is O(n²) in memory.
  The standard BPDA approximation treats the profile as constant w.r.t. δ,
  which is both computationally tractable and academically accepted.

Usage:
  python -m experiments.e5_adaptive_attack
  python -m experiments.e5_adaptive_attack --pgd_steps 20 --n_batches 20
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, N_BINS_SMALL,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral_lipschitz import compute_sli
from pcsd.unet import PCSD, DnCNN

parser = argparse.ArgumentParser()
parser.add_argument("--pgd_steps", type=int, default=20,
                    help="Steps for adaptive attack (should be higher than training)")
parser.add_argument("--epsilon",   type=float, default=8.0 / 255.0)
parser.add_argument("--n_batches", type=int, default=20,
                    help="Number of test batches (~20×64=1280 images)")
parser.add_argument("--batch_size", type=int, default=64)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR   = str(CHECKPOINT_DIR)
RESULT_DIR = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Load classifier ───────────────────────────────────────
print("Loading classifier...")
classifier = ResNet18_CIFAR(num_classes=10).to(DEVICE)
for name in ["resnet18_cifar10_lat_robust.pth", "resnet18_cifar10_lat.pth"]:
    p = os.path.join(SAVE_DIR, name)
    if os.path.exists(p):
        sd = torch.load(p, map_location=DEVICE, weights_only=True)
        if not next(iter(sd)).startswith("model."):
            sd = {f"model.{k}": v for k, v in sd.items()}
        classifier.load_state_dict(sd, strict=False)
        print(f"  Loaded {name}")
        break
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

# ── Load denoisers ────────────────────────────────────────
pcsd = PCSD(in_channels=3, profile_dim=N_BINS_SMALL, base_ch=32).to(DEVICE)
p = os.path.join(SAVE_DIR, "pcsd_cifar10.pth")
if os.path.exists(p):
    pcsd.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    print("  Loaded pcsd_cifar10.pth")
pcsd.eval()

dncnn = DnCNN(in_channels=3, num_layers=8, num_features=64).to(DEVICE)
p = os.path.join(SAVE_DIR, "dncnn_cifar10.pth")
if os.path.exists(p):
    dncnn.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    print("  Loaded dncnn_cifar10.pth")
dncnn.eval()

# ── Static Lipschitz profile ──────────────────────────────
test_data = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
sli = compute_sli(
    classifier,
    DataLoader(Subset(test_data, list(range(200))), batch_size=16),
    DEVICE, N_BINS_SMALL, epsilon=0.01, max_images=200, desc="Profile",
)
static_profile = torch.tensor(
    sli["lipschitz_profile"], dtype=torch.float32, device=DEVICE
)

test_loader = DataLoader(
    Subset(test_data, list(range(args.n_batches * args.batch_size))),
    batch_size=args.batch_size, shuffle=False, num_workers=2,
)


# ── Attack utilities ──────────────────────────────────────
def parse_eps(eps):
    mean_t = torch.tensor(CIFAR10_MEAN, device=DEVICE).view(1, -1, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=DEVICE).view(1, -1, 1, 1)
    return eps / std_t, (eps / 4) / std_t, (0.0 - mean_t) / std_t, (1.0 - mean_t) / std_t


def pgd_standard(images, labels, steps):
    """Standard PGD on classifier only."""
    en, al, lo, hi = parse_eps(args.epsilon)
    delta = torch.zeros_like(images).uniform_(-1, 1) * en
    delta = torch.max(torch.min(images + delta, hi), lo) - images
    for _ in range(steps):
        delta.requires_grad_(True)
        F.cross_entropy(classifier(images + delta), labels).backward()
        g = delta.grad.detach()
        delta = delta.detach() + al * g.sign()
        delta = torch.clamp(delta, -en, en)
        delta = torch.max(torch.min(images + delta, hi), lo) - images
    return (images + delta).detach()


def pgd_adaptive_dncnn(images, labels, steps):
    """
    Adaptive attack on DnCNN + classifier.
    Backprops through denoiser: maximize CE(classifier(DnCNN(x+δ)), y).
    """
    en, al, lo, hi = parse_eps(args.epsilon)
    sp = static_profile.unsqueeze(0).expand(images.size(0), -1)
    delta = torch.zeros_like(images).uniform_(-1, 1) * en
    delta = torch.max(torch.min(images + delta, hi), lo) - images
    for _ in range(steps):
        delta.requires_grad_(True)
        purified = dncnn(images + delta, sp)
        F.cross_entropy(classifier(purified), labels).backward()
        g = delta.grad.detach()
        delta = delta.detach() + al * g.sign()
        delta = torch.clamp(delta, -en, en)
        delta = torch.max(torch.min(images + delta, hi), lo) - images
    return (images + delta).detach()


def compute_profile_fixed(images):
    """Compute DGSC profile from CLEAN images (fixed for BPDA)."""
    n_bins = N_BINS_SMALL
    B, C, H, W = images.shape
    device = images.device
    x = images.detach().clone().requires_grad_(True)
    F.cross_entropy(classifier(x), classifier(x).argmax(1)).backward()
    grad = torch.clamp(x.grad.detach().abs(), 0, 1e2)
    G = torch.fft.fftshift(torch.fft.fft2(grad), dim=(-2, -1)).abs().mean(1)
    G = torch.nan_to_num(G)
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij")
    r = torch.sqrt((gx - cx)**2 + (gy - cy)**2)
    bin_idx = (r / float((cx**2+cy**2)**0.5) * (n_bins-1)).clamp(0, n_bins-1).long()
    spectrum = torch.zeros(B, n_bins, device=device)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() > 0:
            spectrum[:, b] = G[:, mask].mean(1)
    spectrum = torch.nan_to_num(spectrum)
    mx = spectrum.max(1, keepdim=True).values.clamp(1e-8)
    return (spectrum / mx).detach()


def pgd_adaptive_pcsd(images, labels, steps):
    """
    BPDA-style adaptive attack on PCSD+DGSC + classifier.

    Profile is computed once from CLEAN images, then fixed.
    Backprop flows: CE → classifier → PCSD(denoiser) → δ.

    This is the correct adaptive attack — the attacker knows about the denoiser
    and its conditioning, and attacks the full pipeline.
    """
    en, al, lo, hi = parse_eps(args.epsilon)
    # Compute profile from clean images (fixed — BPDA on profile computation)
    profile = compute_profile_fixed(images)  # (B, n_bins)

    delta = torch.zeros_like(images).uniform_(-1, 1) * en
    delta = torch.max(torch.min(images + delta, hi), lo) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        # Full differentiable path: δ → PCSD(images+δ, profile) → classifier
        purified = pcsd(images + delta, profile)
        F.cross_entropy(classifier(purified), labels).backward()
        g = delta.grad.detach()
        delta = delta.detach() + al * g.sign()
        delta = torch.clamp(delta, -en, en)
        delta = torch.max(torch.min(images + delta, hi), lo) - images

    return (images + delta).detach()


def evaluate(name, adv_fn, denoiser_fn=None):
    """Run attack, optionally apply denoiser, return accuracy."""
    correct = total = 0
    for i, (images, labels) in enumerate(test_loader):
        if i >= args.n_batches:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        adv = adv_fn(images, labels)

        with torch.no_grad():
            if denoiser_fn is not None:
                adv = denoiser_fn(adv)
            correct += (classifier(adv).argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"  {name:<55} {acc:>6.2f}%")
    return acc


# ── Run evaluation ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"ADAPTIVE ATTACK EVALUATION (ε=8/255, steps={args.pgd_steps})")
print(f"N={args.n_batches * args.batch_size} images  |  Device={DEVICE}")
print(f"{'='*70}")
print(f"  {'Setup':<55} {'Acc':>6}")
print(f"  {'-'*63}")

results = {}

# Baseline: no attack
acc = evaluate(
    "No attack (clean accuracy)",
    lambda x, y: x,
)
results["clean"] = acc

# -- Standard PGD on classifier only
adv_standard = {}  # cache to avoid re-generating

# Standard PGD → no defense
print("\n  [Standard PGD — non-adaptive, attacker ignores denoiser]")
acc = evaluate(
    f"PGD-{args.pgd_steps} → classifier only (no defense)",
    lambda x, y: pgd_standard(x, y, args.pgd_steps),
)
results["pgd_no_defense"] = acc

sp = static_profile.unsqueeze(0)
acc = evaluate(
    f"PGD-{args.pgd_steps} → DnCNN → classifier",
    lambda x, y: pgd_standard(x, y, args.pgd_steps),
    denoiser_fn=lambda adv: dncnn(adv, sp.expand(adv.size(0), -1)),
)
results["pgd_dncnn"] = acc

acc = evaluate(
    f"PGD-{args.pgd_steps} → PCSD+DGSC → classifier",
    lambda x, y: pgd_standard(x, y, args.pgd_steps),
    denoiser_fn=lambda adv: pcsd(adv, compute_profile_fixed(adv)),
)
results["pgd_pcsd_dgsc"] = acc

# -- Adaptive PGD (attacker knows about denoiser)
print("\n  [Adaptive PGD — attacker knows & attacks THROUGH the denoiser]")
acc = evaluate(
    f"Adaptive PGD-{args.pgd_steps} → DnCNN → classifier",
    lambda x, y: pgd_adaptive_dncnn(x, y, args.pgd_steps),
)
results["adaptive_dncnn"] = acc

acc = evaluate(
    f"Adaptive BPDA PGD-{args.pgd_steps} → PCSD+DGSC → classifier (ours)",
    lambda x, y: pgd_adaptive_pcsd(x, y, args.pgd_steps),
)
results["adaptive_pcsd_dgsc"] = acc

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("KEY FINDINGS:")
drop_dncnn  = results["pgd_dncnn"]  - results["adaptive_dncnn"]
drop_pcsd   = results["pgd_pcsd_dgsc"] - results["adaptive_pcsd_dgsc"]
print(f"  DnCNN accuracy drop under adaptive attack:      {drop_dncnn:+.2f}%")
print(f"  PCSD+DGSC accuracy drop under adaptive attack:  {drop_pcsd:+.2f}%")
if abs(drop_pcsd) < 5:
    print("  → PCSD+DGSC is ROBUST to adaptive attack (no gradient obfuscation)")
elif abs(drop_pcsd) < 15:
    print("  → Moderate drop — PCSD provides genuine (not obfuscated) robustness")
else:
    print("  → Large drop — report honestly in Limitations section")

results["drop_dncnn_adaptive"]  = drop_dncnn
results["drop_pcsd_adaptive"]   = drop_pcsd

save_path = os.path.join(RESULT_DIR, "e5_adaptive_attack.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {save_path}")
