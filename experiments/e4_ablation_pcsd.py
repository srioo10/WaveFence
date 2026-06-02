"""
e4_ablation_pcsd.py — PCSD Ablation Study.

Trains 5 denoiser variants to isolate what drives PCSD's accuracy recovery.
Each variant trains for N epochs (default 10 — enough to show relative differences).

Variants:
  1. PCSD + static profile + pixel loss   → Is U-Net arch alone enough?
  2. PCSD + static profile + cls loss     → Does cls guidance help without DGSC?
  3. PCSD + DGSC      + pixel loss        → Does DGSC help without cls guidance?
  4. PCSD + DGSC      + cls loss (OURS)   → Full method
  5. DnCNN            + cls loss          → Fair comparison: can DnCNN catch up?

Usage:
  python -m experiments.e4_ablation_pcsd
  python -m experiments.e4_ablation_pcsd --epochs 15
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    N_BINS_SMALL,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral_lipschitz import compute_sli
from pcsd.unet import PCSD, DnCNN

# ── Args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10,
                    help="Epochs per variant (10 is sufficient to show relative order)")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
parser.add_argument("--pgd_steps", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lambda_cls", type=float, default=0.5)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR   = str(CHECKPOINT_DIR)
DATA_ROOT  = str(DATA_DIR)
RESULT_DIR = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Ablation variants (in order of complexity) ───────────
VARIANTS = [
    {
        "name":       "PCSD_static_pixel",
        "model":      "pcsd",
        "dynamic":    False,
        "lambda_cls": 0.0,
        "label":      "PCSD + Static Profile + Pixel Loss",
    },
    {
        "name":       "PCSD_static_cls",
        "model":      "pcsd",
        "dynamic":    False,
        "lambda_cls": args.lambda_cls,
        "label":      "PCSD + Static Profile + Cls Loss",
    },
    {
        "name":       "PCSD_dgsc_pixel",
        "model":      "pcsd",
        "dynamic":    True,
        "lambda_cls": 0.0,
        "label":      "PCSD + DGSC + Pixel Loss",
    },
    {
        "name":       "PCSD_dgsc_cls",
        "model":      "pcsd",
        "dynamic":    True,
        "lambda_cls": args.lambda_cls,
        "label":      "PCSD + DGSC + Cls Loss  ← OURS (full)",
    },
    {
        "name":       "DnCNN_cls",
        "model":      "dncnn",
        "dynamic":    False,
        "lambda_cls": args.lambda_cls,
        "label":      "DnCNN + Cls Loss  (fair baseline)",
    },
]


# ── Load frozen classifier ────────────────────────────────
print("Loading classifier...")
classifier_path = None
for name in ["resnet18_cifar10_lat_robust.pth",
             "resnet18_cifar10_lat.pth",
             "resnet18_cifar10_pgd_at.pth",
             "resnet18_cifar10_clean.pth"]:
    p = os.path.join(SAVE_DIR, name)
    if os.path.exists(p):
        classifier_path = p
        break

if classifier_path is None:
    print("ERROR: No classifier found. Train one first.")
    sys.exit(1)

print(f"  Classifier: {classifier_path}")
classifier = ResNet18_CIFAR(num_classes=10).to(DEVICE)
sd = torch.load(classifier_path, map_location=DEVICE, weights_only=True)
if not next(iter(sd.keys())).startswith("model."):
    sd = {f"model.{k}": v for k, v in sd.items()}
classifier.load_state_dict(sd, strict=False)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False


# ── Static Lipschitz profile ──────────────────────────────
print("Computing static Lipschitz profile...")
_test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
_profile_loader = DataLoader(
    Subset(_test_data, list(range(100))), batch_size=16, shuffle=False
)
sli_r = compute_sli(
    classifier, _profile_loader, DEVICE, N_BINS_SMALL,
    epsilon=0.01, max_images=100, desc="Profile",
)
lip_profile_t = torch.tensor(
    sli_r["lipschitz_profile"], dtype=torch.float32, device=DEVICE
)
print(f"  Profile shape: {lip_profile_t.shape}  "
      f"MeanLip: {sli_r['mean_lipschitz']:.4f}")


# ── Data ──────────────────────────────────────────────────
train_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True,
    transform=get_cifar10_transforms(train=True),
)
test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)


# ── Helpers ───────────────────────────────────────────────
def get_static_profile(B):
    return lip_profile_t.unsqueeze(0).expand(B, -1)


def compute_dynamic_profile(adv_images):
    """Dynamic Gradient Spectrum Conditioning (DGSC) — novel contribution."""
    n_bins = N_BINS_SMALL
    B, C, H, W = adv_images.shape
    device = adv_images.device

    x = adv_images.detach().clone().requires_grad_(True)
    F.cross_entropy(classifier(x), classifier(x).argmax(1)).backward()
    grad = torch.clamp(x.grad.detach().abs(), 0.0, 1e2)

    G = torch.fft.fftshift(torch.fft.fft2(grad), dim=(-2, -1)).abs().mean(dim=1)
    G = torch.nan_to_num(G, nan=0.0, posinf=0.0)

    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    bin_idx = (r / float((cx**2 + cy**2)**0.5) * (n_bins - 1)).clamp(0, n_bins - 1).long()

    spectrum = torch.zeros(B, n_bins, device=device)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() > 0:
            spectrum[:, b] = G[:, mask].mean(dim=1)

    spectrum = torch.nan_to_num(spectrum, nan=0.0, posinf=0.0)
    max_val  = spectrum.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    spectrum = spectrum / max_val

    bad = spectrum.sum(dim=1) < 1e-6
    if bad.any():
        spectrum[bad] = torch.full((n_bins,), 1.0 / n_bins, device=device)

    return spectrum.detach()


def generate_adversarial(images, labels):
    device = images.device
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=device).view(1, -1, 1, 1)
    eps    = args.epsilon / std_t
    alpha  = (args.epsilon / 4) / std_t
    lo     = (0.0 - mean_t) / std_t
    hi     = (1.0 - mean_t) / std_t

    delta = torch.zeros_like(images).uniform_(-1, 1) * eps
    delta = torch.max(torch.min(images + delta, hi), lo) - images

    for _ in range(args.pgd_steps):
        delta.requires_grad_(True)
        F.cross_entropy(classifier(images + delta), labels).backward()
        g = delta.grad.detach()
        delta = delta.detach() + alpha * g.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.max(torch.min(images + delta, hi), lo) - images

    return (images + delta).detach()


# ── Train & evaluate one variant ──────────────────────────
def run_variant(variant):
    name       = variant["name"]
    arch       = variant["model"]
    use_dgsc   = variant["dynamic"]
    lam_cls    = variant["lambda_cls"]

    print(f"\n{'='*70}")
    print(f"VARIANT: {variant['label']}")
    print(f"{'='*70}")
    print(f"  arch={arch}  dynamic={use_dgsc}  lambda_cls={lam_cls}")

    # Build fresh denoiser
    if arch == "pcsd":
        denoiser = PCSD(in_channels=3, profile_dim=N_BINS_SMALL, base_ch=32).to(DEVICE)
    else:
        denoiser = DnCNN(in_channels=3, num_layers=8, num_features=64).to(DEVICE)

    optimizer = optim.Adam(denoiser.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_recovery = -999.0
    epoch_logs    = []

    for epoch in range(args.epochs):
        denoiser.train()
        t0 = time.time()
        total_loss = total_cls = total_n = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.size(0)

            adv = generate_adversarial(images, labels)

            # Profile selection
            if use_dgsc:
                profile = compute_dynamic_profile(adv)
            else:
                profile = get_static_profile(B)

            denoised  = denoiser(adv, profile)
            pix_loss  = nn.L1Loss()(denoised, images) + 0.5 * nn.MSELoss()(denoised, images)

            if lam_cls > 0:
                cls_loss = F.cross_entropy(classifier(denoised), labels)
                loss     = pix_loss + lam_cls * cls_loss
            else:
                cls_loss = torch.zeros(1, device=DEVICE)
                loss     = pix_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
            optimizer.step()

            total_loss += pix_loss.item() * B
            total_cls  += cls_loss.item() * B
            total_n    += B

        scheduler.step()

        # ── Eval ──
        denoiser.eval()
        adv_c = den_c = eval_n = 0

        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.size(0)

            adv = generate_adversarial(images, labels)

            if use_dgsc:
                profile = compute_dynamic_profile(adv)
            else:
                profile = get_static_profile(B)

            with torch.no_grad():
                adv_c += (classifier(adv).argmax(1) == labels).sum().item()
                den   = denoiser(adv, profile)
                den_c += (classifier(den).argmax(1) == labels).sum().item()
            eval_n += B

        adv_acc = 100.0 * adv_c / eval_n
        den_acc = 100.0 * den_c / eval_n
        rec     = den_acc - adv_acc
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
              f"L_pix: {total_loss/total_n:.4f} | "
              f"Adv: {adv_acc:.1f}% | Denoised: {den_acc:.1f}% "
              f"(+{rec:.1f}%) | {elapsed:.1f}s")

        epoch_logs.append({"epoch": epoch+1, "adv_acc": adv_acc,
                           "den_acc": den_acc, "recovery": rec})

        if rec > best_recovery:
            best_recovery = rec
            torch.save(denoiser.state_dict(),
                       os.path.join(SAVE_DIR, f"ablation_{name}.pth"))

    print(f"  ✓ Best recovery: +{best_recovery:.2f}%")
    return {"variant": name, "label": variant["label"],
            "arch": arch, "dynamic": use_dgsc, "lambda_cls": lam_cls,
            "best_recovery": best_recovery, "epochs": epoch_logs}


# ── Run all variants ──────────────────────────────────────
print(f"\nABLATION STUDY — {len(VARIANTS)} variants × {args.epochs} epochs")
print(f"Estimated time: ~{len(VARIANTS) * args.epochs * 47 / 60:.0f} min on GPU\n")

all_results = []
for v in VARIANTS:
    r = run_variant(v)
    all_results.append(r)

# ── Summary table ─────────────────────────────────────────
print(f"\n{'='*70}")
print("ABLATION SUMMARY")
print(f"{'='*70}")
print(f"{'Variant':<45} {'Arch':<8} {'DGSC':<6} {'CLS':<6} {'Recovery':>10}")
print("-" * 70)
for r in all_results:
    dgsc = "✓" if r["dynamic"] else "✗"
    cls  = "✓" if r["lambda_cls"] > 0 else "✗"
    marker = "  ← OURS" if r["variant"] == "PCSD_dgsc_cls" else ""
    print(f"  {r['label'][:43]:<43} {r['arch']:<8} {dgsc:<6} {cls:<6} "
          f"{r['best_recovery']:>+8.2f}%{marker}")

print(f"\nKey questions:")

results_by_name = {r["variant"]: r for r in all_results}

r_static_pix = results_by_name.get("PCSD_static_pixel", {}).get("best_recovery", 0)
r_static_cls = results_by_name.get("PCSD_static_cls",   {}).get("best_recovery", 0)
r_dgsc_pix   = results_by_name.get("PCSD_dgsc_pixel",   {}).get("best_recovery", 0)
r_dgsc_cls   = results_by_name.get("PCSD_dgsc_cls",     {}).get("best_recovery", 0)
r_dncnn_cls  = results_by_name.get("DnCNN_cls",         {}).get("best_recovery", 0)

print(f"  Cls loss contribution (static):  "
      f"{r_static_cls - r_static_pix:+.2f}% "
      f"({r_static_pix:.1f}% → {r_static_cls:.1f}%)")
print(f"  DGSC contribution (no cls):      "
      f"{r_dgsc_pix - r_static_pix:+.2f}% "
      f"({r_static_pix:.1f}% → {r_dgsc_pix:.1f}%)")
print(f"  DGSC contribution (with cls):    "
      f"{r_dgsc_cls - r_static_cls:+.2f}% "
      f"({r_static_cls:.1f}% → {r_dgsc_cls:.1f}%)")
print(f"  PCSD vs DnCNN (both use cls):    "
      f"{r_dgsc_cls - r_dncnn_cls:+.2f}% "
      f"({r_dncnn_cls:.1f}% → {r_dgsc_cls:.1f}%)")

# ── Save ──────────────────────────────────────────────────
save_path = os.path.join(RESULT_DIR, "e4_ablation_pcsd.json")
with open(save_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved: {save_path}")
