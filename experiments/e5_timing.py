"""
e5_timing.py — Inference throughput comparison.

Measures images/sec and ms/image for:
  - Bare classifier (ResNet18)
  - DnCNN + classifier
  - PCSD + static profile + classifier
  - PCSD + DGSC + classifier  <- most expensive (requires backward() at inference)

Usage:
  python -m experiments.e5_timing
"""

import os
import sys
import time
import json
import torch
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
import torch.nn.functional as F

torch.manual_seed(SEED)
SAVE_DIR   = str(CHECKPOINT_DIR)
RESULT_DIR = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(RESULT_DIR, exist_ok=True)

N_WARMUP   = 20    # warmup batches (GPU JIT, caching)
N_MEASURE  = 100   # measurement batches
BATCH_SIZE = 32    # realistic inference batch

# ── Load classifier ───────────────────────────────────────
print("Loading classifier...")
ckpt = None
for name in ["resnet18_cifar10_lat_robust.pth", "resnet18_cifar10_lat.pth",
             "resnet18_cifar10_pgd_at.pth"]:
    p = os.path.join(SAVE_DIR, name)
    if os.path.exists(p):
        ckpt = p
        break
assert ckpt, "No classifier checkpoint found."

classifier = ResNet18_CIFAR(num_classes=10).to(DEVICE)
sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
if not next(iter(sd)).startswith("model."):
    sd = {f"model.{k}": v for k, v in sd.items()}
classifier.load_state_dict(sd, strict=False)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

# ── Load PCSD ─────────────────────────────────────────────
pcsd = PCSD(in_channels=3, profile_dim=N_BINS_SMALL, base_ch=32).to(DEVICE)
pcsd_ckpt = os.path.join(SAVE_DIR, "pcsd_cifar10.pth")
if os.path.exists(pcsd_ckpt):
    pcsd.load_state_dict(torch.load(pcsd_ckpt, map_location=DEVICE, weights_only=True))
else:
    print("  WARNING: pcsd_cifar10.pth not found, using random weights")
pcsd.eval()

# ── Load DnCNN ────────────────────────────────────────────
dncnn = DnCNN(in_channels=3, num_layers=8, num_features=64).to(DEVICE)
dncnn_ckpt = os.path.join(SAVE_DIR, "dncnn_cifar10.pth")
if os.path.exists(dncnn_ckpt):
    dncnn.load_state_dict(torch.load(dncnn_ckpt, map_location=DEVICE, weights_only=True))
else:
    print("  WARNING: dncnn_cifar10.pth not found, using random weights")
dncnn.eval()

# ── Static profile ────────────────────────────────────────
test_data = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
profile_loader = DataLoader(Subset(test_data, list(range(100))), batch_size=16)
sli = compute_sli(classifier, profile_loader, DEVICE, N_BINS_SMALL,
                  epsilon=0.01, max_images=100, desc="Profile")
static_profile = torch.tensor(
    sli["lipschitz_profile"], dtype=torch.float32, device=DEVICE
)

# ── Data ──────────────────────────────────────────────────
loader = DataLoader(
    Subset(test_data, list(range(BATCH_SIZE * (N_WARMUP + N_MEASURE)))),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
)
batches = [(x.to(DEVICE), y.to(DEVICE)) for x, y in loader]


def time_pipeline(fn, name):
    """Time a pipeline function over N_MEASURE batches after warmup."""
    # Warmup
    for i in range(N_WARMUP):
        _ = fn(batches[i][0])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    for i in range(N_MEASURE):
        _ = fn(batches[N_WARMUP + i][0])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_images = N_MEASURE * BATCH_SIZE
    ms_per_image  = elapsed / total_images * 1000
    fps           = total_images / elapsed

    print(f"  {name:<45} {ms_per_image:>8.2f} ms/img  {fps:>8.0f} img/s")
    return {"name": name, "ms_per_image": ms_per_image, "fps": fps}


def compute_dynamic_profile(adv_images):
    """DGSC: explicitly uses enable_grad() to compute the profile safely."""
    n_bins = N_BINS_SMALL
    B, C, H, W = adv_images.shape
    device = adv_images.device

    with torch.enable_grad():
        x = adv_images.detach().clone().requires_grad_(True)
        logits = classifier(x)
        loss = F.cross_entropy(logits, logits.argmax(1))
        loss.backward()
        grad = torch.clamp(x.grad.detach().abs(), 0, 1e2)

    G = torch.fft.fftshift(torch.fft.fft2(grad), dim=(-2, -1)).abs().mean(1)
    G = torch.nan_to_num(G)
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij")
    r = torch.sqrt((gx - cx)**2 + (gy - cy)**2)
    bin_idx = (r / float((cx**2 + cy**2)**0.5) * (n_bins - 1)).clamp(0, n_bins-1).long()
    spectrum = torch.zeros(B, n_bins, device=device)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() > 0:
            spectrum[:, b] = G[:, mask].mean(1)
    spectrum = torch.nan_to_num(spectrum)
    mx = spectrum.max(1, keepdim=True).values.clamp(1e-8)
    return (spectrum / mx).detach()


# ── Pipeline Definitions ──────────────────────────────────
sp = static_profile.unsqueeze(0).expand(BATCH_SIZE, -1)


@torch.no_grad()
def base_pipeline(x):
    return classifier(x)


@torch.no_grad()
def dncnn_pipeline(x):
    try:
        return classifier(dncnn(x, sp))
    except TypeError:
        return classifier(dncnn(x))


@torch.no_grad()
def pcsd_static_pipeline(x):
    return classifier(pcsd(x, sp))


def dgsc_pipeline(x):
    # 1. Compute profile WITH gradients allowed
    profile = compute_dynamic_profile(x)
    # 2. Run the rest WITHOUT gradients
    with torch.no_grad():
        return classifier(pcsd(x, profile))


# ── Run timing ────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"INFERENCE THROUGHPUT  (batch={BATCH_SIZE}, device={DEVICE})")
print(f"{'='*70}")
print(f"  {'Pipeline':<45} {'ms/img':>10}  {'img/sec':>10}")
print(f"  {'-'*65}")

results = []
results.append(time_pipeline(base_pipeline,        "ResNet18 classifier (no defense)"))
results.append(time_pipeline(dncnn_pipeline,       "DnCNN + classifier (blind denoiser)"))
results.append(time_pipeline(pcsd_static_pipeline, "PCSD + static profile + classifier"))
results.append(time_pipeline(dgsc_pipeline,        "PCSD + DGSC + classifier (ours)"))

# ── Summary ───────────────────────────────────────────────
print(f"\n  Overhead vs bare classifier:")
base_fps = results[0]["fps"]
for r in results[1:]:
    slowdown = base_fps / r["fps"]
    print(f"    {r['name']:<45} {slowdown:.1f}× slower")

# Save
save_path = os.path.join(RESULT_DIR, "e5_timing.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {save_path}")
