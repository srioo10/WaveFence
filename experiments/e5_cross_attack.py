"""
e5_cross_attack.py — Cross-Attack Generalization Test.

Evaluates PCSD+DGSC vs DnCNN+cls on UNSEEN attack types.
Models were trained on PGD-Linf only. This tests generalization.

Attacks evaluated:
  1. PGD-Linf (known — same as training)
  2. FGSM     (single-step, weaker, slightly different frequency)
  3. PGD-L2   (smooth perturbations, VERY different frequency spectrum)
  4. Random Noise (Gaussian noise — no frequency structure)

Hypothesis:
  DnCNN memorized PGD-Linf spatial patterns. On unseen attacks it drops.
  PCSD+DGSC computes gradient spectrum at inference → adapts to new patterns.

Usage:
  python -m experiments.e5_cross_attack
  python -m experiments.e5_cross_attack --n_batches 20
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, N_BINS_SMALL,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral_lipschitz import compute_sli
from pcsd.unet import PCSD, DnCNN

parser = argparse.ArgumentParser()
parser.add_argument("--n_batches",  type=int, default=15)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epsilon",    type=float, default=8.0 / 255.0)
parser.add_argument("--epsilon_l2", type=float, default=1.0)      # L2 ball radius
parser.add_argument("--pgd_steps",  type=int, default=20)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR   = str(CHECKPOINT_DIR)
RESULT_DIR = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Load classifier ───────────────────────────────────────
classifier = ResNet18_CIFAR(num_classes=10).to(DEVICE)
for name in ["resnet18_cifar10_lat_robust.pth", "resnet18_cifar10_lat.pth"]:
    p = os.path.join(SAVE_DIR, name)
    if os.path.exists(p):
        sd = torch.load(p, map_location=DEVICE, weights_only=True)
        if not next(iter(sd)).startswith("model."):
            sd = {f"model.{k}": v for k, v in sd.items()}
        classifier.load_state_dict(sd, strict=False)
        print(f"Classifier: {name}")
        break
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

# ── Load DGSC detector (clean — high MeanLip for rich gradients) ──────────
detector = ResNet18_CIFAR(num_classes=10).to(DEVICE)
det_path = os.path.join(SAVE_DIR, "resnet18_cifar10_clean.pth")
if os.path.exists(det_path):
    det_sd = torch.load(det_path, map_location=DEVICE, weights_only=True)
    if not next(iter(det_sd)).startswith("model."):
        det_sd = {f"model.{k}": v for k, v in det_sd.items()}
    detector.load_state_dict(det_sd, strict=False)
    print("DGSC Detector: resnet18_cifar10_clean.pth  (MeanLip=0.227 — hyper-sensitive)")
else:
    detector = classifier
    print("DGSC Detector: fallback to robust classifier")
detector.eval()
for p in detector.parameters():
    p.requires_grad = False

# ── Load PCSD ─────────────────────────────────────────────
pcsd = PCSD(in_channels=3, profile_dim=N_BINS_SMALL, base_ch=32).to(DEVICE)
p = os.path.join(SAVE_DIR, "pcsd_cifar10.pth")
if os.path.exists(p):
    pcsd.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    print("PCSD: pcsd_cifar10.pth")
else:
    print("WARNING: pcsd_cifar10.pth not found")
pcsd.eval()

# ── Load DnCNN ────────────────────────────────────────────
dncnn = DnCNN(in_channels=3, num_layers=8, num_features=64).to(DEVICE)
p = os.path.join(SAVE_DIR, "dncnn_cifar10.pth")
if os.path.exists(p):
    dncnn.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    print("DnCNN: dncnn_cifar10.pth")
else:
    print("WARNING: dncnn_cifar10.pth not found")
dncnn.eval()

# ── Static profile ────────────────────────────────────────
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


# ── DGSC ─────────────────────────────────────────────────
def compute_dgsc(adv_images):
    n_bins = N_BINS_SMALL
    B, C, H, W = adv_images.shape
    device = adv_images.device
    x = adv_images.detach().clone().requires_grad_(True)
    F.cross_entropy(detector(x), detector(x).argmax(1)).backward()
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


# ── Attack functions ──────────────────────────────────────
def norm_params(eps=None):
    mean_t = torch.tensor(CIFAR10_MEAN, device=DEVICE).view(1, -1, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=DEVICE).view(1, -1, 1, 1)
    lo = (0.0 - mean_t) / std_t
    hi = (1.0 - mean_t) / std_t
    if eps is not None:
        en = eps / std_t
        al = (eps / 4) / std_t
        return en, al, lo, hi
    return lo, hi


def attack_fgsm(images, labels):
    """FGSM — single step, Linf. Same epsilon as training."""
    en, _, lo, hi = norm_params(args.epsilon)
    images.requires_grad_(True)
    F.cross_entropy(classifier(images), labels).backward()
    adv = images + en * images.grad.sign()
    adv = torch.max(torch.min(adv, hi), lo)
    return adv.detach()


def attack_pgd_linf(images, labels):
    """PGD-Linf — SAME as training attack. Known distribution."""
    en, al, lo, hi = norm_params(args.epsilon)
    delta = torch.zeros_like(images).uniform_(-1, 1) * en
    delta = torch.max(torch.min(images + delta, hi), lo) - images
    for _ in range(args.pgd_steps):
        delta.requires_grad_(True)
        F.cross_entropy(classifier(images + delta), labels).backward()
        g = delta.grad.detach()
        delta = delta.detach() + al * g.sign()
        delta = torch.clamp(delta, -en, en)
        delta = torch.max(torch.min(images + delta, hi), lo) - images
    return (images + delta).detach()


def attack_pgd_l2(images, labels):
    """
    PGD-L2 — VERY different frequency spectrum from Linf.
    L2-constrained: perturbation is smooth, spread across frequencies.
    DnCNN trained on Linf spikes will fail here.
    """
    lo, hi = norm_params()
    eps = args.epsilon_l2
    alpha = eps / 4

    delta = torch.randn_like(images) * 0.01
    # Project onto L2 ball
    dnorm = delta.view(delta.size(0), -1).norm(2, dim=1).clamp(min=1e-8)
    delta = delta * (eps / dnorm.view(-1, 1, 1, 1))

    for _ in range(args.pgd_steps):
        delta.requires_grad_(True)
        F.cross_entropy(classifier(images + delta), labels).backward()
        g = delta.grad.detach()

        # Normalize gradient to unit L2
        gnorm = g.view(g.size(0), -1).norm(2, dim=1).clamp(min=1e-8)
        g = g / gnorm.view(-1, 1, 1, 1)

        delta = delta.detach() + alpha * g

        # Project onto L2 ball of radius eps
        dnorm = delta.view(delta.size(0), -1).norm(2, dim=1).clamp(min=1e-8)
        factor = torch.min(
            torch.ones_like(dnorm),
            eps / dnorm
        )
        delta = delta * factor.view(-1, 1, 1, 1)
        delta = torch.max(torch.min(images + delta, hi), lo) - images

    return (images + delta).detach()


def attack_random_noise(images, labels):
    """Gaussian noise — no adversarial structure at all. Tests noise robustness."""
    lo, hi = norm_params()
    noise = torch.randn_like(images) * (args.epsilon / 2)
    adv = (images + noise).clamp(lo, hi)
    return adv.detach()


ATTACKS = {
    "PGD-Linf (known)":  attack_pgd_linf,
    "FGSM (simpler)":    attack_fgsm,
    "PGD-L2 (unseen)":   attack_pgd_l2,
    "Random Noise":      attack_random_noise,
}


# ── Evaluate ──────────────────────────────────────────────
def run(attack_fn):
    """Return (adv_acc, dncnn_acc, pcsd_acc) over n_batches."""
    adv_c = dn_c = pc_c = total = 0
    for i, (images, labels) in enumerate(test_loader):
        if i >= args.n_batches:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        B = images.size(0)

        adv = attack_fn(images, labels)
        sp  = static_profile.unsqueeze(0).expand(B, -1)
        dgsc_p = compute_dgsc(adv)

        with torch.no_grad():
            adv_c += (classifier(adv).argmax(1) == labels).sum().item()
            dn_c  += (classifier(dncnn(adv, sp)).argmax(1) == labels).sum().item()
            pc_c  += (classifier(pcsd(adv, dgsc_p)).argmax(1) == labels).sum().item()
        total += B

    return (100.0 * adv_c / total,
            100.0 * dn_c  / total,
            100.0 * pc_c  / total)


# ── Run all attacks ───────────────────────────────────────
print(f"\n{'='*72}")
print(f"CROSS-ATTACK GENERALIZATION  (ε_linf=8/255, ε_l2={args.epsilon_l2})")
print(f"Both models trained on PGD-Linf ONLY → testing on unseen attacks")
print(f"PCSD uses Dual-Classifier DGSC: Clean(detect) + Robust(solve)")
print(f"N={args.n_batches * args.batch_size} images | Device={DEVICE}")
print(f"{'='*76}")
print(f"  {'Attack':<25} {'Adv':>6}  {'DnCNN+cls':>10}  {'PCSD+DualDGSC':>14}  {'Δ (PCSD-DnCNN)':>14}")
print(f"  {'-'*72}")

all_results = {}
for attack_name, attack_fn in ATTACKS.items():
    adv_acc, dn_acc, pc_acc = run(attack_fn)
    delta = pc_acc - dn_acc
    marker = " ← PCSD WINS" if delta > 2.0 else (" ← DnCNN wins" if delta < -2.0 else " ← TIE")
    print(f"  {attack_name:<25} {adv_acc:>5.1f}%  {dn_acc:>9.1f}%  {pc_acc:>9.1f}%  {delta:>+12.1f}%{marker}")
    all_results[attack_name] = {
        "adv_acc": adv_acc,
        "dncnn_acc": dn_acc,
        "pcsd_acc": pc_acc,
        "delta": delta,
    }

# Key finding
pgd_l2 = all_results.get("PGD-L2 (unseen)", {})
print(f"\n{'='*72}")
print("KEY FINDING:")
if pgd_l2.get("delta", 0) > 3.0:
    print(f"  ✅ PCSD+DGSC generalizes to PGD-L2 (+{pgd_l2['delta']:.1f}% over DnCNN)")
    print("     DnCNN memorized PGD-Linf patterns. DGSC adapts dynamically.")
elif pgd_l2.get("delta", 0) < -3.0:
    print(f"  ⚠️  DnCNN generalizes better on PGD-L2 ({pgd_l2['delta']:.1f}%)")
else:
    print(f"  Both generalize similarly to PGD-L2 (Δ={pgd_l2.get('delta',0):.1f}%)")
    print("  PCSD's advantage is interpretability, not generalization on this test.")

save_path = os.path.join(RESULT_DIR, "e5_cross_attack.json")
with open(save_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {save_path}")
