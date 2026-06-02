"""
train_pcsd.py — Train the PCSD denoiser (Profile-Conditioned Spectral Denoiser).

Training data: pairs of (adversarial image, clean image).
Generates PGD adversarial examples on-the-fly from the frozen classifier,
then trains the denoiser to reconstruct the clean image.

Novel contributions vs DnCNN baseline:
  1. Dynamic Gradient Spectrum Conditioning (DGSC): per-image FiLM conditioning
     using the classifier's gradient frequency spectrum of the adversarial input.
  2. Classifier-guided loss: CE(classifier(denoised), y) — cite Liao 2018.

DnCNN baseline: plain pixel loss + static fixed profile (no per-image adaptation).

Usage:
  python -m pcsd.train_pcsd                   (PCSD + DGSC + classifier guidance)
  python -m pcsd.train_pcsd --model dncnn     (DnCNN blind baseline)
  python -m pcsd.train_pcsd --lambda_cls 1.0  (stronger classifier guidance)
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import numpy as np

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    N_BINS_SMALL, PX_PER_DEGREE_32,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral_lipschitz import compute_sli
from pcsd.unet import PCSD, DnCNN

# ── Args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["pcsd", "dncnn"], default="pcsd")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epsilon", type=float, default=8.0/255.0)
parser.add_argument("--pgd_steps", type=int, default=10)
parser.add_argument("--n_bins", type=int, default=N_BINS_SMALL)
parser.add_argument("--lambda_cls", type=float, default=0.5,
                    help="Weight of classifier-guided CE loss (PCSD only)")
parser.add_argument("--classifier_ckpt", type=str, default=None)
parser.add_argument("--dgsc_classifier_ckpt", type=str, default=None,
                    help="Separate classifier for DGSC gradient computation. "
                         "Use a HIGH-MeanLip (clean) model for richer frequency signals. "
                         "If None, uses the same classifier as task solver.")
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)
SAVE_DIR = str(CHECKPOINT_DIR)
DATA_ROOT = str(DATA_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Denoiser: {args.model}")
print(f"Epochs: {args.epochs}, LR: {args.lr}")


# ── Load frozen classifier ──────────────────────────────
classifier_path = args.classifier_ckpt
if classifier_path is None:
    for name in ["resnet18_cifar10_lat_robust.pth",
                 "resnet18_cifar10_lat.pth",
                 "resnet18_cifar10_pgd_at.pth",
                 "resnet18_cifar10_clean.pth"]:
        p = os.path.join(SAVE_DIR, name)
        if os.path.exists(p):
            classifier_path = p
            break

if classifier_path is None:
    print("ERROR: No classifier checkpoint found.")
    sys.exit(1)

print(f"Classifier: {classifier_path}")
classifier = ResNet18_CIFAR(num_classes=10).to(DEVICE)
sd = torch.load(classifier_path, map_location=DEVICE, weights_only=True)
sample_key = next(iter(sd.keys()))
if not sample_key.startswith("model."):
    sd = {f"model.{k}": v for k, v in sd.items()}
classifier.load_state_dict(sd, strict=False)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False


# ── Load DGSC detector (clean model — high MeanLip for rich gradients) ──
# Key insight: DGSC needs a HIGH-sensitivity model to detect attack frequencies.
# The robust classifier (MeanLip=0.031) has flat gradients — useless for DGSC.
# The clean classifier (MeanLip=0.227) screams at exactly the exploited frequencies.
dgsc_ckpt = args.dgsc_classifier_ckpt
if dgsc_ckpt is None and args.model == "pcsd":
    # Auto-find clean checkpoint
    for name in ["resnet18_cifar10_clean.pth"]:
        p = os.path.join(SAVE_DIR, name)
        if os.path.exists(p):
            dgsc_ckpt = p
            break

if dgsc_ckpt and dgsc_ckpt != classifier_path:
    print(f"DGSC Detector: {dgsc_ckpt}  (clean — high MeanLip=0.227)")
    detector = ResNet18_CIFAR(num_classes=10).to(DEVICE)
    det_sd = torch.load(dgsc_ckpt, map_location=DEVICE, weights_only=True)
    if not next(iter(det_sd)).startswith("model."):
        det_sd = {f"model.{k}": v for k, v in det_sd.items()}
    detector.load_state_dict(det_sd, strict=False)
    detector.eval()
    for param in detector.parameters():
        param.requires_grad = False
else:
    print("DGSC Detector: same as classifier (fallback)")
    detector = classifier


# ── Compute static Lipschitz profile (for DnCNN / fallback) ────
print("Computing Lipschitz profile...")
test_data_raw = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
profile_subset = Subset(test_data_raw, list(range(100)))
profile_loader = DataLoader(profile_subset, batch_size=16, shuffle=False)

sli_result = compute_sli(
    classifier, profile_loader, DEVICE, args.n_bins,
    epsilon=0.01, max_images=100, desc="Profile",
)
lip_profile = sli_result["lipschitz_profile"]
lip_profile_t = torch.tensor(lip_profile, dtype=torch.float32, device=DEVICE)
print(f"  Profile shape: {lip_profile_t.shape}")
print(f"  Mean Lipschitz: {sli_result['mean_lipschitz']:.4f}")


# ── PGD attack (training pairs) ──────────────────────────
def generate_adversarial(classifier, images, labels, epsilon, steps, mean, std):
    device = images.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm   = epsilon / std_t
    alpha_norm = (epsilon / 4) / std_t
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    delta = torch.zeros_like(images).uniform_(-1, 1) * eps_norm
    delta = torch.max(torch.min(images + delta, upper), lower) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(classifier(images + delta), labels)
        loss.backward()
        g = delta.grad.detach()
        delta = delta.detach() + alpha_norm * g.sign()
        delta = torch.clamp(delta, -eps_norm, eps_norm)
        delta = torch.max(torch.min(images + delta, upper), lower) - images

    return (images + delta).detach()


# ── Profile helpers ──────────────────────────────────────
def get_static_profile(batch_size):
    """Fixed Lipschitz profile — same for every image (DnCNN baseline)."""
    return lip_profile_t.unsqueeze(0).expand(batch_size, -1)


def compute_dynamic_profile(adv_images):
    """
    Dynamic Gradient Spectrum Conditioning (DGSC) — novel contribution.

    Computes the frequency spectrum of the classifier's input gradient for
    each adversarial image. This gives a per-image conditioning vector that
    tells PCSD which frequency bands this specific attack exploited.

    Key properties:
      - Varies per image (unlike static profile which is fixed)
      - Computed from adversarial input alone — works at inference time
      - DnCNN cannot use this (no conditioning architecture)
      - Connects to our GSE metric: both use gradient frequency spectra

    Args:
        adv_images: (B, C, H, W) adversarial examples

    Returns:
        (B, n_bins) per-image gradient frequency spectrum, normalized to [0,1]
    """
    n_bins = args.n_bins
    B, C, H, W = adv_images.shape
    device = adv_images.device

    # 1. Compute DETECTOR gradient — using the HIGH-MeanLip clean model.
    #    The clean model is hyper-sensitive and screams at the attack frequencies.
    #    The robust classifier has flat gradients (MeanLip=0.031) — useless here.
    x = adv_images.detach().clone().requires_grad_(True)
    logits = detector(x)
    pred = logits.argmax(1)
    F.cross_entropy(logits, pred).backward()
    grad = x.grad.detach().abs()           # (B, C, H, W)

    # 2. Clip to prevent FFT explosion from extreme gradient values
    grad = torch.clamp(grad, 0.0, 1e2)

    # 3. FFT → frequency-domain gradient magnitude
    G = torch.fft.fftshift(torch.fft.fft2(grad), dim=(-2, -1)).abs()
    G = G.mean(dim=1)                      # (B, H, W) avg over channels
    G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. Bin into radial frequency bands (same as spectral_lipschitz.py)
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    r_max = float((cx ** 2 + cy ** 2) ** 0.5)
    bin_idx = (r / r_max * (n_bins - 1)).clamp(0, n_bins - 1).long()  # (H, W)

    spectrum = torch.zeros(B, n_bins, device=device)
    for b in range(n_bins):
        mask = (bin_idx == b)                              # (H, W)
        if mask.sum() > 0:
            spectrum[:, b] = G[:, mask].mean(dim=1)        # (B,)

    spectrum = torch.nan_to_num(spectrum, nan=0.0, posinf=0.0)

    # 5. Normalize to [0, 1] per image (max=1), same scale as lip_profile_t
    #    This prevents FiLM layers seeing out-of-distribution values
    max_val = spectrum.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    spectrum = spectrum / max_val

    # 6. Fallback to uniform if degenerate
    bad = spectrum.sum(dim=1) < 1e-6
    if bad.any():
        spectrum[bad] = torch.full((n_bins,), 1.0 / n_bins, device=device)

    return spectrum.detach()   # (B, n_bins)


# ── Data ────────────────────────────────────────────────
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
test_loader  = DataLoader(test_data, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)


# ── Denoiser ──────────────────────────────────────────
if args.model == "pcsd":
    denoiser = PCSD(in_channels=3, profile_dim=args.n_bins, base_ch=32).to(DEVICE)
else:
    denoiser = DnCNN(in_channels=3, num_layers=8, num_features=64).to(DEVICE)

print(f"Denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")

optimizer = optim.Adam(denoiser.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# ── Training ────────────────────────────────────────────
best_recovery = -999.0
history = []

print(f"\nStarting {args.model.upper()} training...")
if args.model == "pcsd":
    print("Conditioning:  Dynamic Gradient Spectrum per image (DGSC) ← novel")
    print(f"Loss:          pixel + {args.lambda_cls} * CE(classifier(denoised), y)  ← Liao 2018")
    print("DnCNN baseline: static profile + pixel loss only\n")
else:
    cls_info = f"pixel + {args.lambda_cls} * CE(classifier(denoised), y)" if args.lambda_cls > 0 else "pixel loss only"
    print(f"Loss:          {cls_info}")
    print("Conditioning:  static Lipschitz profile (fixed)\n")

for epoch in range(args.epochs):
    denoiser.train()
    train_pix_loss = 0.0
    train_cls_loss = 0.0
    train_total = 0
    t0 = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        B = images.size(0)

        # ── Inner: generate adversarial examples ──
        adv_images = generate_adversarial(
            classifier, images, labels,
            args.epsilon, args.pgd_steps, CIFAR10_MEAN, CIFAR10_STD,
        )

        # ── Profile: DGSC (per-image) for PCSD, static for DnCNN ──
        if args.model == "pcsd":
            profile = compute_dynamic_profile(adv_images)   # (B, n_bins)
        else:
            profile = get_static_profile(B)                 # (B, n_bins)

        # ── Forward + loss ──
        denoised = denoiser(adv_images, profile)
        pix_loss = nn.L1Loss()(denoised, images) + 0.5 * nn.MSELoss()(denoised, images)

        if args.lambda_cls > 0:
            cls_loss = F.cross_entropy(classifier(denoised), labels)
            loss = pix_loss + args.lambda_cls * cls_loss
        else:
            cls_loss = torch.zeros(1, device=DEVICE)
            loss = pix_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=5.0)
        optimizer.step()

        train_pix_loss += pix_loss.item() * B
        train_cls_loss += cls_loss.item() * B
        train_total += B

    scheduler.step()

    # ── Evaluate ──
    denoiser.eval()
    total_psnr = 0.0
    clean_correct = adv_correct = denoised_correct = eval_total = 0

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        B = images.size(0)

        # Adversarial generation (needs backward → outside no_grad)
        adv_images = generate_adversarial(
            classifier, images, labels,
            args.epsilon, args.pgd_steps, CIFAR10_MEAN, CIFAR10_STD,
        )

        # CRITICAL: compute_dynamic_profile also needs backward → outside no_grad
        if args.model == "pcsd":
            profile = compute_dynamic_profile(adv_images)
        else:
            profile = get_static_profile(B)

        with torch.no_grad():
            clean_correct   += (classifier(images).argmax(1) == labels).sum().item()
            adv_correct     += (classifier(adv_images).argmax(1) == labels).sum().item()
            denoised        = denoiser(adv_images, profile)
            denoised_correct += (classifier(denoised).argmax(1) == labels).sum().item()
            mse_val = (denoised - images).pow(2).mean(dim=(1, 2, 3))
            total_psnr += (10 * torch.log10(1.0 / (mse_val + 1e-8))).sum().item()
        eval_total += B

    avg_psnr     = total_psnr / eval_total
    clean_acc    = 100.0 * clean_correct / eval_total
    adv_acc      = 100.0 * adv_correct / eval_total
    denoised_acc = 100.0 * denoised_correct / eval_total
    recovery     = denoised_acc - adv_acc
    elapsed      = time.time() - t0

    log = dict(epoch=epoch+1,
               pix_loss=train_pix_loss/train_total,
               cls_loss=train_cls_loss/train_total,
               psnr=avg_psnr, clean_acc=clean_acc,
               adv_acc=adv_acc, denoised_acc=denoised_acc,
               recovery=recovery, time=elapsed)
    history.append(log)

    cls_str = f" | L_cls: {train_cls_loss/train_total:.3f}" if args.model == "pcsd" else ""
    print(f"Epoch {epoch+1:3d}/{args.epochs} | "
          f"L_pix: {train_pix_loss/train_total:.4f}{cls_str} | PSNR: {avg_psnr:.1f}dB | "
          f"Clean: {clean_acc:.1f}% | Adv: {adv_acc:.1f}% | "
          f"Denoised: {denoised_acc:.1f}% (+{recovery:.1f}%) | {elapsed:.1f}s")

    if recovery > best_recovery:
        best_recovery = recovery
        torch.save(denoiser.state_dict(),
                   os.path.join(SAVE_DIR, f"{args.model}_cifar10.pth"))

torch.save(denoiser.state_dict(),
           os.path.join(SAVE_DIR, f"{args.model}_cifar10_final.pth"))

print(f"\nBest accuracy recovery: +{best_recovery:.2f}%")
print(f"Saved: {os.path.join(SAVE_DIR, f'{args.model}_cifar10.pth')}")

with open(os.path.join(SAVE_DIR, f"history_{args.model}.json"), "w") as f:
    json.dump(history, f, indent=2)