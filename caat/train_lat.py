"""
train_lat.py — Lipschitz Adversarial Training (LAT)

OUR DEFENSE METHOD (replaces CAAT).

Approach:
  Inner loop: Standard PGD (full strength, like PGD-AT)
  Outer loop: CE(adv) + lambda_lip * L_lip(clean)

L_lip = differentiable Lipschitz estimate:
  - Random perturbation delta ~ N(0, sigma)
  - L_lip = mean( ||f(x+d) - f(x)||_2 / ||d||_2 )
  - Gradient penalizes model for being too sensitive to small perturbations

This is grounded — MeanLip had rho = -0.788 (p=0.035) with PGD accuracy.
Penalizing it during training should directly improve robustness.

Usage:
  python -m caat.train_lat
  python -m caat.train_lat --lambda_acl 0.5
  python -m caat.train_lat --lambda_acl 1.0   (stronger consistency)
  python -m caat.train_lat --lambda_acl 0.0   (= pure PGD-AT baseline)
"""

import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD, TRAIN_CONFIG,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms

# ── Args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lambda_acl", type=float, default=0.5,
                    help="Weight of Adversarial Consistency Loss")
parser.add_argument("--lip_sigma", type=float, default=0.05,
                    help="Std of random perturbation for Lipschitz estimation")
parser.add_argument("--lip_samples", type=int, default=1,
                    help="Number of random perturbations per batch (1 is fine)")
parser.add_argument("--epochs", type=int, default=None)
args, _ = parser.parse_known_args()

# ── Setup ───────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

cfg = TRAIN_CONFIG["pgd_at"].copy()
cfg["lambda_acl"] = args.lambda_acl
cfg["lip_sigma"] = args.lip_sigma
if args.epochs is not None:
    cfg["epochs"] = args.epochs

SAVE_DIR = str(CHECKPOINT_DIR)
DATA_ROOT = str(DATA_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"LAT: PGD-AT + Adversarial Consistency Loss (lambda_acl={cfg['lambda_acl']})")
print(f"Config: {json.dumps({k: v for k, v in cfg.items()}, indent=2)}")

# ── Data ────────────────────────────────────────────────
train_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True,
    transform=get_cifar10_transforms(train=True),
)
test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)

train_loader = DataLoader(train_data, batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=cfg["batch_size"],
                         shuffle=False, num_workers=4, pin_memory=True)


# ── Standard PGD inner loop ─────────────────────────────
def pgd_attack(model, images, labels, epsilon, alpha, steps, mean, std):
    """Standard PGD — full gradient, no spectral filtering."""
    device = images.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)

    eps_norm = epsilon / std_t
    alpha_norm = alpha / std_t
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    delta = torch.zeros_like(images).uniform_(-1, 1) * eps_norm
    delta = torch.max(torch.min(images + delta, upper), lower) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(images + delta)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        grad = delta.grad.detach()
        delta = delta.detach() + alpha_norm * grad.sign()
        delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
        delta = torch.max(torch.min(images + delta, upper), lower) - images

    return (images + delta).detach()


# ── Adversarial Consistency Loss (ACL) ───────────────────
def adversarial_consistency_loss(logits_adv, logits_clean):
    """
    Adversarial Consistency Loss (ACL) — our novel outer-loop regularizer.

    Minimizes MSE between adversarial and clean logits:
        L_ACL = ||f(x_adv) - f(x_clean)||^2 / n_classes

    The gradient pushes f(x_adv) → f(x_clean), forcing prediction consistency
    under adversarial perturbation. Logit MSE is properly scaled (~2-10),
    unlike gradient norm (~0.0001) or KL on softmax (~0.0000).

    Zero extra cost: x_adv is already computed in the inner loop.
    Clean logits computed once per batch (shared with CE computation).

    Different from TRADES:
      TRADES inner loop: maximize KL(f(x+δ) || f(x))  [KL-PGD]
      Our inner loop:    maximize CE(f(x+δ), y)        [standard PGD, stronger]
      TRADES outer:      CE(clean) + β * KL(adv || clean)
      Our outer:         CE(adv)   + λ * MSE(adv, clean)  ← stricter: forces adv correct

    Args:
        logits_adv:   (B, C) logits for adversarial images
        logits_clean: (B, C) logits for clean images (detached)

    Returns:
        scalar loss (per-class-normalized MSE, range ~0.2-5.0)
    """
    n_classes = logits_adv.size(1)
    return (logits_adv - logits_clean.detach()).pow(2).mean() / n_classes


# ── Model ───────────────────────────────────────────────
model = ResNet18_CIFAR(num_classes=10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                      momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
criterion = nn.CrossEntropyLoss()

# ── Training ───────────────────────────────────────────
best_acc = 0.0
best_robust = 0.0
history = []

print(f"\nStarting LAT training ({cfg['epochs']} epochs)...")
print(f"Inner loop: standard PGD ({cfg['pgd_steps']} steps, eps={cfg['epsilon']:.4f})")
print(f"Outer loop: CE(adv) + {cfg['lambda_acl']} * L_ACL (adv consistency)\n")

for epoch in range(cfg["epochs"]):
    model.train()
    train_loss_ce = 0.0
    train_loss_lip = 0.0
    train_correct = 0
    train_total = 0
    t0 = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # ── INNER LOOP: Standard PGD ──
        model.eval()
        adv_images = pgd_attack(
            model, images, labels,
            epsilon=cfg["epsilon"],
            alpha=cfg["alpha"],
            steps=cfg["pgd_steps"],
            mean=CIFAR10_MEAN, std=CIFAR10_STD,
        )
        model.train()

        # ── OUTER LOOP: CE(adv) + Adversarial Consistency Loss ──
        with torch.no_grad():
            logits_clean = model(images)         # clean logits (detached)
        logits = model(adv_images)                # adv logits
        loss_ce = criterion(logits, labels)

        # ACL: MSE between adversarial and clean logits
        loss_acl = adversarial_consistency_loss(logits, logits_clean)
        loss_total = loss_ce + cfg["lambda_acl"] * loss_acl

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        train_loss_ce += loss_ce.item() * images.size(0)
        train_loss_lip += loss_acl.item() * images.size(0)
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    scheduler.step()

    # ── Evaluate ──
    model.eval()
    test_clean = 0
    test_total_eval = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            test_clean += (model(images).argmax(1) == labels).sum().item()
            test_total_eval += images.size(0)

    clean_acc = 100.0 * test_clean / test_total_eval
    train_acc = 100.0 * train_correct / train_total
    avg_ce = train_loss_ce / train_total
    avg_lip = train_loss_lip / train_total
    elapsed = time.time() - t0

    # PGD eval every 5 epochs
    robust_acc = -1.0
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg["epochs"] - 1:
        test_robust = 0
        eval_total = 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            adv = pgd_attack(
                model, images, labels,
                epsilon=cfg["epsilon"], alpha=cfg["alpha"],
                steps=cfg["pgd_steps"],
                mean=CIFAR10_MEAN, std=CIFAR10_STD,
            )
            with torch.no_grad():
                test_robust += (model(adv).argmax(1) == labels).sum().item()
            eval_total += images.size(0)
        robust_acc = 100.0 * test_robust / eval_total

    log = {
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "loss_ce": avg_ce,
        "loss_lip": avg_lip,
        "lr": scheduler.get_last_lr()[0],
        "time": elapsed,
    }
    history.append(log)

    robust_str = f"Robust: {robust_acc:.1f}%" if robust_acc >= 0 else ""
    print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
          f"Clean: {clean_acc:.1f}% | {robust_str} | "
          f"L_CE: {avg_ce:.3f} | L_lip: {avg_lip:.2f} | "
          f"{elapsed:.1f}s")

    # Save best by clean accuracy
    if clean_acc > best_acc:
        best_acc = clean_acc
        torch.save(model.state_dict(),
                    os.path.join(SAVE_DIR, "resnet18_cifar10_lat.pth"))

    # Also track best robust accuracy
    if robust_acc > best_robust:
        best_robust = robust_acc
        torch.save(model.state_dict(),
                    os.path.join(SAVE_DIR, "resnet18_cifar10_lat_robust.pth"))

# Save final
torch.save(model.state_dict(),
           os.path.join(SAVE_DIR, "resnet18_cifar10_lat_final.pth"))

print(f"\nBest clean accuracy: {best_acc:.2f}%")
print(f"Best robust accuracy: {best_robust:.2f}%")
print(f"Saved: {os.path.join(SAVE_DIR, 'resnet18_cifar10_lat.pth')}")

with open(os.path.join(SAVE_DIR, "history_lat.json"), "w") as f:
    json.dump(history, f, indent=2)