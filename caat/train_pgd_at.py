"""
train_pgd_at.py — PGD Adversarial Training (clean Lightning version)
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import numpy as np

# ── Clean config import ─────────────────────────────────────
from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    TRAIN_CONFIG, DATA_DIR, CHECKPOINT_DIR
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms

# ── Paths ───────────────────────────────────────────────────
DATA_ROOT = str(DATA_DIR)
SAVE_DIR = str(CHECKPOINT_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

cfg = TRAIN_CONFIG["pgd_at"]

print(f"Device: {DEVICE}")
print(f"Config: {json.dumps(cfg, indent=2)}")

# ── Data ────────────────────────────────────────────────────
train_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True,
    transform=get_cifar10_transforms(train=True),
)
test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)

train_loader = DataLoader(
    train_data, batch_size=cfg["batch_size"],
    shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_data, batch_size=cfg["batch_size"],
    shuffle=False, num_workers=4
)

# ── PGD Attack ──────────────────────────────────────────────
def pgd_inner_loop(model, images, labels, epsilon, alpha, steps, mean, std):

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

        # 🔥 FIX: clear gradients
        model.zero_grad()

        logits = model(images + delta)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        grad = delta.grad.detach()
        delta = delta.detach() + alpha_norm * grad.sign()
        delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
        delta = torch.max(torch.min(images + delta, upper), lower) - images

    return (images + delta).detach()

# ── Model ───────────────────────────────────────────────────
model = ResNet18_CIFAR(num_classes=10).to(DEVICE)

optimizer = optim.SGD(
    model.parameters(),
    lr=cfg["lr"],
    momentum=cfg["momentum"],
    weight_decay=cfg["weight_decay"]
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg["epochs"]
)

criterion = nn.CrossEntropyLoss()

# ── Training ────────────────────────────────────────────────
best_acc = 0.0

for epoch in range(cfg["epochs"]):
    model.train()
    train_correct = 0
    train_total = 0
    t0 = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Generate adversarial examples
        model.eval()
        adv_images = pgd_inner_loop(
            model, images, labels,
            cfg["epsilon"], cfg["alpha"], cfg["pgd_steps"],
            CIFAR10_MEAN, CIFAR10_STD
        )
        model.train()

        logits = model(adv_images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    scheduler.step()

    # ── Evaluation ─────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += images.size(0)

    clean_acc = 100 * correct / total
    elapsed = time.time() - t0

    print(f"Epoch {epoch+1}/{cfg['epochs']} | CleanAcc: {clean_acc:.2f}% | Time: {elapsed:.1f}s")

    # ✅ Save best model
    if clean_acc > best_acc:
        best_acc = clean_acc
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "resnet18_cifar10_pgd_at.pth")
        )

    # ✅ Save every epoch (safety)
    torch.save( 
        model.state_dict(),
        os.path.join(SAVE_DIR, f"pgd_epoch_{epoch+1}.pth")
    )

print(f"\nBest clean accuracy: {best_acc:.2f}%")
print(f"Saved to: {os.path.join(SAVE_DIR, 'resnet18_cifar10_pgd_at.pth')}")