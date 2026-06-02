"""
train_clean.py — Clean (standard) training baseline.

Usage:
  Local:  python -m caat.train_clean
  Kaggle: %run caat/train_clean.py

Trains ResNet18 on CIFAR-10 with standard augmentation.
No adversarial training — this is the baseline.
"""

import sys
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import numpy as np

# ── Environment detection ─────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")

if IS_KAGGLE:
    sys.path.insert(0, "/kaggle/working/Project")
    DATA_ROOT = "/kaggle/working/data"
    SAVE_DIR = "/kaggle/working/checkpoints"
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from caat.config import DATA_DIR, CHECKPOINT_DIR
    DATA_ROOT = str(DATA_DIR)
    SAVE_DIR = str(CHECKPOINT_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD, TRAIN_CONFIG,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms

# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

cfg = TRAIN_CONFIG["clean"]
print(f"Device: {DEVICE}")
print(f"Config: {json.dumps(cfg, indent=2)}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True,
    transform=get_cifar10_transforms(train=True),
)
test_data = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)

train_loader = DataLoader(train_data, batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=cfg["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ResNet18_CIFAR(num_classes=10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                      momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────────────────────
best_acc = 0.0
history = []

for epoch in range(cfg["epochs"]):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    t0 = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    scheduler.step()

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_total += images.size(0)

    train_acc = 100.0 * train_correct / train_total
    test_acc = 100.0 * test_correct / test_total
    elapsed = time.time() - t0

    log = {
        "epoch": epoch + 1,
        "train_loss": train_loss / train_total,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "lr": scheduler.get_last_lr()[0],
        "time": elapsed,
    }
    history.append(log)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"TrainAcc: {train_acc:.1f}% | TestAcc: {test_acc:.1f}% | "
              f"LR: {log['lr']:.4f} | Time: {elapsed:.1f}s")

    # Save best
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),
                    os.path.join(SAVE_DIR, "resnet18_cifar10_clean.pth"))

print(f"\nBest test accuracy: {best_acc:.2f}%")
print(f"Model saved to: {os.path.join(SAVE_DIR, 'resnet18_cifar10_clean.pth')}")

# Save training history
with open(os.path.join(SAVE_DIR, "history_clean.json"), "w") as f:
    json.dump(history, f, indent=2)
