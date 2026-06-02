"""
e1_v2_gse_sli.py — Test GSE + SLI (new metrics) across architectures.

FIXES from v1:
  1. ImageNette label mapping (v1 had random-chance accuracy)
  2. Uses GSE + SLI instead of dead PRI/VCI

Tests 7 pretrained models, computes GSE/SLI, runs FGSM+PGD with
correct labels, computes Spearman correlation.

Usage:
  python e1_v2_gse_sli.py --n_images 200
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from scipy.stats import spearmanr

from caat.config import DEVICE, SEED, RESULTS_DIR, DATA_DIR
from caat.spectral import compute_s_model
from caat.spectral_lipschitz import compute_gse, compute_sli

# ── Args ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_images", type=int, default=200)
parser.add_argument("--n_bins", type=int, default=32)
parser.add_argument("--fgsm_eps", type=float, default=4.0/255.0)
parser.add_argument("--pgd_steps", type=int, default=10)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"Device: {DEVICE}")
print(f"N_images: {args.n_images}, N_bins: {args.n_bins}")
print(f"FGSM eps: {args.fgsm_eps:.4f}\n")


# ═══════════════════════════════════════════════════════════
#  ImageNette label fix
# ═══════════════════════════════════════════════════════════

# ImageNette classes → real ImageNet indices
# Sorted alphabetically (as ImageFolder loads them)
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # n01440764 → tench
    1: 217,  # n02102040 → English springer
    2: 482,  # n02979186 → cassette player
    3: 491,  # n03000684 → chain saw
    4: 497,  # n03028079 → church
    5: 566,  # n03394916 → French horn
    6: 569,  # n03417042 → garbage truck
    7: 571,  # n03425413 → gas pump
    8: 574,  # n03445777 → golf ball
    9: 701,  # n03888257 → parachute
}

IMAGENET_TO_LOCAL = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}
IMAGENETTE_INDICES = list(IMAGENETTE_TO_IMAGENET.values())


class ImageNetteWrapper(Dataset):
    """Wraps ImageFolder to map labels to ImageNet class indices."""
    def __init__(self, imagefolder_dataset):
        self.dataset = imagefolder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Map local label (0-9) to ImageNet class index
        imagenet_label = IMAGENETTE_TO_IMAGENET[label]
        return img, imagenet_label


def imagenet_to_local_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Given 1000-class logits, extract only the 10 ImageNette classes
    and return the predicted local class (0-9).
    """
    # Extract logits for the 10 ImageNette classes
    indices = torch.tensor(IMAGENETTE_INDICES, device=logits.device)
    sub_logits = logits[:, indices]  # (B, 10)
    return sub_logits


# ═══════════════════════════════════════════════════════════
#  Attack functions (with correct label handling)
# ═══════════════════════════════════════════════════════════

def fgsm_eval(model, loader, epsilon, mean, std, device):
    """FGSM with correct ImageNette→ImageNet label mapping."""
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm = epsilon / std_t
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    model.eval()
    clean_correct = 0
    fgsm_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += images.size(0)

        # Clean accuracy — check against ImageNet class labels
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(1)
            clean_correct += (preds == labels).sum().item()

        correct_mask = (preds == labels)
        if correct_mask.sum() == 0:
            continue

        imgs_c = images[correct_mask].detach().requires_grad_(True)
        labs_c = labels[correct_mask]

        out = model(imgs_c)
        loss = F.cross_entropy(out, labs_c)
        loss.backward()

        adv = imgs_c.detach() + eps_norm * imgs_c.grad.sign()
        adv = torch.max(torch.min(adv, upper), lower)

        with torch.no_grad():
            fgsm_correct += (model(adv).argmax(1) == labs_c).sum().item()

    return 100.0 * clean_correct / total, 100.0 * fgsm_correct / total


def pgd_eval(model, loader, epsilon, steps, mean, std, device):
    """PGD with correct labels."""
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm = epsilon / std_t
    alpha_norm = (epsilon / 4) / std_t
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t

    model.eval()
    pgd_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += images.size(0)

        with torch.no_grad():
            correct_mask = (model(images).argmax(1) == labels)
        if correct_mask.sum() == 0:
            continue

        imgs_c = images[correct_mask]
        labs_c = labels[correct_mask]

        delta = torch.zeros_like(imgs_c).uniform_(-1, 1) * eps_norm
        delta = torch.max(torch.min(imgs_c + delta, upper), lower) - imgs_c

        for _ in range(steps):
            delta.requires_grad_(True)
            loss = F.cross_entropy(model(imgs_c + delta), labs_c)
            loss.backward()
            g = delta.grad.detach()
            delta = delta.detach() + alpha_norm * g.sign()
            delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
            delta = torch.max(torch.min(imgs_c + delta, upper), lower) - imgs_c

        with torch.no_grad():
            pgd_correct += (model(imgs_c + delta).argmax(1) == labs_c).sum().item()

    return 100.0 * pgd_correct / total


# ═══════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Find ImageNette
imagenette_paths = [
    os.path.join(str(DATA_DIR), "imagenette2-320"),
    "/teamspace/studios/this_studio/imagenette2-320",
    "/teamspace/studios/this_studio/data/imagenette2-320",
]

dataset = None
for p in imagenette_paths:
    val_path = os.path.join(p, "val")
    if os.path.exists(val_path):
        raw = torchvision.datasets.ImageFolder(val_path, transform=transform)
        dataset = ImageNetteWrapper(raw)  # FIX: correct labels
        print(f"Found ImageNette at: {p} ({len(dataset)} images)")
        break

if dataset is None:
    print("Downloading ImageNette...")
    dl_dir = str(DATA_DIR)
    os.makedirs(dl_dir, exist_ok=True)
    os.system(f"wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -O /tmp/in.tgz && tar xzf /tmp/in.tgz -C {dl_dir}")
    val_path = os.path.join(dl_dir, "imagenette2-320", "val")
    raw = torchvision.datasets.ImageFolder(val_path, transform=transform)
    dataset = ImageNetteWrapper(raw)

indices = torch.randperm(len(dataset))[:args.n_images].tolist()
subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=2)
print(f"Using {len(subset)} images\n")


# ═══════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════

MODEL_LIST = [
    ("VGG16",           lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT)),
    ("ResNet18",        lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)),
    ("ResNet50",        lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT)),
    ("DenseNet121",     lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)),
    ("MobileNetV2",     lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)),
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)),
    ("ViT-B/16",        lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)),
]


# ═══════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════

results = []

for model_name, model_fn in MODEL_LIST:
    print(f"\n{'='*65}")
    print(f"  {model_name}")
    print(f"{'='*65}")
    t0 = time.time()

    model = model_fn().to(DEVICE)
    model.eval()

    use_hanning = "ViT" in model_name

    # S_model for GSE
    print(f"  Computing S_model (for GSE)...")
    s_model = compute_s_model(
        model, loader, DEVICE, args.n_bins,
        use_hanning=use_hanning,
        max_images=args.n_images,
        desc=model_name,
    )

    # GSE
    gse_r = compute_gse(s_model)

    # SLI
    print(f"  Computing SLI (band Lipschitz)...")
    sli_r = compute_sli(
        model, loader, DEVICE, args.n_bins,
        epsilon=0.01, max_images=min(50, args.n_images),
        desc=f"{model_name} SLI",
    )

    # Attacks (with CORRECT labels now)
    print(f"  Running FGSM...")
    clean_acc, fgsm_acc = fgsm_eval(
        model, loader, args.fgsm_eps,
        IMAGENET_MEAN, IMAGENET_STD, DEVICE
    )

    print(f"  Running PGD-{args.pgd_steps}...")
    pgd_acc = pgd_eval(
        model, loader, args.fgsm_eps, args.pgd_steps,
        IMAGENET_MEAN, IMAGENET_STD, DEVICE
    )

    elapsed = time.time() - t0

    entry = {
        "model": model_name,
        "clean_acc": clean_acc,
        "fgsm_acc": fgsm_acc,
        "pgd_acc": pgd_acc,
        "gse": gse_r["gse"],
        "sli": sli_r["sli"],
        "mean_lip": sli_r["mean_lipschitz"],
        "max_band": sli_r["max_band"],
        "band_lipschitz": sli_r["band_lipschitz"].tolist(),
        "spectral_pdf": gse_r["spectral_pdf"].tolist(),
        "time": elapsed,
    }
    results.append(entry)

    print(f"  Clean: {clean_acc:.1f}% | FGSM: {fgsm_acc:.1f}% | PGD: {pgd_acc:.1f}%")
    print(f"  GSE: {gse_r['gse']:.4f} | SLI: {sli_r['sli']:.4f} | "
          f"MeanLip: {sli_r['mean_lipschitz']:.2f} | "
          f"MaxBand: {sli_r['max_band']}")
    print(f"  ({elapsed:.0f}s)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
#  Results
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 95)
print("FINAL RESULTS")
print("=" * 95)
print(f"{'Model':<18} {'Clean':>7} {'FGSM':>7} {'PGD':>7} "
      f"{'GSE':>7} {'SLI':>7} {'MeanLip':>9} {'MaxBand':>8}")
print("-" * 95)
for r in results:
    print(f"{r['model']:<18} {r['clean_acc']:>6.1f}% {r['fgsm_acc']:>6.1f}% "
          f"{r['pgd_acc']:>6.1f}% {r['gse']:>7.4f} {r['sli']:>7.2f} "
          f"{r['mean_lip']:>9.2f} {r['max_band']:>8d}")


# ═══════════════════════════════════════════════════════════
#  Correlations
# ═══════════════════════════════════════════════════════════

gse_vals = [r["gse"] for r in results]
sli_vals = [r["sli"] for r in results]
lip_vals = [r["mean_lip"] for r in results]
fgsm_vals = [r["fgsm_acc"] for r in results]
pgd_vals = [r["pgd_acc"] for r in results]
clean_vals = [r["clean_acc"] for r in results]

correlations = {}
pairs = [
    ("GSE", gse_vals, "FGSM", fgsm_vals),
    ("GSE", gse_vals, "PGD", pgd_vals),
    ("SLI", sli_vals, "FGSM", fgsm_vals),
    ("SLI", sli_vals, "PGD", pgd_vals),
    ("MeanLip", lip_vals, "FGSM", fgsm_vals),
    ("MeanLip", lip_vals, "PGD", pgd_vals),
    ("GSE", gse_vals, "Clean", clean_vals),
    ("SLI", sli_vals, "Clean", clean_vals),
]

print("\n" + "=" * 95)
print("SPEARMAN CORRELATIONS")
print("=" * 95)

for m_name, m_vals, a_name, a_vals in pairs:
    res = spearmanr(m_vals, a_vals)
    rho = float(res.statistic)
    p = float(res.pvalue)
    key = f"{m_name}_vs_{a_name}"
    correlations[key] = {"rho": rho, "p": p}
    star = " ***" if abs(rho) > 0.7 else (" **" if abs(rho) > 0.5 else "")
    print(f"  {m_name:>8} vs {a_name:<6}:  rho = {rho:+.4f}  (p = {p:.4f}){star}")


# ═══════════════════════════════════════════════════════════
#  Verdict
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 95)
print("METRIC SPREAD")
print("=" * 95)
gse_spread = max(gse_vals) - min(gse_vals)
sli_spread = max(sli_vals) - min(sli_vals)
lip_spread = max(lip_vals) - min(lip_vals)
print(f"  GSE range:     [{min(gse_vals):.4f}, {max(gse_vals):.4f}]  spread = {gse_spread:.4f}")
print(f"  SLI range:     [{min(sli_vals):.4f}, {max(sli_vals):.4f}]  spread = {sli_spread:.4f}")
print(f"  MeanLip range: [{min(lip_vals):.2f}, {max(lip_vals):.2f}]  spread = {lip_spread:.2f}")
print(f"  (PRI had spread = 0.0273 — these must beat that)")

print("\n" + "=" * 95)
# Check best correlation
best_metric = None
best_rho = 0
for key, val in correlations.items():
    if "PGD" in key and abs(val["rho"]) > abs(best_rho):
        best_rho = val["rho"]
        best_metric = key.split("_vs_")[0]

if abs(best_rho) > 0.5:
    print(f"VERDICT: {best_metric} HAS SIGNAL (|rho| = {abs(best_rho):.4f} with PGD accuracy)")
    if gse_spread > 0.05 or sli_spread > 1.0:
        print(f"  Spread is meaningful. Metrics can differentiate architectures.")
    else:
        print(f"  WARNING: Spread is still narrow. Need more diverse models.")
else:
    print(f"VERDICT: METRICS NEED WORK (best |rho| = {abs(best_rho):.4f})")
    if gse_spread > 0.05 or sli_spread > 1.0:
        print(f"  Spread is OK but correlation is weak. Metrics measure something, but not robustness.")
    else:
        print(f"  Both spread and correlation are weak. Consider fundamentally different approach.")
print("=" * 95)


# ═══════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════

save_dir = os.path.join(str(RESULTS_DIR), "evaluation")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "e1_v2_gse_sli.json")

output = {
    "config": vars(args),
    "results": results,
    "correlations": correlations,
    "spreads": {
        "gse": gse_spread,
        "sli": sli_spread,
        "mean_lip": lip_spread,
    },
}
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to: {save_path}")
