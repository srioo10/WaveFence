"""
e5_qualitative.py — The "Money" Figure: Clean → Adversarial → DGSC → Denoised.

Generates a paper-ready 3×4 grid:
  Row 1: Images     (clean, adversarial, DGSC heatmap, denoised)
  Row 2: FFT spectra (shows attack frequencies, then erasure)
  Row 3: DGSC profile bar chart (proves it found the right bands)

Usage:
  python -m experiments.e5_qualitative
  python -m experiments.e5_qualitative --n_examples 4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from caat.config import (
    DEVICE, SEED, CIFAR10_MEAN, CIFAR10_STD,
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, N_BINS_SMALL,
)
from caat.models import ResNet18_CIFAR, get_cifar10_transforms
from caat.spectral_lipschitz import compute_sli
from pcsd.unet import PCSD

parser = argparse.ArgumentParser()
parser.add_argument("--n_examples", type=int, default=4)
parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
parser.add_argument("--pgd_steps", type=int, default=10)
args, _ = parser.parse_known_args()

torch.manual_seed(SEED)

SAVE_DIR   = str(CHECKPOINT_DIR)
FIG_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

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

# ── Load PCSD ─────────────────────────────────────────────
pcsd = PCSD(in_channels=3, profile_dim=N_BINS_SMALL, base_ch=32).to(DEVICE)
p = os.path.join(SAVE_DIR, "pcsd_cifar10.pth")
if os.path.exists(p):
    pcsd.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    print("PCSD: pcsd_cifar10.pth")
pcsd.eval()

# ── Data ──────────────────────────────────────────────────
test_data = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True,
    transform=get_cifar10_transforms(train=False),
)
loader = DataLoader(test_data, batch_size=args.n_examples * 4,
                    shuffle=False, num_workers=2)

# ── Static profile ────────────────────────────────────────
sli = compute_sli(
    classifier,
    DataLoader(test_data, batch_size=16),
    DEVICE, N_BINS_SMALL,
    epsilon=0.01, max_images=200, desc="Profile",
)
static_profile = torch.tensor(
    sli["lipschitz_profile"], dtype=torch.float32, device=DEVICE
)


# ── PGD attack ────────────────────────────────────────────
def pgd(images, labels, eps, steps):
    mean_t = torch.tensor(CIFAR10_MEAN, device=DEVICE).view(1, -1, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD,  device=DEVICE).view(1, -1, 1, 1)
    en = eps / std_t
    al = (eps / 4) / std_t
    lo = (0.0 - mean_t) / std_t
    hi = (1.0 - mean_t) / std_t

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


# ── DGSC ─────────────────────────────────────────────────
def compute_dgsc(adv_images):
    n_bins = N_BINS_SMALL
    B, C, H, W = adv_images.shape
    device = adv_images.device
    x = adv_images.detach().clone().requires_grad_(True)
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
    bin_idx = (r / float((cx**2 + cy**2)**0.5) * (n_bins - 1)).clamp(0, n_bins-1).long()

    spectrum = torch.zeros(B, n_bins, device=device)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() > 0:
            spectrum[:, b] = G[:, mask].mean(1)
    spectrum = torch.nan_to_num(spectrum)
    mx = spectrum.max(1, keepdim=True).values.clamp(1e-8)
    return (spectrum / mx).detach(), G.detach()   # profile + raw spatial gradient map


# ── Image helpers ─────────────────────────────────────────
def denorm(t):
    """De-normalize CIFAR-10 tensor to [0,1] for display."""
    mean = torch.tensor(CIFAR10_MEAN, device=t.device).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=t.device).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def to_fft_log(img_tensor):
    """Per-channel FFT magnitude (log scale), averaged over channels."""
    G = torch.fft.fftshift(torch.fft.fft2(img_tensor), dim=(-2, -1)).abs()
    G = G.mean(0)  # avg over C
    return torch.log1p(G).cpu().numpy()


def get_pred(x):
    with torch.no_grad():
        return CIFAR10_CLASSES[classifier(x.unsqueeze(0)).argmax(1).item()]


# ── Collect examples ─────────────────────────────────────
print("\nGenerating examples...")
examples = []
for images, labels in loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Find correctly classified clean images
    with torch.no_grad():
        clean_preds = classifier(images).argmax(1)
    correct = (clean_preds == labels).nonzero(as_tuple=True)[0]
    if len(correct) == 0:
        continue

    for idx in correct[:args.n_examples]:
        img   = images[idx:idx+1]
        label = labels[idx:idx+1]

        adv = pgd(img, label, args.epsilon, args.pgd_steps)
        profile, grad_map = compute_dgsc(adv)

        with torch.no_grad():
            denoised = pcsd(adv, profile)

        examples.append({
            "clean":    img.squeeze(0).cpu(),
            "adv":      adv.squeeze(0).cpu(),
            "denoised": denoised.squeeze(0).cpu(),
            "profile":  profile.squeeze(0).cpu().numpy(),
            "grad_map": grad_map.squeeze(0).cpu().numpy(),
            "label":    CIFAR10_CLASSES[label.item()],
            "pred_clean":    get_pred(img.squeeze(0)),
            "pred_adv":      get_pred(adv.squeeze(0)),
            "pred_denoised": get_pred(denoised.squeeze(0)),
        })

    if len(examples) >= args.n_examples:
        break

print(f"  Got {len(examples)} examples")


# ── Figure ────────────────────────────────────────────────
N = len(examples)
COLS = 4  # Clean | Adversarial | DGSC profile | Denoised
ROWS = N * 2  # image row + FFT row per example

fig = plt.figure(figsize=(COLS * 3.5, ROWS * 3.0))
fig.patch.set_facecolor("#0f0f0f")

COL_TITLES = ["Clean Image", "Adversarial\n(PGD-10)", "DGSC Profile\n(detected bands)", "Denoised\n(PCSD+DGSC)"]
COL_COLORS = ["#22c55e", "#ef4444", "#f59e0b", "#6366f1"]

outer = gridspec.GridSpec(N, 1, figure=fig, hspace=0.05)

for ex_idx, ex in enumerate(examples):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, COLS, subplot_spec=outer[ex_idx], hspace=0.05, wspace=0.1
    )

    clean    = denorm(ex["clean"])
    adv_img  = denorm(ex["adv"])
    denoised = denorm(ex["denoised"])

    images_row = [clean, adv_img, None, denoised]
    ffts_row   = [
        to_fft_log(ex["clean"]),
        to_fft_log(ex["adv"]),
        None,
        to_fft_log(ex["denoised"]),
    ]

    for col in range(COLS):
        # Row 0: images
        ax0 = fig.add_subplot(inner[0, col])
        ax0.set_facecolor("#0f0f0f")

        if col != 2:
            ax0.imshow(images_row[col].permute(1, 2, 0).numpy())
        else:
            # DGSC profile as colored bar chart
            profile = ex["profile"][:N_BINS_SMALL - 1]  # drop DC
            bins    = np.arange(len(profile))
            colors  = plt.cm.hot(profile / (profile.max() + 1e-8))
            ax0.bar(bins, profile, color=colors, edgecolor="none")
            ax0.set_facecolor("#1a1a2e")
            ax0.set_yticks([])
            ax0.set_xlabel("Freq band →", color="white", fontsize=7)
            ax0.tick_params(colors="white", labelsize=6)
            for spine in ax0.spines.values():
                spine.set_edgecolor("#333")

        ax0.set_xticks([]); ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_edgecolor(COL_COLORS[col])
            spine.set_linewidth(2)

        if ex_idx == 0:
            ax0.set_title(COL_TITLES[col], color=COL_COLORS[col],
                          fontsize=10, fontweight="bold", pad=4)

        # Prediction labels
        preds = [
            f"✓ {ex['label']}", f"✗ {ex['pred_adv']}",
            "Gradient\nSpectrum", f"{'✓' if ex['pred_denoised'] == ex['label'] else '✗'} {ex['pred_denoised']}"
        ]
        pred_color = [
            "#22c55e", "#ef4444" if ex["pred_adv"] != ex["label"] else "#22c55e",
            "#f59e0b",
            "#22c55e" if ex["pred_denoised"] == ex["label"] else "#ef4444"
        ]
        ax0.set_xlabel(preds[col], color=pred_color[col], fontsize=8, labelpad=2)

        # Row 1: FFT spectra
        ax1 = fig.add_subplot(inner[1, col])
        ax1.set_facecolor("#0f0f0f")

        if col != 2:
            ax1.imshow(ffts_row[col], cmap="inferno", interpolation="nearest")
            ax1.set_title("FFT spectrum", color="#888", fontsize=7, pad=2)
        else:
            # Difference image (adversarial - clean) amplified
            diff = (adv_img - clean).abs().permute(1, 2, 0).numpy()
            diff = diff / (diff.max() + 1e-8)
            ax1.imshow(diff, cmap="hot")
            ax1.set_title("Perturbation (×)", color="#888", fontsize=7, pad=2)

        ax1.set_xticks([]); ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_edgecolor("#333")

plt.suptitle(
    "PCSD+DGSC: Dynamic Gradient Spectrum Conditioning\n"
    "The DGSC profile (col 3) identifies which frequency bands each adversarial example exploits.\n"
    "PCSD uses this to perform targeted denoising, recovering correct classification.",
    color="white", fontsize=10, y=1.01, va="bottom"
)

out = os.path.join(FIG_DIR, "fig_qualitative_dgsc.pdf")
fig.savefig(out, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print(f"\nFigure saved:")
print(f"  PDF: {out}")
print(f"  PNG: {out_png}")
