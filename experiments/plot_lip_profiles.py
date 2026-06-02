"""
plot_lip_profiles.py — Paper-ready Lipschitz profile comparison figure.

Loads eval JSON files and plots per-band Lipschitz profiles for all models.

Usage (on Lightning):
  python experiments/plot_lip_profiles.py

Output:
  figures/lipschitz_profiles.png  (for paper)
  figures/lipschitz_profiles.pdf  (vector, for LaTeX)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load eval JSONs ───────────────────────────────────────
RESULTS_DIR = "results/evaluation"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hardcoded from evaluate.py output (or load from JSON)
models = {
    "Clean": {
        "band_lip": [0.171, 0.200, 0.197, 0.239, 0.195, 0.193,
                     0.216, 0.252, 0.260, 0.272, 0.298, 0.344,
                     0.285, 0.268, 0.248, 0.000],
        "mean_lip": 0.2273,
        "pgd_acc": 0.00,
        "color": "#e74c3c",
        "ls": "--",
        "marker": "s",
    },
    "PGD-AT": {
        "band_lip": [0.075, 0.058, 0.054, 0.048, 0.043, 0.038,
                     0.035, 0.031, 0.031, 0.031, 0.029, 0.029,
                     0.026, 0.027, 0.027, 0.000],
        "mean_lip": 0.0365,
        "pgd_acc": 48.92,
        "color": "#3498db",
        "ls": "-.",
        "marker": "^",
    },
    "LAT+ACL (Ours)": {
        "band_lip": [0.061, 0.050, 0.046, 0.042, 0.038, 0.034,
                     0.033, 0.030, 0.028, 0.028, 0.026, 0.023,
                     0.025, 0.024, 0.023, 0.000],
        "mean_lip": 0.0320,
        "pgd_acc": 48.58,
        "color": "#2ecc71",
        "ls": "-",
        "marker": "o",
    },
}

# Try loading from JSON if available
for fname, name in [("eval_clean.json", "Clean"),
                    ("eval_pgd-at.json", "PGD-AT"),
                    ("eval_lat.json", "LAT+ACL (Ours)")]:
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if "band_lipschitz" in data:
            models[name]["band_lip"] = data["band_lipschitz"]
            models[name]["mean_lip"] = data["mean_lip"]
            models[name]["pgd_acc"] = data.get("pgd_acc", models[name]["pgd_acc"])
            print(f"  Loaded {fname}")

# ── Figure setup ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#f8f9fa")

bands = np.arange(15)  # Exclude band 15 (DC, always 0)
band_labels = [str(b) for b in bands]

# ── LEFT: Per-band Lipschitz profiles ────────────────────
ax = axes[0]
ax.set_facecolor("#ffffff")

for name, info in models.items():
    vals = np.array(info["band_lip"][:15])
    ax.plot(bands, vals,
            color=info["color"],
            linestyle=info["ls"],
            marker=info["marker"],
            markersize=6,
            linewidth=2.0,
            label=f"{name}  (MeanLip={info['mean_lip']:.4f})",
            zorder=3)

# Shade the gap between PGD-AT and LAT (our improvement)
pgd_vals = np.array(models["PGD-AT"]["band_lip"][:15])
lat_vals = np.array(models["LAT+ACL (Ours)"]["band_lip"][:15])
ax.fill_between(bands, lat_vals, pgd_vals, alpha=0.15,
                color="#2ecc71", label="LAT reduction vs PGD-AT")

ax.set_xlabel("Frequency Band (0=low, 14=high)", fontweight="bold")
ax.set_ylabel("Local Lipschitz Constant", fontweight="bold")
ax.set_title("Per-Band Lipschitz Profile\n(lower = more robust)", fontweight="bold")
ax.set_xticks(bands)
ax.set_xticklabels(band_labels, fontsize=9)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_ylim(bottom=0)

# Annotate MeanLip for AT models
ax.annotate("LAT lower in ALL bands ✓",
            xy=(7, 0.030), xytext=(4, 0.060),
            fontsize=9, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))

# ── RIGHT: MeanLip vs PGD Robustness bar chart ───────────
ax2 = axes[1]
ax2.set_facecolor("#ffffff")

model_names = list(models.keys())
mean_lips = [models[n]["mean_lip"] for n in model_names]
pgd_accs = [models[n]["pgd_acc"] for n in model_names]
colors = [models[n]["color"] for n in model_names]

x = np.arange(len(model_names))
w = 0.35

bars1 = ax2.bar(x - w/2, mean_lips, w,
                color=colors, alpha=0.85,
                label="MeanLip (left axis, ↓ better)",
                edgecolor="white", linewidth=1.5)

ax2.set_ylabel("MeanLip (lower = more robust)", fontweight="bold", color="#333")
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontweight="bold")
ax2.set_title("MeanLip vs PGD-10 Robustness\n(our metric vs actual attack accuracy)",
              fontweight="bold")

# Second y-axis for PGD accuracy
ax2r = ax2.twinx()
bars2 = ax2r.bar(x + w/2, pgd_accs, w,
                 color=colors, alpha=0.45,
                 hatch="///", edgecolor=colors, linewidth=1.5,
                 label="PGD-10 Acc % (right axis, ↑ better)")
ax2r.set_ylabel("PGD-10 Accuracy (%)", fontweight="bold", color="#555")
ax2r.set_ylim(0, 80)

# Value labels
for bar, val in zip(bars1, mean_lips):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

for bar, val in zip(bars2, pgd_accs):
    ax2r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Combined legend
patch1 = mpatches.Patch(color="gray", alpha=0.85, label="MeanLip ↓")
patch2 = mpatches.Patch(color="gray", alpha=0.45, hatch="///", label="PGD-10 Acc ↑")
ax2.legend(handles=[patch1, patch2], loc="upper left", framealpha=0.9)

ax2.grid(True, axis="y", alpha=0.3, linestyle=":")

# ── Final layout ──────────────────────────────────────────
fig.suptitle(
    "LAT+ACL reduces spectral sensitivity across ALL frequency bands\n"
    "MeanLip: Clean=0.2273 → PGD-AT=0.0365 → LAT+ACL=0.0320 (−12.3%)",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()

out_png = os.path.join(FIGURES_DIR, "lipschitz_profiles.png")
out_pdf = os.path.join(FIGURES_DIR, "lipschitz_profiles.pdf")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())

print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
