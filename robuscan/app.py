"""
app.py — RobuScan v2 Gradio Web Interface.

Upload a PyTorch CIFAR-10 ResNet18 checkpoint → get:
  - GSE (Gradient Spectral Entropy)
  - MeanLip (Spectral Lipschitz Sensitivity)
  - Per-band Lipschitz profile (bar chart)
  - Letter grade (A+ through F)
  - Actionable recommendations

No adversarial attacks needed. Results in ~30 seconds.

Usage:
  python -m robuscan.app
"""

import sys
import os
import io
import json
import tempfile

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caat.config import (
    DEVICE, N_BINS_SMALL,
    CIFAR10_MEAN, CIFAR10_STD, DATA_DIR,
)
from caat.models import ResNet18_CIFAR
from robuscan.scanner import scan_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    print("Install gradio: pip install gradio")
    sys.exit(1)


# ── Preload dataset ──────────────────────────────────────────────────────────
print("Loading CIFAR-10 test set...")
transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
test_data = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True, transform=transform,
)
subset = Subset(test_data, list(range(200)))
scan_loader = DataLoader(subset, batch_size=16, shuffle=False)
print(f"  Ready ({len(subset)} images)")


# ── Reference baselines (from our paper) ─────────────────────────────────────
BASELINES = {
    "Clean (undefended)": {"mean_lip": 0.2273, "gse": 0.799, "grade": "F"},
    "PGD-AT (baseline)":  {"mean_lip": 0.0365, "gse": 0.799, "grade": "A"},
    "LAT+ACL (ours)":     {"mean_lip": 0.0320, "gse": 0.802, "grade": "A+"},
}


# ── Plot generation ───────────────────────────────────────────────────────────
def create_lipschitz_plot(result, model_name):
    """Generate a 2-panel figure: per-band Lipschitz profile + grade gauge."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f9fa")

    bands     = np.arange(len(result["band_lipschitz"]) - 1)  # exclude DC band 15
    band_vals = np.array(result["band_lipschitz"][:-1])
    max_band  = int(band_vals.argmax())  # recalculate on truncated array

    # ── Left: Per-band Lipschitz bar chart ───────────────
    ax = axes[0]
    ax.set_facecolor("#ffffff")

    colors = []
    for i, v in enumerate(band_vals):
        if i == max_band:
            colors.append("#ef4444")      # red — most vulnerable
        elif v > band_vals.mean() + band_vals.std():
            colors.append("#f97316")      # orange — vulnerable
        else:
            colors.append("#6366f1")      # indigo — normal

    bars = ax.bar(bands, band_vals, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.2)

    # Add baseline lines
    ax.axhline(0.0365, color="#3b82f6", linestyle="--", linewidth=1.5,
               alpha=0.7, label="PGD-AT baseline (0.0365)")
    ax.axhline(0.0320, color="#22c55e", linestyle="--", linewidth=1.5,
               alpha=0.7, label="LAT+ACL ours (0.0320)")
    ax.axhline(result["mean_lip"], color="#f59e0b", linestyle="-", linewidth=2,
               label=f"This model MeanLip ({result['mean_lip']:.4f})")

    ax.set_xlabel("Frequency Band (0=low, 14=high)", fontweight="bold")
    ax.set_ylabel("Local Lipschitz Constant", fontweight="bold")
    ax.set_title(f"Per-Band Lipschitz Profile\n{model_name}", fontweight="bold")
    ax.set_xticks(bands)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)

    # Annotate most vulnerable band
    ax.annotate(f"Most\nvulnerable\n(Band {max_band})",
                xy=(max_band, band_vals[max_band]),
                xytext=(max_band + 2, band_vals[max_band] + 0.005),
                fontsize=8, color="#ef4444",
                arrowprops=dict(arrowstyle="->", color="#ef4444"))

    # ── Right: Comparison bar chart ───────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")

    names = list(BASELINES.keys()) + [f"{model_name} (yours)"]
    values = [b["mean_lip"] for b in BASELINES.values()] + [result["mean_lip"]]
    bar_colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b"]

    hbars = ax2.barh(names, values, color=bar_colors, alpha=0.85,
                     edgecolor="white", linewidth=1.2)

    for bar, val in zip(hbars, values):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("MeanLip (lower = more robust)", fontweight="bold")
    ax2.set_title("MeanLip Comparison\nvs. Known Baselines", fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax2.invert_yaxis()

    grade_colors = {"A+": "#22c55e", "A": "#22c55e", "B": "#84cc16",
                    "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}
    grade = result["grade"]
    gc = grade_colors.get(grade, "#6366f1")

    ax2.text(0.98, 0.02, f"Grade: {grade}",
             transform=ax2.transAxes, fontsize=24, fontweight="bold",
             color=gc, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=gc, linewidth=2))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(buf.read())
    tmp.close()
    return tmp.name


# ── Scan function ─────────────────────────────────────────────────────────────
def run_scan(checkpoint_file, model_name):
    """Main scan function called by Gradio."""
    if checkpoint_file is None:
        return "⚠️ Please upload a model checkpoint (.pth)", None, ""

    if not model_name or not model_name.strip():
        model_name = "Uploaded Model"

    try:
        # Load model
        model = ResNet18_CIFAR(num_classes=10).to(DEVICE)
        ckpt_path = checkpoint_file if isinstance(checkpoint_file, str) else checkpoint_file.name
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        sample_key = next(iter(sd.keys()))
        if not sample_key.startswith("model."):
            sd = {f"model.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()

        # Run scan
        result = scan_model(
            model, scan_loader, DEVICE,
            n_bins=N_BINS_SMALL,
            max_images=200,
        )

        # Grade styling
        grade_emoji = {
            "A+": "🟢", "A": "🟢", "B": "🟡",
            "C": "🟠", "D": "🟠", "F": "🔴",
        }
        emoji = grade_emoji.get(result["grade"], "⚪")

        # Comparison vs baselines
        pgd_at_lip = 0.0365
        lat_lip    = 0.0320
        mean_lip   = result["mean_lip"]

        if mean_lip <= lat_lip:
            comparison = f"✅ **Better than LAT+ACL** ({mean_lip:.4f} < {lat_lip:.4f})"
        elif mean_lip <= pgd_at_lip:
            comparison = f"✅ **Between LAT+ACL and PGD-AT** ({mean_lip:.4f})"
        elif mean_lip <= 0.1:
            comparison = f"⚠️ **Worse than PGD-AT** — consider adversarial training"
        else:
            comparison = f"🔴 **Undefended territory** — adversarial training strongly recommended"

        report = f"""
## {emoji} Robustness Grade: **{result['grade']}** ({result.get('zone', '')})

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **GSE** (Gradient Spectral Entropy) | **{result['gse']:.4f}** | {'Concentrated ← attack target' if result['gse'] < 0.80 else 'Spread ← diverse sensitivity'} |
| **MeanLip** (Mean Lipschitz Sensitivity) | **{result['mean_lip']:.4f}** | Lower = more robust |
| **Most Vulnerable Band** | **Band {result['max_band']}** | {'Low frequency (common target)' if result['max_band'] <= 2 else 'Mid/high frequency'} |

### Compared to Baselines
{comparison}

| Baseline | MeanLip | Grade |
|----------|---------|-------|
| Clean (undefended) | 0.2273 | F |
| PGD-AT | 0.0365 | A |
| LAT+ACL (ours) | 0.0320 | A+ |
| **{model_name}** | **{mean_lip:.4f}** | **{result['grade']}** |

### Recommendation
{result.get('recommendation', 'N/A')}

### What do these metrics mean?
- **GSE** measures how spread vs concentrated the gradient spectrum is. Lower GSE = the model is sensitive to a narrow frequency range = easier to attack.
- **MeanLip** measures average Lipschitz constant across frequency bands. Lower = smoother = more robust to perturbations.
- **MeanLip was validated** with Spearman ρ = −0.788 (p=0.035) against PGD-10 robustness across multiple architectures.
"""

        # Generate plot
        plot_path = create_lipschitz_plot(result, model_name)

        # JSON details
        details = json.dumps({
            "model_name": model_name,
            "gse": round(result["gse"], 4),
            "mean_lip": round(result["mean_lip"], 4),
            "max_band": result["max_band"],
            "grade": result["grade"],
            "zone": result.get("zone", ""),
            "band_lipschitz": [round(v, 4) for v in result["band_lipschitz"]],
        }, indent=2)

        return report, plot_path, details

    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```", None, ""


# ── Gradio Interface ──────────────────────────────────────────────────────────
def build_app():
    with gr.Blocks(
        title="RobuScan v2 — Spectral Robustness Scanner",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald"),
    ) as app:
        gr.Markdown("""
        # 🔬 RobuScan v2 — Spectral Robustness Scanner

        **Upload a CIFAR-10 ResNet18 checkpoint → Get a spectral robustness report in ~30s.**

        No adversarial attacks needed. RobuScan uses two validated diagnostic metrics:
        - **GSE** (Gradient Spectral Entropy) — architectural fingerprint
        - **MeanLip** (Mean Lipschitz Sensitivity) — primary robustness predictor (ρ=−0.788, p=0.035)

        > Works with ResNet18 CIFAR-10 models (.pth checkpoints)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                model_file = gr.File(
                    label="Upload Model Checkpoint (.pth)",
                    file_types=[".pth", ".pt"],
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., MyModel-v1",
                    value="My Model",
                )
                scan_btn = gr.Button(
                    "🔬 Scan Model",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                report_md = gr.Markdown(label="Scan Report")
                spectral_plot = gr.Image(label="Lipschitz Profile Analysis")
                details_json = gr.Code(
                    label="Detailed Results (JSON)",
                    language="json",
                )

        scan_btn.click(
            fn=run_scan,
            inputs=[model_file, model_name],
            outputs=[report_md, spectral_plot, details_json],
        )

        gr.Markdown("""
        ---
        **RobuScan v2** — Part of the *Spectral Adversarial Robustness* framework.

        | Metric | Range | Interpretation |
        |--------|-------|----------------|
        | GSE | 0–1 | Higher = more spread gradient (better) |
        | MeanLip | >0 | Lower = smoother model (more robust) |
        | Grade A+ | MeanLip < 0.032 | Exceeds LAT+ACL baseline |
        | Grade A | MeanLip < 0.040 | Matches PGD-AT level |
        | Grade B | MeanLip < 0.080 | Moderate robustness |
        | Grade F | MeanLip > 0.200 | Undefended |

        *Validated on CIFAR-10 ResNet18 — Spearman ρ=−0.788, p=0.035 vs PGD-10 accuracy*
        """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
