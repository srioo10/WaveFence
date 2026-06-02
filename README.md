# Spectral Robustness Toolkit

This repository contains a research prototype for diagnosing and improving adversarial robustness in vision models using frequency-domain sensitivity analysis. The project includes:

- attack-free spectral diagnostics such as GSE, SLI/MeanLip, and earlier PRI/VCI experiments;
- CIFAR-10 ResNet18 training and evaluation scripts for clean, PGD-AT, LAT, CAAT-style, and denoiser-based defenses;
- PCSD, a profile-conditioned spectral denoiser built around FiLM-style conditioning;
- RobuScan, a Gradio interface that scans a CIFAR-10 ResNet18 checkpoint and returns a spectral robustness report;
- saved JSON summaries and figures for the completed experimental runs;
- a final DIP report in `reports/dip_combined_report.pdf`.

## Repository Status

This is worth uploading to GitHub as a research/prototype repository. The idea is interesting, the code is modular enough to explain the work, and the saved result JSONs make the project inspectable without forcing people to rerun every experiment.

Before making it a polished public release, the main improvement would be to remove hardcoded local paths in `caat/config.py`. It currently points at `/teamspace/studios/this_studio`, which is fine for the original run environment but not portable for every user.

## What To Upload

Upload these:

- `caat/` - model definitions, spectral metrics, attacks, training, and evaluation.
- `pcsd/` - PCSD and DnCNN denoiser code.
- `robuscan/` - scanner backend and Gradio app.
- `experiments/` - reproducible experiment scripts.
- `results/evaluation/*.json` - lightweight saved metrics and tables.
- `figures/` - final plots used to explain the work.
- `reports/dip_combined_report.pdf` - final submitted project report.
- `reports/dip_combined_report.docx` - editable report source, optional but useful.
- `spectral_vulnerability_map.py` - standalone spectral vulnerability script.
- `implementation_plan.md` - original project blueprint.
- `README.md`, `requirements.txt`, `.gitignore`, and `FINAL_REPORT.md`.

Do not upload these:

- `.venv/`, `.codex/`, `.gradio/`, `.qodo/`, `.vscode/`
- `data/`, `imagenette2-320/`, `project_data/`
- `checkpoints/` and all `*.pth` model weights
- `artifacts/paper_ieee/`
- `__pycache__/` folders

The folder `artifacts/paper_ieee` appears to contain a generated paper about a different JTrans/Mamba binary-code project, so it should not be used as this repository's final report unless you intentionally want to keep unrelated artifacts. The correct final report is in `reports/`.

## Project Layout

```text
.
|-- caat/                         # Classifiers, attacks, spectral metrics, training/eval
|-- pcsd/                         # Profile-conditioned spectral denoiser
|-- robuscan/                     # Gradio scanner app
|-- experiments/                  # Experiment runners and plotting scripts
|-- results/evaluation/           # Saved evaluation JSONs
|-- figures/                      # Generated plots
|-- reports/                      # Final DIP report PDF and editable DOCX
|-- spectral_vulnerability_map.py # Standalone spectral vulnerability demo
|-- implementation_plan.md        # Research plan and method notes
|-- FINAL_REPORT.md               # Final project summary
`-- requirements.txt
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source .venv/bin/activate
```

PyTorch installation can vary by CUDA version. If GPU support matters, install the correct PyTorch build from the official PyTorch selector, then install the remaining packages from `requirements.txt`.

## Running The Main Tools

Evaluate a CIFAR-10 checkpoint:

```bash
python -m caat.evaluate --checkpoint checkpoints/resnet18_cifar10_pgd_at.pth --name PGD-AT
```

Run the RobuScan web app:

```bash
python -m robuscan.app
```

Train the PCSD denoiser:

```bash
python -m pcsd.train_pcsd --model pcsd --epochs 20
```

Run the cross-architecture GSE/SLI experiment:

```bash
python -m experiments.e1_v2_gse_sli --n_images 200
```

## Results Snapshot

Saved results show that MeanLip had the strongest signal against PGD robustness in the ImageNette cross-model experiment, with Spearman |rho| = 0.788 and p = 0.035. On CIFAR-10, clean training reached 92.12% clean accuracy but 0.00% PGD accuracy, while PGD-AT reached 77.97% clean accuracy and 48.25% PGD-20 accuracy. LAT+ACL had similar robustness with lower MeanLip.

The final report frames PCSD+DualDGSC against the standard blind DnCNN pixel-loss baseline, where it recovers +35.3 percentage points versus +17.7. The deeper ablation also shows that DnCNN with classifier-guided loss is a very strong fair baseline, so the most honest takeaway is that classifier-guided loss is the main driver and DGSC gives a smaller incremental gain.

## Limitations

- Some scripts are tied to local/Lightning paths through `caat/config.py`.
- Large datasets and weights are not included in Git and must be downloaded or generated.
- Result files mix an earlier PRI/VCI line of work with a later GSE/MeanLip line; the final report separates them.
- The generated IEEE paper artifact currently appears unrelated to this spectral robustness project; use `reports/dip_combined_report.pdf` instead.

## Author

Sooraj S, IIITDM Kancheepuram
