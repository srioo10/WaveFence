# WaveFence - Spectral Robustness Toolkit

WaveFence is a research-oriented toolkit for analyzing and improving adversarial robustness in computer vision models using frequency-domain sensitivity analysis. The project focuses on developing attack-free robustness diagnostics and evaluating how spectral characteristics correlate with adversarial vulnerability in CIFAR-10 and ImageNette models.

The repository includes:

* spectral robustness metrics (GSE, SLI, MeanLip),
* adversarial training/evaluation pipelines,
* the PCSD denoising framework,
* and RobuScan, a Gradio-based robustness analysis interface.

This repository contains a research prototype for diagnosing and improving adversarial robustness in vision models using frequency-domain sensitivity analysis. The project includes:

- attack-free spectral diagnostics such as GSE, SLI/MeanLip, and earlier PRI/VCI experiments;
- CIFAR-10 ResNet18 training and evaluation scripts for clean, PGD-AT, LAT, CAAT-style, and denoiser-based defenses;
- PCSD, a profile-conditioned spectral denoiser built around FiLM-style conditioning;
- RobuScan, a Gradio interface that scans a CIFAR-10 ResNet18 checkpoint and returns a spectral robustness report;
- saved JSON summaries and figures for the completed experimental runs;
- a final report in `reports/dip_combined_report.pdf`.

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

## Repository Status

This project was originally developed and executed within the Lightning AI environment. As a result, certain implementation patterns, execution flow, and syntax conventions may differ from standard standalone Python projects or local development setups.

Some configuration files and runtime assumptions were designed specifically around the original Lightning AI workspace structure.

Before running locally, users may need to:

Update hardcoded paths in caat/config.py
Modify environment-specific directory references
Adjust dataset/model paths for their local system

In particular, the current configuration references:

/teamspace/studios/this_studio

which is specific to the original Lightning AI runtime environment and should be replaced with appropriate local paths.


## Limitations

- Some scripts are tied to local/Lightning paths through `caat/config.py`.
- Large datasets and weights are not included in Git and must be downloaded or generated.
- Result files mix an earlier PRI/VCI line of work with a later GSE/MeanLip line; the final report separates them.
- use `reports/dip_combined_report.pdf` for any furthur clarifications.

## Author

Sooraj S
