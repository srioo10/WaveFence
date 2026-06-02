# Final Report: Spectral Robustness Toolkit

## 1. Project Overview

This project studies adversarial robustness through the frequency behavior of vision models. Instead of relying only on adversarial attacks, it analyzes how model gradients and local Lipschitz sensitivity are distributed across spatial frequency bands. The repository contains diagnostic metrics, training/evaluation scripts, an inference-time denoiser, and a Gradio-based scanner called RobuScan.

The practical goal is a workflow where a user can train or upload a CIFAR-10 ResNet18 checkpoint, scan its spectral sensitivity, estimate robustness risk, and compare it against saved baselines.

The submitted DIP report is available at `reports/dip_combined_report.pdf`, with the editable Word source at `reports/dip_combined_report.docx`.

## 2. Main Components

### Spectral diagnostics

The project includes two generations of metrics:

- PRI/VCI/PGS: human-vision-inspired metrics based on a contrast sensitivity function and model gradient sensitivity.
- GSE/SLI/MeanLip: attack-free spectral metrics based on gradient spectral entropy and band-limited Lipschitz estimates.

The later RobuScan/report pipeline uses GSE and MeanLip because the saved experiments showed a stronger relationship between MeanLip and PGD robustness.

### Training and evaluation

The `caat` package contains:

- CIFAR-adapted ResNet18 and small-image model helpers;
- FGSM/PGD attack code;
- clean, PGD-AT, LAT, and CAAT-related training/evaluation scripts;
- spectral profile computation and robustness grading utilities.

### PCSD denoiser

The `pcsd` package implements a profile-conditioned spectral denoiser using FiLM-style conditioning. It can run with dynamic gradient spectrum conditioning (DGSC), static profiles, classifier-guided loss, or DnCNN-style baselines.

### RobuScan

The `robuscan` package provides a Gradio app for uploading a CIFAR-10 ResNet18 checkpoint and generating a spectral robustness report with:

- GSE;
- MeanLip;
- most sensitive frequency band;
- robustness grade;
- baseline comparison plot.

## 3. Experimental Results

### CIFAR-10 robustness

Saved evaluations show the expected clean-versus-robust tradeoff:

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy | MeanLip |
|---|---:|---:|---:|---:|
| Clean ResNet18 | 92.12% | 8.87% | 0.00% PGD-10 | 0.2273 |
| PGD-AT | 77.97% | 52.45% | 48.25% PGD-20 | 0.0365 |
| LAT | 77.19% | 52.06% | 47.81% PGD-20 | 0.0320 |

The clean model has high standard accuracy but collapses under PGD. PGD-AT and LAT reduce clean accuracy but produce much lower MeanLip and much higher adversarial accuracy.

### Cross-architecture metric validation

The ImageNette experiment compared seven pretrained models. MeanLip showed the strongest relationship with PGD robustness:

| Correlation | Spearman rho | p-value |
|---|---:|---:|
| MeanLip vs PGD | -0.7881 | 0.0353 |
| MeanLip vs FGSM | -0.7456 | 0.0544 |
| GSE vs PGD | -0.5911 | 0.1622 |

This supports MeanLip as the most useful saved diagnostic signal in the current version of the project.

### Denoiser results

The final report presents PCSD+DualDGSC against the standard blind DnCNN pixel-loss baseline, where PCSD+DualDGSC gives a much larger recovery. The deeper ablation also shows that classifier-guided loss is the biggest driver:

| Variant | Best Recovery |
|---|---:|
| Blind DnCNN + pixel loss | +17.7 |
| PCSD + static profile + pixel loss | +12.74 |
| PCSD + DGSC + pixel loss | +12.69 |
| PCSD + static profile + classifier loss | +29.25 |
| PCSD + DGSC + classifier loss | +30.24 |
| DnCNN + classifier loss | +30.68 |

The honest conclusion is nuanced: PCSD+DualDGSC clearly beats the standard blind denoising baseline, but the fair classifier-guided DnCNN baseline is essentially tied or slightly ahead. This makes classifier-guided loss the main scientific finding, while DGSC remains a promising but secondary contribution.

### Natural corruptions

The corruption results show that adversarially trained models improve noise robustness but lose performance on some natural corruptions:

| Model | Clean Accuracy | Mean Corruption Accuracy | MeanLip |
|---|---:|---:|---:|
| Clean | 92.12% | 74.04% | 0.2243 |
| PGD-AT | 77.97% | 70.64% | 0.0357 |
| LAT | 77.19% | 69.96% | 0.0304 |

This means low MeanLip is useful for adversarial robustness, but it does not automatically imply better average corruption robustness.

## 4. GitHub Readiness

The project is worth uploading to GitHub as a research prototype. The best framing is:

> A spectral adversarial robustness research toolkit with attack-free diagnostics, CIFAR-10 robustness baselines, profile-conditioned denoising experiments, and a RobuScan demo app.

Use GitHub for code, figures, result JSONs, and documentation. Keep datasets and model checkpoints out of Git. If you want to share trained checkpoints, publish them separately through GitHub Releases, Google Drive, Hugging Face, or another model artifact host.

The final report PDF is only about 4.5 MB, so it is reasonable to commit it directly. The editable DOCX is also small enough to include if you want future editing convenience.

## 5. Important Caveats

- `caat/config.py` uses a hardcoded project root for the original execution environment.
- Some files contain encoding artifacts from copied symbols; they do not usually break execution, but they make the source look less polished.
- The `artifacts/paper_ieee` folder appears unrelated to this project and should not be presented as the final report.
- The project has strong experimental logs, but it is not yet packaged as an installable Python library.

## 6. Conclusion

The project has enough substance for GitHub: real modules, saved experiments, a usable scanner app, and a clear research direction. The strongest current contribution is the GSE/MeanLip diagnostic pipeline and the RobuScan interface. The PCSD component is promising, but the saved ablation suggests it should be described as exploratory rather than definitively better than DnCNN.

Recommended next step before a public release: make paths configurable, clean encoding artifacts, add a small smoke test, and optionally provide one small downloadable checkpoint outside Git.
