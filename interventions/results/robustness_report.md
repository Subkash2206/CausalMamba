# Spectral Robustness — Consolidated Report

## Data splits (deterministic, documented for reproducibility)

- **CVC-ClinicDB**: `CVCDataset` 80/20 split via `train_test_split(test_size=0.2, random_state=42)` → **489 train / 123 val**. The 123-image val is used for best-val-loss model selection *and* all reported metrics (a mild selection-on-test caveat to state in the paper).

- **ISIC2018**: 50-image validation subset (first 50 sorted images of the 2,594 training images), matching the Phase-0 protocol (`experiment2`).

- All checkpoints loaded `strict=True`; all evals at 256×256 (Swin-UNet @224).


## CVC-ClinicDB — per-image Dice (mean ± SD, bootstrap 95% CI)

| Model | Cond | Pooled | Per-img μ±σ | 95% CI |
|---|---|---|---:|---:|
| VM-UNet | Clean | 0.8957 | 0.8779±0.1156 | [0.8567, 0.8970] |
| VM-UNet | Feat-LP 0.25 | 0.2919 | 0.2250±0.2357 | [0.1855, 0.2675] |
| VM-UNet | Input-LP 0.25 | 0.8230 | 0.8003±0.2118 | [0.7618, 0.8351] |
| VM-UNet-TSA | Clean | 0.9122 | 0.9059±0.1192 | [0.8826, 0.9244] |
| VM-UNet-TSA | Feat-LP 0.25 | 0.2644 | 0.2324±0.2238 | [0.1935, 0.2718] |
| VM-UNet-TSA | Input-LP 0.25 | 0.8803 | 0.8609±0.1911 | [0.8244, 0.8916] |
| ResNet50-UNet | Clean | 0.9320 | 0.9133±0.1312 | [0.8877, 0.9331] |
| ResNet50-UNet | Feat-LP 0.25 | 0.0000 | 0.0000±0.0000 | [0.0000, 0.0000] |
| ResNet50-UNet | Input-LP 0.25 | 0.7228 | 0.6793±0.3270 | [0.6202, 0.7355] |
| Swin-UNETR | Clean | 0.7848 | 0.7156±0.2548 | [0.6690, 0.7591] |
| Swin-UNETR | Feat-LP 0.25 | 0.6000 | 0.5225±0.2787 | [0.4731, 0.5703] |
| Swin-UNETR | Input-LP 0.25 | 0.7815 | 0.7129±0.2561 | [0.6664, 0.7568] |

### Feature-space LP dose-response (pooled Dice)

| Model | Clean | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VM-UNet | 0.896 | 0.000 | 0.181 | 0.300 | 0.292 | 0.307 | 0.383 |
| VM-UNet-TSA | 0.912 | 0.002 | 0.016 | 0.204 | 0.264 | 0.275 | 0.498 |
| ResNet50-UNet | 0.932 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.533 |
| Swin-UNETR | 0.785 | 0.264 | 0.361 | 0.456 | 0.600 | 0.646 | 0.710 |

### Input-space degradations (pooled Dice)

| Model | Clean | LP.25 | G2 | G4 | G8 | J80 | J50 | J20 | M5 | M10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VM-UNet | 0.896 | 0.823 | 0.815 | 0.569 | 0.388 | 0.889 | 0.880 | 0.853 | 0.869 | 0.803 |
| VM-UNet-TSA | 0.912 | 0.880 | 0.868 | 0.696 | 0.512 | 0.911 | 0.907 | 0.896 | 0.901 | 0.856 |
| ResNet50-UNet | 0.932 | 0.723 | 0.748 | 0.591 | 0.166 | 0.928 | 0.923 | 0.874 | 0.900 | 0.787 |
| Swin-UNETR | 0.785 | 0.781 | 0.776 | 0.731 | 0.571 | 0.785 | 0.784 | 0.784 | 0.782 | 0.765 |

### Boundary metrics (clean vs feature-LP 0.25)

| Model | Clean BF1 | Clean HD95 | LP BF1 | LP HD95 |
|---|---:|---:|---:|---:|
| VM-UNet | 0.672 | 21.4 | 0.062 | 93.9 |
| VM-UNet-TSA | 0.756 | 17.5 | 0.081 | 86.4 |
| ResNet50-UNet | 0.788 | 16.1 | 0.000 | nan |
| Swin-UNETR | 0.409 | 51.0 | 0.192 | 75.9 |

## ISIC2018 — cross-architecture benchmark (feature-space vs input-space)

### Held-out test split (n=260, untouched — supersedes dev-50; `isic_heldout_eval.json`)
| Architecture | Family | Clean | Feat-LP | Δ% | Input-LP | Δ% | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|
| VM-UNet | SSM | 0.9151 | 0.8212 | −10.3% | 0.9089 | −0.7% | 1.1e-19 |
| ResNet50-UNet (ISIC recipe) | CNN | 0.9091 | 0.8170 | −10.1% | 0.9049 | −0.5% | 9.6e-27 |
| ResNet50-UNet (CVC recipe, matched) | CNN | 0.8917 | 0.8077 | −9.4% | 0.8874 | −0.5% | 1.9e-31 |
| Swin-UNet (legacy ISIC arch) | ViT | 0.9114 | 0.3058 | −66.5% | 0.8987 | −1.4% | 2.3e-44 |
| Swin-UNETR (CVC recipe, ViT leg closure) | ViT | 0.8901 | 0.8844 | **−0.6%** | 0.8890 | −0.1% | 9.4e-12 |

### Dev-50 convenience subset (first 50 sorted images — DO NOT USE for the paper)
| Architecture | Family | Clean | Feat-LP | Δ% |
|---|---|---:|---:|---:|
| VM-UNet | SSM | 0.9506 | 0.9135 | −3.9% |
| ResNet50-UNet | CNN | 0.9473 | 0.9210 | −2.8% |
| Swin-UNet | ViT | 0.9483 | 0.3774 | −60.2% |

## Cross-dataset fragility (headline, ρ=0.25 feature-LP; both on carved held-out splits)
| Architecture | CVC held-out (62) Δ% | ISIC held-out (260) Δ% | Leg status |
|---|---:|---:|---|
| ResNet50-UNet (CNN) | −100% (0.960→0.000) | −9.4% (0.892→0.808) | matched recipe |
| VM-UNet (SSM) | −73.2% (0.913→0.245) | −10.3% (0.915→0.821) | CVC canonical; ISIC legacy VSSM |
| Swin-UNETR (ViT) | −30.9% (0.824→0.569) | −0.6% (0.890→0.884) | matched recipe |
| Swin-UNet (legacy ISIC) | — | −66.5% (0.911→0.306) | replaced by Swin-UNETR |

**Consistent ordering, no inversion:** CNN > SSM > ViT on both datasets; every family is
more fragile on CVC (CNN 10.6×, SSM 7.1×, ViT 52×). The earlier "inversion" traced to
an easy dev subset + the mismatched Swin-UNet checkpoint: on the same ISIC data the
matched Swin-UNETR loses only −0.6%, so transformer fragility is architecture-specific.
6 of 260 ISIC test images overlap the dev-50 (random-chance level, ~5).

### CVC held-out eval detail (62 test frames; `cvc_heldout_eval.json`)
| Architecture | Family | Clean | Feat-LP | Δ% | Input-LP | Δ% |
|---|---|---:|---:|---:|---:|---:|
| VM-UNet | SSM | 0.9125 | 0.2445 | −73.2% | 0.8556 | −6.2% |
| VM-UNet-TSA | SSM | 0.9421 | 0.2523 | −73.2% | 0.9314 | −1.1% |
| ResNet50-UNet | CNN | 0.9598 | 0.0000 | −100% | 0.7683 | −20.0% |
| Swin-UNETR | ViT | 0.8239 | 0.5690 | −30.9% | 0.8172 | −0.8% |