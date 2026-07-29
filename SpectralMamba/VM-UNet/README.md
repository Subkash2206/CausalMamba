# Spectral Fingerprints of Vision Architectures

## TL;DR: Unmasking Spectral Artifacts
This study performs a rigorous spectral audit of Mamba-based architectures (VM-UNet). We discover that the previously reported "Spectral Debt" (a global correlation between aliasing and boundary precision) was largely an **evaluation artifact** caused by uncentered 2D-FFTs (DC component bias). Once mean-centered, the linear correlation between total alias volume and Boundary F1 collapses to near-zero (+0.05). However, we reveal a unique **Dual-Stage Spectral Behavior** in Mamba: it exhibits high aliasing in early encoding stages (Level 1) but transitions to aggressive low-pass filtering in deep stages (Level 4), unlike the more uniform spectral profiles of CNNs and Transformers.

![Evidence Chain](../results/figures/band_decomposition.png)
*1. Spectral Evolution: Mamba transitions from high-frequency debt at Level 1 to aggressive smoothing at Level 4.*

![Spectral Leakage](../results/figures/power_spectrum_grid.png)
*2. Level 1 Leakage: 2D FFTs (mean-centered) reveal Mamba's high-frequency energy accumulation in early stages.*

![Statistical Impact](../results/figures/avr_bf1_scatter.png)
*3. Statistical Correction: Mean-centering removes activation bias, showing that global aliasing does not predict boundary precision (r ≈ 0).*

## Abstract
This repository contains a comparative spectral analysis of three dominant architectural paradigms in medical image segmentation: Convolutional Neural Networks (UNet-ResNet50), Vision Transformers (Swin-Tiny), and State-Space Models (VM-UNet). Following a methodological correction—enforcing mean-centered feature maps to remove DC-offset bias—we re-evaluate the "Spectral Debt" hypothesis. Our key finding is the **"Correlation Collapse"**: the pooled AVR–BF1 Pearson r is **+0.0108 (p=0.6704)**, statistically indistinguishable from zero. While Mamba exhibits a unique dual-stage fingerprint (Level 1 AVR **0.4401**, Level 4 AVR **0.1196**), this does not translate to a global boundary precision deficit. Mamba's O(N) linear scaling advantage over Transformer's O(N²) quadratic attention is therefore not offset by any statistically verified spectral cost.

## 1. Key Performance Metrics


| Architecture | Paradigm | Global Dice (↑) | Boundary F1 (BF1) (↑) | Mean AVR (↓) |
| :--- | :--- | :---: | :---: | :---: |
| **VM-UNet (Mamba)** | Selective Scan | **0.9027** | 0.4939 | **0.2638** |
| **Swin-Tiny** | Attention | 0.9023 | **0.5259** | 0.3510 |
| **UNet-ResNet50** | Convolutions | 0.9000 | 0.4470 | 0.2862 |

*Source of truth: `results/boundary_results.csv`*

## 2. Core Scientific Findings

### A. Dual-Stage Spectral Dynamics
By mean-centering feature maps before computing the 2D-FFT, we isolate genuine high-frequency content from activation magnitude biases. This reveals that Mamba's "Spectral Debt" is concentrated in early stages.

#### Stage-wise Alias Volume Ratio (AVR, mean-centered)
| Resolution Level | UNet AVR | Swin-UNet AVR | VM-UNet AVR |
| :--- | :---: | :---: | :---: |
| **Level 1 (~64×64)** | 0.3344 | 0.3802 | **0.4401** |
| **Level 2 (~32×32)** | 0.3477 | 0.3525 | **0.3551** |
| **Level 3 (~16×16)** | 0.2831 | 0.3442 | **0.1406** |
| **Level 4 (~8×8)**   | 0.1795 | 0.3270 | **0.1196** |

*Source: `results/avr_stagewise_results_matched.csv`*

#### Frequency Band Decomposition, Mamba (mean-centered)
| Stage | Low Band (<0.25 Ny) | Mid Band (0.25-0.75 Ny) | High Band (>0.75 Ny) |
| :--- | :---: | :---: | :---: |
| **Level 1 (Early)** | 32.15% | 45.08% | **22.76%** |
| **Level 4 (Deep)** | **69.35%** | 27.35% | **3.29%** |

### B. Statistical Correction: The Correlation Collapse
To test the causal link between aliasing and edge precision, we performed a per-image correlation analysis. Once DC-bias is removed, the pooled correlation collapses. 

#### Correlation Results — Mean AVR vs. BF1 (mean-centered FFT)
| Population | Pearson *r* | *p*-value | Significant? |
| :--- | :---: | :---: | :--- |
| **VM-UNet (Mamba)** | +0.0998 | 0.0229 | Yes |
| **Pooled (Overall)** | **+0.0108** | 0.6704 | **No** |
| **Partial (Controlled)**| **-0.0001** | 0.9956 | **No** |

## 3. Visualization Gallery
All figures are automatically generated and saved in `results/figures/`:
*   `band_decomposition.png`: Shows the stage-wise energy shift.
*   `avr_bf1_scatter.png`: The "Correlation Collapse" scatter plot.
*   `power_spectrum_grid.png`: Heatmaps of the power spectra.
*   `shift_consistency_curves.png`: Robustness decay curves.
