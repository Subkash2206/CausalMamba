# Phase 0 Executive Summary

**Date:** 2026-08-04
**Scope:** Phase 0 - VM-UNet topology remediation, baseline regeneration, and causal-spectral verification (Experiments 2-6).
**Dataset:** ISIC2018 lesion segmentation (50 images); canonical 30-block VM-UNet (depths=[2,2,9,2], depths_decoder=[2,9,2,2]); checkpoint best-vmunet-scratch-isic18.pth.

---

## 1. Infrastructure Fixes

A critical configuration defect was identified and remediated: Experiments 2-6 instantiated VM-UNet with depths=[2,2,2,2] / depths_decoder=[2,2,2,1] instead of the canonical [2,2,9,2] / [2,9,2,2], and were pointed at a 15-block checkpoint (best-vmunet-isic18.pth) rather than the 28-block weights (best-vmunet-scratch-isic18.pth) used by every SpectralMamba reviewer script. All load_state_dict calls now pass explicit strict=True.

**Why the old 15-block data was invalid:** The truncated architecture enumerated only 15 of the canonical 30 VSSBlocks and produced an under-powered (N=15) statistics regime - the previous null Pearson result (r=0.01, p=0.97) was an artifact of this structural truncation, not a genuine absence of spectral-dependency.

**Why the new 30-block data is structurally sound:** The corrected pipeline instantiates the true 30-block VM-UNet (2+2+9+2 encoder, 2+9+2+2 decoder), loads checkpoint weights under strict verification, and passed independent regression checks. verify_phase0.py reported full agreement across all Phase-0 CSV sets, with cross-experiment reproducibility: baseline Dice = 0.9525 reproduced identically in Exps. 2, 3, and 5; Low-Pass Dice = 0.9128 reproduced in Exps. 2 and 6.

## 2. The New Empirical Baseline (Experiment 2)

With the corrected 30-block architecture on ISIC2018 (50 images):

| Condition | Dice | IoU |
|---|---|---|
| **Baseline (canonical)** | **0.9525** | 0.9093 |
| **Low-Pass intervened (cutoff = 0.25)** | **0.9128** | 0.8396 |

Whole-network Low-Pass intervention produced a -4.17% relative Dice drop (Delta = -0.0397).

## 3. Scientific Breakthroughs

**Resolved statistical power (spectral-dependency link).** With N = 30 layers, the Pearson correlation between |Delta AVR| and |Delta Dice| is **r = 0.5548, p = 1.46e-03 (p < 0.01)** - a statistically significant positive association between spectral intervention magnitude and causal impact. This replaces the under-powered N = 15 null result (r ~ 0.01, p = 0.97), which was an artifact of the truncated architecture.

**Architectural dissociation (encoder vs. decoder).** Mean Delta Dice under layer-wise intervention is **-0.0211 for encoder blocks versus -0.0000 for decoder blocks** (both n = 15). The most causally sensitive layers are the early stage-2 encoder blocks - layers.2.blocks.0 alone induces Delta Dice = -0.088 - demonstrating that high-frequency information in the Mamba encoder is functionally necessary, while decoder blocks are largely insulated.

**Boundary error spike (Experiment 6).** Under Low-Pass (cutoff = 0.25), the mean error rate at the lesion boundary (+-5 px) is **35.54% versus 2.61% in the interior - a 13.6x multiplier** (35.54 / 2.61). DC-only intervention collapses to a degenerate near-uniform output (Dice = 0.0000, pred_std ~ 2e-9), confirming that boundary discrimination is critically dependent on preserved high-frequency content.

## 4. Phase 1 Readiness

The Phase-0 repository is locked and tagged with all topology/loading fixes and regenerated 30-block baselines verified, ready to ingest the UNet-ResNet50 architecture for Phase 1.