# Experiment 1: Synthetic Low-Pass Validation

## Purpose
Demonstrate that the FrequencyIntervention causally alters model predictions
on a controlled synthetic dataset.

## Method
- Model: SimpleFeatureNet (synthetic CNN, 16689 params)
- Dataset: 32 synthetic samples (3x64x64), binary masks (15% foreground)
- Intervention: Low-pass mask (cutoff=0.25) on all Conv2d layers
- Comparison: Baseline (no intervention) vs Low-pass

## Key Result
- Max |diff| between baseline and low-pass predictions: 2.34e-02
- Mean |diff|: 1.09e-02
- Low-pass reduces AVR to 0.0 across all layers
- Dice unchanged (0.0 on both due to random data below threshold)
- CONFIRMED: Intervention causally alters forward pass outputs

## Why No Saved Artifacts
This experiment was performed on synthetic data for validation purposes.
The numerical results are reported above and in the associated paper section.
The experiment script is self-contained and reproduces in seconds.

## Associated Files
- Script: experiments/experiment1_synthetic_validation.py
