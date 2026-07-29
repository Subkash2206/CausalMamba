# Experiment 0: Identity Validation

## Purpose
Validate that the FrequencyIntervention framework is mathematically transparent
when using an identity (all-ones) mask.

## Method
- Framework: FFT -> x1 -> IFFT on all VSSBlock outputs
- Model: SimpleFeatureNet (synthetic CNN, 16689 params)
- Dataset: 32 synthetic samples (3x64x64)
- Mask: All-ones (identity)

## Key Result
- Max |diff| between baseline and identity: 7.45e-08
- Mean |diff|: 1.54e-08
- Dice difference: 0.00e+00
- Max |Delta AVR| across layers: 5.96e-08
- All tolerances passed -> Framework is TRANSPARENT

## Why No Saved Artifacts
This experiment was performed inline and verified in-memory.
The numerical results are reported above and referenced in the paper.
The experiment script is self-contained and reproduces these values in seconds.

## Associated Files
- Script: experiments/experiment0_identity_validation.py
- Figure: paper/figures/fig2_identity_validation.png
