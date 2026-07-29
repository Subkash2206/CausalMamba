# Master Manifest

## Experiments

| # | Experiment | Finding | Script |
|---|-----------|---------|--------|
| 0 | Identity validation | max diff = 7.45e-08 | experiments/experiment0_identity_validation.py |
| 1 | Synthetic validation | Intervention alters predictions | experiments/experiment1_synthetic_validation.py |
| 2 | Whole-network LP | Dice 0.941 -> 0.874 (-7.08%) | experiments/experiment2_real_lowpass.py |
| 3 | Layer-wise LP | Encoder 12x more important | experiments/experiment3_layerwise.py |
| 4 | Cutoff sweep | Nonlinear threshold 0.10-0.20 | experiments/experiment4_cutoff_sweep.py |
| 5 | Robustness | All conclusions confirmed | experiments/experiment5_robustness.py |
| 6 | DC baseline + boundary | DC: Dice=0.06; Boundaries: 16.4x | experiments/experiment6_dc_boundary.py |