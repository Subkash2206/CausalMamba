# Experiment Documentation

## Experiment 0: Identity Validation
- **Goal:** Verify FFT->x1->IFFT is mathematically transparent
- **Method:** Apply all-ones mask to all VSSBlocks
- **Result:** max |diff| = 7.45e-08 (float32)
- **Figure:** fig2_identity_validation.png
- **File:** experiments/experiment0_identity_validation.py

## Experiment 1: Synthetic Low-Pass Validation
- **Goal:** Confirm intervention causally alters predictions
- **Method:** Low-pass on synthetic CNN + random data
- **Result:** Predictions change (max |diff| = 2.34e-02)
- **File:** experiments/experiment1_intervention_validation.py

## Experiment 2: Whole-Network Low-Pass
- **Goal:** Measure segmentation impact of suppressing high frequencies everywhere
- **Method:** Low-pass (cutoff=0.25) on all 15 VSSBlocks
- **Result:** Dice 0.941 -> 0.874 (-7.08%)
- **Figure:** fig3_whole_network.png
- **Table:** table2_whole_network.csv
- **File:** experiments/experiment2_real_lowpass.py

## Experiment 3: Layer-Wise Intervention
- **Goal:** Identify which VSSBlocks are most causally important
- **Method:** Intervene on one block at a time (cutoff=0.25)
- **Result:** First encoder block most sensitive; encoder 12x > decoder
- **Figure:** fig4_layerwise_importance.png
- **Table:** table3_layerwise.csv
- **File:** experiments/experiment3_layerwise.py

## Experiment 4: Cutoff Sweep
- **Goal:** Characterize performance vs spectral retention
- **Method:** Evaluate cutoffs [0.10, 0.20, ..., 0.80]
- **Result:** Nonlinear threshold at 0.10-0.20; saturation at 0.50
- **Figure:** fig5_cutoff_sweep.png
- **Table:** table4_cutoff_sweep.csv
- **File:** experiments/experiment4_cutoff_sweep.py

## Experiment 5: Robustness Verification
- **Goal:** Verify conclusions at different cutoffs
- **Method:** Replicate Exp 3 with cutoff=0.50
- **Result:** All 4 core conclusions confirmed
- **File:** experiments/experiment5_robustness.py

## Experiment 6: DC-Only Baseline + Boundary Analysis
- **Goal:** Establish performance floor and spatial error distribution
- **Protocol A:** DC-only (cutoff=0.01) -> Dice=0.0604 (degenerate)
- **Protocol B:** Boundary error analysis -> 16.4x errors at boundaries
- **Figure:** fig7_boundary_errors.png
- **Table:** table6_dc_baseline.csv, table7_boundary_analysis.csv
- **File:** experiments/experiment6_dc_boundary.py