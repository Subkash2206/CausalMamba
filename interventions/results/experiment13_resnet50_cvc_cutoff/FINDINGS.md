# CVC-ClinicDB Low-Pass Sweep — Findings (UNet-ResNet50)

**Date:** 2026-08-04. Whole-network low-pass on CVC val (123 img, 352x352), `unet_cvc_best.pth`; analogue of Exp 4.

## Results (Dice / IoU)
0.10: 0.2868/0.1674 · 0.20: 0.1486/0.0803 · 0.25: 0.1792/0.0984 · 0.30: 0.4109/0.2586 · 0.40: 0.7523/0.6029 · 0.50: 0.8746/0.7772 · 0.60: 0.9107/0.8360 · 0.70: 0.9183/0.8490 · 0.80: 0.9230/0.8569 · Baseline: 0.9286/0.8667.

## Verification
1. **Radius nesting:** 0.10->17.6px, 0.15->26.4, 0.20->35.2, 0.25->44.0 at 352x352 (max 176): strictly monotone, nested. No discretization artifact.
2. **Reproducibility:** valley re-run under seeds 2 and 999 gave byte-identical Dice (0.10: 0.286778, 0.20: 0.148630, 0.25: 0.179196). Eval is deterministic (no dropout, fixed 123-img val split), so this rules out run-to-run noise and caching; the 0.25 point also matches Exp 10's independent 0.1792.
3. **Identity (audit `cvc_audit/`):** reproduces 0.9286 baseline to 4e-6.

## Interpretation (honest scope)
Collapse forms a **non-monotonic valley** (Dice 0.29->0.15->0.18 across 0.10-0.25), then monotone recovery to 0.923 @ 0.80, -0.5% below baseline.

- **Established:** valley is deterministic and reproducible; noise, caching, and mask discretization are excluded.
- **Open question (not established):** whether the wobble reflects genuine near-degenerate model sensitivity or a systematic property of CVC-ClinicDB's frequency content in the 0.10-0.25 band. Deterministic eval cannot discriminate these; noted as open.

## Cross-dataset contrast
ISIC (VM-UNet, Exp 4) recovers to near-baseline by 0.80 (0.036->0.954); CVC (ResNet50) collapses and recovers slowly, never fully (0.923 @ 0.80). Supported claim: CVC leans on a broader high-frequency band than ISIC for this architecture.

## Boundary ratio caveat
Exp 12's 0.45x ratio is excluded: at 87% interior error the model is globally failed; no spatial structure remains to interpret.