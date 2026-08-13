# Plan Progress & Runbook — 2026-08-13

## Status update (14:30 local — post-training / post-eval)
- **Training DONE 13:05**: CVC-recipe ISIC CNN ran epochs 33–100 from the epoch-32 resume
  (best val loss **0.1106 at epoch 58**). Checkpoint:
  `interventions/checkpoints/unet_isic_cvcrecipe_best.pth`.
- **Held-out ISIC eval DONE 14:16**: all 4 models on the untouched 260-image test split →
  `interventions/results/isic_heldout_eval.json`.
  - **Headline change:** the earlier dev-50 "<4% CNN/SSM on ISIC" was a subset artifact.
    On the held-out split: CNN −9.4% (matched recipe), SSM −10.3%, ViT −66.5%.
  - The **rank-flip inversion holds**: CVC CNN(−100) > SSM(−67) > ViT(−24); ISIC
    ViT(−66) ≫ SSM(−10) ≈ CNN(−9). All per-image effects Wilcoxon p < 1e-19.
  - Skeleton (ISBI_SKELETON_v2.md) + robustness_report.md updated with held-out numbers;
    dev-50 table retained only as a "DO NOT USE" reference.
- **HD95 metric note:** tta_boundary_study's `hausdorff_95()` is a max-Hausdorff
  (directed_hausdorff = 100th pct), despite the name. The held-out eval now uses an
  EDT-based exact-equivalent (verified diff=0.0) that runs in microseconds — the
  directed_hausdorff version took minutes per large ISIC lesion.

## Running now (or last state)
- Nothing running; GPU idle. All Phase-2 deliverables complete.

## Delivered this session (commits)
| Artifact | Path | Purpose |
|---|---|---|
| Held-out splits | `interventions/results/splits/{isic,cvc}_split.json` | ISIC 2075/259/260, CVC 489/61/62 (seed 42) |
| Held-out ISIC eval | `interventions/experiments/eval_isic_heldout.py` | `--models all\|existing\|standardized`; `--max_images N`; fast EDT-HD95 |
| Inversion summarizer | `interventions/experiments/summarize_inversion.py` | Table-2 rows + Wilcoxon from CVC + ISIC JSONs |
| ISIC retrain (CNN) | `interventions/train_unet_isic18_cvcrecipe.py` | CVC recipe; resume support |
| ISIC retrain (SSM, cloud) | `interventions/train_vmunet_isic18_cvcrecipe.py` | Canonical VSSM + CVC recipe — **needs cloud** |
| Phase-0 audit | `interventions/results/PHASE0B_RECIPE_AUDIT.md` | VSSM divergence + CNN recipe mismatch |
| ISBI skeleton v2 | `interventions/results/ISBI_SKELETON_v2.md` | 4-page skeleton, rank-flip inversion, held-out Table 2 |
| Held-out results | `interventions/results/isic_heldout_eval.json` | Raw per-image + pooled numbers |

## Eval notes (do not repeat mistakes)
- **Hook order:** clean pass must run BEFORE `attach_lp` (deterministic intervention →
  clean == feat if hooks attached first).
- **ISIC images are ~29 MP:** pre-resize to 256 at load time or the loader OOMs.
- **Contention:** eval OOMs/starves when concurrent with training on the 6 GB GPU.
- **Slow HD95:** never use `directed_hausdorff` on full masks for large lesions (minutes
  per image); use the EDT equivalent (`hd95_fast` in eval_isic_heldout.py).

## Remaining before Oct 26
1. **Cloud decision (blocked on you):** VM-UNet ISIC canonical retrain
   (`interventions/train_vmunet_isic18_cvcrecipe.py` + `CLOUD_RETRAIN.md`) — closes the
   SSM implementation caveat; enables the Phase-3 ViT decision.
2. Optional: evaluate CVC models on the new carved CVC test split (62) to remove the
   selection-on-test caveat from the CVC column of Table 2.
3. Lock author list; assemble citations (10–12); render 300-dpi figures.

## Known blockers
- VM-UNet ISIC canonical retrain infeasible locally (~3.5–4 h/epoch). Needs cloud.

