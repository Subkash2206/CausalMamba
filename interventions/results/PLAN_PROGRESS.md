# Plan Progress & Runbook — 2026-08-13 (night session)

## Running now
- **ResNet50-UNet ISIC retrain (CVC recipe)** — PID 13028, started 02:31 local.
  Log: `interventions/logs/train_unet_isic_cvcrecipe.log` (epoch lines flush ~every
  epoch; ~3.3 min/epoch → 100 epochs done ≈ **08:10–08:20 local**).
  Best checkpoint: `interventions/checkpoints/unet_isic_cvcrecipe_best.pth` (gitignored).
  Health check: `nvidia-smi` (expect ~2 GB, GPU busy), tail the log for `saved best`.

## Delivered this session (commit `30e6262`)
| Artifact | Path | Purpose |
|---|---|---|
| Held-out splits | `interventions/results/splits/{isic,cvc}_split.json` | ISIC 2075/259/260, CVC 489/61/62 (seed 42) |
| Held-out ISIC eval | `interventions/experiments/eval_isic_heldout.py` | Phase-2 eval on untouched test split; `--models all\|existing\|standardized`; `--max_images N` smoke |
| ISIC retrain (CNN) | `interventions/train_unet_isic18_cvcrecipe.py` | CVC recipe (random init, BCEWithLogits, cosine, 256, bs8, 100ep, seed 42) — **running** |
| ISIC retrain (SSM, cloud) | `interventions/train_vmunet_isic18_cvcrecipe.py` | Canonical VSSM + CVC recipe (AdamW, BCE+Dice, cosine) — **needs cloud** |
| Phase-0 audit | `interventions/results/PHASE0B_RECIPE_AUDIT.md` | VSSM divergence (canonical vs JIT-loop; 0.906 sim, 6/30 blocks) + CNN recipe mismatch |
| ISBI skeleton v2 | `interventions/results/ISBI_SKELETON_v2.md` | 4-page skeleton, CNN-anchored inversion, 3 tables, matched-vs-caveated legs |
| Cloud runbook | `interventions/CLOUD_RETRAIN.md` | Steps to launch the canonical VM-UNet ISIC retrain on a ≥16 GB GPU |

## Eval notes (do not repeat mistakes)
- **Hook order matters:** clean pass must run BEFORE `attach_lp` (intervention is
  deterministic → attaching hooks first makes clean == feat, silently zeroing the effect).
- **ISIC images are ~29 MP:** pre-resize to 256 at load time or the loader OOMs on RAM.
- **Contention:** the eval OOMs/starves when run concurrently with the ResNet50 training
  on the 6 GB GPU. Run the full eval only AFTER training finishes.

## Post-training steps (when `train_unet_isic_cvcrecipe.log` shows epoch 100)
1. Confirm checkpoint exists and clean Dice ≈ 0.93–0.95:
   ```powershell
   Get-Content interventions/logs/train_unet_isic_cvcrecipe.log | Select-Object -Last 3
   Test-Path interventions/checkpoints/unet_isic_cvcrecipe_best.pth
   ```
2. Run the full held-out eval (GPU idle now — fast, ~20 min):
   ```powershell
   python interventions/experiments/eval_isic_heldout.py --models all
   ```
   → `interventions/results/isic_heldout_eval.json` (VM-UNet, ResNet50-ISIC-recipe,
   ResNet50-CVC-recipe, Swin-UNet).
3. Update ISBI_SKELETON_v2.md Table 2 ISIC column with held-out numbers; commit.
4. Cloud decision: VM-UNet ISIC canonical retrain (see CLOUD_RETRAIN.md) — critical path
   for a matched SSM leg and the Phase-3 ViT decision.

## Known blockers
- VM-UNet ISIC canonical retrain infeasible locally (~3.5–4 h/epoch). Needs cloud.
