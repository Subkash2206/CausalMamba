# Plan Progress & Runbook — 2026-08-13

## Status update (10:30 local — post-interruption recovery)
- **System rebooted at 04:31** (Event 577/578, kernel-initiated reboot — likely Windows
  Update) killed the first training run at epoch 46/100. Best checkpoint (epoch 32,
  val loss 0.1156, saved 03:57) survived.
- **Resumed training** at 10:30 from the epoch-32 best weights (fresh Adam at the correct
  cosine position, lr 7.7e-5, best-val floor 0.1156). Runs epochs 33–100.
  Log: `interventions/logs/train_unet_isic_cvcrecipe_resume.log` (+`_err.log`).
  ~2.2–3.3 min/epoch → **expected completion ≈ 13:00–14:30 local**.
- **Resume support added** (`b7a7281`): `--resume` (full state file OR raw weights),
  `--resume_epoch`, `--resume_best_val`; full model+optimizer+scheduler state saved every
  5 epochs to `interventions/checkpoints/unet_isic_cvcrecipe_state_latest.pt`.
  Recovery from any future interruption is now one command:
  ```
  python interventions/train_unet_isic18_cvcrecipe.py --resume interventions/checkpoints/unet_isic_cvcrecipe_state_latest.pt
  ```
- Sleep timeouts are already 0 (never) on AC/DC — the reboot was not sleep-related.

## Running now (or last state)
- **ResNet50-UNet ISIC retrain (CVC recipe)** — resumed run, PID rotated; log
  `interventions/logs/train_unet_isic_cvcrecipe_resume.log`.
  Best checkpoint: `interventions/checkpoints/unet_isic_cvcrecipe_best.pth` (gitignored).
- Held-out eval deferred until this finishes (GPU contention OOMs on the 6 GB card).

## Delivered this session (commits)
| Artifact | Path | Purpose |
|---|---|---|
| Held-out splits | `interventions/results/splits/{isic,cvc}_split.json` | ISIC 2075/259/260, CVC 489/61/62 (seed 42) |
| Held-out ISIC eval | `interventions/experiments/eval_isic_heldout.py` | Phase-2 eval on untouched test split; `--models all\|existing\|standardized`; `--max_images N` smoke |
| ISIC retrain (CNN) | `interventions/train_unet_isic18_cvcrecipe.py` | CVC recipe; **resume support** added |
| ISIC retrain (SSM, cloud) | `interventions/train_vmunet_isic18_cvcrecipe.py` | Canonical VSSM + CVC recipe — **needs cloud** |
| Phase-0 audit | `interventions/results/PHASE0B_RECIPE_AUDIT.md` | VSSM divergence + CNN recipe mismatch |
| ISBI skeleton v2 | `interventions/results/ISBI_SKELETON_v2.md` | 4-page skeleton, CNN-anchored inversion, 3 tables |
| Cloud runbook | `interventions/CLOUD_RETRAIN.md` | Launch canonical VM-UNet ISIC retrain on ≥16 GB GPU |

## Eval notes (do not repeat mistakes)
- **Hook order:** clean pass must run BEFORE `attach_lp` (deterministic intervention →
  clean == feat if hooks attached first).
- **ISIC images are ~29 MP:** pre-resize to 256 at load time or the loader OOMs.
- **Contention:** eval OOMs/starves when concurrent with training on the 6 GB GPU. Run
  only after training finishes.

## Post-training steps (when resume log shows epoch 100)
1. Confirm checkpoint + clean Dice:
   ```powershell
   Get-Content interventions/logs/train_unet_isic_cvcrecipe_resume.log | Select-Object -Last 3
   Test-Path interventions/checkpoints/unet_isic_cvcrecipe_best.pth
   ```
2. Run the full held-out eval (GPU idle — ~20 min):
   ```powershell
   python interventions/experiments/eval_isic_heldout.py --models all
   ```
   → `interventions/results/isic_heldout_eval.json` (VM-UNet, ResNet50-ISIC-recipe,
   ResNet50-CVC-recipe, Swin-UNet).
3. Update ISBI_SKELETON_v2.md Table 2 ISIC column with held-out numbers; commit.
4. Cloud decision: VM-UNet ISIC canonical retrain (CLOUD_RETRAIN.md) — critical path for
   a matched SSM leg and the Phase-3 ViT decision.

## Known blockers
- VM-UNet ISIC canonical retrain infeasible locally (~3.5–4 h/epoch). Needs cloud.

