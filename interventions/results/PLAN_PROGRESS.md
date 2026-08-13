# Plan Progress & Runbook — 2026-08-13

## Status update (18:45 local — ViT leg closed, SSM leg running CVC-matched subset)
- **ViT leg CLOSED 17:30**: Swin-UNETR trained on ISIC (CVC recipe, 100 ep, 2.3 h, best
  val loss 0.1339) and evaluated on the held-out test: clean 0.890, feat-LP **−0.6%**
  (p=9.4e-12). **Headline change:** the legacy Swin-UNet's −66.5% is architecture-specific,
  not transformer-general. With all legs matched there is **no cross-dataset inversion**:
  ordering is CNN > SSM > ViT on BOTH datasets, and every family is more fragile on CVC
  (CNN 10.6×, SSM 7.1×, ViT 52×). Skeleton/report/summarizer reframed accordingly.
- **SSM leg RUNNING 18:39 — user decision: CVC-matched 490-image subset (~3 days).**
  Rationale: 8–15 epochs was rejected (LR still 95% of peak at ep 15; recreates the
  recipe confound); a 490-image subset keeps the full 100-epoch cosine recipe AND mirrors
  the CVC leg's 489-image data volume exactly. Run:
  `python interventions/train_vmunet_isic18_cvcrecipe.py --subset 490` (PID 22480),
  ~42 min/epoch → ~3 days. Cached dataset `train_490.pt`, resume state every 5 epochs.
  Log `interventions/logs/train_vmunet_isic_cvcrecipe.log`; resume:
  `python interventions/train_vmunet_isic18_cvcrecipe.py --subset 490 --resume
  interventions/checkpoints/vmunet_isic_cvcrecipe_state_latest.pt`.
- Cached dataset module `interventions/isic_dataset.py` (thread-parallel build, persisted
  to `interventions/cache/isic256/*.pt`, gitignored) cut per-epoch 29MP decode costs.

## Running now (or last state)
- **VM-UNet ISIC canonical retrain (490-image CVC-matched subset)** — PID 22480 (started
  18:39), GPU busy ~3 days.

## Delivered this session (commits)
| Artifact | Path | Purpose |
|---|---|---|
| Cached ISIC dataset | `interventions/isic_dataset.py` | 256px tensor cache (disk+RAM), kills per-epoch 29MP decode |
| Swin-UNETR ISIC trainer | `interventions/train_swinunetr_isic18_cvcrecipe.py` | ViT leg closure (CVC recipe + ImageNet norm) |
| VM-UNet ISIC trainer | `interventions/train_vmunet_isic18_cvcrecipe.py` | + cache, cudnn.benchmark, --max_batches/--max_epochs, resume |
| Held-out ISIC eval | `interventions/experiments/eval_isic_heldout.py` | + Swin-UNETR build/attach support |
| Held-out results | `interventions/results/isic_heldout_eval.json` | 5 models incl. Swin-UNETR (−0.6%) |
| Skeleton v3 reframe | `interventions/results/ISBI_SKELETON_v2.md` | No-inversion story: consistent CNN>SSM>ViT ordering; Swin-UNet anomaly |
| Summarizer | `interventions/experiments/summarize_inversion.py` | Prefers recipe-matched rows |

## Eval notes (do not repeat mistakes)
- **Hook order:** clean pass must run BEFORE `attach_lp` (deterministic intervention →
  clean == feat if hooks attached first).
- **ISIC images are ~29 MP:** use ISICCacheDataset or pre-resize; raw decode is ~4 s/img.
- **Contention:** eval OOMs/starves when concurrent with training on the 6 GB GPU.
- **Swin-UNETR hooks** are the swinViT block layout (nhwc) from cross_arch_cvc_eval.
- **`directed_hausdorff` on full masks is minutes/image** for large lesions; use the
  EDT-equivalent `hd95_fast` (exact, diff=0.0).

## Remaining before Oct 26
1. **~9–10 days:** VM-UNet ISIC canonical retrain (running) → replaces the legacy-VSSM
   ISIC SSM row; rerun held-out eval (--models all) when done.
2. Write the full 4-page paper from the reframed skeleton (no-inversion thesis).
3. Lock author list; assemble citations (10–12); render 300-dpi figures.

## Known blockers
- None active. The SSM ISIC row carries a legacy-VSSM caveat until the running retrain
  completes (~9–10 days).

