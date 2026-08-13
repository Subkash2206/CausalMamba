# Plan Progress & Runbook — 2026-08-13

## Status update (19:00 local — SSM retrain KILLED by decision; paper ships with caveat)
- **ViT leg CLOSED 17:30**: Swin-UNETR trained on ISIC (CVC recipe, 100 ep, 2.3 h, best
  val loss 0.1339) and evaluated on the held-out test: clean 0.890, feat-LP **−0.6%**
  (p=9.4e-12). The legacy Swin-UNet's −66.5% is architecture-specific, not
  transformer-general. With matched CNN + ViT legs there is **no cross-dataset inversion**:
  ordering is CNN > SSM > ViT on BOTH datasets, and every family is more fragile on CVC
  (CNN 10.6×, SSM 7.1×, ViT 52×).
- **SSM ISIC canonical retrain KILLED 18:55** (user decision — laptop needed for other
  projects; only ~16 min of epoch-1 work lost). The paper proceeds with the ISIC SSM row
  from the legacy VSSM checkpoint (−10.3%, held-out n=260), honestly caveated in Table 2
  and methods. Rationale: the claim is corroborating (headline anchored on the matched
  CNN + ViT legs); the two VSSM implementations agree to 0.906 feature similarity, so the
  legacy number (−10.3%, consistent with the CNN's −9.4%) is expected to be representative.
  If a reviewer asks, the retrain can be run later (script is resume-safe; $5–15 rented
  GPU = half a day) with 2 months of deadline slack.
- The `--subset 490` flag and cached dataset remain available for a future run:
  `python interventions/train_vmunet_isic18_cvcrecipe.py --subset 490 --resume
  interventions/checkpoints/vmunet_isic_cvcrecipe_state_latest.pt`.

## Running now (or last state)
- **Nothing running; GPU free.** All Phase-2 eval work complete. Laptop fully usable.

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
1. Write the full 4-page paper from the reframed skeleton (no-inversion thesis; SSM ISIC
   row caveated with the legacy-VSSM footnote).
2. Optional (only if a reviewer asks): run the canonical ISIC SSM retrain on cloud or in a
   laptop GPU gap (~3 days local / half a day rented).
3. Lock author list; assemble citations (10–12); render 300-dpi figures.

## Known blockers
- None active. One documented caveat in Table 2 (ISIC SSM = legacy VSSM implementation).

