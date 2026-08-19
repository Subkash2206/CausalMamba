# Plan Progress & Runbook — 2026-08-13

## Status update (20:10 local — paper package COMPLETE; draft ready)
- **Kaggle SSM retrain ABANDONED** (session failures + state-file loss at epoch 17; no
  recovery). The paper ships with the legacy-VSSM caveat on the ISIC SSM row (Table 2 †),
  exactly as agreed. The package + recipe remain on Kaggle if ever needed (~4–5 sessions).
- **Paper package generated** (`interventions/results/paper_v2/`):
  - `ISBI_PAPER_DRAFT.md` — complete 4-page draft (abstract, intro, methods, results,
    discussion, 3 tables, 2 figures, 11 references; ~2,200 words).
  - `tables/table{1,2,3}_*.csv` — generated from the result JSONs by
    `experiments/export_isbi_package.py` (no hardcoded numbers).
  - `figures/fig1_cvc_dose_response.png` + `fig2_cross_dataset_fragility.png` — 300 dpi.
- Headline (final): consistent CNN > SSM > ViT ordering on both datasets; CVC ≫ ISIC
  magnitude; no inversion. Swin-UNet −66.5% vs Swin-UNETR −0.6% = within-family warning.

## Delivered this session (commits)
| Artifact | Path | Purpose |
|---|---|---|
| Paper export script | `interventions/experiments/export_isbi_package.py` | Tables + 300-dpi figures from JSONs |
| Paper package | `interventions/results/paper_v2/` | Draft + 3 table CSVs + 2 figures |
| Kaggle runbook | `interventions/KAGGLE_RUNBOOK.md` | Cloud path (deferred/abandoned) |

## Remaining before Oct 26
1. Lock author list; format the draft to IEEE/ISBI LaTeX (two-column).
2. ~~Optional: example-mask overlays for Fig. 2~~ **Done** — ISIC + CVC qualitative panels at 300 dpi (`make_isic_robustness_figures.py`, `make_robustness_figures.py`).
3. Optional (only if a reviewer asks): canonical ISIC SSM retrain (Kaggle package intact).

## Delivered this session (commits)
| Artifact | Path | Purpose |
|---|---|---|
| Cached ISIC dataset | `interventions/isic_dataset.py` | 256px tensor cache (disk+RAM), kills per-epoch 29MP decode |
| Swin-UNETR ISIC trainer | `interventions/train_swinunetr_isic18_cvcrecipe.py` | ViT leg closure (CVC recipe + ImageNet norm) |
| VM-UNet ISIC trainer | `interventions/train_vmunet_isic18_cvcrecipe.py` | + cache, cudnn.benchmark, --max_batches/--max_epochs, resume |
| Held-out ISIC eval | `interventions/experiments/eval_isic_heldout.py` | + Swin-UNETR build/attach support; `--models all` complete file; `--merge_existing` upsert for partial runs |
| Held-out results | `interventions/results/isic_heldout_eval.json` | 5 models incl. Swin-UNETR (−0.6%) |
| Skeleton v3 reframe | `interventions/results/ISBI_SKELETON_v2.md` | No-inversion story: consistent CNN>SSM>ViT ordering; Swin-UNet anomaly |
| Summarizer | `interventions/experiments/summarize_inversion.py` | Prefers recipe-matched rows |
| ISIC qualitative figure | `interventions/experiments/make_isic_robustness_figures.py` | 300-dpi clean/input-LP/feat-LP panels, 5 held-out models, small/med/large lesions |
| Draft cleanup | `interventions/results/paper_v2/ISBI_PAPER_DRAFT.md` | metrics paragraph moved to 2.4; citations [6][7][11]; Table 2 VM-UNet CVC 0.912->0.244 |

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
3. Lock author list; assemble citations (10–12); ~~render 300-dpi figures~~ **done** (all
   qualitative + quantitative figures are 300 dpi).

## Known blockers
- None active. One documented caveat in Table 2 (ISIC SSM = legacy VSSM implementation).

