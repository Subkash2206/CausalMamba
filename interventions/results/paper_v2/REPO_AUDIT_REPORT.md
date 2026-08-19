# Complete Repo Audit — ISBI Paper (2026-08-13)

**Auditor:** automated pass (scripts `audit_isbi_numbers.py`, `export_isbi_package.py`) + manual review.
**Scope:** every experiment dir under `interventions/results/`, `results/paper/`, every result JSON, every checkpoint, the paper draft `paper_v2/ISBI_PAPER_DRAFT.md`, and the figures.

---

## 0. Executive Summary

1. **The paper is complete and self-consistent.** All 76 numeric cells in the draft's three tables were verified against the source JSONs — **0 mismatches remain** (one rounding-convention mismatch in Table 2 was found and fixed; see §3).
2. **Bootstrap CIs computed** for all 9 held-out model rows (§4) — these are NEW (only Wilcoxon p was reported before).
3. **Experiment census:** ~28 experiment directories + 8 checkpoint files. Classification: **6 used** in the paper, **~12 unused-but-relevant** (supporting/mechanistic), **~10 safely dead** (superseded or negative-result).
4. **Example-mask figures:** `make_robustness_figures.py` already generates clean-vs-feature-LP prediction triplets with the CURRENT CVC checkpoints (`cvc_sample_*.png`, 150 dpi). Usable for the paper after a 300-dpi re-run; the ISIC side would need a new analogous script (§5).
5. **Regeneration chain** is fully scripted from checkpoints to tables/figures (§6); two manual steps remain (one JSON row-merge, the 300-dpi re-run).

---

## 1. Experiment Inventory (what was run, on what, key numbers)

Legend: **CVC** = CVC-ClinicDB, **ISIC** = ISIC2018. "canonical VM-UNet" = 30-block VM-UNet with the canonical vectorized VSSM.

### Phase 0 — original SPIE-era analysis (ISIC-50 dev subset, canonical VM-UNet)
| Dir | Measured | Model/Dataset | Key numbers |
|---|---|---|---|
| `experiment0_identity` | identity (all-ones) mask sanity | VM-UNet / ISIC-50 | max \|diff\| ~ 7.45e-08 (pass) |
| `experiment1_intervention_validation` | synthetic low-pass validation | VM-UNet / synthetic | intervention provably alters predictions |
| `experiment2` | whole-network LP @0.25 | VM-UNet (canonical) / ISIC-50 | 0.9525 -> 0.9128 (**-4.17%**) |
| `experiment2_whole_network` | whole-network LP (older run) | VM-UNet / ISIC-50 | 0.9408 -> 0.8742 (-7.08%) - superseded by exp2 |
| `experiment3` / `experiment3_layerwise` | per-VSSBlock LP @0.25 | VM-UNet / ISIC-50 | top layer layers.0.blocks.0 dDice -0.0295; encoder >> decoder; |dAVR|-vs-|dDice| r=0.5548, p=1.46e-3 |
| `experiment4` / `experiment4_cutoff_sweep` | cutoff sweep whole vs single layer | VM-UNet / ISIC-50 | whole: 0.453@0.1 -> 0.932@0.5 -> 0.940@0.8; single-layer robust |
| `experiment5` / `experiment5_robustness` | robustness verification (top-1/top-3 match) | VM-UNet / ISIC-50 | top1_matches=true |
| `experiment6` / `experiment6_dc_boundary` | DC-only + boundary analysis | VM-UNet / ISIC-50 | DC-only Dice 0.060; boundary-5px err 40.7% vs interior 1.4% (16.4x) |

### Phase 1 — ResNet50-UNet causal (ISIC-50, then CVC at 352 px)
| Dir | Measured | Model/Dataset | Key numbers |
|---|---|---|---|
| `experiment7_resnet50` | whole-network LP @0.25 | ResNet50-UNet / ISIC-50 | 0.9459 -> 0.9176 (-2.99%) |
| `experiment8_resnet50_layerwise` | layerwise LP | ResNet50-UNet / ISIC-50 | top encoder.layer2[-1] dDice -0.0154 |
| `experiment9_resnet50_boundary` | boundary errors (clean) | ResNet50-UNet / ISIC-50 | bnd-5px 33.9% vs interior 0.53% |
| `experiment10_resnet50_cvc_causal` | whole-network LP @0.25 @352 | ResNet50-UNet / CVC-352 val(123) | 0.9286 -> 0.1792 (**-80.70%**) |
| `experiment11_resnet50_cvc_layerwise` | layerwise @352 | ResNet50-UNet / CVC-352 | top encoder.layer4[-1] dDice -0.249 |
| `experiment12_resnet50_cvc_boundary` | boundary errors (intervened) | ResNet50-UNet / CVC-352 | bnd-5px 50.7%, interior 87.3% (collapsed) |

### Phase 2 — CVC unification at 256 px + VM-UNet CVC training
| Dir | Measured | Model/Dataset | Key numbers |
|---|---|---|---|
| `experiment13_resnet50_cvc_cutoff` | ResNet50 CVC cutoff sweep (352 then 256) | ResNet50-UNet / CVC-123 | at 256: collapses by 0.1-0.3, recovers 0.533@0.4 |
| `best-swinunetr-cvc-256/` | CVC Swin-UNETR retrained @256 | Swin-UNETR / CVC | checkpoint best-swinunetr-cvc-256.pth (clean 0.785 @123-val) |
| (log) `SpectralMamba/results/vmunet_progress.txt` | VM-UNet CVC training 100 ep | canonical VM-UNet / CVC-489 | best val Dice 0.9163 @ep75; final 0.9122 |

### Phase 3 — causal analysis (CVC, 123-val, canonical VM-UNet)
| Dir | Measured | Key numbers |
|---|---|---|
| `experiment14_vmunet_cvc_causal` | whole-network LP @0.25 | 0.8957 -> 0.2919 (**-67.41%**) |
| `experiment14_..._seed123` / `_seed2027` | "multi-seed" replication | **bit-identical to seed 42** (same frozen checkpoint -> no-op) -> **deleted from paper** |
| `experiment15_vmunet_cvc_boundary_baseline` | boundary-vs-interior error (clean) | bnd-5px 20.38%, bnd-10px 13.93%, bnd-20px 8.83%, interior 0.102%, background 0.31% -> **86.5x** |
| `experiment15_..._intervened` | boundary errors under LP | bnd-5px 48.1%, interior 54.0% (errors homogenize under collapse) |
| `experiment16_vmunet_cvc_size_confound` | dDice by lesion size | Small **-0.734** (n=41), Medium -0.632, Large -0.593 |

### Phase 4 — robustness suite (CVC, 123-val)
| Dir/File | Measured | Key numbers |
|---|---|---|
| `cvc_robustness_eval.json` | **Table 1 + Table 3 sources**: feature-LP dose-response {0.10-0.40}, input-LP, degradations (G2/4/8, J80/50/20, M5/10), boundary BF1/HD95 | see Section 3 |
| `cross_arch_cvc_eval.json` | feature-LP @0.25, 4 models, 123-val | VM-UNet -67.4%, TSA -71.0%, ResNet50 -100%, Swin-UNETR -23.5% |
| `input_space_baselines_eval.json` | input-LP, CNN/ViT | ResNet50 -22.4%; Swin-UNETR -0.4% |
| `input_space_tsa_eval.json` | TSA defense input-LP | -8.1% -> -3.5% |
| `robustness_stats.json` | Wilcoxon + bootstrap on 123-val (CVC) and **dev-50 (ISIC)** | CVC feat-LP all p<1e-18; **ISIC entries superseded** by held-out |
| `best-vmunet-cvc-tsa-finetune/` | TSA fine-tuned checkpoint | used in Tables 1/3 |
| `best-vmunet-cvc-reg/` + `experiment17_vmunet_cvc_reg_eval*` | regularized VM-UNet @128/256 | clean 0.720@128 / 0.345@256 -> **worse than baseline; negative result, not in paper** |

### Phase 2 of the plan (2026-08-13) — held-out evals + recipe-matched retrains (**the paper's numbers**)
| Dir/File | Measured | Key numbers |
|---|---|---|
| `splits/{isic,cvc}_split.json` | deterministic 80/10/10 carves | ISIC 2075/259/260; CVC 489/61/62 |
| `cvc_heldout_eval.json` | **Table 2 CVC column** (62-test) | CNN 0.960->0.000 (-100%); SSM 0.912->0.244 (-73.2%); TSA -73.2%; ViT 0.824->0.569 (-30.9%) |
| `isic_heldout_eval.json` | **Table 2 ISIC column** (260-test) | VM-UNet -10.3%; ResNet50(ISIC-recipe) -10.1%; ResNet50(CVC-recipe) 0.892->0.808 (-9.4%); Swin-UNet -66.5%; Swin-UNETR 0.890->0.884 (**-0.6%**) |
| `interventions/checkpoints/unet_isic_cvcrecipe_best.pth` | ISIC CNN retrain (CVC recipe, 100 ep) | best val 0.1106 @ep58 |
| `interventions/checkpoints/swinunetr_isic_cvcrecipe_best.pth` | ISIC ViT retrain (CVC recipe, 100 ep) | best val 0.1339 @ep37 |
| `paper_v2/` | draft + tables + figures (auto-generated) | Sections 3-6 |
| `audit_pipeline.py`, `cvc_audit`, `cvc_audit_256`, `verify_*.py` | pipeline audits | 6/6 audit passes (hook identity, valley determinism, checkpoint strict-load, resolution) |

### `results/paper/` — OLD SPIE-era package (superseded)
`figures/{diagram,fig2_identity_validation,fig3_whole_network,fig4_layerwise_importance,fig5_cutoff_sweep,fig6_avr_vs_importance,fig7_boundary_errors}.png`, `tables/table{2..7}_*.csv`, `metadata/*_manifest.json`. Generated for the earlier **SPIE Medical Imaging** framing (ISIC-50, canonical VM-UNet only) and **do NOT reflect the current cross-dataset paper** — dead for ISBI, kept only as historical artifacts.


## 2. Usage Classification

**USED in `paper_v2/ISBI_PAPER_DRAFT.md`:**
- `splits/*`, `cvc_heldout_eval.json`, `isic_heldout_eval.json`, `cvc_robustness_eval.json`, the two ISIC retrain checkpoints + the 4 CVC checkpoints, `paper_v2/*`.

**UNUSED-BUT-RELEVANT (supporting / mechanistic / could be cited or extended):**
- `experiment14` (causal basis of the SSM CVC row), `experiment15` (boundary-locus 86.5x - one line in 3.1), `experiment16` (size aggravator - one line in 3.1), `experiment2/3/4/6` (Phase-0 single-dataset causal chain - useful for related-work/prior-work), `experiment7/8/9` (CNN ISIC-50 predecessor), `experiment10/11/12` (CNN CVC-352 predecessor), `cross_arch_cvc_eval.json` (123-val cross-check), `input_space_*` (input-LP baselines feeding Table 3), `make_robustness_figures.py` + `cvc_sample_*.png` (example-mask figure), `robustness_stats.json` (CVC rows).

**SAFELY DEAD (superseded / negative / no-op):**
- `experiment14_seed123/2027` (no-op control - deleted from paper), `experiment2_whole_network` (superseded by exp2), `experiment17*` + `best-vmunet-cvc-reg` (regularization negative), `isic_cross_arch_eval.json` dev-50 numbers (superseded by held-out; retained only as disclosed reference), `robustness_stats.json` ISIC rows (dev-50), `results/paper/*` (old SPIE framing), `PHASE0_EXECUTIVE_SUMMARY.md` (older phase summary), the 15-block-checkpoint-era artifacts in `SpectralMamba/results` (pre-topology-fix).

## 3. Line-by-line Number Verification (draft vs source JSON)

**Method:** `audit_isbi_numbers.py` parses the draft's three markdown tables numerically (normalizes Unicode minus/arrows) and compares every cell against the source JSONs (tolerance 5e-4 for Dice/BF1, 0.05 for HD95 px, 0.1 for %).

**Result: 76/76 numeric cells OK, 0 issues.**

- **Table 1** (28 cells: 4 models x {clean + 6 cutoffs}) vs `cvc_robustness_eval.json` -> all exact.
- **Table 2** (18 cells: 3 matched legs x {CVC clean/LP/d%, ISIC clean/LP/d%}) vs `cvc_heldout_eval.json` + `isic_heldout_eval.json` -> all exact **after one fix**.
- **Table 3** (32 cells: 4 models x {clean, input, feat, d%, BF1-clean, BF1-LP, HD95-clean, HD95-LP}) vs `cvc_robustness_eval.json` -> all exact.
- **Abstract claims** (-100%, -73%, -31%, -9.4%, -10.3%, -0.6%, 86.5, -66, 10.6, 7.1, 52) -> all present.

**The one issue found & fixed (2026-08-13):** draft Table 2 listed the VM-UNet CVC feature-LP as "0.913 -> 0.245". The JSON value is 0.9125/0.2445; the auto-generated CSV (`export_isbi_package.py`, Python `round`, banker's rounding) gives **0.912/0.244**. The draft (hand-written, round-half-up) now matches the CSV. Same cell fixed in `ISBI_SKELETON_v2.md`.

**Rounding-convention note for the paper:** the auto-generated CSVs use Python `round` (banker's rounding), so 0.9125 -> 0.912 and 0.2445 -> 0.244. When writing the LaTeX tables by hand, use the CSV values verbatim.

## 4. 95% Bootstrap CIs on mean per-image Delta-Dice @ rho=0.25 (B=10,000, seed 42)

NEW - computed by `audit_isbi_numbers.py` from the per-image arrays in the held-out JSONs.

| Dataset | Model | mean Delta | 95% CI | pooled d% |
|---|---|---|---|---|
| CVC (62) | VM-UNet | +0.6989 | [+0.6393, +0.7551] | -73.2% |
| CVC (62) | VM-UNet-TSA | +0.6983 | [+0.6390, +0.7566] | -73.2% |
| CVC (62) | ResNet50-UNet | +0.9558 | [+0.9476, +0.9627] | -100.0% |
| CVC (62) | Swin-UNETR | +0.3037 | [+0.2467, +0.3626] | -30.9% |
| ISIC (260) | VM-UNet | +0.0928 | [+0.0688, +0.1180] | -10.3% |
| ISIC (260) | ResNet50 (ISIC recipe) | +0.0949 | [+0.0762, +0.1150] | -10.1% |
| ISIC (260) | ResNet50 (CVC recipe) | +0.1357 | [+0.1167, +0.1552] | -9.4% |
| ISIC (260) | Swin-UNet | +0.6363 | [+0.6131, +0.6583] | -66.5% |
| ISIC (260) | Swin-UNETR | +0.0099 | [+0.0051, +0.0148] | -0.6% |

All CIs exclude zero -> all per-image drops significant (consistent with the Wilcoxon p < 1e-19 reported). Swin-UNETR on ISIC, while small (-0.6% pooled / mean +0.0099), still has a CI bounded away from zero.


## 5. Qualitative Example-Mask Figures (clean vs feature-LP)

**Status: CURRENT and USABLE for both datasets, all at 300 dpi.**
- `interventions/experiments/make_robustness_figures.py` generates **Input / prediction
  grids under Clean, Input-LP (0.25), Feature-LP (0.25)** for the 4 CVC models
  (current checkpoints) -> `results/figures/cvc_sample_{10,40,80}.png` (4470x2821, 300 dpi)
  + `vmunet_feature_spectrum.png`.
- `interventions/experiments/make_isic_robustness_figures.py` mirrors that style for the
  5 ISIC held-out models (VM-UNet, ResNet50-UNet ISIC/CVC recipes, Swin-UNETR,
  Swin-UNet) on 3 held-out test images chosen by lesion size (small/medium/large) ->
  `results/paper_v2/figures/isic_sample_{small,medium,large}.png` (5370x2821, 300 dpi),
  using the exact `eval_isic_heldout.py` protocol (normalization, per-model resize,
  hooks, apply_tsa) so the panels match the audited numbers.
- `cvc_audit/sample_*.png` and `cvc_audit_256/sample_*.png` are audit strips (1920x480),
  not paper figures. `paper/figures/fig*.png` are the OLD SPIE-era figures (stale models)
  - do not reuse.

## 6. Regeneration Order (checkpoints -> tables/figures)

### Script chain (in order)
1. **Splits:** `python interventions/experiments/carve_splits.py` -> `results/splits/{isic,cvc}_split.json`.
2. **Training** (checkpoints already built; re-run only if retraining):
   - CVC CNN: `tta_boundary_study/src/train_cvc_256.py`
   - CVC SSM: `interventions/train_vmunet_cvc.py` (352px, checkpointing + AMP)
   - CVC TSA: `interventions/train_vmunet_cvc_tsa.py`
   - CVC ViT: `interventions/train_swinunetr_cvc_256.py`
   - ISIC CNN: `interventions/train_unet_isic18_cvcrecipe.py` (done; best val 0.1106)
   - ISIC ViT: `interventions/train_swinunetr_isic18_cvcrecipe.py` (done; best val 0.1339)
   - ISIC SSM canonical: `interventions/train_vmunet_isic18_cvcrecipe.py --subset 490` (**deferred**; legacy checkpoint used instead)
3. **CVC robustness (Tables 1 & 3 source):** `python interventions/experiments/eval_cvc_robustness.py` -> `cvc_robustness_eval.json` (dose-response, degradations, boundary; 123-val).
4. **CVC held-out (Table 2 CVC column):** `python interventions/experiments/eval_cvc_heldout.py` -> `cvc_heldout_eval.json`.
5. **ISIC held-out (Table 2 ISIC column):** `python interventions/experiments/eval_isic_heldout.py --models all` -> `isic_heldout_eval.json`.
6. **Stats:** `python interventions/experiments/audit_isbi_numbers.py` (bootstrap CIs + draft verification); `summarize_inversion.py` (Table-2 markdown rows).
7. **Paper package:** `python interventions/experiments/export_isbi_package.py` -> `paper_v2/tables/*.csv` + `paper_v2/figures/*.png` (300 dpi).
8. **Qualitative figures (all 300 dpi):** `python interventions/experiments/make_robustness_figures.py` -> `results/figures/cvc_sample_*.png`; `python interventions/experiments/make_isic_robustness_figures.py` -> `paper_v2/figures/isic_sample_*.png`.

### Manual steps / caveats
1. **No manual JSON merge needed:** `eval_isic_heldout.py --models all` (default) writes the complete 5-model `isic_heldout_eval.json` in one pass; partial runs use `--merge_existing` to upsert rows into the existing file (dedupe by name, verified end-to-end).
2. **All figures are 300 dpi** (both figure scripts); no re-run needed.
3. **GPU constraint:** the held-out evals OOM/starve if run concurrently with training on the 6 GB GPU (run on an idle GPU).
4. **Cache:** `interventions/cache/isic256/*.pt` is gitignored - a fresh checkout must rebuild it (first `ISICCacheDataset` use) or copy it over; without it, the raw 29 MP ISIC images are decoded per epoch.
5. **Canonical VSSM path:** always use `SpectralMamba/models/vmunet/vmamba.py` (canonical vectorized scan), never the `SpectralMamba/VM-UNet/...` copy (legacy JIT loop).
6. **Splits must be seed-42; the 123-val used by Tables 1/3 is `CVCDataset`'s own split, not the carved 62-test.**
7. **No `mamba-ssm` dependency** (pure-PyTorch fallback); `einops` + `timm` required.

---

*Audit artifacts: `interventions/experiments/audit_isbi_numbers.py` (rerunnable), `paper_v2/REPO_AUDIT_REPORT.md` (this file). Commit of the one draft fix: see git log.*

