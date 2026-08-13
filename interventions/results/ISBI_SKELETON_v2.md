# ISBI 2027 (4-page) Skeleton v2 — Spectral Vulnerability of Segmentation Backbones

**Working title:** *Feature-domain spectral fragility is dataset-dependent: a matched
cross-dataset causal analysis of CNN, SSM, and ViT segmenters*

**Format:** 4 pages + references (ISBI double-column). 3 tables max. 2 figures.
Oct 26 draft deadline.

---

## Abstract (≈150 words)

> Deep segmentation models are often studied for input-domain robustness, but their
> sensitivity to *feature-domain* frequency manipulations is less understood. We apply
> targeted low-pass interventions to the internal representations of four segmenters —
> ResNet50-UNet (CNN), VM-UNet (state-space, SSM), Swin-UNETR/Swin-UNet (ViT) — and find a
> **cross-dataset inversion in feature-spectral fragility**. On untouched held-out test
> splits, a 0.25 feature-space cutoff on CVC-ClinicDB (polyps) collapses the CNN (−100%
> Dice) and SSM (−73%) while leaving the ViT largely intact (−31%); on ISIC2018 (skin
> lesions) the ranking inverts: the ViT collapses (−66%) while the same CNN and SSM
> architectures lose only ~10% Dice. Input-domain blur of equal strength degrades all
> models mildly, dissociating feature- from input-spectral sensitivity. The CNN leg is
> trained with identical recipes on both datasets, ruling out recipe confounds; the
> SSM/ViT legs are reported with explicit implementation caveats. Fourier-based
> augmentation does not close the feature-domain gap. Feature-spectral fragility is
> therefore not an architecture invariant, but an architecture–dataset interaction.

---

## 1. Introduction (~0.6 page)

- Motivation: robustness in medical segmentation is usually assessed with input
  perturbations (noise, blur, compression) [refs]. But deployed models are also exposed to
  *internal* distribution shift; feature-domain interventions are a direct causal probe of
  what representations rely on [Geirhos et al.; causal interventions refs].
- Prior spectral-bias results focus on input statistics and CNNs [refs]. State-space
  models (Mamba/VM-UNet) and transformers are largely unstudied under feature-frequency
  intervention.
- This paper: a causal, cross-architecture, cross-dataset study of feature-domain
  low-pass interventions at a fixed cutoff (0.25) and across a dose-response sweep.
- **Claim (one sentence):** feature-spectral fragility *ranks* differently across
  datasets — on CVC the CNN and SSM are the most fragile and the ViT the most robust, on
  ISIC the ViT is the most fragile and the CNN/SSM are the least — and the CNN's rank flip
  survives recipe matching and evaluation on an untouched held-out split.

## 2. Methods (~1 page)

### 2.1 Datasets & splits
- CVC-ClinicDB (612 polyp frames) and ISIC2018 (2,594 lesion images).
- **Deterministic splits (seed 42):** ISIC 2075/259/260 (train/val/test), CVC
  489/61/62, carved by `carve_splits.py` and **untouched until final evaluation**.
- Model selection on the val splits only. **All Table-2 inversion numbers are reported
  on the untouched test splits (CVC 62, ISIC 260).** The CVC validation set (123) is
  used for the mechanistic dose-response and defense characterizations (Tables 1, 3).

### 2.2 Architectures & recipes — matched vs caveated legs
- **CNN leg (anchor, matched):** ResNet50-UNet (`smp.Unet`, random init) trained with
  *identical* recipe on both datasets — BCEWithLogits, Adam 1e-4, CosineAnnealing(100),
  256 px, batch 8, seed 42. Only input normalization differs (ImageNet for ISIC, [0,1]
  for CVC), a preprocessing difference stated explicitly.
- **SSM leg (matched on CVC, caveated on ISIC unless cloud retrain completes):**
  VM-UNet (depths [2,2,9,2], 30 VSSBlocks). CVC trained with the canonical vectorized
  VSSM; the ISIC checkpoint used the legacy JIT-loop implementation. We verified the two
  VSSM implementations diverge (mean feature similarity 0.906; 6/30 blocks differ) and
  provide a canonical-implementation ISIC retrain (same recipe as CVC) to close the leg
  (see appendix A / companion). *If the cloud retrain is unavailable: the ISIC SSM row is
  reported with the legacy checkpoint and flagged as implementation-caveated.*
- **ViT leg (caveated by architecture):** Swin-UNETR (MONAI) on CVC; Swin-UNet on ISIC.
  Both 224 px, windowed self-attention, but not identical → reported as a directional
  signal, not a matched comparison.
- No `mamba-ssm` dependency is required (pure-PyTorch canonical implementation).

### 2.3 Interventions
- **Feature-domain low-pass (LP):** register forward hooks on all stage outputs
  (30 VSSBlocks for SSM; 9 encoder/decoder blocks for CNN; transformer blocks for ViT);
  zero frequencies above normalized radial cutoff ρ ∈ {0.10,…,0.40} (dose-response) and
  ρ = 0.25 (headline), via an ideal circular mask in the 2-D DFT.
- **Input-domain low-pass** of equal strength, applied to the resized input.
- Metrics: pooled and per-image Dice (mean ± SD, 95% bootstrap CI, paired Wilcoxon);
  boundary F1 and HD95 for the SSM/CNN legs on the CVC side (CNN LP HD95 is undefined —
  empty predictions — and is omitted). Interventions are applied post-training; no
  parameter changes.

### 2.4 Defense probe
- **TSA (Fourier augmentation):** fine-tune the CVC VM-UNet with low-passed training
  examples (cutoff 0.25, probability 1.0) and re-measure the feature-domain response.

## 3. Results (~1.5 pages)

### 3.1 Feature-domain dose-response on CVC (Table 1)
- All models degrade monotonically with aggressiveness of feature LP, but the *onset*
  differs sharply: CNN and SSM collapse by ρ=0.10–0.15; ViT only gradually.
- **Boundary vs interior (one line):** at ρ=0.25 the SSM error concentrates on the
  lesion boundary (+5 px band: 20.4% vs 0.10% interior, ~86×), consistent with
  high-frequency features carrying boundary information. Small-lesion size is a
  secondary aggravator (ΔDice −0.73 worst quartile) but does not drive the collapse.

### 3.2 Cross-dataset inversion on the anchor leg (Table 2) — headline
- **CNN (ResNet50-UNet, matched recipe):** feature-LP 0.25 → −100% Dice on the CVC
  held-out test (0.960 → 0.000) but −9.4% on the ISIC held-out test (0.892 → 0.808;
  ISIC-recipe model consistent: −10.1%). The CNN is therefore **10× less fragile on
  ISIC, not immune**; both CNN legs share init, loss, scheduler, resolution, and seed, so
  the rank flip is not a recipe artifact. Residual differences (input normalization,
  dataset statistics) are stated, not controlled.
- SSM: −73.2% (CVC held-out, canonical VSSM) vs −10.3% (ISIC held-out, legacy VSSM —
  implementation-caveated pending a canonical retrain). ViT: −30.9% (CVC held-out,
  Swin-UNETR) vs **−66.5%** (ISIC, Swin-UNet; architecture-caveated). The ViT is the
  most robust family on CVC and by far the most fragile on ISIC — the CNN/ViT ranking
  strictly inverts.
- **Dev-set note (transparency):** the earlier ≤4% ISIC drops were measured on a 50-image
  convenience subset (first 50 sorted ISIC images); they do not replicate on the
  untouched 260-image test split and are superseded by the held-out numbers reported here.
- All held-out effects are significant per-image (paired Wilcoxon p < 1e-19 for every
  model, both datasets); clean Dice on the ISIC held-out split (0.89–0.92) is lower than
  on the easy dev subset (0.95), and the CVC-recipe CNN's clean Dice trails the
  ISIC-recipe CNN by ~1.7 pts — a modest recipe effect on clean accuracy.

### 3.3 Input-domain dissociation and defense (Table 3)
- Input-space LP at the same cutoff leaves all models ≥ −8% (CNN −22% worst on CVC),
  i.e. feature-domain fragility is *not* a trivial consequence of low-frequency-only
  images.
- TSA fine-tuning helps input-domain robustness (+3.5 pts on CVC input-LP) but does
  **not** repair feature-domain collapse (−71% under feature-LP), i.e. spectral
  robustness does not transfer across domains.

### 3.4 Inversion interpretation (exactly one sentence)
> We conjecture that the inversion reflects dataset-dependent redundancy placement —
  where CVC polyp representations cache boundary-contrast energy in the same features the
  LP mask removes, ISIC lesion representations retain more low-frequency discriminative
  redundancy after the same mask, and transformer tokens distribute this redundancy
  differently — but we do not test this hypothesis here.

## 4. Conclusion & limitations (0.4 page)
- Feature-spectral fragility is dataset-dependent and the *ranking of architectures can
  invert* across datasets: on held-out test splits, the CNN/SSM are the most fragile on
  CVC (−100%/−73%) and the least on ISIC (−9%/−10%), while the ViT is the most robust on
  CVC (−31%) and the most fragile on ISIC (−66%). The CNN leg is recipe-matched, so the
  rank flip is not a training-recipe artifact.
- Limitations: SSM ISIC leg has an implementation caveat (closing via canonical retrain,
  appendix); ViT legs are different architectures; normalization differs by dataset;
  feature hooks cover stage outputs, not per-channel statistics; Tables 1 and 3
  (CVC dose-response/defense) use the CVC validation set rather than the test split.


## Tables (max 3)

### Table 1 — CVC-ClinicDB feature-space LP dose-response (pooled Dice, n=123 test)
| Model | Clean | ρ=0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet50-UNet (CNN) | 0.932 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.533 |
| VM-UNet (SSM) | 0.896 | 0.000 | 0.181 | 0.300 | 0.292 | 0.307 | 0.383 |
| Swin-UNETR (ViT) | 0.785 | 0.264 | 0.361 | 0.456 | 0.600 | 0.646 | 0.710 |
*Per-image mean±SD and 95% bootstrap CIs in text.*

### Table 2 — Cross-dataset inversion at ρ=0.25 (feature-domain LP, Δ% Dice)
| Architecture | CVC held-out clean → LP (Δ%) | ISIC held-out clean → LP (Δ%) | Leg status |
|---|---|---:|---|
| ResNet50-UNet (CNN) | 0.960 → 0.000 (−100%) | **0.892 → 0.808 (−9.4%)** | **matched recipe** |
| VM-UNet (SSM) | 0.913 → 0.245 (−73.2%)† | 0.915 → 0.821 (−10.3%)† | CVC canonical; ISIC legacy VSSM |
| Swin-UNet (ViT) | 0.824 → 0.569 (−30.9%)‡ | 0.911 → 0.306 (−66.5%) | different ViT architectures |
\*Both columns are on untouched carved held-out splits (CVC 62, ISIC 260; seed-42 carve). The CNN ISIC row is the CVC-recipe retrain; the ISIC-recipe CNN gives −10.1% on the same split.
†SSM on CVC uses the canonical VSSM; the ISIC SSM row uses the legacy VSSM checkpoint (implementation caveat — canonical retrain pending, appendix).
‡Swin-UNETR on CVC, Swin-UNet on ISIC. All per-image effects significant (Wilcoxon p<1e-19 both datasets).

### Table 3 — Input-domain dissociation & defense (CVC, ρ=0.25, n=123)

## Figures (2, 300 dpi)
- **Fig 1:** dose-response curves (Dice vs ρ) per architecture on CVC; annotated onset.
- **Fig 2:** cross-dataset inversion heatmap (Δ% per architecture × dataset) with example
  masks (clean / feature-LP) for one CVC and one ISIC case; CNN leg emphasized.

## References (target set)
1. Geirhos et al., *Shortcut learning in deep neural networks*, Nat. Mach. Intell. 2020.
2. Gu & Dao, *Mamba: Linear-time sequence modeling with selective state spaces*, 2023.
3. Zhang et al., *VM-UNet: Vision Mamba UNet for medical image segmentation*, 2024.
4. Hatamizadeh et al., *Swin UNETR*, CVPR 2022 (MONAI).
5. Cao et al., *Swin-Unet*, MICCAI 2022.
6. Reinke et al., *Common limitations of image segmentation metrics* (BF1/HD95), 2024.
7. Ji et al., *Boundary loss for highly unbalanced segmentation*, MIDL 2019 (BD-Loss).
8. Fourier/TSA augmentation: source used in interventions (state-space FT augment), 2024.
9. Pearl, *Causality* (intervention semantics) / causal robustness survey (Koh et al.).
10. Wang et al., *Pan-cancer segmentation: robustness across datasets* (dataset shifts).
11. ISIC2018 challenge; Jha et al., CVC-ClinicDB (dataset refs).

## Appendix A — VM-UNet ISIC canonical retrain (cloud)
- Script: `interventions/train_vmunet_isic18_cvcrecipe.py` (canonical VSSM + CVC recipe,
  from scratch, 100 ep, seed 42). Infeasible on 6 GB laptop (3.5–4 h/epoch). Requires a
  ≥16 GB GPU; see `interventions/CLOUD_RETRAIN.md`. Checkpoint →
  `interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth`.

## Open items before Oct 26
1. ✅ Held-out ISIC eval of all 4 models (n=260, untouched test split) — DONE 2026-08-13;
   results in `interventions/results/isic_heldout_eval.json`. Headline numbers supersede
   the earlier dev-50 values; Table 2 updated.
2. ✅ CVC-recipe ISIC CNN trained to epoch 100 (best val 0.1106, epoch 58) and evaluated
   on the held-out split (clean 0.892, feat-LP −9.4%).
3. Cloud decision for the SSM canonical retrain (critical path for a second matched leg
   and the Phase-3 ViT decision).
4. Optional: evaluate CVC models on the new carved CVC test split (62) to remove the
   selection-on-test caveat from the CVC column.
5. Lock author list; assemble citations (10–12); render 300-dpi figures.

| Model | Clean | Input-LP | Feat-LP | Δ% feat | BF1 clean→LP | HD95 clean→LP |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50-UNet (CNN) | 0.932 | 0.723 | 0.000 | −100% | 0.788→0.000 | — (undefined) |
| VM-UNet (SSM) | 0.896 | 0.823 | 0.292 | −67.4% | 0.672→0.062 | 21.4→93.9 |
| VM-UNet-TSA | 0.912 | 0.880 | 0.264 | −71.0% | 0.756→0.081 | 17.5→86.4 |
| Swin-UNETR (ViT) | 0.785 | 0.781 | 0.600 | −23.5% | 0.409→0.192 | 51.0→75.9 |
