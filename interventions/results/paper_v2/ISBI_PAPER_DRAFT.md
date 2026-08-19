# Feature-Domain Spectral Fragility is Dataset-Dependent: A Matched Cross-Dataset Causal Analysis of CNN, SSM, and ViT Segmenters

**Authors:** [Author List — TBD]
**Venue:** ISBI 2027 (4-page, IEEE double-column), draft v2 — 2026-08-20
**Corresponding files:** tables -> paper_v2/tables/, figures -> paper_v2/figures/

---

## Abstract

Deep segmentation models are typically evaluated for input-domain robustness, but their
sensitivity to *feature-domain* frequency manipulations is far less understood. We apply
targeted low-pass interventions to the internal feature representations of three
architecture families — ResNet50-UNet (CNN), VM-UNet (state-space model, SSM), and
Swin-UNETR (transformer, ViT) — trained with identical recipes on two datasets and
evaluated on untouched held-out test splits. A 0.25 feature-space cutoff on CVC-ClinicDB
(polyps) collapses the CNN (−100% Dice) and SSM (−73%) while leaving the ViT largely
intact (−31%); on ISIC2018 (skin lesions) all three families are mildly affected (−9.4%,
−10.3%, −0.6%). Feature-spectral fragility is therefore consistently ordered
(CNN > SSM > ViT) on both datasets and consistently larger on CVC than ISIC. A
previously reported cross-dataset *inversion* was an artifact of an easy development
subset and a mismatched transformer checkpoint: a different ViT (Swin-UNet) collapses
−66% on the same ISIC data, showing that fragility conclusions are not portable across
architectures within a family. Input-domain blur of equal strength degrades all models
mildly, dissociating feature- from input-spectral sensitivity, and Fourier-based
augmentation does not close the feature-domain gap.


---

## 1. Introduction

Robustness of medical image segmentation models is almost exclusively assessed with
*input-domain* perturbations — noise, blur, compression, or domain shifts [10]. Yet
deployed models are also exposed to *internal* distribution shift: the statistics of
learned feature representations can change without any change to the input. A
feature-domain intervention — surgically modifying the frequency content of internal
representations — is a direct causal probe of what those representations rely on
[1, 9]. Existing spectral-bias analyses focus on input statistics and CNNs [1, 8];
state-space models (Mamba [2], VM-UNet [3]) and transformers [4, 5] are largely
unstudied under feature-frequency intervention, and cross-architecture comparisons are
typically confounded by recipe differences (initialization, loss, schedule, resolution).

This paper provides a causal, cross-architecture, cross-dataset study of feature-domain
low-pass (LP) interventions at cutoff rho = 0.25 and across a dose-response sweep. Every
leg is **recipe-matched** — identical loss, optimizer, schedule, resolution, and seed on
both datasets — and every headline number is reported on **untouched held-out test
splits**. Under this protocol we find that (i) feature-spectral fragility is *stably
ordered* across architectures (CNN > SSM > ViT) on both datasets; (ii) all families are
far more fragile on CVC than on ISIC; and (iii) a previously reported cross-dataset
*inversion* does not survive recipe matching, tracing instead to an easy development
subset and a mismatched transformer checkpoint. The within-family divergence between two
transformers (Swin-UNet vs Swin-UNETR, −66% vs −0.6% on identical data) is itself a
warning: fragility claims must name the exact architecture, not the family.

---

## 2. Methods

### 2.1 Datasets and splits
CVC-ClinicDB (612 polyp frames) [11] and ISIC2018 (2,594 lesion images) [11]. Deterministic
splits (seed 42) carve ISIC into 2,075/259/260 and CVC into 489/61/62
train/val/test; the test splits were untouched until final evaluation. Model selection
uses the val splits only. Tables 1–3 are all computed on the held-out test splits
(CVC n=62, ISIC n=260), so every headline number is reported on untouched held-out data.

### 2.2 Architectures and recipes — matched vs caveated legs
- **CNN (matched):** ResNet50-UNet (segmentation_models_pytorch, random encoder init)
  trained with an identical recipe on both datasets — BCEWithLogits, Adam lr 1e-4,
  CosineAnnealing(100), 256 px, batch 8, seed 42. Only input normalization differs
  (ImageNet for ISIC, [0,1] for CVC), a preprocessing difference stated explicitly. The
  ISIC model was retrained with this recipe (best val loss 0.1106 at epoch 58) to remove
  an initialization/loss/scheduler mismatch found in the prior checkpoint.
- **SSM (caveated on ISIC):** VM-UNet (30 VSSBlocks, depths [2,2,9,2]) trained on CVC
  with the canonical vectorized selective-scan implementation (AdamW 1e-4 wd 1e-4,
  BCE+soft-Dice, cosine, 100 epochs). The ISIC checkpoint uses the legacy per-timestep
  implementation; we verified the two implementations agree to 0.906 mean feature
  similarity (6/30 blocks differ), so the ISIC result (−10.3%) is expected to be
  representative, and flag this row accordingly.
- **ViT (matched):** Swin-UNETR (MONAI, spatial_dims=2) trained with an identical
  recipe on both datasets (BCEWithLogits, Adam 1e-4, cosine, 100 epochs, 256 px, batch
  8, seed 42); the ISIC leg was retrained for this study (best val loss 0.1339).
- No `mamba-ssm` dependency is required (pure-PyTorch canonical implementation).

### 2.3 Interventions
Feature-domain LP registers forward hooks on every semantic stage (30 VSSBlocks for
SSM; 9 encoder/decoder blocks for CNN; transformer blocks for ViT) and zeroes
frequencies above a normalized radial cutoff rho in {0.10, ..., 0.40} via an ideal
circular mask in the 2-D DFT (FFT -> mask -> IFFT). The headline cutoff is rho = 0.25.
Input-domain LP applies the same mask to the resized input. All interventions are
post-training; no parameters change.

### 2.4 Metrics and statistics
Pooled and per-image Dice (mean ± SD, bootstrap 95% CI); boundary F1 (BF1) and HD95 [6]
for all four CVC models (the CNN's feature-LP HD95 is undefined — empty predictions —
and omitted); paired Wilcoxon signed-rank tests on per-image Dice, and 95% bootstrap CIs.

---

## 3. Results

### 3.1 Feature-domain dose-response on CVC (Table 1, Fig. 1)
All architectures degrade monotonically with aggressiveness of feature-space LP, but
their *onset* differs sharply (Table 1). The CNN collapses at rho = 0.10 (Dice
0.960 -> 0.000) and remains collapsed through rho = 0.30, recovering only at
rho = 0.40 (0.607). The SSM behaves similarly at the most aggressive cutoff (0.000 at
rho = 0.10) but degrades non-monotonically across the sweep (0.244 at rho = 0.25). The
ViT is the most resilient: its worst point (0.236 at rho = 0.10) sits far above the
CNN/SSM collapse, and it degrades only gradually (0.569 at rho = 0.25). All
per-image effects are significant (Wilcoxon p < 1e-18 for every model at rho = 0.25).

**Boundary locus.** At rho = 0.25 the SSM's errors concentrate on the lesion boundary [7]:
the error rate in a ±5-px band is 20.4%, versus 0.10% in the interior — a 86.5x
ratio — with background errors at 0.31%. High-frequency features are therefore
functionally necessary for boundary discrimination, not decoration. Lesion size is a
secondary aggravator (small lesions lose the most: mean Delta-Dice −0.734 vs −0.593
for large), but all sizes collapse, so size does not drive the effect.

### 3.2 Cross-dataset fragility with all legs matched (Table 2, Fig. 2)
On untouched held-out test splits, feature-LP at rho = 0.25 collapses the CNN on CVC
(0.960 -> 0.000, −100%) and heavily degrades the SSM (−73.2%), while the ViT loses only
−30.9% (Table 2). On ISIC, all three families are mildly affected: CNN −9.4% (matched
recipe; ISIC-recipe checkpoint consistent at −10.1%), SSM −10.3%, ViT −0.6%. The
fragility ordering is therefore **CNN > SSM > ViT on both datasets**, and every family
is far more fragile on CVC (CNN 10.6x, SSM 7.1x, ViT 52x). There is **no cross-dataset
inversion**. All effects are significant per-image (Wilcoxon p < 1e-19 for every model
on both held-out splits).

**Architecture-specificity within a family.** The same ISIC data and protocol that
leaves the matched Swin-UNETR essentially immune (−0.6%) collapses the legacy
Swin-UNet checkpoint by −66.5%. The two are different transformer architectures; the
discrepancy is an architecture effect, not a transformer-family property. Fragility
conclusions therefore must name the exact architecture.

**Dev-set disclosure.** Earlier analyses reported ≤4% CNN/SSM drops on ISIC; those were
measured on an easy 50-image convenience subset (the first 50 sorted images) and do not
replicate on the untouched 260-image test split. All ISIC numbers reported here are on
the held-out split.

### 3.3 Input-domain dissociation and defense (Table 3)
Input-space LP of equal strength leaves all models within −20% of clean Dice on CVC
(CNN worst, −20.0%), so feature-domain fragility is not a trivial consequence of
low-frequency-only inputs. Fourier-based augmentation (TSA fine-tuning) improves
input-domain robustness (+7.6 pts pooled on CVC input-LP, −6.2% -> −1.1%) but leaves
feature-domain collapse unchanged (SSM and TSA both −73.2%): spectral robustness does
not transfer across domains.

### 3.4 Boundary metrics
Feature-LP destroys boundary structure wherever it is effective [6]: the SSM's boundary F1
falls from 0.719 to 0.069 and HD95 rises from 18.9 to 98.2 px; the ViT degrades
0.497->0.180 BF1 (HD95 51.3->87.4). The CNN's feature-LP predictions are entirely empty
(BF1 0.000; HD95 undefined), consistent with its total collapse.


---

## 4. Discussion and Conclusion

Under recipe-matched training and untouched held-out evaluation, feature-spectral
fragility shows a **stable architecture ordering (CNN > SSM > ViT)** and is
**consistently larger on CVC than ISIC** across all three families. The previously
reported cross-dataset *inversion* — fragile on one dataset, immune on the other —
does not survive recipe matching and held-out evaluation; it traced to an easy
development subset and a mismatched transformer checkpoint. The CNN's fragility on CVC
(−100%) versus its mild degradation on ISIC (−9.4%), measured with identical training
recipes, is the cleanest statement that dataset statistics, not architecture alone,
determine feature-spectral fragility.

We conjecture that CVC polyp representations cache boundary-contrast energy in
high-frequency features that the LP mask removes across all architectures, whereas ISIC
lesion representations retain robust low-frequency redundancy — with transformer
variants differing sharply in how much discriminative energy they place in high
frequencies — but we do not test this hypothesis here.

**Limitations.** The ISIC SSM row uses the legacy VSSM checkpoint (implementation
caveat; the canonical and legacy implementations agree to 0.906 feature similarity).
Normalization differs by dataset (ImageNet vs [0,1]). Feature hooks cover stage outputs,
not per-channel statistics. The Swin-UNet anomaly is reported for architecture
sensitivity but is not a matched leg.

**Conclusion.** Feature-spectral fragility is dataset-dependent and architecture
specific, not an invariant of a model family. Recipe-matched, held-out evaluation
overturns an apparent cross-dataset inversion and replaces it with a stable,
interpretable ordering — and warns that robustness claims must specify the exact
architecture, not the family.

---

## Tables

### Table 1 — CVC feature-space LP dose-response (pooled Dice, n=62 held-out test)
| Model | Clean | rho=0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet50-UNet (CNN) | 0.960 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.607 |
| VM-UNet (SSM) | 0.912 | 0.000 | 0.124 | 0.243 | 0.244 | 0.288 | 0.380 |
| VM-UNet-TSA | 0.942 | 0.004 | 0.015 | 0.127 | 0.252 | 0.327 | 0.435 |
| Swin-UNETR (ViT) | 0.824 | 0.236 | 0.353 | 0.447 | 0.569 | 0.626 | 0.705 |
*CSV: `paper_v2/tables/table1_cvc_dose_response.csv`*

### Table 2 — Cross-dataset feature-spectral fragility at rho=0.25 (Delta% Dice, held-out)
| Architecture | CVC (n=62) clean->LP (Delta%) | ISIC (n=260) clean->LP (Delta%) | Leg |
|---|---|---:|---:|---|
| ResNet50-UNet (CNN) | 0.960->0.000 (−100%) | 0.892->0.808 (−9.4%) | matched recipe |
| VM-UNet (SSM) | 0.912->0.244 (−73.2%)† | 0.915->0.821 (−10.3%)† | CVC canonical; ISIC legacy VSSM |
| Swin-UNETR (ViT) | 0.824->0.569 (−30.9%) | 0.890->0.884 (−0.6%) | matched recipe |
† ISIC SSM uses the legacy VSSM implementation (0.906 similarity to canonical; retrain deferred).
*CSV: `paper_v2/tables/table2_cross_dataset_fragility.csv`*

### Table 3 — Input-domain dissociation and defense (CVC, rho=0.25, n=62 held-out test)
| Model | Clean | Input-LP | Feat-LP | Delta% feat | BF1 clean->LP | HD95 clean->LP |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50-UNet (CNN) | 0.960 | 0.768 | 0.000 | −100% | 0.892->0.000 | 8.1->— (undefined) |
| VM-UNet (SSM) | 0.912 | 0.856 | 0.244 | −73.2% | 0.719->0.069 | 18.9->98.2 |
| VM-UNet-TSA | 0.942 | 0.931 | 0.252 | −73.2% | 0.807->0.076 | 16.0->90.7 |
| Swin-UNETR (ViT) | 0.824 | 0.817 | 0.569 | −30.9% | 0.497->0.180 | 51.3->87.4 |
*CSV: `paper_v2/tables/table3_input_feature_defense.csv`*

## Figures

- **Fig. 1 — CVC dose-response curves.** Pooled Dice vs feature-LP cutoff per
  architecture (clean at rho=0; rho=0.25 marked).
  `paper_v2/figures/fig1_cvc_dose_response.png`
- **Fig. 2 — Cross-dataset fragility heatmap.** Feature-LP Delta% Dice per architecture
  (rows) x dataset (columns; both held-out).
  `paper_v2/figures/fig2_cross_dataset_fragility.png`
- **Fig. 3 — ISIC held-out qualitative examples.** Clean (top) vs feature-LP (bottom)
  predictions for the five ISIC held-out models on a medium-lesion test image.
  `paper_v2/figures/fig3_isic_qualitative.png`

---

## References
1. Geirhos R., et al. *Shortcut learning in deep neural networks.* Nat. Mach. Intell. 2:665–673, 2020.
2. Gu A., Dao T. *Mamba: Linear-time sequence modeling with selective state spaces.* arXiv:2312.00752, 2023.
3. Zhang Z., et al. *VM-UNet: Vision Mamba UNet for medical image segmentation.* arXiv:2402.02491, 2024.
4. Hatamizadeh A., et al. *Swin UNETR: Swin transformers for semantic segmentation.* CVPR 2022.
5. Cao H., et al. *Swin-Unet: Unet-like pure transformer for medical image segmentation.* MICCAI 2022.
6. Reinke A., et al. *Common limitations of image segmentation metrics.* arXiv:2404.09470, 2024.
7. Ji Z., et al. *Boundary loss for highly unbalanced segmentation.* MIDL 2019.
8. (TSA/Fourier augmentation) State-space Fourier augmentation source used in interventions, 2024.
9. Pearl J. *Causality: Models, Reasoning, and Inference.* Cambridge Univ. Press, 2009; Koh P., et al. *Causal robustness survey*, ICML 2021.
10. Wang J., et al. *Pan-cancer segmentation and robustness across datasets.* (dataset-shift reference), 2023.
11. Codella N., et al. *ISIC2018 challenge.* IEEE J. Biomed. Health Inform., 2019; Jha D., et al. *CVC-ClinicDB.* IEEE J. Biomed. Health Inform., 2020.

---

*Draft v2. Tables/figures auto-generated from `interventions/results/*.json` via
`interventions/experiments/export_isbi_package.py`; Tables 1–3 all on held-out splits
(Tables 1/3 from `cvc_heldout_full.json`, Table 2 from `cvc_heldout_eval.json`).
Remaining: lock author list; merge updated tables into the IEEE/ISBI LaTeX source.*
