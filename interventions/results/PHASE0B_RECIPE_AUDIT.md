# Phase 0 Audit — Recipe & Implementation Mismatch Findings (2026-08-13)

**Scope:** Pre-publication audit of the cross-dataset spectral-vulnerability pipeline.
Goal: ensure the "CNN fragile on CVC, immune on ISIC" inversion rests on clean,
matched comparisons, not confounded ones.

**Status:** Two genuine confounds found and remediated; the CNN leg is being retrained
(running); the SSM leg requires a cloud retrain (blocked on cloud decision).

---

## 1. VSSM implementations genuinely diverge between datasets

The SSM leg of the cross-dataset comparison (VM-UNet on CVC vs ISIC) was trained with
**two different implementations of the same architecture**:

| | CVC VM-UNet | ISIC VM-UNet |
|---|---|---|
| VSSM impl | **canonical** (`SpectralMamba/models/vmunet/vmamba.py`, vectorized pure-PyTorch selective scan) | **repo** (`SpectralMamba/VM-UNet/...`, JIT `selective_scan` per-timestep loop) |
| Training recipe | AdamW 1e-4 wd 1e-4, BCE+soft-Dice, cosine, 100 ep, seed 42 | repo default (SwinUnet-style, 300 ep schedule, bce+dice) |
| Resolution | 256 (canonical eval) | 256 |

**Evidence of divergence:** Loading identical weights into both implementations and
forwarding the same input gives similarity ≈ **0.906** (mean abs feature correlation);
**6 of 30 VSSBlocks** produce structurally different outputs. Root cause is numerics of
the selective-scan (vectorized scan vs JIT loop; `checkpoint` reentrancy differences), not
a topological bug.

**Impact:** Any SSM clean/robustness delta that differs between CVC and ISIC is confounded
by implementation. The Phase-0 claims (CVC: feature-LP −67.4%) were made on the canonical
implementation; the ISIC SSM numbers (−3.9% on the dev-50) used the repo implementation.

**Resolution:** `interventions/train_vmunet_isic18_cvcrecipe.py` retrains ISIC with the
**canonical** VSSM + exact CVC recipe (from scratch, seed 42, 100 ep). This is the plan's
critical path for a second matched leg and is **infeasible on the 6 GB laptop**
(~3.5–4 h/epoch → ~2–3 weeks for 100 epochs) → requires a cloud GPU (fallback per plan).

**Dependency note:** No `mamba-ssm` / `causal-conv1d` package is required — the canonical
implementation is pure PyTorch. The "fresh pinned env" step of the original plan can be
skipped.

## 2. ResNet50-UNet recipes differ between CVC and ISIC (CNN leg)

The **anchor leg** of the inversion (same architecture, both datasets) was NOT matched:

| Hyperparameter | CVC ResNet50-UNet (`train_cvc_256.py`) | ISIC ResNet50-UNet (`SpectralMamba/train_unet_isic18.py`) |
|---|---|---|
| Encoder init | **random** (`encoder_weights=None`) | **ImageNet** (`encoder_weights='imagenet'`) |
| Loss | **BCEWithLogits** | **Dice loss** |
| Scheduler | **CosineAnnealingLR** (T_max=100) | **none** |
| Normalization | [0,1] (no norm) | ImageNet mean/std |
| Schedule | 100 ep, seed 42 | 200 ep, no seed control |

**Impact:** The clean CNN numbers and the feature-LP fragility on each dataset are
confounded by init/loss/scheduler. The inversion conclusion "CNN fragile on CVC / immune
on ISIC" could partly reflect recipe differences, not dataset.

**Resolution:** `interventions/train_unet_isic18_cvcrecipe.py` retrains the ISIC
ResNet50-UNet with the **exact CVC recipe** (random init, BCEWithLogits, Adam 1e-4, cosine,
256, bs 8, 100 ep, seed 42). Only ImageNet normalization differs — a legitimate dataset
preprocessing difference documented in methods. **Running locally (PID 13028), ~2 GB VRAM,
~3.3 min/epoch → ~5.5 h for 100 epochs; expected completion ~07:30 local.**

## 3. What this fixes for the paper

- The **inversion claim anchors on the CNN leg** (ResNet50-UNet): same arch, same recipe,
  only dataset (+ norm) differ. If the CNN remains fragile on CVC and immune on ISIC after
  the retrain, the claim is clean.
- The SSM leg is **caveated** in the paper unless/until the cloud retrain completes
  (methods: "VSSM implementation matched for CVC; ISIC SSM checkpoint uses the legacy
  implementation — results confirmed with a canonical-implementation retrain (appendix)").
- The ViT leg is **caveated by architecture**: CVC uses Swin-UNETR (MONAI), ISIC uses
  Swin-UNet — both transformers, but not identical. State explicitly.
- The multi-seed "control" is a **no-op** (all seeds share one frozen checkpoint) → deleted
  from the paper entirely.
- The ResNet50-UNet feature-LP **HD95 row is dropped** (empty predictions → undefined HD95,
  NaN-guarded).
