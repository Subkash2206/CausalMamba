# Cloud Runbook — VM-UNet ISIC retrain (canonical VSSM + CVC recipe)

**Purpose:** Close the SSM leg of the cross-dataset inversion. The ISIC VM-UNet checkpoint
in the repo (`SpectralMamba/VM-UNet/best-ckpt/best-vmunet-scratch-isic18.pth`) was trained
with the *legacy* JIT-loop VSSM; the CVC VM-UNet uses the *canonical* vectorized VSSM. We
proved the two implementations diverge (feature similarity ≈ 0.906; 6/30 blocks differ).
To make the SSM leg implementation-matched, retrain ISIC with the canonical VSSM and the
exact CVC training recipe.

**Why not locally:** at 256 px, micro-batch 2 + grad-accum 4, the ~2,075-image ISIC train
split runs ~3.5–4 h/epoch → ~2–3 weeks for 100 epochs on the 6 GB laptop. A ≥16 GB GPU
finishes in ~4–8 h.

**No exotic deps:** canonical VSSM is pure PyTorch (`SpectralMamba/models/vmunet/vmamba.py`).
No `mamba-ssm`, `causal-conv1d`, or pinned-env rebuild needed. Requirements: `torch`,
`torchvision`, `numpy`, `PIL`, `timm`.

---

## Steps

1. **Transfer the repo** (or just `interventions/` + `SpectralMamba/models/vmunet/` +
   `SpectralMamba/VM-UNet/data/isic18/` + `interventions/results/splits/isic_split.json`).

2. **Verify the split** (should print `train=2075 val=259 test=260`):
   ```bash
   python -c "import json; s=json.load(open('interventions/results/splits/isic_split.json')); print({k:len(v) for k,v in s.items()})"
   ```

3. **Launch training** (GPU node, e.g. A100/L4/RTX 4090, ≥16 GB VRAM, ≥8 CPU threads):
   ```bash
   cd /path/to/repo
   python interventions/train_vmunet_isic18_cvcrecipe.py 2>&1 | tee interventions/logs/train_vmunet_isic_cvcrecipe.log
   ```
   - 100 epochs, from scratch, AdamW 1e-4 wd 1e-4, BCE + soft-Dice, CosineAnnealing,
     seed 42, effective batch 8, AMP + gradient checkpointing enabled by default.
   - Best val loss checkpoint → `interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth`.

4. **Expected health signals:**
   - Epoch time on a modern datacenter GPU: ≤ 5 min → full run ≤ 8 h.
   - Clean val Dice should converge toward 0.93–0.95 (ISIC 256 px, canonical VSSM);
     the legacy checkpoint reached 0.95 on a 50-image dev set.
   - If VRAM OOMs: set `MICRO_BATCH=1` (keeps effective batch via accumulation).

5. **Hand back the checkpoint** to the laptop repo and run the held-out eval:
   ```bash
   python interventions/experiments/eval_isic_heldout.py
   ```
   The eval auto-includes the model (named "VM-UNet (canonical VSSM, CVC recipe)") once the
   checkpoint exists — add it to `MODELS` in `eval_isic_heldout.py` if not auto-detected.

## What this unlocks
- A second **matched leg** (SSM: canonical implementation + identical recipe on both
  datasets) → the inversion table (Table 2) no longer carries the SSM implementation
  caveat.
- The **Phase-3 ViT decision**: with a clean SSM row, the ViT's ISIC-only collapse
  (−60%) vs SSM/CNN immunity (<4%) becomes interpretable as architecture-level rather
  than implementation-level.

## Verification checklist
- [ ] Split prints 2075/259/260.
- [ ] First epoch completes in ≤5 min on the cloud GPU.
- [ ] Final checkpoint exists; `torch.load(...)` with `strict=True` loads into
      `VMUNet(num_classes=1, input_channels=3, depths=[2,2,9,2], depths_decoder=[2,9,2,2],
      drop_path_rate=0.2, use_checkpoint=False)`.
- [ ] Held-out eval JSON contains the canonical SSM row.
