# Kaggle Runbook — ISIC VM-UNet canonical retrain (CVC-matched 490-image subset)

**Goal:** Train the canonical-VSSM VM-UNet on 490 ISIC images (100 epochs, CVC recipe)
to replace the legacy-VSSM ISIC SSM row in Table 2. ~30–40 GPU-hours on Kaggle
(~1–1.25 weeks of the free 30 h/week quota; can be shortened with the batch-4 patch).

**Package:** `causal_mamba_kaggle.zip` (~115 MB) — on your Desktop.
Contains (uncompressed): the training script, cached 256px ISIC tensors
(`train_490.pt` 490 MB, `val.pt` 259 MB), the split JSON, and the canonical
`vmamba.py`/`vmunet.py`. **No raw images, no mamba-ssm** (pure-PyTorch fallback).

---

## 1. Create the notebook
1. kaggle.com → **Notebooks** → **New Notebook**.
2. Right-hand **Settings**:
   - Accelerator: **GPU T4 x2** (P100 if offered)
   - Internet: **ON**

## 2. Upload + unzip + install deps
**Cell 1:**
```python
import zipfile, glob, os
os.chdir('/kaggle/working')
z = glob.glob('/kaggle/working/*.zip')
print(z)
if z:
    with zipfile.ZipFile(z[0], 'r') as fh:
        fh.extractall('/kaggle/working')
print(os.listdir('/kaggle/working'))
```
**Cell 2:**
```python
!pip install -q einops timm
```

## 3. (Optional, ~20–30% faster) batch 4 on the 16 GB GPU — effective batch stays 8
**Cell 3:**
```python
p = '/kaggle/working/interventions/train_vmunet_isic18_cvcrecipe.py'
s = open(p).read()
open(p, 'w').write(s.replace('MICRO_BATCH, ACC_STEPS = 2, 4', 'MICRO_BATCH, ACC_STEPS = 4, 2'))
print('patched to batch 4 / acc 2')
```

## 4. Run training
**Cell 4:**
```python
!python /kaggle/working/interventions/train_vmunet_isic18_cvcrecipe.py --subset 490
```
Each epoch prints `Epoch NN/100 | Train … | Val … | Dice … | Xh --> Saved best`.
Expected **~15–25 min/epoch on T4** → ~30–40 h for 100 epochs.

## 5. Session limits & RESUME (Kaggle sessions die after ~9 h)
- State (model+optimizer+scheduler+epoch) auto-saves every 5 epochs to
  `/kaggle/working/interventions/checkpoints/vmunet_isic_cvcrecipe_state_latest.pt`.
- When a session ends, grab that file (and `vmunet_isic_cvcrecipe_best.pth`) from the
  notebook version's **Output**.
- **Next session:** re-upload the zip + drop the state file back, then run:
```python
!python /kaggle/working/interventions/train_vmunet_isic18_cvcrecipe.py --subset 490 \
  --resume /kaggle/working/interventions/checkpoints/vmunet_isic_cvcrecipe_state_latest.pt
```
- Repeat until `Epoch 100/100` prints `Done. Best val loss …`.
- More robust: put the zip in a **private Kaggle Dataset** and Add Input each session
  (update the dataset with the new state file between sessions).

## 6. Bring the model home + evaluate
1. Download `interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth` (~177 MB).
2. Save it on the laptop as `interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth`.
3. Locally (GPU free):
   ```powershell
   python interventions/experiments/eval_isic_heldout.py --models all
   ```
4. The eval auto-includes the canonical SSM row → the legacy-VSSM asterisk on the ISIC
   SSM row disappears. Re-run `summarize_inversion.py` and update Table 2.

## Gotchas
- Internet ON or `pip install` fails.
- A mid-epoch interruption costs ≤1 epoch (state saves at epoch boundaries).
- Free-tier T4 quota (30 h/week) pauses the run until the weekly reset — the state file
  is safe.
- Do NOT run the batch-4 patch mid-run after resuming from a batch-2 state — keep the
  same batch config for the whole run (effective batch 8 either way).
