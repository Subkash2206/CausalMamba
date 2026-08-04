# Configuration Audit Report — interventions/

**Date:** 2026-08-04
**Status:** ✅ ALL VIOLATIONS FIXED (fixes verified via `py_compile` + audit re-scan)
**Scope:** All Python scripts under `interventions/` (experiments 2–6, `avr_analysis_intervention.py`).
**Constraints checked:**
1. `depths` must be explicitly `[2, 2, 9, 2]` on every VM-UNet instantiation.
2. `strict=True` must be used whenever weights are loaded via `torch.load` / `load_state_dict`.

---

## 0. Preliminary note

`interventions/utils/` is **empty** — there is no `model_config.py` or `utils.py` under
`interventions/`. Model configuration is inlined as a local `_NC` dict inside each experiment
script. The canonical reference configuration lives at
`SpectralMamba/models/vmunet/vmunet.py` (defaults `depths=[2, 2, 9, 2]`,
`depths_decoder=[2, 9, 2, 2]`), and every reviewer script under `SpectralMamba/` explicitly
uses `depths=[2,2,9,2]`, `depths_decoder=[2,9,2,2]`.

---

## 1. VM-UNet instantiations & `depths` verification

Legend: ✅ OK — `depths` explicitly set to `[2, 2, 9, 2]` (or canonical default used);
✅ FIXED — was non-canonical, now corrected to `[2, 2, 9, 2]`.

| File | Line(s) | Instantiation | `depths` used | Verdict |
|---|---|---|---|---|
| `experiments/avr_analysis_intervention.py` | 37 | `VMUNet()` | default `[2, 2, 9, 2]` (no override) | ✅ OK (uses canonical default, but see strict audit below) |
| `experiments/experiment2_real_lowpass.py` | `_NC` at 101; 173, 221, 315 | `VMUNet(...)` via `_NC` | `[2, 2, 9, 2]` (was `[2,2,2,2]`) | ✅ FIXED |
| `experiments/experiment3_layerwise.py` | `_NC` at 96; 131, 182, 251 | `VMUNet(**_NC)` | `[2, 2, 9, 2]` (was `[2,2,2,2]`) | ✅ FIXED |
| `experiments/experiment4_cutoff_sweep.py` | `_NC` at 101; 136, 184, 248, 287 | `VMUNet(**_NC)` | `[2, 2, 9, 2]` (was `[2,2,2,2]`) | ✅ FIXED |
| `experiments/experiment5_robustness.py` | `_NC` at 90; 120, 159, 261, 323 | `VMUNet(**_NC)` | `[2, 2, 9, 2]` (was `[2,2,2,2]`) | ✅ FIXED |
| `experiments/experiment6_dc_boundary.py` | `_NC` at 98; 125, 155, 235 | `VMUNet(**_NC)` | `[2, 2, 9, 2]` (was `[2,2,2,2]`) | ✅ FIXED |

### `depths_decoder` correction (secondary finding)

Every `_NC` dict has also been corrected from `depths_decoder=[2, 2, 2, 1]` to the
canonical `depths_decoder=[2, 9, 2, 2]` in experiments 2–6.

**Corroborating evidence the wrong topology was actually used:** the committed results in
`results/experiment2_whole_network/results.csv` list only **15** VSSBlocks
(`layers.0…layers.3` × 2 blocks + `layers_up.0…layers_up.3` minus one = 15), which matches the
`depths=[2,2,2,2]` / `depths_decoder=[2,2,2,1]` topology — NOT the canonical 20-block
`[2,2,9,2]` / `[2,9,2,2]` topology.

---

## 2. `strict=True` verification

PyTorch's `load_state_dict` defaults to `strict=True`, but the constraint requires it to be
**explicit**. **All** `load_state_dict` calls in the audited target files now pass explicit
`strict=True`:

| File | Line(s) | Status |
|---|---|---|
| `experiments/avr_analysis_intervention.py` | 45, 47, 49 | ✅ FIXED |
| `experiments/experiment2_real_lowpass.py` | 208, 228, 322 | ✅ FIXED |
| `experiments/experiment3_layerwise.py` | 137, 183, 252 | ✅ FIXED |
| `experiments/experiment4_cutoff_sweep.py` | 142, 185, 249, 288 | ✅ FIXED |
| `experiments/experiment5_robustness.py` | 126, 160, 262, 324 | ✅ FIXED |
| `experiments/experiment6_dc_boundary.py` | 131, 156, 236 | ✅ FIXED |

Note: several of these load a live `model_ref.state_dict()` (copying weights between models)
rather than a checkpoint file, but all now pass explicit `strict=True` per the constraint.
`torch.load` calls themselves (no `strict` kwarg exists there) are at
`experiment2_real_lowpass.py:184`, `experiment3_layerwise.py:134`,
`experiment4_cutoff_sweep.py:139`, `experiment5_robustness.py:123`,
`experiment6_dc_boundary.py:128`, `avr_analysis_intervention.py:43`.

**Out of scope (noted only):** `experiment0_identity_validation.py` and
`experiment1_intervention_validation.py` contain `load_state_dict` calls without explicit
`strict=True`, but they operate on `SimpleFeatureNet` (synthetic validation models), not
VM-UNet, and were not in the target file list. PyTorch defaults to `strict=True` there;
no behavioral change is expected.

---

## 3. Status of required fixes

1. ✅ **`_NC` dicts in experiments 2–6** — corrected:
   - `depths: [2, 2, 9, 2]` (was `[2, 2, 2, 2]`)
   - `depths_decoder: [2, 9, 2, 2]` (was `[2, 2, 2, 1]`)
   - Applied in `experiment2_real_lowpass.py`, `experiment3_layerwise.py`,
     `experiment4_cutoff_sweep.py`, `experiment5_robustness.py`,
     `experiment6_dc_boundary.py`.
2. ✅ **`strict=True` added** to every `load_state_dict` call in all 6 target files.
3. ✅ **Syntax verified** — all modified files pass `py_compile`.
4. ⏳ **Re-run experiments 2–6** — the previously committed results in
   `results/experiment2_whole_network/`, `results/experiment3_layerwise/`,
   `results/experiment4_cutoff_sweep/`, `results/experiment5_robustness/`,
   `results/experiment6_dc_boundary/` were produced with the wrong architecture and are
   invalid for the canonical `depths=[2, 2, 9, 2]` checkpoint.
5. ⏳ **Verify the re-run results** against the reference outputs using
   `python verify_phase0.py <reference_dir> <verification_dir>`.
