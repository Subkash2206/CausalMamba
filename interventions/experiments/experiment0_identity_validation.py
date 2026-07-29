"""
Experiment 0: Identity Intervention Validation.

Verifies that inserting the FrequencyIntervention framework into a forward
graph does NOT alter model behaviour when the intervention uses an all-ones
(identity) mask.

Pipeline:
    1. Build a multi-layer CNN with skip-connection-like structure.
    2. Run baseline inference (no intervention).
    3. Run inference with identity intervention on every conv layer.
    4. Compare:
        - Predictions (max |diff|, mean |diff|)
        - Per-layer AVR before vs after intervention
        - ΔAVR across all layers
    5. Report whether identity intervention is transparent.

This script uses the same FrequencyIntervention class and hook mechanism
that is integrated into SpectralMamba/tools/avr_analysis_intervention.py.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

# Add parent directory to resolve interventions/ package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interventions.intervention import FrequencyIntervention


# ---------------------------------------------------------------------------
# Synthetic multi-layer CNN (simulates a tiny segmentation backbone)
# ---------------------------------------------------------------------------

class SimpleFeatureNet(nn.Module):
    """Minimal CNN with multiple feature-extraction stages.

    Architecture:
        conv1 (3→16, 3×3)  → relu → conv2 (16→16, 3×3)
        → down 2× (avg pool)
        → conv3 (16→32, 3×3) → relu → conv4 (32→32, 3×3)
        → down 2× (avg pool)
        → conv5 (32→1, 1×1)  →  output (logits)

    This provides multiple intermediate feature maps at varying resolutions
    where hooks can be inserted, simulating the SpectralMamba VSSBlock layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.avg_pool2d(x, 2)
        x = self.conv5(x)
        return x


# ---------------------------------------------------------------------------
# AVR computation (matches SpectralMamba's convention)
# ---------------------------------------------------------------------------

def compute_avr(fmap: torch.Tensor) -> float:
    """Average Volume Ratio: fraction of energy above Nyquist."""
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2

    cy, cx = H // 2, W // 2
    y = torch.arange(H, device='cpu').view(1, 1, H, 1)
    x = torch.arange(W, device='cpu').view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
    mask = mask.expand(B, C, H, W)

    high_freq_energy = (power * mask).sum()
    total_energy = power.sum()
    return (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0


# ---------------------------------------------------------------------------
# Dice score
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Resize prediction to match target spatial dimensions (due to pooling)
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum().item()
    return (2.0 * intersection) / (pred_bin.sum().item() + target.sum().item() + 1e-8)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"Experiment 0: Identity Intervention Validation")
    print(f"{'='*70}")
    print(f"Device: {device}\n")

    # -- Seed for reproducibility -------------------------------------------
    torch.manual_seed(42)

    # -- Create synthetic dataset (32 samples, 3×64×64) ---------------------
    B = 8  # batch size
    N_SAMPLES = 32
    dataset_x = torch.randn(N_SAMPLES, 3, 64, 64)
    # Synthetic binary masks
    dataset_y = (torch.rand(N_SAMPLES, 1, 64, 64) > 0.85).float()

    # -- Initialise model ---------------------------------------------------
    model = SimpleFeatureNet().to(device)
    model.eval()
    print(f"Model: SimpleFeatureNet ({sum(p.numel() for p in model.parameters())} params)")
    print(f"Dataset: {N_SAMPLES} samples, {B}-batch, 3×64×64\n")

    # =======================================================================
    # Phase 1: Baseline (no intervention)
    # =======================================================================
    print("-" * 70)
    print("Phase 1: Baseline (no intervention)")
    print("-" * 70)

    model_baseline = SimpleFeatureNet().to(device)
    model_baseline.load_state_dict(model.state_dict())  # identical weights
    model_baseline.eval()

    baseline_preds = []
    baseline_dice = []
    baseline_avr = defaultdict(list)

    # Register hooks that only capture features (no intervention)
    baseline_features = {}

    def make_baseline_hook(name):
        def hook(module, inp, out):
            baseline_features[name] = out.detach().cpu()
        return hook

    baseline_handles = []
    for layer_name, module in model_baseline.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(make_baseline_hook(layer_name))
            baseline_handles.append(handle)

    with torch.no_grad():
        for i in range(0, N_SAMPLES, B):
            x_batch = dataset_x[i:i+B].to(device)
            y_batch = dataset_y[i:i+B].to(device)
            pred = model_baseline(x_batch)
            baseline_preds.append(pred.cpu())
            baseline_dice.append(dice_score(pred.cpu(), y_batch.cpu()))

            for name in sorted(baseline_features.keys()):
                baseline_avr[name].append(compute_avr(baseline_features[name]))

    # Clean up baseline hooks
    for h in baseline_handles:
        h.remove()

    baseline_preds = torch.cat(baseline_preds, dim=0)
    mean_baseline_dice = sum(baseline_dice) / len(baseline_dice)
    print(f"  Mean Dice: {mean_baseline_dice:.6f}")

    # =======================================================================
    # Phase 2: Identity intervention (all-ones mask)
    # =======================================================================
    print("-" * 70)
    print("Phase 2: Identity intervention (all-ones mask)")
    print("-" * 70)

    model_interv = SimpleFeatureNet().to(device)
    model_interv.load_state_dict(model.state_dict())  # identical weights
    model_interv.eval()

    # Identity mask: pass all frequencies unchanged
    identity_mask = lambda h, w, device, dtype: torch.ones(
        1, 1, h, w, device=device, dtype=dtype
    )
    intervention = FrequencyIntervention(identity_mask, check_nan=True)

    interv_preds = []
    interv_dice = []
    interv_avr_before = defaultdict(list)
    interv_avr_after = defaultdict(list)

    def make_interv_hook(name, intervention):
        def hook(module, inp, out):
            # Normalise to (B, C, H, W) if needed (simulates VSSBlock permute)
            fmap = out[0] if isinstance(out, tuple) else out
            # Store pre-intervention feature
            before = fmap.detach().cpu()
            # Apply intervention
            modified = intervention(fmap)
            after = modified.detach().cpu()
            # Store both
            _interv_hook_store[name] = (before, after)
            # Return modified tensor to the graph
            return modified
        return hook

    # Use a list to store hook captures (workaround for closure mutability)
    _interv_hook_store = {}

    interv_handles = []
    for layer_name, module in model_interv.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(
                make_interv_hook(layer_name, intervention)
            )
            interv_handles.append(handle)

    with torch.no_grad():
        for i in range(0, N_SAMPLES, B):
            x_batch = dataset_x[i:i+B].to(device)
            y_batch = dataset_y[i:i+B].to(device)

            # Reset store for this batch
            _interv_hook_store.clear()

            pred = model_interv(x_batch)
            interv_preds.append(pred.cpu())
            interv_dice.append(dice_score(pred.cpu(), y_batch.cpu()))

            # Accumulate AVR for each layer
            for name in sorted(_interv_hook_store.keys()):
                before, after = _interv_hook_store[name]
                interv_avr_before[name].append(compute_avr(before))
                interv_avr_after[name].append(compute_avr(after))

    # Clean up
    for h in interv_handles:
        h.remove()

    interv_preds = torch.cat(interv_preds, dim=0)
    mean_interv_dice = sum(interv_dice) / len(interv_dice)

    # =======================================================================
    # Validation Report
    # =======================================================================
    print("\n" + "=" * 70)
    print("Validation Report")
    print("=" * 70)

    # 1. Prediction comparison
    pred_diff = (interv_preds - baseline_preds).abs()
    max_pred_diff = pred_diff.max().item()
    mean_pred_diff = pred_diff.mean().item()
    print(f"\n1. Prediction Comparison")
    print(f"   Max  |diff| between baseline and identity: {max_pred_diff:.2e}")
    print(f"   Mean |diff| between baseline and identity: {mean_pred_diff:.2e}")

    # 2. Dice comparison
    dice_diff = abs(mean_interv_dice - mean_baseline_dice)
    print(f"\n2. Dice Score Comparison")
    print(f"   Baseline Dice:             {mean_baseline_dice:.10f}")
    print(f"   Identity Intervention Dice: {mean_interv_dice:.10f}")
    print(f"   Absolute difference:        {dice_diff:.2e}")

    # 3. Per-layer AVR analysis
    print(f"\n3. Per-Layer AVR Analysis")
    print(f"   {'Layer':<10} | {'Resolution':<12} | {'AVR Before':<12} | {'AVR After':<12} | {'ΔAVR':<10}")
    print(f"   {'-'*10} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}")

    max_delta_avr = 0.0
    all_deltas = []

    layer_names_sorted = sorted(set(
        list(baseline_avr.keys()) + list(interv_avr_before.keys())
    ))

    for name in layer_names_sorted:
        # Baseline AVR (from Phase 1)
        bl_avrs = baseline_avr.get(name, [0.0])
        bl_mean = sum(bl_avrs) / len(bl_avrs)

        # Intervention AVRs (from Phase 2)
        iv_before = interv_avr_before.get(name, [0.0])
        iv_after = interv_avr_after.get(name, [0.0])
        iv_before_mean = sum(iv_before) / len(iv_before)
        iv_after_mean = sum(iv_after) / len(iv_after)

        delta = iv_after_mean - iv_before_mean
        all_deltas.append(abs(delta))
        if abs(delta) > max_delta_avr:
            max_delta_avr = abs(delta)

        # Resolution from first sample
        sample_fmap = baseline_features.get(name, torch.zeros(1, 1, 8, 8))
        res_str = f"{sample_fmap.shape[2]}x{sample_fmap.shape[3]}"

        print(f"   {name:<10} | {res_str:<12} | {iv_before_mean:<12.6f} | {iv_after_mean:<12.6f} | {delta:+.2e}")

    mean_delta_avr = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
    print(f"\n4. AVR Discrepancy Summary")
    print(f"   Max  |ΔAVR| across all layers: {max_delta_avr:.2e}")
    print(f"   Mean |ΔAVR| across all layers: {mean_delta_avr:.2e}")

    # =======================================================================
    # Final Verdict
    # =======================================================================
    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)

    # Thresholds: floating-point tolerance for float32
    PRED_TOL = 1e-5
    DICE_TOL = 1e-8
    AVR_TOL = 1e-6

    pred_ok = max_pred_diff < PRED_TOL
    dice_ok = dice_diff < DICE_TOL
    avr_ok = max_delta_avr < AVR_TOL

    if pred_ok and dice_ok and avr_ok:
        print(f"\n  ✓ Identity intervention successfully preserves model behaviour.")
        print(f"  ✓ Predictions differ by at most {max_pred_diff:.2e} (tolerance {PRED_TOL:.0e})")
        print(f"  ✓ Dice differs by {dice_diff:.2e} (tolerance {DICE_TOL:.0e})")
        print(f"  ✓ Max |ΔAVR| = {max_delta_avr:.2e} (tolerance {AVR_TOL:.0e})")
        print(f"\n  The FrequencyIntervention framework is TRANSPARENT when using")
        print(f"  an identity (all-ones) mask. Proceed to frequency suppression experiments.")
    else:
        print(f"\n  ✗ Identity intervention introduced measurable discrepancies:")
        if not pred_ok:
            print(f"    - Predictions: max |diff| = {max_pred_diff:.2e} > {PRED_TOL:.0e}")
        if not dice_ok:
            print(f"    - Dice: diff = {dice_diff:.2e} > {DICE_TOL:.0e}")
        if not avr_ok:
            print(f"    - Max |ΔAVR| = {max_delta_avr:.2e} > {AVR_TOL:.0e}")
        print(f"\n  Investigate before proceeding.")

    print()


if __name__ == "__main__":
    main()