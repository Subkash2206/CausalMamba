"""
Experiment 1: Whole-Network Low-Pass Intervention.

Measures how suppressing high-frequency information throughout the encoder
affects segmentation performance and spectral statistics.

Configuration:
    - Mask type: low-pass
    - Cutoff: 0.25 (normalised)
    - Applied to: all Conv2d layers (encoder stages)
    - Dataset: synthetic 3×64×64 (same as Experiment 0)

Output:
    - Per-layer AVR_before, AVR_after, ΔAVR, % reduction
    - Global Dice comparison vs baseline
    - Interpretation
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask


# ---------------------------------------------------------------------------
# Network (identical to Experiment 0)
# ---------------------------------------------------------------------------

class SimpleFeatureNet(nn.Module):
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
# Metrics
# ---------------------------------------------------------------------------

def compute_avr(fmap: torch.Tensor) -> float:
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2
    y = torch.arange(H).view(1, 1, H, 1)
    x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)
    mask = mask.expand(B, C, H, W)
    high_freq_energy = (power * mask).sum()
    total_energy = power.sum()
    return (high_freq_energy / total_energy).item() if total_energy > 0 else 0.0


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum().item()
    return (2.0 * intersection) / (pred_bin.sum().item() + target.sum().item() + 1e-8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Experiment 1: Whole-Network Low-Pass Intervention (cutoff=0.25)")
    print("=" * 80)
    print(f"Device: {device}\n")

    torch.manual_seed(42)

    # -- Dataset -----------------------------------------------------------
    B = 8
    N_SAMPLES = 32
    dataset_x = torch.randn(N_SAMPLES, 3, 64, 64)
    dataset_y = (torch.rand(N_SAMPLES, 1, 64, 64) > 0.85).float()

    model = SimpleFeatureNet().to(device)
    model.eval()
    print(f"Model: SimpleFeatureNet ({sum(p.numel() for p in model.parameters())} params)")
    print(f"Dataset: {N_SAMPLES} samples, {B}-batch, 3×64×64\n")

    # ======================================================================
    # Baseline (no intervention)
    # ======================================================================
    print("-" * 80)
    print("Baseline (no intervention)")
    print("-" * 80)

    model_bl = SimpleFeatureNet().to(device)
    model_bl.load_state_dict(model.state_dict())
    model_bl.eval()

    baseline_preds = []
    baseline_dice = []
    baseline_avr = defaultdict(list)
    baseline_features = {}

    def make_bl_hook(name):
        def hook(module, inp, out):
            baseline_features[name] = out.detach().cpu()
        return hook

    handles_bl = []
    for ln, mod in model_bl.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles_bl.append(mod.register_forward_hook(make_bl_hook(ln)))

    with torch.no_grad():
        for i in range(0, N_SAMPLES, B):
            xb = dataset_x[i:i+B].to(device)
            yb = dataset_y[i:i+B].to(device)
            pred = model_bl(xb)
            baseline_preds.append(pred.cpu())
            baseline_dice.append(dice_score(pred.cpu(), yb.cpu()))
            for nm in sorted(baseline_features.keys()):
                baseline_avr[nm].append(compute_avr(baseline_features[nm]))

    for h in handles_bl:
        h.remove()

    baseline_preds = torch.cat(baseline_preds, dim=0)
    mean_bl_dice = sum(baseline_dice) / len(baseline_dice)
    print(f"  Mean Dice: {mean_bl_dice:.6f}")

    # ======================================================================
    # Low-pass intervention (cutoff=0.25)
    # ======================================================================
    print("-" * 80)
    print("Low-pass intervention (cutoff=0.25)")
    print("-" * 80)

    model_lp = SimpleFeatureNet().to(device)
    model_lp.load_state_dict(model.state_dict())
    model_lp.eval()

    # Low-pass mask with cutoff=0.25
    lp_mask_fn = lambda h, w, device, dtype: lowpass_mask(
        h, w, 0.25, device=device, dtype=dtype
    )
    intervention = FrequencyIntervention(lp_mask_fn, check_nan=True)

    lp_preds = []
    lp_dice = []
    lp_avr_before = defaultdict(list)
    lp_avr_after = defaultdict(list)

    _store = {}

    def make_lp_hook(name, intervention):
        def hook(module, inp, out):
            fmap = out[0] if isinstance(out, tuple) else out
            before = fmap.detach().cpu()
            modified = intervention(fmap)
            after = modified.detach().cpu()
            _store[name] = (before, after)
            return modified
        return hook

    handles_lp = []
    for ln, mod in model_lp.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles_lp.append(
                mod.register_forward_hook(make_lp_hook(ln, intervention))
            )

    with torch.no_grad():
        for i in range(0, N_SAMPLES, B):
            xb = dataset_x[i:i+B].to(device)
            yb = dataset_y[i:i+B].to(device)
            _store.clear()
            pred = model_lp(xb)
            lp_preds.append(pred.cpu())
            lp_dice.append(dice_score(pred.cpu(), yb.cpu()))
            for nm in sorted(_store.keys()):
                bf, af = _store[nm]
                lp_avr_before[nm].append(compute_avr(bf))
                lp_avr_after[nm].append(compute_avr(af))

    for h in handles_lp:
        h.remove()

    lp_preds = torch.cat(lp_preds, dim=0)
    mean_lp_dice = sum(lp_dice) / len(lp_dice)

    # ======================================================================
    # Results
    # ======================================================================
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    # -- Per-layer AVR table -----------------------------------------------
    print("\n--- Per-Layer Spectral Statistics ---")
    print(f"{'Layer':<10} | {'Res':<8} | {'AVR Before':<12} | {'AVR After':<12} | {'ΔAVR':<12} | {'Reduction':<10}")
    print("-" * 70)

    layer_names = sorted(set(list(baseline_avr.keys()) + list(lp_avr_before.keys())))
    deltas = []

    for nm in layer_names:
        iv_bf = lp_avr_before.get(nm, [0.0])
        iv_af = lp_avr_after.get(nm, [0.0])
        mean_bf = sum(iv_bf) / len(iv_bf)
        mean_af = sum(iv_af) / len(iv_af)
        delta = mean_af - mean_bf
        pct = ((mean_bf - mean_af) / (mean_bf + 1e-12)) * 100.0
        deltas.append(abs(delta))
        sf = baseline_features.get(nm, torch.zeros(1, 1, 8, 8))
        res = f"{sf.shape[2]}x{sf.shape[3]}"
        print(f"{nm:<10} | {res:<8} | {mean_bf:<12.6f} | {mean_af:<12.6f} | {delta:<+12.2e} | {pct:<+10.2f}%")

    # -- Global metrics ----------------------------------------------------
    print("\n--- Global Metrics ---")
    pred_diff = (lp_preds - baseline_preds).abs()
    print(f"{'Metric':<25} {'Baseline':<15} {'Low-pass':<15} {'Diff':<12}")
    print("-" * 70)
    print(f"{'Dice':<25} {mean_bl_dice:<15.6f} {mean_lp_dice:<15.6f} {mean_lp_dice - mean_bl_dice:<+12.2e}")
    print(f"{'Max |pred diff|':<25} {'—':<15} {'—':<15} {pred_diff.max().item():<+12.2e}")
    print(f"{'Mean |pred diff|':<25} {'—':<15} {'—':<15} {pred_diff.mean().item():<+12.2e}")

    # -- Summary -----------------------------------------------------------
    print("\n--- Summary ---")
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    print(f"  Mean |ΔAVR| across all layers: {mean_delta:.2e}")
    print(f"  Maximum |ΔAVR| across all layers: {max_delta:.2e}")

    # ======================================================================
    # Interpretation
    # ======================================================================
    print("\n" + "=" * 80)
    print("Discussion")
    print("=" * 80)

    # 1. Does low-pass filtering reduce AVR?
    all_reduced = all(
        lp_avr_after.get(nm, [0.0])[0] <= lp_avr_before.get(nm, [0.0])[0] + 1e-8
        for nm in layer_names
    )
    if all_reduced:
        print("\n  1. Low-pass filtering reduces AVR across ALL layers.")
        print(f"     Maximum ΔAVR = {max_delta:.2e}.")
    else:
        print("\n  1. Low-pass filtering does NOT reduce AVR uniformly.")
        for nm in layer_names:
            bf = lp_avr_before.get(nm, [0.0])[0]
            af = lp_avr_after.get(nm, [0.0])[0]
            if af > bf + 1e-8:
                print(f"     Layer {nm}: AVR increased from {bf:.6f} to {af:.6f}.")

    # 2. Does reducing AVR improve or degrade segmentation?
    dice_change = mean_lp_dice - mean_bl_dice
    if dice_change > 0:
        print(f"  2. AVR reduction improves Dice by {dice_change:+.4f}.")
    elif dice_change < 0:
        print(f"  2. AVR reduction degrades Dice by {dice_change:+.4f}.")
    else:
        print(f"  2. Dice unchanged ({dice_change:+.2e}).")

    # 3. Which layers exhibit the largest spectral change?
    layer_deltas = []
    for nm in layer_names:
        iv_bf = lp_avr_before.get(nm, [0.0])
        iv_af = lp_avr_after.get(nm, [0.0])
        mean_bf = sum(iv_bf) / len(iv_bf)
        mean_af = sum(iv_af) / len(iv_af)
        layer_deltas.append((abs(mean_af - mean_bf), nm, mean_bf, mean_af))
    layer_deltas.sort(reverse=True)
    print(f"  3. Layers ranked by spectral change (|ΔAVR|):")
    for i, (d, nm, bf, af) in enumerate(layer_deltas):
        pct = ((bf - af) / (bf + 1e-12)) * 100.0
        print(f"     {i+1}. {nm:<10} |ΔAVR| = {d:<12.2e}  ({pct:+.1f}%)")

    print()


if __name__ == "__main__":
    main()