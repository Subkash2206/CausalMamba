# Methodology

## Causal Frequency Intervention

### Pipeline

Feature Tensor (B, C, H, W) -> FFT -> Shift -> Mask -> IFFT -> Modified Tensor

### Key Design Decisions

1. **Model-agnostic:** The intervention operates on tensors, not model internals. It has zero knowledge of VM-UNet, Swin-UNet, or any specific architecture.

2. **Pure functions:** FFT utilities and mask generation are pure functions with no state, making them easy to test and verify.

3. **Tensor shape conventions:** All operations expect (B, C, H, W) layout. VM-UNet's VSSBlock outputs (B, H, W, C) — the hook handles permutation.

4. **FP preservation:** All operations use .clone() and .contiguous() to prevent in-place tensor modification.

### Mask Types

- **lowpass_mask(h, w, cutoff):** Circular low-pass with radius = cutoff x min(H,W)/2
- **highpass_mask(h, w, cutoff):** Element-wise complement of lowpass
- **bandpass_mask(h, w, c_low, c_high):** Annular band between two cutoffs
- **bandstop_mask(h, w, c_low, c_high):** Complement of bandpass

### AVR Definition

Average Volume Ratio (AVR) measures the fraction of spectral energy above the Nyquist boundary. For a feature map of size HxW:
- DC is at (H//2, W//2)
- Nyquist boundary is at H/4 and W/4 from DC
- AVR = (sum of power above Nyquist) / (total power)