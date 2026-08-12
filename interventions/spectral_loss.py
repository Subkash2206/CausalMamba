"""
High-Frequency Consistency Loss (Phase 4 - WACV Stretch, Constructive Arm).

Regularizes the VM-UNet encoder to produce stable high-frequency feature
representations under input perturbation (Gaussian noise). For two feature
maps extracted at the same encoder location (clean vs perturbed input), we
compute the high-pass magnitude spectra and penalise their squared distance.

Pipeline per feature map:
    cast fp32 -> (NHWC -> NCHW if needed) -> 2D FFT (ortho) -> fftshift
    -> apply circular high-pass mask (cutoff_radius=0.25) -> magnitude
    -> MSE between clean and perturbed magnitudes

CRITICAL AMP SAFETY RULE:
    torch.fft crashes on fp16 tensors, so every input is explicitly cast to
    torch.float32 immediately before the FFT.

The ``norm='ortho'`` choice makes the FFT orthonormal (Parseval), keeping the
magnitude scale comparable to the spatial feature scale so the auxiliary loss
does not overwhelm the segmentation loss.
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the interventions package importable both as `python interventions/spectral_loss.py`
# and when imported from a training script run elsewhere in the repo.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:  # prefer the canonical core path; fall back to the top-level alias
    from interventions.core.masks import highpass_mask
except ImportError:  # pragma: no cover
    from interventions.masks import highpass_mask


class HighFrequencyConsistencyLoss(nn.Module):
    """MSE between high-pass magnitude spectra of clean vs perturbed features.

    Parameters
    ----------
    cutoff_radius : float, default=0.25
        Normalised high-pass cutoff (fraction of Nyquist). Frequencies with
        distance from DC greater than this radius are kept.

    Inputs
    ------
    features_clean : torch.Tensor
        Feature maps extracted from the clean (unperturbed) forward pass,
        shape (B, C, H, W) or (B, H, W, C).
    features_perturbed : torch.Tensor
        Feature maps extracted from the perturbed forward pass, same shape as
        ``features_clean``.
    """

    def __init__(self, cutoff_radius: float = 0.25) -> None:
        super().__init__()
        self.cutoff_radius = cutoff_radius

    def forward(
        self,
        features_clean: torch.Tensor,
        features_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        # ---- AMP SAFETY: FFT requires fp32; never feed fp16 to torch.fft ----
        clean = features_clean.float()
        pert = features_perturbed.float()

        # Accept NHWC (B, H, W, C) feature maps, e.g. raw VSSBlock outputs.
        if clean.dim() == 4 and clean.shape[1] < clean.shape[-1]:
            clean = clean.permute(0, 3, 1, 2).contiguous()
            pert = pert.permute(0, 3, 1, 2).contiguous()

        B, C, H, W = clean.shape

        # ---- 2D FFT (orthonormal) + centre DC for the centred high-pass mask ----
        spec_clean = torch.fft.fftshift(
            torch.fft.fft2(clean, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        spec_pert = torch.fft.fftshift(
            torch.fft.fft2(pert, dim=(-2, -1), norm='ortho'), dim=(-2, -1))

        # Circular high-pass mask (1, 1, H, W), broadcast over (B, C, H, W).
        mask = highpass_mask(H, W, self.cutoff_radius,
                             device=clean.device, dtype=clean.dtype)

        # ---- isolate high frequencies and compare magnitudes ----
        mag_clean = spec_clean.abs() * mask
        mag_pert = spec_pert.abs() * mask

        return F.mse_loss(mag_clean, mag_pert)


if __name__ == '__main__':
    # Quick self-test: shape, dtype safety, and gradient flow.
    torch.manual_seed(0)
    criterion = HighFrequencyConsistencyLoss(cutoff_radius=0.25)

    # fp16 inputs must NOT crash (AMP safety) and must match fp32 results.
    a16 = torch.randn(2, 96, 32, 32, dtype=torch.float16, requires_grad=True)
    b16 = torch.randn(2, 96, 32, 32, dtype=torch.float16)
    loss = criterion(a16, b16)
    loss.backward()
    assert a16.grad is not None and torch.isfinite(a16.grad).all()
    print(f'[PASS] fp16 inputs -> fp32 FFT, loss={loss.item():.6f}, finite grads')

    # NHWC inputs (B, H, W, C) also accepted.
    a_nhwc = torch.randn(2, 32, 32, 96, requires_grad=True)
    loss2 = criterion(a_nhwc, torch.randn(2, 32, 32, 96))
    loss2.backward()
    print(f'[PASS] NHWC inputs accepted, loss={loss2.item():.6f}')

    # Identical features -> zero loss.
    x = torch.randn(2, 96, 32, 32)
    print(f'[PASS] identical features loss = {criterion(x, x).item():.2e}')
