"""
Causal frequency intervention on feature tensors.

Pipeline: spatial tensor → FFT → apply frequency mask → IFFT → spatial tensor.

The intervention knows nothing about models, datasets, or experiments.
It only accepts tensors and returns tensors.

Example
-------
    >>> from fft import fft2_feature, ifft2_feature, fftshift_feature, ifftshift_feature
    >>> from masks import lowpass_mask
    >>> mask_fn = lambda h, w, device, dtype: lowpass_mask(h, w, 0.25, device=device, dtype=dtype)
    >>> intervention = FrequencyIntervention(mask_fn)
    >>> x = torch.randn(2, 16, 64, 64)
    >>> y = intervention(x)
    >>> y.shape == x.shape
    True
"""

from typing import Callable, Optional

import torch

from .fft import fft2_feature, ifft2_feature, fftshift_feature, ifftshift_feature


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class InterventionError(RuntimeError):
    """Raised when an intervention receives invalid input or produces invalid output."""
    pass


# ---------------------------------------------------------------------------
# Intervention class
# ---------------------------------------------------------------------------


class FrequencyIntervention:
    """Apply a frequency-domain mask to a feature tensor.

    The full pipeline is::

        fft2 → fftshift → apply mask → ifftshift → ifft2

    Parameters
    ----------
    mask_fn : Callable[[int, int, torch.device, torch.dtype], torch.Tensor]
        A callable that returns a mask of shape ``(1, 1, H, W)`` with values
        in ``[0, 1]``.  The mask is generated from the spatial dimensions
        of the incoming tensor, so the same ``FrequencyIntervention`` can be
        applied to tensors of varying sizes.

        The signature is::

            mask_fn(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor

    check_nan : bool, default=True
        If True, raise ``InterventionError`` when the output contains NaN
        or Inf values.

    Raises
    ------
    InterventionError
        If the mask returned by ``mask_fn`` has an invalid shape or contains
        values outside the range ``[0, 1]``.
    """

    def __init__(
        self,
        mask_fn: Callable[[int, int, torch.device, torch.dtype], torch.Tensor],
        *,
        check_nan: bool = True,
    ) -> None:
        self._mask_fn = mask_fn
        self._check_nan = check_nan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor:
        """Apply the frequency intervention to a feature tensor.

        Parameters
        ----------
        x : torch.Tensor
            4D real-valued float tensor of shape ``(B, C, H, W)``.
            Must be on CPU or CUDA.

        Returns
        -------
        torch.Tensor
            Modified feature tensor with the same shape, dtype, and device
            as the input.  Always a new tensor (never a view of or in-place
            modification of ``x``).

        Raises
        ------
        InterventionError
            If input validation fails, the mask has an unexpected shape
            or values, or the output contains NaN/Inf.
        """
        self._validate_input(x)

        # 1. Spatial → frequency domain.
        spectrum = self.to_frequency(x)

        # 2. Generate the mask based on the current spatial dimensions.
        _, _, h, w = x.shape
        mask = self._mask_fn(h, w, x.device, x.dtype)
        self._validate_mask(mask, h, w)

        # 3. Apply the mask.
        spectrum_masked = self.apply_mask(spectrum, mask)

        # 4. Frequency → spatial domain.
        y = self.to_spatial(spectrum_masked, x.shape)

        # 5. Validate the output.
        self._validate_output(y, x)

        return y

    # ------------------------------------------------------------------
    # Pipeline stages (protected, overridable for extensibility)
    # ------------------------------------------------------------------

    def to_frequency(self, x: torch.Tensor, /) -> torch.Tensor:
        """Convert a spatial feature tensor to the frequency domain.

        Pipeline: fft2 → fftshift.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued float tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Complex-valued tensor of shape ``(B, C, H, W)`` with DC at centre.
        """
        # fft2 operates on the spatial dimensions.
        X = fft2_feature(x)
        X = fftshift_feature(X)
        return X

    def apply_mask(self, spectrum: torch.Tensor, mask: torch.Tensor, /) -> torch.Tensor:
        """Apply a frequency mask to a centred spectrum.

        Parameters
        ----------
        spectrum : torch.Tensor
            Complex tensor of shape ``(B, C, H, W)`` with DC at centre.
        mask : torch.Tensor
            Float tensor of shape ``(1, 1, H, W)`` with values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Masked complex spectrum of the same shape as ``spectrum``.
        """
        # Element-wise multiplication broadcasts the (1, 1, H, W) mask
        # over the batch and channel dimensions.
        return spectrum * mask

    def to_spatial(
        self,
        spectrum: torch.Tensor,
        original_shape: tuple[int, int, int, int],
        /,
    ) -> torch.Tensor:
        """Convert a centred spectrum back to the spatial domain.

        Pipeline: ifftshift → ifft2.

        Parameters
        ----------
        spectrum : torch.Tensor
            Complex tensor of shape ``(B, C, H, W)`` with DC at centre.
        original_shape : tuple[int, int, int, int]
            The shape of the original input tensor.  Used only for
            consistency checks; the spatial size is inferred from
            the input argument.

        Returns
        -------
        torch.Tensor
            Real-valued float tensor of shape ``(B, C, H, W)``.
        """
        X = ifftshift_feature(spectrum)
        y = ifft2_feature(X)
        return y

    # ------------------------------------------------------------------
    # Input / output validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        """Check that ``x`` is a valid input tensor.

        Raises
        ------
        InterventionError
            If any precondition is violated.
        """
        if not isinstance(x, torch.Tensor):
            raise InterventionError(
                f"Expected torch.Tensor, got {type(x).__name__}."
            )
        if x.dim() != 4:
            raise InterventionError(
                f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor "
                f"with shape {tuple(x.shape)}."
            )
        if x.is_complex():
            raise InterventionError(
                f"Expected real-valued tensor, got complex dtype {x.dtype}."
            )
        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            raise InterventionError(
                f"Expected floating-point dtype, got {x.dtype}."
            )
        if x.device.type not in ("cpu", "cuda"):
            raise InterventionError(
                f"Expected CPU or CUDA tensor, got {x.device}."
            )
        if x.shape[2] < 2 or x.shape[3] < 2:
            raise InterventionError(
                f"Spatial dimensions must be at least 2×2, got {x.shape[2]}×{x.shape[3]}."
            )

    @staticmethod
    def _validate_mask(mask: torch.Tensor, h: int, w: int) -> None:
        """Check that the mask has the expected shape and value range.

        Raises
        ------
        InterventionError
            If the mask is invalid.
        """
        if not isinstance(mask, torch.Tensor):
            raise InterventionError(
                f"mask_fn must return a torch.Tensor, got {type(mask).__name__}."
            )
        if mask.dim() != 4:
            raise InterventionError(
                f"Mask must be 4D (1, 1, H, W), got {mask.dim()}D tensor "
                f"with shape {tuple(mask.shape)}."
            )
        if mask.shape[2] != h or mask.shape[3] != w:
            raise InterventionError(
                f"Mask spatial dimensions ({mask.shape[2]}×{mask.shape[3]}) "
                f"do not match input ({h}×{w})."
            )
        if mask.min() < 0.0 or mask.max() > 1.0:
            raise InterventionError(
                f"Mask values must be in [0, 1], "
                f"got range [{mask.min().item():.3f}, {mask.max().item():.3f}]."
            )

    @staticmethod
    def _validate_output(y: torch.Tensor, x: torch.Tensor) -> None:
        """Check that the output matches the input in shape, dtype, and device.

        Raises
        ------
        InterventionError
            If any property differs or NaN/Inf is detected.
        """
        if y.shape != x.shape:
            raise InterventionError(
                f"Output shape {tuple(y.shape)} differs from input shape "
                f"{tuple(x.shape)}."
            )
        if y.dtype != x.dtype:
            raise InterventionError(
                f"Output dtype {y.dtype} differs from input dtype {x.dtype}."
            )
        if y.device != x.device:
            raise InterventionError(
                f"Output device {y.device} differs from input device {x.device}."
            )
        if y.is_complex():
            raise InterventionError(
                "Output is complex-valued; expected real-valued tensor."
            )
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise InterventionError("Output contains NaN or Inf values.")


# ---------------------------------------------------------------------------
# Self-contained test suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from .masks import lowpass_mask, highpass_mask

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running intervention tests on {_device} ...")

    # Helper: create a mask_fn from a masks module function.
    def _make_mask_fn(mask_func, **kwargs):
        def _fn(h, w, device, dtype):
            return mask_func(h, w, **kwargs, device=device, dtype=dtype)
        return _fn

    B, C, H, W = 2, 16, 64, 64
    x = torch.randn(B, C, H, W, device=_device)

    # --- 1. Shape preservation --------------------------------------------
    null_fn = _make_mask_fn(lowpass_mask, cutoff=1.0)
    intervention = FrequencyIntervention(null_fn)
    y = intervention(x)
    assert y.shape == (B, C, H, W), f"Shape changed: {y.shape}"
    print("  [PASS] Shape preservation")

    # --- 2. Dtype preservation --------------------------------------------
    assert y.dtype == x.dtype, f"dtype changed: {y.dtype} vs {x.dtype}"
    print("  [PASS] Dtype preservation")

    # --- 3. Device preservation -------------------------------------------
    assert y.device.type == _device.type, f"Device changed: {y.device} vs {_device}"
    print("  [PASS] Device preservation")

    # --- 4. Low-pass mask changes the tensor ------------------------------
    lp_fn = _make_mask_fn(lowpass_mask, cutoff=0.25)
    intervention_lp = FrequencyIntervention(lp_fn)
    y_lp = intervention_lp(x)
    diff = (x - y_lp).abs().max().item()
    assert diff > 1e-6, (
        f"Low-pass intervention did not change the tensor (max|diff| = {diff:.2e})"
    )
    print(f"  [PASS] Low-pass mask changes the tensor (max|diff| = {diff:.2e})")

    # --- 5. Null mask (all ones) reconstructs the input ------------------
    # An all-ones mask in the frequency domain should pass every frequency
    # through unchanged, so the output must be near-identical to the input.
    null_fn = lambda h, w, device, dtype: torch.ones(1, 1, h, w, device=device, dtype=dtype)
    intervention_null = FrequencyIntervention(null_fn)
    y_null = intervention_null(x)
    err = (x - y_null).abs().max().item()
    assert err < 1e-5, (
        f"Null mask did not reconstruct input (max|diff| = {err:.2e})"
    )
    print(f"  [PASS] Null mask reconstructs the input (max|diff| = {err:.2e})")

    # --- 6. High-pass removes low frequencies, changing the tensor --------
    hp_fn = _make_mask_fn(highpass_mask, cutoff=0.25)
    intervention_hp = FrequencyIntervention(hp_fn)
    y_hp = intervention_hp(x)
    diff_hp = (x - y_hp).abs().max().item()
    assert diff_hp > 1e-6, (
        f"High-pass intervention did not change the tensor (max|diff| = {diff_hp:.2e})"
    )
    print(f"  [PASS] High-pass mask changes the tensor (max|diff| = {diff_hp:.2e})")

    # --- 7. Error: invalid input shape ------------------------------------
    _caught = 0
    try:
        FrequencyIntervention(null_fn)(torch.randn(2, 16, 64))
    except InterventionError:
        _caught += 1
    try:
        FrequencyIntervention(null_fn)(torch.randn(2, 16, 64, 64, dtype=torch.complex64))
    except InterventionError:
        _caught += 1
    try:
        FrequencyIntervention(null_fn)(torch.randn(2, 16, 1, 64))
    except InterventionError:
        _caught += 1
    assert _caught == 3, f"Expected 3 input validation errors, caught {_caught}"
    print("  [PASS] Input validation rejects invalid tensors")

    print(f"\nAll tests passed on {_device}.")