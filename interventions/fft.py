"""
Pure Fourier utilities for 2D feature tensors.

All functions operate on tensors of shape (B, C, H, W) and transform
only the spatial dimensions (H, W). Batch and channel dimensions are
always preserved.

Every function is:
    - Pure (no side effects, no state)
    - Device-agnostic (works on CPU and CUDA)
    - Free of model-specific knowledge
    - Fully type-annotated

Example
-------
    >>> x = torch.randn(2, 16, 64, 64)
    >>> X = fft2_feature(x)           # complex spectrum
    >>> X = fftshift_feature(X)       # DC centered
    >>> X = ifftshift_feature(X)      # DC back to corners
    >>> y = ifft2_feature(X)          # reconstruct
    >>> torch.allclose(x, y, atol=1e-6)
    True
"""

from typing import Tuple
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fft2_feature(x: torch.Tensor) -> torch.Tensor:
    """Compute the 2D FFT over the spatial dimensions of a feature tensor.

    Parameters
    ----------
    x : torch.Tensor
        4D real-valued tensor of shape (B, C, H, W).
        Must be on CPU or CUDA. Must be a floating-point dtype.

    Returns
    -------
    torch.Tensor
        Complex-valued tensor of shape (B, C, H, W) containing the
        unshifted 2D Fourier spectrum. Same device and dtype-compatible
        with the input.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.Tensor``.
    ValueError
        If ``x`` is not 4-dimensional or is not real-valued.

    Notes
    -----
    The transform is applied independently to each of the (B * C) channels.
    The DC component is at the corners of the result; use
    :py:func:`fftshift_feature` to move it to the center.
    """
    _validate_real_4d(x)

    # fft2 returns complex128 for float64 input, complex64 for float32/float16.
    # PyTorch applies the transform over the last two dimensions by default
    # when given dim=(-2, -1), which is exactly what we want for (H, W).
    return torch.fft.fft2(x, dim=(-2, -1))


def ifft2_feature(X: torch.Tensor) -> torch.Tensor:
    """Compute the inverse 2D FFT to recover a real-valued feature tensor.

    Parameters
    ----------
    X : torch.Tensor
        4D complex-valued tensor of shape (B, C, H, W) representing a
        frequency spectrum. Typically produced by :py:func:`fft2_feature`
        followed by optional shifts and masks.

    Returns
    -------
    torch.Tensor
        Real-valued tensor of shape (B, C, H, W). Matches the dtype
        that was passed to the forward :py:func:`fft2_feature`.

    Raises
    ------
    TypeError
        If ``X`` is not a ``torch.Tensor``.
    ValueError
        If ``X`` is not 4-dimensional or is not complex-valued.

    Notes
    -----
    Small imaginary residuals arising from floating-point arithmetic are
    discarded via :py:func:`torch.view_as_real`. The real part is returned.
    """
    _validate_complex_4d(X)

    # ifft2 is the exact inverse of fft2 when applied with the same dims.
    # The imag part should be ~zero for a consistent FFT/IFFT pair, but we
    # explicitly take .real to guarantee a real-valued output.
    y = torch.fft.ifft2(X, dim=(-2, -1))
    return y.real


def fftshift_feature(X: torch.Tensor) -> torch.Tensor:
    """Shift the DC component of a 2D spectrum to the spatial centre.

    Parameters
    ----------
    X : torch.Tensor
        4D tensor (complex or real) of shape (B, C, H, W) whose DC
        component is at the corners (standard FFT layout).

    Returns
    -------
    torch.Tensor
        Tensor of the same shape, dtype, and device as the input, with
        the DC component at frequency indices ``(H // 2, W // 2)``.

    Raises
    ------
    TypeError
        If ``X`` is not a ``torch.Tensor``.
    ValueError
        If ``X`` is not 4-dimensional.
    """
    _validate_4d(X)

    # fftshift swaps the first and third quadrants of the spatial
    # dimensions. PyTorch's fftshift operates on the specified dims.
    return torch.fft.fftshift(X, dim=(-2, -1))


def ifftshift_feature(X: torch.Tensor) -> torch.Tensor:
    """Undo the DC-centering shift performed by :py:func:`fftshift_feature`.

    Parameters
    ----------
    X : torch.Tensor
        4D tensor (complex or real) of shape (B, C, H, W) whose DC
        component is at the centre.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape, dtype, and device as the input, with
        the DC component returned to the corners.

    Raises
    ------
    TypeError
        If ``X`` is not a ``torch.Tensor``.
    ValueError
        If ``X`` is not 4-dimensional.
    """
    _validate_4d(X)

    # ifftshift is the exact inverse of fftshift.
    return torch.fft.ifftshift(X, dim=(-2, -1))


# ---------------------------------------------------------------------------
# Validation helpers (module-internal)
# ---------------------------------------------------------------------------


def _validate_4d(x: torch.Tensor) -> None:
    """Check that ``x`` is a 4-dimensional ``torch.Tensor``."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(x).__name__}."
        )
    if x.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), got {x.dim()}D tensor "
            f"with shape {tuple(x.shape)}."
        )


def _validate_real_4d(x: torch.Tensor) -> None:
    """Check that ``x`` is a 4D real-valued ``torch.Tensor``."""
    _validate_4d(x)
    if x.is_complex():
        raise ValueError(
            f"Expected real-valued tensor, got complex dtype "
            f"{x.dtype}. Use ifft2_feature for the inverse pass."
        )


def _validate_complex_4d(X: torch.Tensor) -> None:
    """Check that ``X`` is a 4D complex-valued ``torch.Tensor``."""
    _validate_4d(X)
    if not X.is_complex():
        raise ValueError(
            f"Expected complex-valued tensor (from fft2_feature), "
            f"got real dtype {X.dtype}."
        )


# ---------------------------------------------------------------------------
# Self-contained test suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running FFT tests on {_device} ...")

    # Test vector: random 4D float tensor on the active device.
    B, C, H, W = 2, 16, 64, 64
    x = torch.randn(B, C, H, W, device=_device)
    dtype_in = x.dtype

    # --- 1. Shape preservation -------------------------------------------
    X = fft2_feature(x)
    assert X.shape == (B, C, H, W), f"fft2_feature changed shape: {X.shape}"
    assert X.is_complex(), "fft2_feature did not return complex tensor"

    y = ifft2_feature(X)
    assert y.shape == (B, C, H, W), f"ifft2_feature changed shape: {y.shape}"
    assert not y.is_complex(), "ifft2_feature returned complex tensor"
    assert y.dtype == dtype_in, (
        f"ifft2_feature changed dtype: {y.dtype} vs {dtype_in}"
    )

    Xs = fftshift_feature(X)
    assert Xs.shape == (B, C, H, W), f"fftshift_feature changed shape: {Xs.shape}"

    Xr = ifftshift_feature(Xs)
    assert Xr.shape == (B, C, H, W), f"ifftshift_feature changed shape: {Xr.shape}"
    print("  [PASS] Shape preservation")

    # --- 2. Device preservation -------------------------------------------
    assert X.device.type == _device.type, (
        f"fft2_feature moved device: {X.device} vs {_device}"
    )
    assert y.device.type == _device.type, (
        f"ifft2_feature moved device: {y.device} vs {_device}"
    )
    assert Xs.device.type == _device.type, (
        f"fftshift_feature moved device: {Xs.device} vs {_device}"
    )
    assert Xr.device.type == _device.type, (
        f"ifftshift_feature moved device: {Xr.device} vs {_device}"
    )
    print("  [PASS] Device preservation")

    # --- 3. FFT + IFFT reconstruction (cycle consistency) ----------------
    y = ifft2_feature(fft2_feature(x))
    err = (x - y).abs().max().item()
    assert err < 1e-5, (
        f"Reconstruction error too large: max|diff| = {err:.2e}"
    )
    assert torch.allclose(x, y, atol=1e-5), (
        "FFT + IFFT did not reconstruct the original tensor"
    )
    print(f"  [PASS] FFT + IFFT reconstruction (max|diff| = {err:.2e})")

    # --- 4. fftshift + ifftshift cycle consistency -----------------------
    Xr = ifftshift_feature(fftshift_feature(X))
    err_shift = (X - Xr).abs().max().item()
    assert err_shift < 1e-6, (
        f"Shift cycle error too large: max|diff| = {err_shift:.2e}"
    )
    print(f"  [PASS] fftshift + ifftshift cycle (max|diff| = {err_shift:.2e})")

    # --- 5. Verify that spatial dimensions are the ones being transformed -
    # If we zero out a single spatial pixel in the input, the spectrum
    # should differ from the unmodified case.
    x2 = x.clone()
    x2[:, :, 32, 32] += 10.0
    X2 = fft2_feature(x2)
    diff = (X - X2).abs().sum().item()
    assert diff > 0.0, "FFT did not react to spatial modification"
    print("  [PASS] FFT responds to spatial changes")

    print(f"All {5} tests passed on {_device}.")