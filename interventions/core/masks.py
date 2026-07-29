"""
Frequency-domain mask generation for causal intervention.

All masks are generated in the shifted frequency domain (DC at centre),
producing floating-point tensors in [0, 1] with shape (1, 1, H, W).

Every function is:
    - Pure (no side effects, no state)
    - Device-agnostic (mask device is set by the caller)
    - Independent of FFT, models, and interventions
    - Fully type-annotated

The mask tensors broadcast naturally over (B, C, H, W) spectra because
the batch and channel dimensions are singletons.

Example
-------
    >>> mask = lowpass_mask(64, 64, cutoff=0.25, device=torch.device('cpu'))
    >>> spectrum = torch.randn(2, 16, 64, 64, dtype=torch.complex64)
    >>> filtered = spectrum * mask   # shape (2, 16, 64, 64)
"""

from typing import Optional
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lowpass_mask(
    h: int,
    w: int,
    cutoff: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a centred low-pass frequency mask (circular).

    Frequencies whose distance from DC is less than or equal to the cutoff
    are kept (mask = 1). Higher frequencies are suppressed (mask = 0).

    Parameters
    ----------
    h : int
        Height of the spatial dimensions (> 1).
    w : int
        Width of the spatial dimensions (> 1).
    cutoff : float
        Normalised cutoff frequency in the range (0, 1].  A value of 1.0
        corresponds to the full frequency extent (i.e. no filtering).
    device : torch.device, optional
        Target device for the mask tensor.  If None, defaults to CPU.
    dtype : torch.dtype, optional
        Target dtype for the mask tensor.  If None, defaults to
        ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Mask of shape (1, 1, H, W) with values in {0, 1}.

    Raises
    ------
    ValueError
        If ``h <= 1``, ``w <= 1``, or ``cutoff`` is not in (0, 1].
    """
    _validate_spatial(h, w)
    _validate_cutoff(cutoff)

    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)  # the farthest full-radius from centre

    # Clamp the cutoff so that its physical radius does not exceed the
    # image boundary.  This ensures the circle always fits inside the
    # frequency grid.
    radius = float(cutoff) * float(max_radius)

    grid_y, grid_x = _frequency_grid(h, w, device=device)
    dist = torch.sqrt(grid_y ** 2 + grid_x ** 2)  # shape (H, W)

    mask = (dist <= radius).to(dtype=dtype or torch.float32)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def highpass_mask(
    h: int,
    w: int,
    cutoff: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a centred high-pass frequency mask (circular).

    Frequencies whose distance from DC is greater than the cutoff are kept
    (mask = 1).  Low frequencies are suppressed (mask = 0).

    The result is the element-wise complement of the low-pass mask::

        highpass = 1.0 - lowpass

    Parameters
    ----------
    h : int
        Height of the spatial dimensions (> 1).
    w : int
        Width of the spatial dimensions (> 1).
    cutoff : float
        Normalised cutoff frequency in the range (0, 1].  A value of 0.0
        corresponds to keeping all frequencies (no filtering).
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor
        Mask of shape (1, 1, H, W) with values in {0, 1}.

    Raises
    ------
    ValueError
        If ``h <= 1``, ``w <= 1``, or ``cutoff`` is not in (0, 1].
    """
    _validate_spatial(h, w)
    _validate_cutoff(cutoff)

    low = lowpass_mask(h, w, cutoff, device=device, dtype=dtype)
    return 1.0 - low


def bandpass_mask(
    h: int,
    w: int,
    cutoff_low: float,
    cutoff_high: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a centred band-pass frequency mask (circular annulus).

    Frequencies whose distance from DC lies in the interval
    ``(cutoff_low, cutoff_high]`` are kept (mask = 1).  Frequencies
    inside or outside the annulus are suppressed (mask = 0).

    Equivalent to::

        bandpass = lowpass(cutoff_high) - lowpass(cutoff_low)

    Parameters
    ----------
    h : int
    w : int
    cutoff_low : float
        Inner (lower) normalised cutoff in (0, 1).
    cutoff_high : float
        Outer (upper) normalised cutoff in (0, 1).  Must be greater than
        ``cutoff_low``.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor
        Mask of shape (1, 1, H, W) with values in {0, 1}.

    Raises
    ------
    ValueError
        If ``h <= 1``, ``w <= 1``, cutoff parameters are out of range,
        or ``cutoff_low >= cutoff_high``.
    """
    _validate_spatial(h, w)
    _validate_cutoff(cutoff_low)
    _validate_cutoff(cutoff_high)
    if cutoff_low >= cutoff_high:
        raise ValueError(
            f"bandpass requires cutoff_low < cutoff_high, "
            f"got {cutoff_low} >= {cutoff_high}."
        )

    inner = lowpass_mask(h, w, cutoff_low, device=device, dtype=dtype)
    outer = lowpass_mask(h, w, cutoff_high, device=device, dtype=dtype)
    return outer - inner  # values are {0, 1}


def bandstop_mask(
    h: int,
    w: int,
    cutoff_low: float,
    cutoff_high: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a centred band-stop (notch) frequency mask.

    Frequencies whose distance from DC lies in the interval
    ``(cutoff_low, cutoff_high]`` are suppressed (mask = 0).
    All other frequencies are kept (mask = 1).

    Equivalent to::

        bandstop = 1.0 - bandpass

    Parameters
    ----------
    h : int
    w : int
    cutoff_low : float
        Inner (lower) normalised cutoff in (0, 1).
    cutoff_high : float
        Outer (upper) normalised cutoff in (0, 1).  Must be greater than
        ``cutoff_low``.
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    torch.Tensor
        Mask of shape (1, 1, H, W) with values in {0, 1}.

    Raises
    ------
    ValueError
        If ``h <= 1``, ``w <= 1``, cutoff parameters are out of range,
        or ``cutoff_low >= cutoff_high``.
    """
    _validate_spatial(h, w)
    # Validation of cutoffs and ordering is delegated to bandpass_mask.

    bp = bandpass_mask(h, w, cutoff_low, cutoff_high, device=device, dtype=dtype)
    return 1.0 - bp


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _frequency_grid(
    h: int,
    w: int,
    *,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return coordinate grids (y, x) relative to the frequency centre.

    The centre (DC) is at index ``(h // 2, w // 2)``.  Coordinates are
    unsigned distances in pixels.

    Returns
    -------
    grid_y : torch.Tensor  (H, W)
    grid_x : torch.Tensor  (H, W)
    """
    cy, cx = h // 2, w // 2

    # Create 1-D index vectors and compute distance from centre.
    # FFT convention: the centre is at h // 2, w // 2 after fftshift.
    y = torch.arange(h, device=device, dtype=torch.float32) - cy
    x = torch.arange(w, device=device, dtype=torch.float32) - cx

    # Broadcasting yields full (H, W) grids without explicit meshgrid.
    grid_y = y.view(-1, 1).expand(h, w)
    grid_x = x.view(1, -1).expand(h, w)
    return grid_y.abs(), grid_x.abs()


def _validate_spatial(h: int, w: int) -> None:
    """Raise ValueError if spatial dimensions are too small."""
    if not isinstance(h, int) or not isinstance(w, int):
        raise TypeError(f"h and w must be ints, got {type(h).__name__}, {type(w).__name__}.")
    if h <= 1:
        raise ValueError(f"h must be > 1, got {h}.")
    if w <= 1:
        raise ValueError(f"w must be > 1, got {w}.")


def _validate_cutoff(c: float) -> None:
    """Raise ValueError if cutoff is not in (0, 1]."""
    if not (0.0 < c <= 1.0):
        raise ValueError(f"Cutoff must be in (0, 1], got {c}.")


# ---------------------------------------------------------------------------
# Self-contained test suite
# ---------------------------------------------------------------------------


def _print_stats(name: str, mask: torch.Tensor, h: int, w: int) -> None:
    """Print summary statistics for a mask (used by tests only)."""
    total_pixels = h * w
    active = mask.sum().item()
    print(
        f"    {name:30s}: "
        f"min={mask.min().item():.3f}, "
        f"max={mask.max().item():.3f}, "
        f"mean={mask.mean().item():.4f}, "
        f"active={int(active):6d}/{total_pixels} "
        f"({100.0 * active / total_pixels:.1f}%)"
    )


if __name__ == "__main__":

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running mask tests on {_device} ...")

    H, W = 128, 128

    # --- 1. Correct output shape ------------------------------------------
    lp = lowpass_mask(H, W, 0.25, device=_device)
    assert lp.shape == (1, 1, H, W), f"lowpass shape: {lp.shape}"
    hp = highpass_mask(H, W, 0.25, device=_device)
    assert hp.shape == (1, 1, H, W), f"highpass shape: {hp.shape}"
    bp = bandpass_mask(H, W, 0.1, 0.25, device=_device)
    assert bp.shape == (1, 1, H, W), f"bandpass shape: {bp.shape}"
    bs = bandstop_mask(H, W, 0.1, 0.25, device=_device)
    assert bs.shape == (1, 1, H, W), f"bandstop shape: {bs.shape}"
    print("  [PASS] Correct output shape")

    # --- 2. Values in [0, 1] ----------------------------------------------
    for name, mask in [("lowpass", lp), ("highpass", hp),
                        ("bandpass", bp), ("bandstop", bs)]:
        assert mask.min() >= 0.0, f"{name} has negative values"
        assert mask.max() <= 1.0, f"{name} has values > 1"
    print("  [PASS] All values in [0, 1]")

    # --- 3. lowpass and highpass are complementary ------------------------
    assert torch.allclose(lp + hp, torch.ones_like(lp)), (
        "lowpass + highpass != 1.0"
    )
    print("  [PASS] lowpass + highpass = 1 (complementary)")

    # --- 4. bandpass and bandstop are complementary -----------------------
    assert torch.allclose(bp + bs, torch.ones_like(bp)), (
        "bandpass + bandstop != 1.0"
    )
    print("  [PASS] bandpass + bandstop = 1 (complementary)")

    # --- 5. bandpass is the difference of two lowpass masks ---------------
    inner = lowpass_mask(H, W, 0.1, device=_device)
    outer = lowpass_mask(H, W, 0.25, device=_device)
    assert torch.allclose(bp, outer - inner), (
        "bandpass != lowpass(0.25) - lowpass(0.1)"
    )
    print("  [PASS] bandpass = lowpass(c_high) - lowpass(c_low)")

    # --- 6. Mask statistics (sanity) --------------------------------------
    print("\n  --- Statistics (128×128, cutoff=0.25) ---")
    _print_stats("lowpass", lp, H, W)
    _print_stats("highpass", hp, H, W)
    _print_stats("bandpass (0.1–0.25)", bp, H, W)
    _print_stats("bandstop (0.1–0.25)", bs, H, W)

    # --- 7. Error cases ---------------------------------------------------
    _errors = 0
    try:
        lowpass_mask(1, 64, 0.25)
    except ValueError:
        _errors += 1
    try:
        lowpass_mask(64, 1, 0.25)
    except ValueError:
        _errors += 1
    try:
        lowpass_mask(64, 64, 0.0)
    except ValueError:
        _errors += 1
    try:
        lowpass_mask(64, 64, 1.5)
    except ValueError:
        _errors += 1
    try:
        bandpass_mask(64, 64, 0.3, 0.1)
    except ValueError:
        _errors += 1
    assert _errors == 5, f"Expected 5 errors caught, got {_errors}"
    print("  [PASS] All 5 error cases correctly rejected")

    print(f"\nAll tests passed on {_device}.")

