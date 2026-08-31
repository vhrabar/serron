"""
Shared fixtures, constants and pure-PyTorch reference implementations for the test suite
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from serron import BorderMode

# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])

PREFERRED_DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")

# --------------------------------------------------------------------------- #
# API surface the parametrised tests iterate over
# --------------------------------------------------------------------------- #

PRIMITIVES = ["erosion", "dilation"]
COMPOSITES = ["opening", "closing", "gradient", "top_hat", "black_hat"]
ALL_OPS = PRIMITIVES + COMPOSITES
BORDERS = [BorderMode.REFLECT, BorderMode.REPLICATE, BorderMode.CONSTANT]


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Every available device (``cpu`` plus ``cuda`` when present)."""
    return torch.device(request.param)


# --------------------------------------------------------------------------- #
# Reference implementations (pure PyTorch)
# --------------------------------------------------------------------------- #


def _pad(x: torch.Tensor, pt: int, pb: int, pl: int, pr: int, border: BorderMode, fill: float) -> torch.Tensor:
    """Pad an ``(N, C, H, W)`` tensor, one edge at a time.

    The pads are deliberately allowed to be asymmetric: an even-sized
    structuring element anchors at ``k // 2``, and that only lines up with the
    kernel's ``di - k // 2`` tap range when the top/left pad can differ from the
    bottom/right one.

    :param x: input batch of shape ``(N, C, H, W)``.
    :param pt: rows to add above.
    :param pb: rows to add below.
    :param pl: columns to add on the left.
    :param pr: columns to add on the right.
    :param border: how the padded region is filled.
    :param fill: value to use when ``border`` is ``CONSTANT``.
    :returns: the padded tensor.
    :raises ValueError: if ``border`` is not a supported mode.
    """
    pad = (pl, pr, pt, pb)
    if border is BorderMode.REPLICATE:
        return F.pad(x, pad, mode="replicate")
    if border is BorderMode.REFLECT:
        return F.pad(x, pad, mode="reflect")
    if border is BorderMode.CONSTANT:
        return F.pad(x, pad, mode="constant", value=fill)
    raise ValueError(f"unsupported border mode: {border!r}")


def _sweep(x: torch.Tensor, kernel: torch.Tensor, border: BorderMode, *, dilate: bool) -> torch.Tensor:
    """Run one additive grayscale erosion or dilation pass.

    The output keeps the input's spatial size and the anchor sits at
    ``kh // 2`` / ``kw // 2``, which is what the compiled kernels do for both
    odd and even kernels.

    :param x: input batch of shape ``(N, C, H, W)``.
    :param kernel: structuring element, either ``(C, kh, kw)`` or a shared
        ``(kh, kw)`` that gets broadcast across the channels.
    :param border: padding mode used for the sweep.
    :param dilate: ``True`` for dilation (max of ``patch + SE``), ``False`` for
        erosion (min of ``patch - SE``).
    :returns: the result, same shape as ``x``.
    :raises ValueError: if ``x`` is not 4-D, or the kernel's channel count does
        not match the input.
    """
    if x.dim() != 4:
        raise ValueError(f"expected a 4-D (N, C, H, W) input, got shape {tuple(x.shape)}")
    c = x.shape[1]
    kh, kw = kernel.shape[-2:]
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).expand(c, kh, kw)
    if kernel.shape[0] != c:
        raise ValueError(f"kernel channels {kernel.shape[0]} != input channels {c}")

    pt, pb = kh // 2, (kh - 1) // 2
    pl, pr = kw // 2, (kw - 1) // 2
    fill = float("-inf") if dilate else float("inf")
    xp = _pad(x, pt, pb, pl, pr, border, fill)

    # Sliding windows, shape (N, C, H, W, kh, kw).
    patches = xp.unfold(2, kh, 1).unfold(3, kw, 1)
    se = kernel.reshape(1, c, 1, 1, kh, kw).to(patches)
    if dilate:
        return (patches + se).amax(dim=(-1, -2))
    return (patches - se).amin(dim=(-1, -2))


def reference_erosion(x: torch.Tensor, kernel: torch.Tensor, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    return _sweep(x, kernel, border, dilate=False)


def reference_dilation(x: torch.Tensor, kernel: torch.Tensor, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    return _sweep(x, kernel, border, dilate=True)


def reference(op: str, x: torch.Tensor, kernel: torch.Tensor, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """Build the reference result for a named op out of the two primitives.

    :param op: op name, e.g. ``"erosion"``, ``"opening"`` or ``"top_hat"``.
    :param x: input batch of shape ``(N, C, H, W)``.
    :param kernel: structuring element handed to the primitives.
    :param border: padding mode.
    :returns: the reference output for ``op``.
    :raises KeyError: if ``op`` is not one of the known names.
    """

    def ero(t: torch.Tensor) -> torch.Tensor:
        return reference_erosion(t, kernel, border)

    def dil(t: torch.Tensor) -> torch.Tensor:
        return reference_dilation(t, kernel, border)

    builders = {
        "erosion": lambda: ero(x),
        "dilation": lambda: dil(x),
        "opening": lambda: dil(ero(x)),
        "closing": lambda: ero(dil(x)),
        "gradient": lambda: dil(x) - ero(x),
        "top_hat": lambda: x - dil(ero(x)),
        "black_hat": lambda: ero(dil(x)) - x,
    }
    if op not in builders:
        raise KeyError(f"unknown op {op!r}; expected one of {sorted(builders)}")
    return builders[op]()


# --------------------------------------------------------------------------- #
# Data factories
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng() -> torch.Generator:
    """Seeded CPU generator, so a failing case reproduces on the next run."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0x5E770)
    return g


def make_image(
    rng: torch.Generator,
    shape: tuple[int, ...] = (2, 3, 9, 11),
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = PREFERRED_DEVICE,
) -> torch.Tensor:
    """Draw a random image on CPU with ``rng``, then move it to ``device``.

    Drawing on CPU first keeps the values identical no matter where the test
    ends up running.

    :param rng: seeded generator to draw from.
    :param shape: image shape, ``(N, C, H, W)``.
    :param dtype: floating dtype of the result.
    :param device: device the returned tensor lands on.
    :returns: the random image.
    """
    return torch.randn(*shape, generator=rng, dtype=dtype).to(device)


def make_se(
    rng: torch.Generator,
    channels: int,
    size: int = 3,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = PREFERRED_DEVICE,
    shared: bool = False,
) -> torch.Tensor:
    """Draw a random additive structuring element.

    :param rng: seeded generator to draw from.
    :param channels: channel count; ignored when ``shared`` is ``True``.
    :param size: side length of the (square) kernel.
    :param dtype: floating dtype of the result.
    :param device: device the returned tensor lands on.
    :param shared: return a single ``(size, size)`` kernel shared across
        channels instead of a per-channel ``(channels, size, size)`` one.
    :returns: the structuring element.
    """
    shape = (size, size) if shared else (channels, size, size)
    return torch.randn(*shape, generator=rng, dtype=dtype).to(device)


def flat_se(
    channels: int,
    size: int = 3,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = PREFERRED_DEVICE,
    shared: bool = False,
) -> torch.Tensor:
    """Build a flat (all-zero) additive structuring element.

    With every tap at zero, erosion and dilation collapse to a plain min / max
    over the whole window.

    :param channels: channel count; ignored when ``shared`` is ``True``.
    :param size: side length of the (square) kernel.
    :param dtype: floating dtype of the result.
    :param device: device the returned tensor lands on.
    :param shared: return a single ``(size, size)`` kernel instead of a
        per-channel one.
    :returns: the flat structuring element.
    """
    shape = (size, size) if shared else (channels, size, size)
    return torch.zeros(*shape, dtype=dtype, device=device)
