"""
Algorithm selection as the compiled ops see it, for the autotuner.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TypedDict

import torch

from . import _flatness
from ._cmake_ops import ops
from .enums import Algorithm

__all__ = [
    "DeviceFingerprint",
    "device_fingerprint",
    "feasible_algorithms",
    "get_separable_min_k",
    "heuristic_algorithm",
    "set_separable_min_k",
]


class DeviceFingerprint(TypedDict):
    """What a tuning result depends on besides the problem."""

    name: str
    capability: str
    sm_count: int
    max_optin_smem: int
    cuda: str
    serron: str


def _flat(kernel: torch.Tensor, flat: bool | None) -> bool:
    return _flatness.is_flat(kernel) if flat is None else flat


def feasible_algorithms(
    input_: torch.Tensor, kernel: torch.Tensor, *, flat: bool | None = None, backward: bool = False
) -> tuple[Algorithm, ...]:
    """
    Algorithms that can run a call, in ascending order of their encoding.

    A forced algorithm outside this set makes the op raise, so these are the candidates worth benchmarking.

    :param input_: CUDA image of the call; its device and dtype are what count.
    :param kernel: CUDA structuring element of the call.
    :param flat: whether ``kernel`` is flat; read off ``kernel`` when ``None``.
    :param backward: whether the call is the backward pass, which needs more shared memory in its line passes.
    :returns: the feasible algorithms, never :attr:`Algorithm.AUTO`.
    """
    mask: int = ops.feasible_algos(input_, kernel, _flat(kernel, flat), backward)
    return tuple(algo for algo in Algorithm if algo is not Algorithm.AUTO and mask >> algo & 1)


def heuristic_algorithm(
    input_: torch.Tensor, kernel: torch.Tensor, *, flat: bool | None = None, backward: bool = False
) -> Algorithm:
    """
    The algorithm the library runs for a call when it is not told which.

    :param input_: CUDA image of the call.
    :param kernel: CUDA structuring element of the call.
    :param flat: whether ``kernel`` is flat; read off ``kernel`` when ``None``.
    :param backward: whether the call is the backward pass.
    :returns: the algorithm auto resolves to.
    """
    return Algorithm(ops.heuristic_algo(input_, kernel, _flat(kernel, flat), backward))


def get_separable_min_k(like: torch.Tensor, *, backward: bool = False) -> int:
    """
    Window length from which the separable path is preferred.

    :param like: CUDA tensor whose device and dtype pick the threshold.
    :param backward: whether to read the backward pass's threshold.
    :returns: the window length, which ``SERRON_SEPARABLE_MIN_K`` overrides when set.
    """
    result: int = ops.get_separable_min_k(like, backward)
    return result


def set_separable_min_k(like: torch.Tensor, min_k: int, *, backward: bool = False) -> int:
    """
    Set the window length from which the separable path is preferred, for one device, dtype and pass.

    The setting lasts for the process.

    :param like: CUDA tensor whose device and dtype pick the threshold.
    :param min_k: new window length, or a value ``<= 0`` to restore the default.
    :param backward: whether to set the backward pass's threshold.
    :returns: the threshold in effect afterwards, which ``SERRON_SEPARABLE_MIN_K`` overrides when set.
    """
    result: int = ops.set_separable_min_k(like, backward, min_k)
    return result


def device_fingerprint(device: torch.device | int | None = None) -> DeviceFingerprint:
    """
    What a tuning result depends on besides the problem: the GPU, the toolkit and this library.

    :param device: CUDA device, the current one when ``None``.
    :returns: JSON-serialisable fields for a cache key: device name, compute capability, SM count, opt-in shared
    memory, CUDA version and serron version.
    :raises RuntimeError: if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("serron: device_fingerprint needs a CUDA device")
    props = torch.cuda.get_device_properties(device)
    try:
        serron_version = _version("serron")
    except PackageNotFoundError:
        serron_version = "unknown"
    return DeviceFingerprint(
        name=props.name,
        capability=f"{props.major}.{props.minor}",
        sm_count=props.multi_processor_count,
        max_optin_smem=int(getattr(props, "shared_memory_per_block_optin")),  # noqa: B009
        cuda=torch.version.cuda or "unknown",
        serron=serron_version,
    )
