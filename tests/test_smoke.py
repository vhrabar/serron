"""
Smoke tests: the package imports, exposes the API it advertises, and its layers' construction.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import serron
from serron import BorderMode, Operation
from serron import structuring_element as se
from serron.modules import _MorphologyNd
from tests.conftest import DEVICES

MODULES = [serron.Erosion2d, serron.Dilation2d, serron.Opening2d, serron.Closing2d]
FUNCTIONALS = ["erosion", "dilation", "opening", "closing", "gradient", "top_hat", "black_hat"]
SE_BUILDERS = ["square", "cross", "disk", "diamond", "from_tensor"]


def test_import_package() -> None:
    assert serron.__doc__
    assert serron.functional is importlib.import_module("serron.functional")
    assert serron.structuring_element is importlib.import_module("serron.structuring_element")


def test_public_api_matches_all() -> None:
    for name in serron.__all__:
        assert hasattr(serron, name), f"serron.__all__ advertises missing attribute {name!r}"


def test_all_is_sorted_and_unique() -> None:
    assert serron.__all__ == sorted(serron.__all__)
    assert len(serron.__all__) == len(set(serron.__all__))


@pytest.mark.parametrize("name", FUNCTIONALS)
def test_functional_exported(name: str) -> None:
    assert callable(getattr(serron.functional, name))
    assert callable(getattr(serron, name))


@pytest.mark.parametrize("name", SE_BUILDERS)
def test_structuring_element_builders_exported(name: str) -> None:
    assert callable(getattr(se, name))


def test_operation_enum_values() -> None:
    assert {op.value for op in Operation} == {
        "erode",
        "dilate",
        "open",
        "close",
        "gradient",
        "top_hat",
        "black_hat",
    }


def test_border_mode_enum_values() -> None:
    assert {b.value for b in BorderMode} == {"reflect", "replicate", "constant"}


@pytest.mark.parametrize("cls", MODULES)
@pytest.mark.parametrize("device", DEVICES)
def test_module_construction(cls: type[_MorphologyNd], device: str) -> None:
    channels, kernel_size = 4, 3
    layer = cls(channels, kernel_size, device=device)

    assert isinstance(layer, torch.nn.Module)
    assert layer.weight.shape == (channels, kernel_size, kernel_size)
    assert layer.weight.requires_grad
    assert layer.weight.device.type == device
    assert list(dict(layer.named_parameters())) == ["weight"]


@pytest.mark.parametrize("cls", MODULES)
def test_module_default_border_and_dtype(cls: type[_MorphologyNd]) -> None:
    layer = cls(2, 5)
    assert layer.border is BorderMode.REPLICATE
    assert layer.weight.dtype == torch.float32
    assert torch.all(layer.weight == 0)


@pytest.mark.parametrize("cls", MODULES)
def test_module_custom_dtype_and_border(cls: type[_MorphologyNd]) -> None:
    layer = cls(1, 3, border=BorderMode.REFLECT, dtype=torch.float64)
    assert layer.border is BorderMode.REFLECT
    assert layer.weight.dtype == torch.float64
