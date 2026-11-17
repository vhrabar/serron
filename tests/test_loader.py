"""
Which compiled extension the loader picks for the installed torch.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from serron import _cmake_ops


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The base names of the libraries ``_load_c_extension`` loads; nothing is really loaded."""
    calls: list[str] = []
    monkeypatch.setattr(_cmake_ops, "_check_torch_version", lambda: None)
    monkeypatch.setattr(_cmake_ops, "_load", lambda path: calls.append(os.path.basename(path)))
    return calls


def _wheel(monkeypatch: pytest.MonkeyPatch, *stems: str, with_rocm: bool | None = None) -> None:
    """Make the wheel contain exactly the libraries named by ``stems``; ``with_rocm`` sets what its build info says."""
    monkeypatch.setattr(_cmake_ops, "_find_library", lambda stem: f"/wheel/{stem}.so" if stem in stems else None)
    info = None if with_rocm is None else SimpleNamespace(WITH_ROCM=with_rocm, TORCH_VERSION="2.15.0")
    monkeypatch.setattr(_cmake_ops, "_build_info", lambda: info)


def _torch(monkeypatch: pytest.MonkeyPatch, *, cuda: str | None = None, hip: str | None = None, rocm: str | None = None) -> None:
    """Make the installed torch look like a CPU, CUDA or ROCm build."""
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.version, "rocm", rocm, raising=False)


def test_cpu_torch_loads_only_the_base_library(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch)
    _wheel(monkeypatch, "_C", "_C_cuda", "_C_rocm")

    _cmake_ops._load_c_extension()

    assert loaded == ["_C.so"]


def test_cuda_torch_loads_the_cuda_library_and_not_the_rocm_one(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch, cuda="13.4")
    _wheel(monkeypatch, "_C", "_C_cuda", "_C_rocm")

    _cmake_ops._load_c_extension()

    assert loaded == ["_C.so", "_C_cuda.so"]


def test_cuda_torch_without_the_cuda_library_still_loads(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch, cuda="13.4")
    _wheel(monkeypatch, "_C", "_C_rocm")

    _cmake_ops._load_c_extension()

    assert loaded == ["_C.so"]


def test_rocm_torch_loads_the_rocm_library_and_not_the_cuda_one(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    # A ROCm 10 build reports the HIP runtime's 7.15 as hip, and 10.0.0 as the release.
    _torch(monkeypatch, hip="7.15.26333", rocm="10.0.0")
    _wheel(monkeypatch, "_C", "_C_cuda", "_C_rocm")

    _cmake_ops._load_c_extension()

    assert loaded == ["_C.so", "_C_rocm.so"]


def test_rocm_torch_without_the_rocm_library_fails_at_import(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch, hip="7.15.26333", rocm="10.0.0")
    _wheel(monkeypatch, "_C", "_C_cuda")

    with pytest.raises(ImportError, match="_C_rocm"):
        _cmake_ops._load_c_extension()


def test_rocm_torch_with_a_build_that_left_rocm_out_says_so(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch, hip="7.15.26333", rocm="10.0.0")
    _wheel(monkeypatch, "_C", "_C_cuda", with_rocm=False)

    with pytest.raises(ImportError, match="SERRON_ROCM"):
        _cmake_ops._load_c_extension()


def test_rocm_older_than_ten_is_refused_before_any_library_loads(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    _torch(monkeypatch, hip="7.2.53211", rocm="7.2.1")
    _wheel(monkeypatch, "_C", "_C_rocm")

    with pytest.raises(ImportError, match=r"ROCm 10 or newer.*ROCm 7\.2\.1"):
        _cmake_ops._load_c_extension()


def test_rocm_without_a_release_attribute_falls_back_to_the_hip_version(
    monkeypatch: pytest.MonkeyPatch, loaded: list[str]
) -> None:
    _torch(monkeypatch, hip="6.4.43483", rocm=None)
    _wheel(monkeypatch, "_C", "_C_rocm")

    with pytest.raises(ImportError, match=r"ROCm 6\.4\.43483"):
        _cmake_ops._load_c_extension()
