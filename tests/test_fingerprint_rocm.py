"""
The tuning fingerprint and the seed file on a ROCm torch, with the device properties patched in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from serron import _seeds as seeds
from serron import _tuning_cache as cache
from serron._algorithms import DeviceFingerprint, device_fingerprint


def _rocm(monkeypatch: pytest.MonkeyPatch, arch: str = "gfx942:sramecc+:xnack-", release: str = "10.0.0") -> None:
    """Make torch look like a ROCm 10 build with an MI300X as the current device."""
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)
    monkeypatch.setattr(torch.version, "rocm", release, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = SimpleNamespace(
        name="AMD Instinct MI300X",
        major=9,
        minor=4,
        multi_processor_count=304,
        shared_memory_per_block=65536,
        gcnArchName=arch,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: props)


def test_rocm_fingerprint_names_the_architecture_and_the_rocm_release(monkeypatch: pytest.MonkeyPatch) -> None:
    _rocm(monkeypatch)

    fingerprint = device_fingerprint()

    assert fingerprint["name"] == "AMD Instinct MI300X"
    assert fingerprint["capability"] == "gfx942"
    assert fingerprint["sm_count"] == 304
    assert fingerprint["max_optin_smem"] == 65536
    assert fingerprint["rocm"] == "10.0.0"
    assert "cuda" not in fingerprint


def test_rocm_architecture_feature_flags_do_not_change_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    _rocm(monkeypatch, arch="gfx942:sramecc+:xnack-")
    with_flags = device_fingerprint()
    _rocm(monkeypatch, arch="gfx942")

    assert device_fingerprint() == with_flags


def test_a_cuda_and_a_rocm_machine_do_not_share_a_cache_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    _rocm(monkeypatch)
    rocm = device_fingerprint()
    cuda = DeviceFingerprint(
        name=rocm["name"],
        capability="9.4",
        sm_count=rocm["sm_count"],
        max_optin_smem=rocm["max_optin_smem"],
        serron=rocm["serron"],
        cuda="13.4",
    )

    assert cache.cache_path(rocm) != cache.cache_path(cuda)


def test_a_different_rocm_release_is_a_different_cache_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    _rocm(monkeypatch, release="10.0.0")
    first = device_fingerprint()
    _rocm(monkeypatch, release="10.1.0")

    assert cache.cache_path(device_fingerprint()) != cache.cache_path(first)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx90a", "gfx1100", "gfx1036"])
def test_seed_file_accepts_rocm_architectures(architecture: str) -> None:
    parsed = seeds.parse_seeds(
        {"version": 1, "devices": {architecture: {"thresholds": {"float32:fwd": 11}}, "12.0": {"thresholds": {"float32:fwd": 9}}}}
    )

    assert parsed[architecture] == {"float32:fwd": 11}
    assert parsed["12.0"] == {"float32:fwd": 9}


@pytest.mark.parametrize("bad", ["gfx", "GFX942", "gfx-942", "sm_120", "942"])
def test_seed_file_rejects_malformed_architectures(bad: str) -> None:
    with pytest.raises(ValueError, match=r"compute capability.*ROCm architecture"):
        seeds.parse_seeds({"version": 1, "devices": {bad: {"thresholds": {"float32:fwd": 11}}}})
