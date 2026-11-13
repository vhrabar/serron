"""
Thresholds shipped with the library, and how they combine with what a machine saved.
"""

from __future__ import annotations

import copy
import importlib.resources
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import serron
from benchmarks import make_seed_table
from serron import _autotune as autotune
from serron import _seeds as seeds
from serron import _tuning_cache as cache
from serron._algorithms import device_fingerprint, get_separable_min_k, set_separable_min_k
from serron._autotune import DEFAULT_KS, calibrate
from tests.conftest import requires_cuda

# Captured at import, before the autouse fixture in conftest replaces it for every test.
_REAL_SEED_THRESHOLDS = seeds.seed_thresholds

_VALID = {
    "version": 1,
    "devices": {
        "12.0": {
            "measured_on": "Some GPU",
            "measured_with": {"serron": "0.5.2"},
            "thresholds": {"float32:fwd": 11, "float32:bwd": 21, "float16:fwd": 13},
        },
        "9.0": {"thresholds": {"float32:fwd": 15}},
    },
}


# --------------------------------------------------------------------------- #
# The file and its validation
# --------------------------------------------------------------------------- #


def test_the_shipped_file_is_a_resource_of_the_package() -> None:
    assert importlib.resources.files("serron").joinpath("_seeds.json").is_file()


def test_the_shipped_seeds_are_valid_and_leave_the_sweep_range_open() -> None:
    raw = json.loads(importlib.resources.files("serron").joinpath("_seeds.json").read_text(encoding="utf-8"))
    parsed = seeds.parse_seeds(raw)

    assert parsed, "a seed file with no architecture ships nothing"
    for capability, thresholds in parsed.items():
        for key, value in thresholds.items():
            # One past the longest length measured means separable never won: a limit of the sweep, not a result.
            assert min(DEFAULT_KS) <= value <= max(DEFAULT_KS), f"{capability} {key} = {value}"


def test_the_shipped_seeds_cover_every_calibrated_dtype_and_pass() -> None:
    expected = {
        autotune.threshold_key(dtype, backward=backward) for dtype in autotune.DEFAULT_DTYPES for backward in (False, True)
    }

    for capability, thresholds in seeds.parse_seeds(
        json.loads(importlib.resources.files("serron").joinpath("_seeds.json").read_text(encoding="utf-8"))
    ).items():
        assert set(thresholds) == expected, capability


def test_parse_accepts_a_valid_file() -> None:
    assert seeds.parse_seeds(_VALID) == {
        "12.0": {"float32:fwd": 11, "float32:bwd": 21, "float16:fwd": 13},
        "9.0": {"float32:fwd": 15},
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(version=2), "version"),
        (lambda d: d.pop("version"), "version"),
        (lambda d: d.update(devices=[]), "'devices'"),
        (lambda d: d["devices"].update({"sm_120": {"thresholds": {"float32:fwd": 11}}}), "compute capability"),
        (lambda d: d["devices"].update({"12": {"thresholds": {"float32:fwd": 11}}}), "compute capability"),
        (lambda d: d["devices"].update({"8.0": {"thresholds": {}}}), "non-empty"),
        (lambda d: d["devices"].update({"8.0": {}}), "non-empty"),
        (lambda d: d["devices"].update({"8.0": []}), "non-empty"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32": 11}), "threshold key"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:sideways": 11}), "threshold key"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:fwd": "11"}), "positive integer"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:fwd": 0}), "positive integer"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:fwd": -3}), "positive integer"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:fwd": 11.0}), "positive integer"),
        (lambda d: d["devices"]["12.0"]["thresholds"].update({"float32:fwd": True}), "positive integer"),
    ],
)
def test_parse_rejects_a_malformed_file(mutate: object, message: str) -> None:
    data = copy.deepcopy(_VALID)
    mutate(data)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        seeds.parse_seeds(data)


def test_parse_rejects_a_file_that_is_not_an_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        seeds.parse_seeds([])


def test_a_damaged_shipped_file_warns_once_and_yields_no_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def joinpath(self, _name: str) -> Broken:
            return self

        def read_text(self, encoding: str = "utf-8") -> str:
            return "{not json"

    monkeypatch.setattr(seeds.resources, "files", lambda _pkg: Broken())
    seeds.load_seeds.cache_clear()
    try:
        with pytest.warns(UserWarning, match="ignoring the shipped calibration seeds"):
            assert seeds.load_seeds() == {}
        assert seeds.load_seeds() == {}, "cached: no second warning"
    finally:
        seeds.load_seeds.cache_clear()


def test_seed_thresholds_returns_a_copy() -> None:
    first = _REAL_SEED_THRESHOLDS("12.0")
    assert first is not None
    first.clear()

    assert _REAL_SEED_THRESHOLDS("12.0"), "changing the answer must not change the loaded seeds"


def test_an_architecture_nobody_calibrated_has_no_seed() -> None:
    assert _REAL_SEED_THRESHOLDS("0.1") is None


# --------------------------------------------------------------------------- #
# Merging and writing
# --------------------------------------------------------------------------- #


def test_merge_replaces_one_architecture_and_keeps_the_others() -> None:
    entry = seeds.build_entry("New GPU", {"float32:fwd": 7}, {"serron": "0.6.0"})

    merged = seeds.merge_entry(_VALID, "12.0", entry)

    assert merged["devices"]["12.0"] == entry
    assert merged["devices"]["9.0"] == _VALID["devices"]["9.0"]
    assert _VALID["devices"]["12.0"]["measured_on"] == "Some GPU", "the input is not modified"


def test_merge_into_an_empty_file_makes_a_valid_one() -> None:
    merged = seeds.merge_entry({}, "8.9", seeds.build_entry("RTX 4090", {"float32:bwd": 21}, {}))

    assert seeds.parse_seeds(merged) == {"8.9": {"float32:bwd": 21}}


def test_merge_refuses_an_entry_that_would_not_parse() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        seeds.merge_entry({}, "8.9", seeds.build_entry("RTX 4090", {"float32:bwd": 0}, {}))


def test_dumps_is_stable_sorted_and_ends_in_a_newline() -> None:
    text = seeds.dumps(_VALID)

    assert text.endswith("}\n")
    assert json.loads(text) == _VALID
    assert text == seeds.dumps(json.loads(text)), "a round trip does not change a byte"
    assert text.index('"12.0"') < text.index('"9.0"') or text.index('"9.0"') < text.index('"12.0"')


def test_build_entry_sorts_thresholds_so_regeneration_diffs_cleanly() -> None:
    entry = seeds.build_entry("GPU", {"float32:fwd": 11, "bfloat16:bwd": 21}, {})

    assert list(entry["thresholds"]) == ["bfloat16:bwd", "float32:fwd"]


# --------------------------------------------------------------------------- #
# How a seed combines with what the machine saved (needs a GPU)
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Empty cache, nothing looked up yet, and a seed in force for this GPU's architecture."""
    monkeypatch.setenv("SERRON_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("SERRON_AUTOTUNE", raising=False)
    monkeypatch.delenv("SERRON_SEPARABLE_MIN_K", raising=False)
    seed = {"float32:fwd": 7, "float32:bwd": 9, "float16:fwd": 15, "float16:bwd": 17}
    monkeypatch.setattr(seeds, "seed_thresholds", lambda capability: dict(seed) if capability else None)
    autotune._forget_loaded()
    yield seed

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        like = torch.empty(0, dtype=dtype, device="cuda")
        set_separable_min_k(like, 0)
        set_separable_min_k(like, 0, backward=True)
    autotune._forget_loaded()


def _touch() -> None:
    """Run an op, which is what makes the library look for thresholds."""
    serron.erosion(torch.randn(1, 1, 16, 16, device="cuda"), torch.zeros(3, 3, device="cuda"))


def _threshold(dtype: torch.dtype, *, backward: bool = False) -> int:
    return get_separable_min_k(torch.empty(0, dtype=dtype, device="cuda"), backward=backward)


@requires_cuda
def test_a_seed_is_in_force_after_the_first_op(seeded: dict[str, int]) -> None:
    _touch()

    assert _threshold(torch.float32) == 7
    assert _threshold(torch.float32, backward=True) == 9
    assert _threshold(torch.float16) == 15
    assert _threshold(torch.float16, backward=True) == 17


@requires_cuda
def test_a_dtype_the_seed_omits_keeps_the_built_in_default(seeded: dict[str, int]) -> None:
    default = _threshold(torch.bfloat16)

    _touch()

    assert _threshold(torch.bfloat16) == default


@requires_cuda
def test_what_the_machine_saved_beats_the_seed_key_by_key(seeded: dict[str, int]) -> None:
    cache.store_thresholds(device_fingerprint(), {"float32:fwd": 5})

    _touch()

    assert _threshold(torch.float32) == 5, "saved wins where both have a value"
    assert _threshold(torch.float32, backward=True) == 9, "the seed fills what was not saved"
    assert _threshold(torch.float16) == 15


@requires_cuda
def test_a_corrupt_cache_still_leaves_the_seed(seeded: dict[str, int]) -> None:
    cache.cache_path(device_fingerprint()).write_text("{not json")

    with pytest.warns(UserWarning, match="serron: ignoring"):
        _touch()

    assert _threshold(torch.float32) == 7


@requires_cuda
def test_autotune_off_ignores_seeds_too(seeded: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERRON_AUTOTUNE", "0")
    default = _threshold(torch.float32)

    _touch()

    assert _threshold(torch.float32) == default


@requires_cuda
def test_an_architecture_without_a_seed_keeps_the_defaults(seeded: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeds, "seed_thresholds", lambda capability: None)
    default = _threshold(torch.float32)

    _touch()

    assert _threshold(torch.float32) == default


@requires_cuda
def test_the_environment_override_beats_the_seed() -> None:
    code = (
        "import torch, serron\n"
        "from serron._algorithms import get_separable_min_k\n"
        "serron.erosion(torch.randn(1, 1, 16, 16, device='cuda'), torch.zeros(3, 3, device='cuda'))\n"
        "print(get_separable_min_k(torch.empty(0, device='cuda')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "SERRON_SEPARABLE_MIN_K": "7", "SERRON_CACHE_DIR": str(Path(os.environ["SERRON_CACHE_DIR"]))},
    )

    assert out.stdout.strip().splitlines()[-1] == "7"


@requires_cuda
def test_calibrating_one_dtype_leaves_the_seed_for_the_others(seeded: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autotune, "sweep_separable_threshold", lambda *_a, **_k: 13)

    calibrate(dtypes=[torch.float32])

    assert _threshold(torch.float32) == 13, "what was measured replaces the seed"
    assert _threshold(torch.float16) == 15, "what was not measured keeps it"
    assert cache.load_thresholds(device_fingerprint()) == {"float32:fwd": 13, "float32:bwd": 13}, (
        "seeds are not saved as if measured"
    )


@requires_cuda
def test_calibrate_measures_even_where_a_seed_exists(seeded: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    swept: list[tuple[torch.dtype, bool]] = []

    def fake(dtype: torch.dtype, *, backward: bool, **_: object) -> int:
        swept.append((dtype, backward))
        return 11

    monkeypatch.setattr(autotune, "sweep_separable_threshold", fake)

    calibrate(dtypes=[torch.float32])

    assert swept == [(torch.float32, False), (torch.float32, True)]


@requires_cuda
def test_the_seed_shipped_for_this_gpu_is_what_a_fresh_process_uses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capability = device_fingerprint()["capability"]
    shipped = _REAL_SEED_THRESHOLDS(capability)
    if shipped is None:
        pytest.skip(f"no seed is shipped for compute capability {capability}")
    code = (
        "import torch, serron\n"
        "from serron._algorithms import get_separable_min_k\n"
        "serron.erosion(torch.randn(1, 1, 16, 16, device='cuda'), torch.zeros(3, 3, device='cuda'))\n"
        "print(get_separable_min_k(torch.empty(0, device='cuda')), "
        "get_separable_min_k(torch.empty(0, device='cuda'), backward=True))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"SERRON_SEPARABLE_MIN_K", "SERRON_AUTOTUNE"}}
    env["SERRON_CACHE_DIR"] = str(tmp_path)

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)

    assert out.stdout.strip().splitlines()[-1] == f"{shipped['float32:fwd']} {shipped['float32:bwd']}"


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #


@pytest.fixture
def generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[tuple[torch.dtype, bool]]]:
    """The generator with the sweep faked and a scratch seed file, plus a log of what it measured."""
    swept: list[tuple[torch.dtype, bool]] = []

    def fake(dtype: torch.dtype, *, backward: bool, **_: object) -> int:
        swept.append((dtype, backward))
        return 21 if backward else 11

    monkeypatch.setattr(make_seed_table, "sweep_separable_threshold", fake)
    return tmp_path / "seeds.json", swept


@requires_cuda
def test_generator_writes_this_gpus_entry(generator: tuple[Path, list[tuple[torch.dtype, bool]]]) -> None:
    output, swept = generator

    assert make_seed_table.main(["--output", str(output), "--reps", "2"]) == 0

    capability = device_fingerprint()["capability"]
    data = json.loads(output.read_text())
    assert seeds.parse_seeds(data)[capability] == {
        "bfloat16:bwd": 21,
        "bfloat16:fwd": 11,
        "float16:bwd": 21,
        "float16:fwd": 11,
        "float32:bwd": 21,
        "float32:fwd": 11,
    }
    assert len(swept) == 6
    assert data["devices"][capability]["measured_on"] == device_fingerprint()["name"]
    assert data["devices"][capability]["measured_with"]["reps"] == 2


@requires_cuda
def test_generator_keeps_other_architectures(generator: tuple[Path, list[tuple[torch.dtype, bool]]]) -> None:
    output, _ = generator
    output.write_text(seeds.dumps(_VALID | {"devices": {"9.0": _VALID["devices"]["9.0"]}}))

    make_seed_table.main(["--output", str(output)])

    devices = json.loads(output.read_text())["devices"]
    assert devices["9.0"] == _VALID["devices"]["9.0"]
    assert device_fingerprint()["capability"] in devices


@requires_cuda
def test_check_passes_when_the_shipped_entry_still_holds(generator: tuple[Path, list[tuple[torch.dtype, bool]]]) -> None:
    output, _ = generator
    make_seed_table.main(["--output", str(output)])
    before = output.read_text()

    assert make_seed_table.main(["--check", "--output", str(output)]) == 0
    assert output.read_text() == before, "--check never writes"


@requires_cuda
def test_check_fails_and_names_the_drift(
    generator: tuple[Path, list[tuple[torch.dtype, bool]]], capsys: pytest.CaptureFixture[str]
) -> None:
    output, _ = generator
    capability = device_fingerprint()["capability"]
    output.write_text(seeds.dumps({"version": 1, "devices": {capability: {"thresholds": {"float32:fwd": 5, "float32:bwd": 21}}}}))

    assert make_seed_table.main(["--check", "--output", str(output)]) == 1

    shown = capsys.readouterr().out
    assert "float32:fwd" in shown
    assert "differs" in shown


@requires_cuda
def test_check_fails_when_this_architecture_has_no_entry(
    generator: tuple[Path, list[tuple[torch.dtype, bool]]],
) -> None:
    output, _ = generator
    output.write_text(seeds.dumps(_VALID | {"devices": {"1.0": {"thresholds": {"float32:fwd": 3}}}}))

    assert make_seed_table.main(["--check", "--output", str(output)]) == 1


@requires_cuda
def test_generator_refuses_a_result_that_only_means_the_sweep_ended(
    generator: tuple[Path, list[tuple[torch.dtype, bool]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = generator
    monkeypatch.setattr(make_seed_table, "sweep_separable_threshold", lambda *_a, **_k: 64)

    with pytest.raises(SystemExit, match="never won"):
        make_seed_table.main(["--output", str(output), "--ks", "3,63"])

    assert not output.exists(), "nothing is written for a result that is not a measurement"
