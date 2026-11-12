"""
Crossover search and the on-disk cache behind ``serron.calibrate``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import serron
from serron import _tuning_cache as cache
from serron import autotune
from serron._algorithms import (
    DeviceFingerprint,
    device_fingerprint,
    get_separable_min_k,
    set_separable_min_k,
)
from serron.autotune import (
    HYSTERESIS,
    apply_thresholds,
    calibrate,
    crossover,
    ensure_loaded,
    sweep_separable_threshold,
    threshold_key,
)
from tests.conftest import requires_cuda

_KS = [3, 5, 7, 11, 21]


def _fingerprint(**changes: str | int) -> DeviceFingerprint:
    fingerprint = DeviceFingerprint(
        name="GPU", capability="9.0", sm_count=132, max_optin_smem=232448, cuda="13.2", serron="0.5.1"
    )
    return DeviceFingerprint(**{**fingerprint, **changes})  # type: ignore[typeddict-item]


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SERRON_CACHE_DIR", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# crossover
# --------------------------------------------------------------------------- #


def test_crossover_is_first_length_whose_win_lasts() -> None:
    assert crossover(_KS, [2, 2, 2, 1, 1], [1, 1, 3, 3, 3]) == 7


def test_crossover_ignores_a_win_that_does_not_last() -> None:
    # Separable wins at k=5 by noise but loses again at k=7.
    assert crossover(_KS, [2, 1, 2, 1, 1], [1, 3, 1, 3, 3]) == 11


def test_crossover_needs_the_hysteresis_margin() -> None:
    barely = 1 / (1 + HYSTERESIS / 2)
    clearly = 1 / (1 + 2 * HYSTERESIS)

    assert crossover([3, 5], [1, barely], [1, 1]) is None
    assert crossover([3, 5], [1, clearly], [1, 1]) == 5


def test_crossover_is_none_when_separable_loses_at_the_longest_length() -> None:
    assert crossover(_KS, [1, 1, 1, 1, 2], [3, 3, 3, 3, 1]) is None


def test_crossover_is_the_shortest_length_when_separable_always_wins() -> None:
    assert crossover(_KS, [1] * 5, [3] * 5) == 3


def test_crossover_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        crossover([3, 5], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="ascending"):
        crossover([5, 3], [1.0, 1.0], [1.0, 1.0])


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #


def test_cache_dir_honours_the_environment(cache_root: Path) -> None:
    assert cache.cache_dir() == cache_root


def test_cache_dir_defaults_under_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERRON_CACHE_DIR", raising=False)

    assert cache.cache_dir() == Path.home() / ".cache" / "serron"


def test_cache_path_differs_for_every_fingerprint_field(cache_root: Path) -> None:
    base = cache.cache_path(_fingerprint())

    for change in ({"name": "other"}, {"capability": "8.0"}, {"sm_count": 1}, {"cuda": "12.0"}, {"serron": "9"}):
        assert cache.cache_path(_fingerprint(**change)) != base, change


def test_cold_cache_loads_nothing_silently(cache_root: Path, recwarn: pytest.WarningsRecorder) -> None:
    assert cache.load_thresholds(_fingerprint()) is None
    assert not recwarn.list


def test_store_then_load_round_trips(cache_root: Path) -> None:
    thresholds = {"float32:fwd": 11, "float32:bwd": 21}

    cache.store_thresholds(_fingerprint(), thresholds)

    assert cache.load_thresholds(_fingerprint()) == thresholds


def test_store_keeps_entries_it_does_not_mention(cache_root: Path) -> None:
    cache.store_thresholds(_fingerprint(), {"float32:fwd": 11})
    cache.store_thresholds(_fingerprint(), {"float16:fwd": 15, "float32:fwd": 13})

    assert cache.load_thresholds(_fingerprint()) == {"float32:fwd": 13, "float16:fwd": 15}


def test_store_leaves_no_temporary_files(cache_root: Path) -> None:
    cache.store_thresholds(_fingerprint(), {"float32:fwd": 11})

    assert [p.name for p in cache_root.iterdir()] == [cache.cache_path(_fingerprint()).name]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{not json",
        "[]",
        json.dumps({"version": 1}),
        json.dumps({"version": 99, "fingerprint": dict(_fingerprint()), "thresholds": {}}),
        json.dumps({"version": 1, "fingerprint": dict(_fingerprint()), "thresholds": {"float32:fwd": "11"}}),
        json.dumps({"version": 1, "fingerprint": dict(_fingerprint()), "thresholds": {"float32:fwd": 0}}),
        json.dumps({"version": 1, "fingerprint": dict(_fingerprint(name="other")), "thresholds": {"float32:fwd": 11}}),
    ],
    ids=["empty", "not_json", "wrong_type", "missing_keys", "future_version", "string_value", "zero", "foreign"],
)
def test_bad_cache_falls_back_with_a_warning(cache_root: Path, text: str) -> None:
    cache.cache_path(_fingerprint()).write_text(text)

    with pytest.warns(UserWarning, match="serron: ignoring"):
        assert cache.load_thresholds(_fingerprint()) is None


def test_store_repairs_a_corrupt_file(cache_root: Path) -> None:
    cache.cache_path(_fingerprint()).write_text("{not json")

    with pytest.warns(UserWarning):
        cache.store_thresholds(_fingerprint(), {"float32:fwd": 11})

    assert cache.load_thresholds(_fingerprint()) == {"float32:fwd": 11}


def _write_many(root: str, dtype: str, count: int) -> None:
    os.environ["SERRON_CACHE_DIR"] = root
    for i in range(count):
        cache.store_thresholds(_fingerprint(), {f"{dtype}:fwd": i + 1})


def test_concurrent_writers_never_leave_a_broken_file(cache_root: Path, recwarn: pytest.WarningsRecorder) -> None:
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_write_many, args=(str(cache_root), name, 25)) for name in ("a", "b", "c")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert all(worker.exitcode == 0 for worker in workers)
    loaded = cache.load_thresholds(_fingerprint())
    assert loaded is not None
    assert set(loaded) <= {"a:fwd", "b:fwd", "c:fwd"}
    assert not recwarn.list
    assert [p.suffix for p in cache_root.iterdir()] == [".json"]


def test_unwritable_directory_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("SERRON_CACHE_DIR", str(blocker / "sub"))

    with pytest.raises(OSError):
        cache.store_thresholds(_fingerprint(), {"float32:fwd": 11})


# --------------------------------------------------------------------------- #
# sweep and apply (need a GPU)
# --------------------------------------------------------------------------- #


def test_threshold_key_names_dtype_and_pass() -> None:
    assert threshold_key(torch.float32, backward=False) == "float32:fwd"
    assert threshold_key(torch.bfloat16, backward=True) == "bfloat16:bwd"


@requires_cuda
@pytest.mark.parametrize("backward", [False, True])
def test_sweep_returns_a_length_in_or_just_past_the_range(backward: bool) -> None:
    found = sweep_separable_threshold(torch.float32, backward=backward, ks=[3, 9, 21], shape=(1, 1, 64, 64), reps=3)

    assert found in {3, 9, 21, 22}


@requires_cuda
def test_sweep_leaves_the_applied_thresholds_alone() -> None:
    like = torch.empty(0, device="cuda")
    before = get_separable_min_k(like)

    sweep_separable_threshold(torch.float32, backward=False, ks=[3, 9], shape=(1, 1, 32, 32), reps=2)

    assert get_separable_min_k(like) == before


@requires_cuda
def test_sweep_rejects_an_empty_range() -> None:
    with pytest.raises(RuntimeError, match="could be measured"):
        sweep_separable_threshold(torch.float32, backward=False, ks=[], shape=(1, 1, 16, 16), reps=1)


@requires_cuda
def test_apply_thresholds_sets_each_dtype_and_pass_and_skips_unknown_keys() -> None:
    if os.environ.get("SERRON_SEPARABLE_MIN_K"):
        pytest.skip("SERRON_SEPARABLE_MIN_K overrides every threshold")
    f32 = torch.empty(0, dtype=torch.float32, device="cuda")
    f16 = torch.empty(0, dtype=torch.float16, device="cuda")

    try:
        apply_thresholds({"float32:fwd": 7, "float32:bwd": 15, "float16:fwd": 9, "notadtype:fwd": 3, "float32:sideways": 5})

        assert get_separable_min_k(f32) == 7
        assert get_separable_min_k(f32, backward=True) == 15
        assert get_separable_min_k(f16) == 9
    finally:
        for like in (f32, f16):
            set_separable_min_k(like, 0)
            set_separable_min_k(like, 0, backward=True)


# --------------------------------------------------------------------------- #
# calibrate and lazy loading (need a GPU)
# --------------------------------------------------------------------------- #

_DTYPES = (torch.float32, torch.float16)
_FAST = {"ks": [3, 9, 21], "shape": (1, 1, 32, 32), "reps": 2}


@pytest.fixture
def fresh_state(cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Empty cache, nothing looked up yet, and the thresholds put back afterwards."""
    monkeypatch.delenv("SERRON_AUTOTUNE", raising=False)
    monkeypatch.delenv("SERRON_SEPARABLE_MIN_K", raising=False)
    autotune._forget_loaded()
    yield

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        like = torch.empty(0, dtype=dtype, device="cuda")
        set_separable_min_k(like, 0)
        set_separable_min_k(like, 0, backward=True)
    autotune._forget_loaded()


def _saved(thresholds: dict[str, int]) -> None:
    cache.store_thresholds(device_fingerprint(), thresholds)


@requires_cuda
def test_calibrate_applies_saves_and_returns_the_table(fresh_state: None) -> None:
    table = calibrate(dtypes=_DTYPES, **_FAST)

    assert set(table) == {"float32:fwd", "float32:bwd", "float16:fwd", "float16:bwd"}
    assert cache.load_thresholds(device_fingerprint()) == table
    f32 = torch.empty(0, device="cuda")
    assert get_separable_min_k(f32) == table["float32:fwd"]
    assert get_separable_min_k(f32, backward=True) == table["float32:bwd"]


@requires_cuda
def test_calibrate_reuses_saved_entries_and_measures_only_the_rest(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved({"float32:fwd": 7, "float32:bwd": 9})
    swept: list[tuple[torch.dtype, bool]] = []

    def fake_sweep(dtype: torch.dtype, *, backward: bool, **_: object) -> int:
        swept.append((dtype, backward))
        return 13

    monkeypatch.setattr(autotune, "sweep_separable_threshold", fake_sweep)

    table = calibrate(dtypes=_DTYPES)

    assert swept == [(torch.float16, False), (torch.float16, True)]
    assert table == {"float32:fwd": 7, "float32:bwd": 9, "float16:fwd": 13, "float16:bwd": 13}


@requires_cuda
def test_calibrate_force_measures_everything_again(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _saved({"float32:fwd": 7, "float32:bwd": 9})
    monkeypatch.setattr(autotune, "sweep_separable_threshold", lambda *_a, **_k: 15)

    table = calibrate(dtypes=[torch.float32], force=True)

    assert table == {"float32:fwd": 15, "float32:bwd": 15}
    assert cache.load_thresholds(device_fingerprint()) == table


@requires_cuda
def test_calibrate_survives_an_unwritable_cache(fresh_state: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("SERRON_CACHE_DIR", str(blocker / "sub"))
    monkeypatch.setattr(autotune, "sweep_separable_threshold", lambda *_a, **_k: 13)

    with pytest.warns(UserWarning) as record:
        table = calibrate(dtypes=[torch.float32])

    assert any("could not save" in str(w.message) for w in record)
    assert get_separable_min_k(torch.empty(0, device="cuda")) == table["float32:fwd"]


@requires_cuda
def test_first_cuda_op_loads_the_saved_thresholds(fresh_state: None) -> None:
    _saved({"float32:fwd": 5, "float32:bwd": 6})
    x = torch.randn(1, 1, 16, 16, device="cuda")

    serron.erosion(x, torch.zeros(3, 3, device="cuda"))

    assert get_separable_min_k(x) == 5
    assert get_separable_min_k(x, backward=True) == 6


@requires_cuda
def test_saved_thresholds_change_what_auto_picks(fresh_state: None) -> None:
    from serron._algorithms import heuristic_algorithm
    from serron.enums import Algorithm

    x = torch.randn(1, 3, 32, 32, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")
    assert heuristic_algorithm(x, kernel) == Algorithm.TILED

    _saved({"float32:fwd": 5})
    serron.dilation(x, kernel)

    assert heuristic_algorithm(x, kernel) == Algorithm.SEPARABLE


@requires_cuda
def test_reconstruction_loads_the_saved_thresholds_too(fresh_state: None) -> None:
    _saved({"float32:fwd": 5})
    x = torch.rand(1, 1, 16, 16, device="cuda")

    serron.reconstruction_by_dilation(x * 0.5, x, torch.zeros(3, 3, device="cuda"))

    assert get_separable_min_k(x) == 5


@requires_cuda
def test_lookup_happens_once_per_device(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    real = cache.load_thresholds
    monkeypatch.setattr(cache, "load_thresholds", lambda fp: calls.append(fp) or real(fp))
    x = torch.randn(1, 1, 16, 16, device="cuda")
    kernel = torch.zeros(3, 3, device="cuda")

    for _ in range(3):
        serron.erosion(x, kernel)

    assert len(calls) == 1


@requires_cuda
def test_autotune_off_ignores_the_saved_thresholds(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERRON_AUTOTUNE", "0")
    default = get_separable_min_k(torch.empty(0, device="cuda"))
    _saved({"float32:fwd": 5})

    serron.erosion(torch.randn(1, 1, 16, 16, device="cuda"), torch.zeros(3, 3, device="cuda"))

    assert get_separable_min_k(torch.empty(0, device="cuda")) == default


@requires_cuda
def test_bad_cache_never_breaks_an_op(fresh_state: None) -> None:
    cache.cache_path(device_fingerprint()).write_text("{not json")
    x = torch.randn(1, 1, 16, 16, device="cuda")

    with pytest.warns(UserWarning, match="serron: ignoring"):
        out = serron.erosion(x, torch.zeros(3, 3, device="cuda"))

    assert out.shape == x.shape


@requires_cuda
def test_no_lookup_while_capturing_a_graph(fresh_state: None) -> None:
    _saved({"float32:fwd": 5})
    like = torch.empty(0, device="cuda")
    default = get_separable_min_k(like)
    x = torch.zeros(4, device="cuda")
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ensure_loaded(x.device)
        x + 1

    assert get_separable_min_k(like) == default, "nothing was applied during capture"
    ensure_loaded(x.device)
    assert get_separable_min_k(like) == 5, "the next call outside capture does the lookup"


@requires_cuda
def test_sweep_refuses_to_run_while_capturing_a_graph(fresh_state: None) -> None:
    x = torch.zeros(4, device="cuda")
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x + 1
        with pytest.raises(RuntimeError, match="captured"):
            sweep_separable_threshold(torch.float32, backward=False, **_FAST)


def test_ensure_loaded_ignores_cpu_tensors() -> None:
    ensure_loaded(torch.device("cpu"))


def test_calibrate_needs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        calibrate()
