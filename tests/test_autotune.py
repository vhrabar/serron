"""
Crossover search and the on-disk cache behind ``serron.calibrate``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Callable, Iterator
from math import prod as math_prod
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import serron
import serron.autograd as serron_autograd
from serron import _autotune as autotune
from serron import _flatness
from serron import _tuning_cache as cache
from serron._algorithms import (
    DeviceFingerprint,
    device_fingerprint,
    get_separable_min_k,
    set_separable_min_k,
)
from serron._autotune import (
    HYSTERESIS,
    LAUNCH_BOUND_ELEMENTS,
    apply_thresholds,
    calibrate,
    crossover,
    ensure_loaded,
    select_algorithm,
    shape_key,
    sweep_separable_threshold,
    threshold_key,
    tune_problem,
)
from serron._autotune import (
    autotune as autotune_shapes,
)
from serron._cmake_ops import ops
from serron.autograd import _ErodeFunction
from serron.enums import Algorithm
from tests.conftest import make_image, requires_cuda

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
        json.dumps({"version": 1, "fingerprint": dict(_fingerprint()), "thresholds": {"float32:fwd": -1}}),
        json.dumps({"version": 1, "fingerprint": dict(_fingerprint(name="other")), "thresholds": {"float32:fwd": 11}}),
    ],
    ids=["empty", "not_json", "wrong_type", "missing_keys", "future_version", "string_value", "negative", "foreign"],
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


# --------------------------------------------------------------------------- #
# Per-problem tuning
# --------------------------------------------------------------------------- #

_SHAPE = (1, 3, 128, 128)  # 49152 elements, under the real minimum, so tests lower it
_BIG = (1, 3, 256, 256)  # 196608


@pytest.fixture
def fake_times(monkeypatch: pytest.MonkeyPatch) -> dict[Algorithm, float]:
    """Replace timing with a table the test edits, and record how often a problem is timed."""
    table: dict[Algorithm, float] = {Algorithm.ELEMENT: 3.0, Algorithm.TILED: 2.0, Algorithm.SEPARABLE: 1.0}
    monkeypatch.setattr(autotune, "_time_candidates", lambda run, candidates, reps: {algo: table[algo] for algo in candidates})
    return table


@requires_cuda
def test_shape_key_buckets_by_size_class() -> None:
    a = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=1 << 20)
    b = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=(1 << 21) - 1)

    assert a == b == "shape:float32:fwd:flat:5x5:p21"
    assert a != shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=1 << 21)
    assert a != shape_key(torch.float32, backward=True, flat=True, k_h=5, k_w=5, numel=1 << 20)
    assert a != shape_key(torch.float16, backward=False, flat=True, k_h=5, k_w=5, numel=1 << 20)
    assert a != shape_key(torch.float32, backward=False, flat=False, k_h=5, k_w=5, numel=1 << 20)


@requires_cuda
def test_tune_problem_picks_a_clear_winner(fake_times: dict[Algorithm, float]) -> None:
    # k=5 flat: the heuristic picks tiled, separable is twice as fast.
    assert tune_problem(_SHAPE, torch.float32, 5, 5, flat=True, backward=False) == Algorithm.SEPARABLE


@requires_cuda
def test_tune_problem_keeps_the_heuristic_inside_the_margin(fake_times: dict[Algorithm, float]) -> None:
    fake_times[Algorithm.SEPARABLE] = fake_times[Algorithm.TILED] / (1 + HYSTERESIS / 2)

    assert tune_problem(_SHAPE, torch.float32, 5, 5, flat=True, backward=False) == Algorithm.AUTO


@requires_cuda
def test_tune_problem_needs_more_than_one_candidate(fake_times: dict[Algorithm, float]) -> None:
    # A non-flat SE too big for the tiled kernel leaves only the element-wise one.
    assert tune_problem((1, 1, 16, 16), torch.float32, 1001, 1001, flat=False, backward=False) == Algorithm.AUTO


@requires_cuda
def test_tune_problem_refuses_to_run_while_capturing_a_graph(fresh_state: None) -> None:
    x = torch.zeros(4, device="cuda")
    torch.cuda.synchronize()

    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        x + 1
        with pytest.raises(RuntimeError, match="captured"):
            tune_problem(_SHAPE, torch.float32, 5, 5, flat=True, backward=False)


@requires_cuda
def test_tune_problem_on_real_hardware_returns_a_feasible_algorithm(fresh_state: None) -> None:
    for backward in (False, True):
        winner = tune_problem(_BIG, torch.float32, 21, 21, flat=True, backward=backward, reps=3)

        assert winner in {Algorithm.AUTO, Algorithm.ELEMENT, Algorithm.TILED, Algorithm.SEPARABLE}


@requires_cuda
def test_autotune_records_saves_and_skips_launch_bound_shapes(fresh_state: None, fake_times: dict[Algorithm, float]) -> None:
    results = autotune_shapes([_BIG, (1, 1, 32, 32)], ks=[5], flat=[True])

    fwd = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=math_prod(_BIG))
    bwd = shape_key(torch.float32, backward=True, flat=True, k_h=5, k_w=5, numel=math_prod(_BIG))
    assert results == {fwd: int(Algorithm.SEPARABLE), bwd: int(Algorithm.SEPARABLE)}, "the small shape is skipped"
    assert cache.load_thresholds(device_fingerprint()) == results


@requires_cuda
def test_autotune_does_not_time_what_is_already_known(
    fresh_state: None, monkeypatch: pytest.MonkeyPatch, fake_times: dict[Algorithm, float]
) -> None:
    calls: list[object] = []
    real = autotune.tune_problem
    monkeypatch.setattr(autotune, "tune_problem", lambda *a, **k: calls.append(a) or real(*a, **k))

    autotune_shapes([_BIG], ks=[5], flat=[True])
    assert len(calls) == 2
    autotune_shapes([_BIG], ks=[5], flat=[True])
    assert len(calls) == 2, "a second call reuses the first"
    autotune_shapes([_BIG], ks=[5], flat=[True], force=True)
    assert len(calls) == 4


@requires_cuda
def test_autotune_results_reach_a_later_process(fresh_state: None, fake_times: dict[Algorithm, float]) -> None:
    autotune_shapes([_BIG], ks=[5], flat=[True])
    autotune._forget_loaded()  # a new process: nothing in memory, the file is all that is left
    x = torch.randn(_BIG, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")

    ensure_loaded(x.device)

    assert select_algorithm(x, kernel, True, backward=False) == int(Algorithm.SEPARABLE)


@requires_cuda
def test_autotune_survives_an_unwritable_cache(
    fresh_state: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_times: dict[Algorithm, float]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("SERRON_CACHE_DIR", str(blocker / "sub"))

    with pytest.warns(UserWarning) as record:
        results = autotune_shapes([_BIG], ks=[5], flat=[True])

    assert any("could not save" in str(w.message) for w in record)
    assert results


@requires_cuda
def test_autotune_needs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        autotune_shapes([_BIG])


@requires_cuda
def test_select_algorithm_is_a_noop_with_nothing_tuned(fresh_state: None) -> None:
    x = torch.randn(_BIG, device="cuda")

    assert select_algorithm(x, torch.zeros(3, 5, 5, device="cuda"), True, backward=False) == 0


@requires_cuda
def test_select_algorithm_ignores_cpu_inputs(fresh_state: None) -> None:
    autotune._shape_table["0|anything"] = 1

    assert select_algorithm(torch.randn(1, 1, 8, 8), torch.zeros(3, 3), True, backward=False) == 0


@requires_cuda
def test_tuned_algorithm_is_what_the_op_runs(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    def spy(name: str, real: object) -> object:
        def call(*args: object) -> object:
            seen[name] = int(args[-1])  # type: ignore[call-overload]
            return real(*args)  # type: ignore[operator]

        return call

    monkeypatch.setattr(
        serron_autograd, "ops", SimpleNamespace(erode=spy("fwd", ops.erode), erode_backward=spy("bwd", ops.erode_backward))
    )
    x = torch.randn(_BIG, device="cuda", requires_grad=True)
    kernel = torch.zeros(3, 5, 5, device="cuda", requires_grad=True)
    fwd = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=x.numel())
    bwd = shape_key(torch.float32, backward=True, flat=True, k_h=5, k_w=5, numel=x.numel())
    autotune._shape_table.update({f"0|{fwd}": int(Algorithm.ELEMENT), f"0|{bwd}": int(Algorithm.TILED)})

    serron.erosion(x, kernel).sum().backward()

    assert seen == {"fwd": int(Algorithm.ELEMENT), "bwd": int(Algorithm.TILED)}


@requires_cuda
def test_forcing_an_algorithm_beats_the_tuned_one(fresh_state: None) -> None:
    x = torch.randn(_BIG, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")
    fwd = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=x.numel())
    autotune._shape_table[f"0|{fwd}"] = int(Algorithm.ELEMENT)
    seen: list[int] = []
    real = ops.erode
    spy = SimpleNamespace(erode=lambda *a: (seen.append(int(a[-1])), real(*a))[1])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(serron_autograd, "ops", spy)
        _ErodeFunction.apply(x, kernel, 1, int(Algorithm.TILED))

    assert seen == [int(Algorithm.TILED)]


@requires_cuda
@pytest.mark.parametrize("op", ["erosion", "dilation"])
def test_tuned_algorithm_gives_the_same_bits(op: str, fresh_state: None, rng: torch.Generator) -> None:
    x = make_image(rng, _BIG, device=torch.device("cuda"))
    kernel = torch.zeros(3, 21, 21, device="cuda")
    expected = getattr(serron, op)(x, kernel)

    for algo in (Algorithm.ELEMENT, Algorithm.TILED, Algorithm.SEPARABLE):
        key = shape_key(x.dtype, backward=False, flat=True, k_h=21, k_w=21, numel=x.numel())
        autotune._shape_table[f"0|{key}"] = int(algo)
        assert torch.equal(getattr(serron, op)(x, kernel), expected), algo.name


# --------------------------------------------------------------------------- #
# Lazy tuning
# --------------------------------------------------------------------------- #


@pytest.fixture
def lazy(fresh_state: None, monkeypatch: pytest.MonkeyPatch, fake_times: dict[Algorithm, float]) -> list[dict[str, object]]:
    """Lazy tuning on, with the problems that get timed recorded."""
    monkeypatch.setenv("SERRON_AUTOTUNE", "1")
    calls: list[dict[str, object]] = []
    real = autotune.tune_problem
    monkeypatch.setattr(autotune, "tune_problem", lambda *a, **k: calls.append(k) or real(*a, **k))
    return calls


@requires_cuda
def test_lazy_tuning_tunes_a_problem_once_and_saves_it(lazy: list[dict[str, object]]) -> None:
    x = torch.randn(_BIG, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")

    for _ in range(3):
        serron.erosion(x, kernel)

    assert len(lazy) == 1
    key = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=x.numel())
    assert cache.load_thresholds(device_fingerprint()) == {key: int(Algorithm.SEPARABLE)}


@requires_cuda
def test_lazy_tuning_also_tunes_the_backward(lazy: list[dict[str, object]]) -> None:
    x = torch.randn(_BIG, device="cuda", requires_grad=True)
    kernel = torch.zeros(3, 5, 5, device="cuda", requires_grad=True)

    serron.erosion(x, kernel).sum().backward()

    assert [call["backward"] for call in lazy] == [False, True]
    assert len(cache.load_thresholds(device_fingerprint()) or {}) == 2


@requires_cuda
def test_lazy_tuning_is_off_by_default(
    fresh_state: None, monkeypatch: pytest.MonkeyPatch, fake_times: dict[Algorithm, float]
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(autotune, "tune_problem", lambda *a, **k: calls.append(a))

    serron.erosion(torch.randn(_BIG, device="cuda"), torch.zeros(3, 5, 5, device="cuda"))

    assert not calls
    assert cache.load_thresholds(device_fingerprint()) is None


@requires_cuda
def test_lazy_tuning_skips_launch_bound_sizes(lazy: list[dict[str, object]]) -> None:
    serron.erosion(torch.randn(1, 1, 64, 64, device="cuda"), torch.zeros(5, 5, device="cuda"))

    assert not lazy
    assert cache.load_thresholds(device_fingerprint()) is None


@requires_cuda
def test_lazy_tuning_never_runs_during_graph_capture(lazy: list[dict[str, object]]) -> None:
    x = torch.randn(_BIG, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")
    torch.cuda.synchronize()
    ensure_loaded(x.device)

    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        x + 1
        assert select_algorithm(x, kernel, True, backward=False) == 0

    assert not lazy
    assert cache.load_thresholds(device_fingerprint()) is None


@requires_cuda
def test_lazy_tuning_failure_falls_back_to_the_heuristic(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERRON_AUTOTUNE", "1")

    def boom(*_a: object, **_k: object) -> Algorithm:
        raise RuntimeError("boom")

    monkeypatch.setattr(autotune, "tune_problem", boom)
    x = torch.randn(_BIG, device="cuda")

    with pytest.warns(UserWarning, match="tuning failed"):
        out = serron.erosion(x, torch.zeros(3, 5, 5, device="cuda"))
    assert out.shape == x.shape
    # The failure is remembered, so it neither warns nor retries on every call.
    serron.erosion(x, torch.zeros(3, 5, 5, device="cuda"))


@requires_cuda
def test_lazy_tuning_leaves_the_callers_tensors_alone(lazy: list[object], rng: torch.Generator) -> None:
    x = make_image(rng, _BIG, device=torch.device("cuda")).requires_grad_()
    kernel = torch.zeros(3, 5, 5, device="cuda", requires_grad=True)
    before = x.detach().clone()

    out = serron.erosion(x, kernel)
    out.sum().backward()

    assert torch.equal(x.detach(), before)
    assert x.grad is not None
    assert torch.equal(out, serron.erosion(before, torch.zeros(3, 5, 5, device="cuda")))


@requires_cuda
def test_autotune_off_ignores_saved_problem_winners(fresh_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    key = shape_key(torch.float32, backward=False, flat=True, k_h=5, k_w=5, numel=math_prod(_BIG))
    _saved({key: int(Algorithm.ELEMENT)})
    monkeypatch.setenv("SERRON_AUTOTUNE", "0")
    x = torch.randn(_BIG, device="cuda")

    serron.erosion(x, torch.zeros(3, 5, 5, device="cuda"))

    assert not autotune._shape_table


@requires_cuda
def test_the_issues_launch_bound_example_is_skipped_and_one_row_more_is_not(
    lazy: list[dict[str, object]], fake_times: dict[Algorithm, float]
) -> None:
    edge = (1, 1, 256, 256)
    just_over = (1, 1, 257, 256)
    assert math_prod(edge) == LAUNCH_BOUND_ELEMENTS < math_prod(just_over)
    kernel = torch.zeros(5, 5, device="cuda")

    serron.erosion(torch.randn(edge, device="cuda"), kernel)
    assert not lazy, "1x1x256x256 is launch-bound"
    assert autotune_shapes([edge], ks=[5], flat=[True]) == {}

    serron.erosion(torch.randn(just_over, device="cuda"), kernel)
    assert len(lazy) == 1
    assert len(autotune_shapes([just_over], ks=[5], flat=[True])) == 2


@requires_cuda
@pytest.mark.parametrize("tuned", [False, True], ids=["cold_table", "tuned_entry"])
def test_a_captured_erosion_replays_the_eager_result(lazy: list[dict[str, object]], tuned: bool) -> None:
    x = torch.randn(_BIG, device="cuda")
    kernel = _flatness.mark(torch.zeros(3, 21, 21, device="cuda"), True)  # a known flatness needs no read-back
    ensure_loaded(x.device)  # lazy tuning is now armed, as it is after any earlier call
    if tuned:
        key = shape_key(x.dtype, backward=False, flat=True, k_h=21, k_w=21, numel=x.numel())
        autotune._shape_table[f"0|{key}"] = int(Algorithm.ELEMENT)
    expected = serron.erosion(x, kernel)  # warms the kernels, and tunes the cold problem outside the capture
    lazy.clear()
    if not tuned:
        autotune._shape_table.clear()  # cold again, so the capture below is what meets an unseen problem

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = serron.erosion(x, kernel)
    graph.replay()
    torch.cuda.synchronize()

    assert not lazy, "nothing is timed inside a capture"
    assert torch.equal(captured, expected)


# --------------------------------------------------------------------------- #
# Screening candidates
# --------------------------------------------------------------------------- #


def _fake_timer(
    monkeypatch: pytest.MonkeyPatch, speeds: dict[Algorithm, float]
) -> tuple[Callable[[Algorithm], None], list[tuple[Algorithm, int]]]:
    """
    Replace ``time_ms`` so a candidate takes ``speeds[algo]``.

    :returns: a runner to hand to ``_time_candidates``, and the log of ``(algorithm, repetitions)`` asked for.
    """
    calls: list[tuple[Algorithm, int]] = []
    current: list[Algorithm] = []

    def run(algo: Algorithm) -> None:
        current[:] = [algo]

    def fake(fn: Callable[[], object], reps: int = 30, warmup: int = 1) -> float:
        fn()
        calls.append((current[0], reps))
        return speeds[current[0]]

    monkeypatch.setattr(autotune, "time_ms", fake)
    return run, calls


def test_screening_gives_full_repetitions_only_to_candidates_near_the_best(monkeypatch: pytest.MonkeyPatch) -> None:
    speeds = {Algorithm.ELEMENT: 9.0, Algorithm.TILED: 1.1, Algorithm.SEPARABLE: 1.0}
    run, calls = _fake_timer(monkeypatch, speeds)

    times = autotune._time_candidates(run, list(speeds), reps=20)

    assert times == speeds, "every candidate gets a time, a dropped one keeps its quick time"
    full = {algo for algo, reps in calls if reps == 20}
    quick = [algo for algo, reps in calls if reps == autotune.SCREEN_REPS]
    assert full == {Algorithm.TILED, Algorithm.SEPARABLE}
    assert sorted(quick, key=int) == [Algorithm.ELEMENT, Algorithm.TILED, Algorithm.SEPARABLE]


def test_screening_keeps_a_candidate_at_the_slowdown_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    speeds = {Algorithm.TILED: 2.0 * autotune.SCREEN_SLOWDOWN / 2, Algorithm.SEPARABLE: 1.0}
    run, calls = _fake_timer(monkeypatch, speeds)

    autotune._time_candidates(run, list(speeds), reps=20)

    assert {algo for algo, reps in calls if reps == 20} == set(speeds)


def test_screening_is_skipped_when_there_is_nothing_to_save(monkeypatch: pytest.MonkeyPatch) -> None:
    speeds = {Algorithm.ELEMENT: 9.0, Algorithm.TILED: 1.0}

    run, calls = _fake_timer(monkeypatch, speeds)
    autotune._time_candidates(run, list(speeds), reps=autotune.SCREEN_REPS)
    assert {reps for _, reps in calls} == {autotune.SCREEN_REPS}, "too few reps to be worth a second pass"

    calls.clear()
    autotune._time_candidates(run, [Algorithm.TILED], reps=20)
    assert calls == [(Algorithm.TILED, 20)], "a lone candidate has nothing to be screened against"
