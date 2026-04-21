"""Tests for PIP-PD pivot detection and fractal/PIP hybrid merging."""
from __future__ import annotations

import math

import numpy as np
import pytest

from chartslp.pivots import (
    DIST_ED,
    DIST_PD,
    DIST_VD,
    detect_pivots_hybrid,
    detect_pivots_pip_pd,
    find_pips,
)

# ---------------------------------------------------------------------------
# find_pips — core algorithm
# ---------------------------------------------------------------------------

def test_find_pips_endpoints_always_first() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pips_x, pips_y = find_pips(series, n_pips=2, dist_measure=DIST_PD)
    assert pips_x == [0, len(series) - 1]
    assert pips_y == [1.0, 6.0]


def test_find_pips_triangle_picks_apex() -> None:
    """Perfect triangle 1-2-3-4-5-4-3-2-1: apex at index 4 should be picked first."""
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    pips_x, pips_y = find_pips(series, n_pips=3, dist_measure=DIST_PD)
    assert 4 in pips_x
    assert pips_y[pips_x.index(4)] == pytest.approx(5.0)


def test_find_pips_v_valley_picks_bottom() -> None:
    series = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    pips_x, _ = find_pips(series, n_pips=3, dist_measure=DIST_PD)
    assert 4 in pips_x  # deepest point


def test_find_pips_returns_at_most_n_pips() -> None:
    series = np.sin(np.linspace(0, 4 * math.pi, 40))
    for target in (3, 5, 10, 15):
        pips_x, pips_y = find_pips(series, n_pips=target, dist_measure=DIST_PD)
        assert len(pips_x) == len(pips_y)
        assert len(pips_x) <= target


def test_find_pips_sorted_by_index() -> None:
    series = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0])
    pips_x, _ = find_pips(series, n_pips=5, dist_measure=DIST_PD)
    assert pips_x == sorted(pips_x)


def test_find_pips_handles_all_three_distance_modes() -> None:
    series = np.array([1.0, 3.0, 2.0, 5.0, 2.0, 6.0, 3.0])
    for mode in (DIST_ED, DIST_PD, DIST_VD):
        pips_x, pips_y = find_pips(series, n_pips=4, dist_measure=mode)
        assert pips_x[0] == 0
        assert pips_x[-1] == len(series) - 1
        assert len(set(pips_x)) == len(pips_x)  # no duplicates


def test_find_pips_empty_and_singleton() -> None:
    assert find_pips(np.array([]), n_pips=5, dist_measure=DIST_PD) == ([], [])
    assert find_pips(np.array([3.14]), n_pips=5, dist_measure=DIST_PD) == ([0], [3.14])


def test_find_pips_n_pips_exceeds_length() -> None:
    series = np.array([1.0, 2.0, 3.0])
    pips_x, pips_y = find_pips(series, n_pips=10, dist_measure=DIST_PD)
    assert len(pips_x) <= len(series)


def test_pd_formula_matches_manual_calculation() -> None:
    """PD of point (1, 5) from line connecting (0, 0) and (2, 0) should be 5.0."""
    series = np.array([0.0, 5.0, 0.0])
    pips_x, pips_y = find_pips(series, n_pips=3, dist_measure=DIST_PD)
    assert 1 in pips_x
    assert pips_y[pips_x.index(1)] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# detect_pivots_pip_pd — candle-level wrapper
# ---------------------------------------------------------------------------

def _make_candle(i: int, price: float) -> dict:
    return {
        "time": 1_700_000_000 + i * 60,
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
        "volume": 1000.0,
    }


def test_detect_pivots_pip_pd_short_candles_returns_empty() -> None:
    assert detect_pivots_pip_pd([], n_pips=10) == []
    assert detect_pivots_pip_pd([_make_candle(i, 100) for i in range(3)], n_pips=10) == []


def test_detect_pivots_pip_pd_returns_high_and_low_kinds() -> None:
    candles = [_make_candle(i, 100 + 10 * math.sin(i * 0.5)) for i in range(30)]
    pivots = detect_pivots_pip_pd(candles, n_pips=8)
    kinds = {p["kind"] for p in pivots}
    assert kinds == {"high", "low"}


def test_detect_pivots_pip_pd_sorted_by_index() -> None:
    candles = [_make_candle(i, 100 + 5 * math.sin(i * 0.3)) for i in range(40)]
    pivots = detect_pivots_pip_pd(candles, n_pips=12)
    indices = [p["index"] for p in pivots]
    assert indices == sorted(indices)


def test_detect_pivots_pip_pd_respects_timeframe_budget() -> None:
    candles = [_make_candle(i, 100 + 5 * math.sin(i * 0.2)) for i in range(50)]
    pivots_1m = detect_pivots_pip_pd(candles, timeframe="1m")
    pivots_1d = detect_pivots_pip_pd(candles, timeframe="1d")
    # 1d gets more pips (32) than 1m (20), so fewer candles per pip at equal data
    assert len(pivots_1d) >= len(pivots_1m)


def test_detect_pivots_pip_pd_pivot_prices_match_candle_wicks() -> None:
    candles = [_make_candle(i, 100 + math.sin(i * 0.4) * 5) for i in range(30)]
    pivots = detect_pivots_pip_pd(candles, n_pips=6)
    for p in pivots:
        bar = candles[p["index"]]
        if p["kind"] == "high":
            assert p["price"] == bar["high"]
        else:
            assert p["price"] == bar["low"]


# ---------------------------------------------------------------------------
# detect_pivots_hybrid — merge correctness
# ---------------------------------------------------------------------------

def test_hybrid_empty_fractal_falls_back_to_pip() -> None:
    candles = [_make_candle(i, 100 + math.sin(i * 0.3) * 5) for i in range(30)]
    merged = detect_pivots_hybrid(candles, fractal_pivots=[], timeframe="1h")
    assert len(merged) > 0


def test_hybrid_preserves_all_fractal_pivots() -> None:
    candles = [_make_candle(i, 100 + math.sin(i * 0.3) * 5) for i in range(30)]
    fractal = [
        {"index": 5, "time": candles[5]["time"], "price": 105.0, "kind": "high"},
        {"index": 15, "time": candles[15]["time"], "price": 95.0, "kind": "low"},
    ]
    merged = detect_pivots_hybrid(candles, fractal_pivots=fractal, timeframe="1h")
    fractal_indices = {(p["index"], p["kind"]) for p in fractal}
    merged_lookup = {(p["index"], p["kind"]) for p in merged}
    assert fractal_indices.issubset(merged_lookup)


def test_hybrid_drops_pip_with_same_index_as_fractal() -> None:
    """When PIP picks the exact same bar as a fractal, drop the PIP duplicate."""
    candles = [_make_candle(i, 100 + math.sin(i * 0.3) * 5) for i in range(30)]
    pip_only = detect_pivots_pip_pd(candles, timeframe="1h")
    if not pip_only:
        pytest.skip("PIP returned no pivots; nothing to dedupe")
    # Seed fractal with the same index as a PIP-PD pick, same kind.
    target = pip_only[0]
    fractal = [{"index": target["index"], "time": target["time"],
                "price": target["price"], "kind": target["kind"]}]
    merged = detect_pivots_hybrid(candles, fractal_pivots=fractal, timeframe="1h")
    matches = [p for p in merged if p["index"] == target["index"] and p["kind"] == target["kind"]]
    assert len(matches) == 1


def test_hybrid_drops_pip_close_in_both_index_and_price() -> None:
    """PIP near a fractal (same kind, <=2 bars, within price tol) must dedupe."""
    candles = [_make_candle(i, 100 + math.sin(i * 0.3) * 5) for i in range(30)]
    pip_only = detect_pivots_pip_pd(candles, timeframe="1h")
    if not pip_only:
        pytest.skip("PIP returned no pivots; nothing to dedupe")
    target = pip_only[0]
    # Place a fractal 1 bar away with near-identical price → triggers dedupe
    nearby_idx = target["index"] + 1 if target["index"] + 1 < len(candles) else target["index"] - 1
    fractal = [{
        "index": nearby_idx,
        "time": candles[nearby_idx]["time"],
        "price": target["price"],  # identical price = within any tolerance
        "kind": target["kind"],
    }]
    merged = detect_pivots_hybrid(candles, fractal_pivots=fractal, timeframe="1h")
    # The original PIP at target["index"] should have been dropped
    same_kind = [p for p in merged if p["kind"] == target["kind"]]
    matching_pip = [p for p in same_kind if p["index"] == target["index"]]
    assert len(matching_pip) == 0, f"PIP at {target['index']} should dedupe against fractal at {nearby_idx}"


def test_hybrid_result_sorted() -> None:
    candles = [_make_candle(i, 100 + math.sin(i * 0.5) * 5) for i in range(40)]
    fractal = [
        {"index": 10, "time": candles[10]["time"], "price": 105.0, "kind": "high"},
        {"index": 20, "time": candles[20]["time"], "price": 95.0, "kind": "low"},
    ]
    merged = detect_pivots_hybrid(candles, fractal_pivots=fractal, timeframe="1h")
    indices = [p["index"] for p in merged]
    assert indices == sorted(indices)


def test_hybrid_no_duplicate_indices() -> None:
    candles = [_make_candle(i, 100 + math.cos(i * 0.4) * 4) for i in range(30)]
    fractal = [{"index": 10, "time": candles[10]["time"], "price": 104.0, "kind": "high"}]
    merged = detect_pivots_hybrid(candles, fractal_pivots=fractal, timeframe="1h")
    # A bar can be both a high pivot and low pivot simultaneously, but same kind
    # at same index should never duplicate
    pairs = [(p["index"], p["kind"]) for p in merged]
    assert len(pairs) == len(set(pairs))


def test_hybrid_preserves_dual_kind_pip_at_same_bar() -> None:
    """Regression guard: a bar that is both high-series PIP and low-series PIP must survive merge.

    Previously `existing_idx` keyed only on index silently dropped the second
    kind. Key must be (index, kind) so dual-kind bars pass through.
    """
    candles = [_make_candle(i, 100 + math.sin(i * 0.3) * 5) for i in range(30)]
    pip_only = detect_pivots_pip_pd(candles, timeframe="1h")
    shared = {p["index"] for p in pip_only if p["kind"] == "high"} & {
        p["index"] for p in pip_only if p["kind"] == "low"
    }
    if not shared:
        pytest.skip("PIP output has no dual-kind bars on this input; nothing to test")

    merged = detect_pivots_hybrid(candles, fractal_pivots=[], timeframe="1h")
    merged_shared = {p["index"] for p in merged if p["kind"] == "high"} & {
        p["index"] for p in merged if p["kind"] == "low"
    }
    assert shared.issubset(merged_shared), (
        f"Dual-kind bars dropped. pip_only={shared}, merged={merged_shared}"
    )


def test_hybrid_monotonic_series_returns_few_pivots() -> None:
    """Strict monotonic series has no interior PIP — pivot list should stay minimal."""
    candles = [_make_candle(i, 100.0 + i * 2.0) for i in range(20)]
    merged = detect_pivots_hybrid(candles, fractal_pivots=[], timeframe="1h")
    # No turning points — expect endpoints only per kind. Allow some slack for
    # wick noise producing minor turns on derived high/low series.
    assert len(merged) <= 6
