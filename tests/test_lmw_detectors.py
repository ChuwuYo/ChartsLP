"""Direct unit tests for LMW-strict detector functions.

Complements the synthetic-pattern baseline (test_pattern_synthetic.py) by
pinning per-detector behaviour on hand-crafted inputs: standard detection,
boundary tolerance cases, rejection of counter-examples, and empty/short
input handling.
"""
from __future__ import annotations

import pytest

from chartslp.lmw_detectors import (
    _fit_line,
    detect_ascending_triangle_lmw,
    detect_bear_flag_lmw,
    detect_bull_flag_lmw,
    detect_descending_triangle_lmw,
    detect_falling_wedge_lmw,
    detect_head_and_shoulders_lmw,
    detect_inverse_head_and_shoulders_lmw,
    detect_rising_wedge_lmw,
    detect_symmetrical_triangle_lmw,
)

BASE_TIME = 1_700_000_000


def _mkc(i: int, open_: float, close: float, high: float | None = None, low: float | None = None,
         vol: float = 1000.0) -> dict:
    hi = high if high is not None else max(open_, close) + 0.5
    lo = low if low is not None else min(open_, close) - 0.5
    return {
        "index": i,
        "time": BASE_TIME + i * 3600,
        "open": float(open_),
        "high": float(hi),
        "low": float(lo),
        "close": float(close),
        "volume": float(vol),
    }


def _mkp(index: int, kind: str, price: float) -> dict:
    return {"index": index, "kind": kind, "time": BASE_TIME + index * 3600, "price": price}


def _line_candles(count: int, start: float = 100.0, slope: float = 0.2) -> list[dict]:
    return [_mkc(i, start + slope * i, start + slope * i + 0.1) for i in range(count)]


# ---------------------------------------------------------------------------
# _fit_line
# ---------------------------------------------------------------------------

def test_fit_line_slope_intercept() -> None:
    pivots = [_mkp(0, "high", 100.0), _mkp(10, "high", 110.0)]
    slope, intercept, rmse = _fit_line(pivots)
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(100.0)
    assert rmse == pytest.approx(0.0)


def test_fit_line_noisy_has_positive_rmse() -> None:
    pivots = [_mkp(0, "high", 100), _mkp(10, "high", 109),
              _mkp(20, "high", 121), _mkp(30, "high", 129)]
    slope, _, rmse = _fit_line(pivots)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert rmse > 0.0


def test_fit_line_empty_and_singleton() -> None:
    assert _fit_line([]) == (0.0, 0.0, 0.0)
    s, i, r = _fit_line([_mkp(0, "high", 42.0)])
    assert s == 0.0 and i == 42.0 and r == 0.0


# ---------------------------------------------------------------------------
# Head and Shoulders (top)
# ---------------------------------------------------------------------------

def _hs_top_pivots() -> list[dict]:
    # Classic H&S: 108 - 100 - 114 - 100 - 108
    return [
        _mkp(10, "high", 108.0),
        _mkp(20, "low",  100.0),
        _mkp(30, "high", 114.0),
        _mkp(40, "low",  100.0),
        _mkp(50, "high", 108.0),
    ]


def _hs_inv_pivots() -> list[dict]:
    return [
        _mkp(10, "low",  92.0),
        _mkp(20, "high", 100.0),
        _mkp(30, "low",  86.0),
        _mkp(40, "high", 100.0),
        _mkp(50, "low",  92.0),
    ]


def test_head_and_shoulders_lmw_detects_standard() -> None:
    pivots = _hs_top_pivots()
    candles = _line_candles(60, start=95.0, slope=0.05)
    result = detect_head_and_shoulders_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "head_and_shoulders"
    assert result["primaryPattern"]["bias"] == "bearish"


def test_inverse_head_and_shoulders_lmw_detects_standard() -> None:
    pivots = _hs_inv_pivots()
    candles = _line_candles(60, start=105.0, slope=-0.05)
    result = detect_inverse_head_and_shoulders_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "inverse_head_and_shoulders"
    assert result["primaryPattern"]["bias"] == "bullish"


def test_head_and_shoulders_rejects_asymmetric_shoulders() -> None:
    # Right shoulder 30% higher than left — way outside 3% sym_tol
    pivots = [
        _mkp(10, "high", 108.0),
        _mkp(20, "low", 100.0),
        _mkp(30, "high", 114.0),
        _mkp(40, "low", 100.0),
        _mkp(50, "high", 140.0),
    ]
    candles = _line_candles(60)
    assert detect_head_and_shoulders_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_head_and_shoulders_rejects_head_not_higher() -> None:
    # Head equals shoulders — not a valid H&S
    pivots = [
        _mkp(10, "high", 108.0),
        _mkp(20, "low", 100.0),
        _mkp(30, "high", 108.0),
        _mkp(40, "low", 100.0),
        _mkp(50, "high", 108.0),
    ]
    candles = _line_candles(60)
    assert detect_head_and_shoulders_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_head_and_shoulders_rejects_wrong_kind_sequence() -> None:
    # All highs, no lows — kind sequence does not match
    pivots = [_mkp(10 + i * 10, "high", 100 + i) for i in range(5)]
    candles = _line_candles(60)
    assert detect_head_and_shoulders_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_head_and_shoulders_empty_pivots() -> None:
    candles = _line_candles(60)
    assert detect_head_and_shoulders_lmw([], candles, 1.0, 1.0, 0.5, 1.0) is None


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------

def test_ascending_triangle_detects_flat_top_rising_bottom() -> None:
    # Flat top at 110, rising bottom from 98 to 106
    pivots = [
        _mkp(5, "high", 110.0),  _mkp(12, "low", 98.0),
        _mkp(18, "high", 110.0), _mkp(25, "low", 101.0),
        _mkp(31, "high", 110.0), _mkp(38, "low", 104.0),
        _mkp(44, "high", 110.0), _mkp(50, "low", 106.0),
    ]
    candles = _line_candles(60, start=100.0, slope=0.1)
    result = detect_ascending_triangle_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "ascending_triangle"


def test_ascending_triangle_rejects_rising_top() -> None:
    # Both top and bottom rising — not an ascending triangle
    pivots = [
        _mkp(5, "high", 108.0),  _mkp(12, "low", 98.0),
        _mkp(18, "high", 110.0), _mkp(25, "low", 100.0),
        _mkp(31, "high", 112.0), _mkp(38, "low", 102.0),
        _mkp(44, "high", 114.0), _mkp(50, "low", 104.0),
    ]
    candles = _line_candles(60)
    assert detect_ascending_triangle_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_descending_triangle_detects_falling_top_flat_bottom() -> None:
    pivots = [
        _mkp(5, "high", 112.0),  _mkp(12, "low", 100.0),
        _mkp(18, "high", 109.0), _mkp(25, "low", 100.0),
        _mkp(31, "high", 106.0), _mkp(38, "low", 100.0),
        _mkp(44, "high", 103.0), _mkp(50, "low", 100.0),
    ]
    candles = _line_candles(60, start=105.0, slope=-0.05)
    result = detect_descending_triangle_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "descending_triangle"


def test_symmetrical_triangle_detects_converging() -> None:
    # Top falling from 115→108, bottom rising from 95→102
    pivots = [
        _mkp(5, "high", 115.0),  _mkp(12, "low", 95.0),
        _mkp(18, "high", 112.0), _mkp(25, "low", 97.0),
        _mkp(31, "high", 110.0), _mkp(38, "low", 100.0),
        _mkp(44, "high", 108.0), _mkp(50, "low", 102.0),
    ]
    candles = _line_candles(60, start=105.0, slope=0.0)
    result = detect_symmetrical_triangle_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "symmetrical_triangle"


def test_symmetrical_triangle_rejects_diverging() -> None:
    # Broadening (top rising, bottom falling) — not symmetrical triangle
    pivots = [
        _mkp(5, "high", 108.0),  _mkp(12, "low", 100.0),
        _mkp(18, "high", 111.0), _mkp(25, "low", 97.0),
        _mkp(31, "high", 114.0), _mkp(38, "low", 94.0),
        _mkp(44, "high", 117.0), _mkp(50, "low", 91.0),
    ]
    candles = _line_candles(60)
    assert detect_symmetrical_triangle_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


# ---------------------------------------------------------------------------
# Wedges
# ---------------------------------------------------------------------------

def test_rising_wedge_detects_converging_uphill() -> None:
    # Both rising, top slower (ratio 0.3)
    pivots = [
        _mkp(5, "high", 106.0),  _mkp(12, "low", 96.0),
        _mkp(18, "high", 107.0), _mkp(25, "low", 100.0),
        _mkp(31, "high", 108.0), _mkp(38, "low", 103.0),
        _mkp(44, "high", 109.0), _mkp(50, "low", 106.0),
    ]
    candles = _line_candles(60, start=100.0, slope=0.1)
    result = detect_rising_wedge_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "rising_wedge"
    assert result["primaryPattern"]["bias"] == "bearish"


def test_rising_wedge_rejects_parallel_channel() -> None:
    # Same slope on both lines — parallel, not converging → channel not wedge
    pivots = [
        _mkp(5, "high", 108.0),  _mkp(12, "low", 98.0),
        _mkp(18, "high", 110.0), _mkp(25, "low", 100.0),
        _mkp(31, "high", 112.0), _mkp(38, "low", 102.0),
        _mkp(44, "high", 114.0), _mkp(50, "low", 104.0),
    ]
    candles = _line_candles(60)
    assert detect_rising_wedge_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_falling_wedge_detects_converging_downhill() -> None:
    # Both falling, top faster (steeper negative)
    pivots = [
        _mkp(5, "high", 114.0),  _mkp(12, "low", 106.0),
        _mkp(18, "high", 111.0), _mkp(25, "low", 104.0),
        _mkp(31, "high", 108.0), _mkp(38, "low", 102.0),
        _mkp(44, "high", 105.0), _mkp(50, "low", 100.0),
    ]
    candles = _line_candles(60, start=110.0, slope=-0.1)
    result = detect_falling_wedge_lmw(pivots, candles, 1.0, 1.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "falling_wedge"
    assert result["primaryPattern"]["bias"] == "bullish"


def test_wedge_rejects_short_pivots() -> None:
    short = [_mkp(5, "high", 100.0), _mkp(10, "low", 99.0)]
    candles = _line_candles(20)
    assert detect_rising_wedge_lmw(short, candles, 1.0, 1.0, 0.5, 1.0) is None
    assert detect_falling_wedge_lmw(short, candles, 1.0, 1.0, 0.5, 1.0) is None


def test_triangle_empty_candles() -> None:
    pivots = _hs_top_pivots()  # reuse; doesn't matter, candles [] is the focus
    assert detect_ascending_triangle_lmw(pivots, [], 1.0, 1.0, 0.5, 1.0) is None
    assert detect_descending_triangle_lmw(pivots, [], 1.0, 1.0, 0.5, 1.0) is None
    assert detect_symmetrical_triangle_lmw(pivots, [], 1.0, 1.0, 0.5, 1.0) is None


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _bull_flag_candles() -> list[dict]:
    """12-bar up pole (+20), then 28-bar slight down-drift flag."""
    candles = []
    # Pole: 100 → 120 over 12 bars
    for i in range(12):
        p = 100 + (i / 11) * 20
        candles.append(_mkc(i, p - 0.1, p + 0.1, high=p + 0.4, low=p - 0.4, vol=2000))
    # Flag: 119 → 116 drift over 28 bars
    for i in range(28):
        p = 119 - (i / 27) * 3
        candles.append(_mkc(12 + i, p - 0.05, p + 0.05, high=p + 0.3, low=p - 0.3, vol=800))
    return candles


def _bear_flag_candles() -> list[dict]:
    candles = []
    for i in range(12):
        p = 120 - (i / 11) * 20
        candles.append(_mkc(i, p + 0.1, p - 0.1, high=p + 0.4, low=p - 0.4, vol=2000))
    for i in range(28):
        p = 101 + (i / 27) * 3
        candles.append(_mkc(12 + i, p + 0.05, p - 0.05, high=p + 0.3, low=p - 0.3, vol=800))
    return candles


def test_bull_flag_lmw_detects_standard() -> None:
    result = detect_bull_flag_lmw([], _bull_flag_candles(), 1.0, 2.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "bull_flag"
    assert result["primaryPattern"]["bias"] == "bullish"


def test_bear_flag_lmw_detects_standard() -> None:
    result = detect_bear_flag_lmw([], _bear_flag_candles(), 1.0, 2.0, 0.5, 1.0)
    assert result is not None
    assert result["primaryPattern"]["type"] == "bear_flag"
    assert result["primaryPattern"]["bias"] == "bearish"


def test_bull_flag_rejects_flat_pole() -> None:
    # Flat pole (no strong direction) → reject
    candles = [_mkc(i, 100.0, 100.1, vol=1000) for i in range(40)]
    assert detect_bull_flag_lmw([], candles, 1.0, 2.0, 0.5, 1.0) is None


def test_bull_flag_rejects_same_direction_flag() -> None:
    # Up pole + up flag (no counter-trend) → not a flag
    candles = []
    for i in range(12):
        p = 100 + (i / 11) * 20
        candles.append(_mkc(i, p - 0.1, p + 0.1, vol=2000))
    for i in range(28):
        p = 120 + (i / 27) * 5  # still going up!
        candles.append(_mkc(12 + i, p - 0.05, p + 0.05, vol=800))
    assert detect_bull_flag_lmw([], candles, 1.0, 2.0, 0.5, 1.0) is None


def test_bull_flag_rejects_too_strong_counter() -> None:
    # Up pole then strong counter-trend drop (|flag| > 0.4 |pole|) → not a flag
    candles = []
    for i in range(12):
        p = 100 + (i / 11) * 20
        candles.append(_mkc(i, p - 0.1, p + 0.1, vol=2000))
    for i in range(28):
        p = 120 - (i / 27) * 18  # drops 18 — more than 40% of pole's 20
        candles.append(_mkc(12 + i, p - 0.05, p + 0.05, vol=800))
    assert detect_bull_flag_lmw([], candles, 1.0, 2.0, 0.5, 1.0) is None


def test_flag_rejects_insufficient_candles() -> None:
    short = [_mkc(i, 100 + i, 100 + i) for i in range(20)]
    assert detect_bull_flag_lmw([], short, 1.0, 2.0, 0.5, 1.0) is None
    assert detect_bear_flag_lmw([], short, 1.0, 2.0, 0.5, 1.0) is None
