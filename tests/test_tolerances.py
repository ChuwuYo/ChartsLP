"""Tests for ATR-adaptive tolerance resolution."""
from __future__ import annotations

import pytest

from chartslp.tolerances import (
    Tolerances,
    avg_candle_range,
    resolve_multiplier,
)


def _candles(count: int, range_size: float = 1.0) -> list[dict]:
    out = []
    for i in range(count):
        base = 100.0 + i * 0.1
        out.append({
            "time": 1_700_000_000 + i * 60,
            "open": base,
            "high": base + range_size / 2,
            "low": base - range_size / 2,
            "close": base + 0.05,
            "volume": 1_000.0,
        })
    return out


# ---------------------------------------------------------------------------
# avg_candle_range
# ---------------------------------------------------------------------------

def test_avg_candle_range_empty_returns_floor() -> None:
    assert avg_candle_range([]) == pytest.approx(1e-9)


def test_avg_candle_range_basic() -> None:
    candles = _candles(10, range_size=2.0)
    assert avg_candle_range(candles) == pytest.approx(2.0, rel=0.01)


def test_avg_candle_range_scales_with_input() -> None:
    small = avg_candle_range(_candles(10, range_size=0.5))
    big = avg_candle_range(_candles(10, range_size=5.0))
    assert big > small
    assert big / small == pytest.approx(10.0, rel=0.01)


def test_avg_candle_range_skips_malformed_bars() -> None:
    candles = _candles(5, range_size=1.0)
    # Inject an invalid candle where high < low
    candles.append({
        "time": 0, "open": 0, "high": 99.0, "low": 100.0, "close": 99.5, "volume": 1.0,
    })
    assert avg_candle_range(candles) == pytest.approx(1.0, rel=0.02)


def test_avg_candle_range_never_zero() -> None:
    """Flat candles (high == low on every bar) still get a positive floor."""
    flat = [{"time": i, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}
            for i in range(10)]
    # high == low is accepted (>=), mean is 0, floor enforces > 0
    assert avg_candle_range(flat) > 0


# ---------------------------------------------------------------------------
# resolve_multiplier
# ---------------------------------------------------------------------------

def test_resolve_multiplier_base_values() -> None:
    assert resolve_multiplier("pivot_symmetry_tol") == pytest.approx(0.020)
    assert resolve_multiplier("breakout_buffer") == pytest.approx(0.080)


def test_resolve_multiplier_tf_override_1m_breakout() -> None:
    base = resolve_multiplier("breakout_buffer")
    override = resolve_multiplier("breakout_buffer", timeframe="1m")
    assert override > base
    assert override == pytest.approx(0.120)


def test_resolve_multiplier_tf_override_1d_symmetry() -> None:
    """Daily pivots are cleaner, use a stricter symmetry tolerance."""
    base = resolve_multiplier("pivot_symmetry_tol")
    strict = resolve_multiplier("pivot_symmetry_tol", timeframe="1d")
    assert strict < base
    assert strict == pytest.approx(0.015)


def test_resolve_multiplier_unknown_tf_falls_back() -> None:
    """Unknown TFs (e.g. made-up '7h') should use base multipliers."""
    assert resolve_multiplier("pivot_symmetry_tol", timeframe="7h") == pytest.approx(0.020)


def test_resolve_multiplier_asset_class_identity_stub() -> None:
    """P3 stub: all asset_class scalars are 1.0, P4 will tune them."""
    for ac in ("crypto", "stock", "forex", "etf", "commodity"):
        assert resolve_multiplier("pivot_symmetry_tol", asset_class=ac) == pytest.approx(0.020)


def test_resolve_multiplier_unknown_knob_raises() -> None:
    with pytest.raises(KeyError):
        resolve_multiplier("nonexistent_knob")


# ---------------------------------------------------------------------------
# Tolerances bundle
# ---------------------------------------------------------------------------

def test_tolerances_for_candles_uses_avg_range() -> None:
    candles = _candles(20, range_size=2.0)
    tol = Tolerances.for_candles(candles, timeframe="1h")
    assert tol.avg_range == pytest.approx(2.0, rel=0.01)


def test_tolerances_all_knobs_scale_with_avg_range() -> None:
    small = Tolerances.for_candles(_candles(10, range_size=0.5), timeframe="1h")
    big = Tolerances.for_candles(_candles(10, range_size=5.0), timeframe="1h")
    for knob in ("pivot_symmetry_tol", "pivot_grouping", "breakout_buffer",
                 "duplicate_price_tol", "retest_tol"):
        assert getattr(big, knob) > getattr(small, knob), f"{knob} did not scale"
        assert getattr(big, knob) / getattr(small, knob) == pytest.approx(10.0, rel=0.01), (
            f"{knob} scale factor wrong"
        )


def test_tolerances_1m_breakout_buffer_wider_than_1h() -> None:
    candles = _candles(20, range_size=1.0)
    tf_1m = Tolerances.for_candles(candles, timeframe="1m")
    tf_1h = Tolerances.for_candles(candles, timeframe="1h")
    assert tf_1m.breakout_buffer > tf_1h.breakout_buffer


def test_tolerances_1d_symmetry_tol_stricter_than_1h() -> None:
    candles = _candles(20, range_size=1.0)
    tf_1d = Tolerances.for_candles(candles, timeframe="1d")
    tf_1h = Tolerances.for_candles(candles, timeframe="1h")
    assert tf_1d.pivot_symmetry_tol < tf_1h.pivot_symmetry_tol


def test_tolerances_empty_candles_returns_safe_floor() -> None:
    tol = Tolerances.for_candles([], timeframe="1h")
    # Every knob computes avg_range (1e-9) * mult (>0), so all > 0 but tiny
    assert tol.pivot_symmetry_tol > 0
    assert tol.breakout_buffer > 0


def test_tolerances_slots_prevents_attr_abuse() -> None:
    tol = Tolerances.for_candles(_candles(5), timeframe="1h")
    with pytest.raises(AttributeError):
        tol.extra_field = 123  # type: ignore[attr-defined]


def test_tolerances_asset_class_and_tf_interact_cleanly() -> None:
    """asset_class and timeframe are independent pipeline stages; both must apply."""
    candles = _candles(20, range_size=1.0)
    # Both tf override + asset scalar — currently asset is 1.0 so effective
    # equals tf-only, but this guards against regression if P4 changes scalars.
    both = Tolerances.for_candles(candles, timeframe="1m", asset_class="crypto")
    tf_only = Tolerances.for_candles(candles, timeframe="1m")
    assert both.breakout_buffer == pytest.approx(tf_only.breakout_buffer)


def test_tolerances_negative_avg_range_floored() -> None:
    """Malformed avg_range (negative) is clamped to the positive floor."""
    tol = Tolerances(avg_range=-5.0, timeframe="1h")
    assert tol.avg_range == pytest.approx(1e-9)
    assert tol.pivot_symmetry_tol > 0


def test_avg_candle_range_large_input_smoke() -> None:
    """Performance smoke test: 10000 candles must not crash or hang."""
    candles = _candles(10_000, range_size=1.5)
    result = avg_candle_range(candles)
    assert result == pytest.approx(1.5, rel=0.01)
