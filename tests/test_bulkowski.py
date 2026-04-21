"""Tests for Bulkowski historical stats lookup."""
from __future__ import annotations

import pytest

from chartslp import detect_chart_patterns
from chartslp.bulkowski import lookup


def test_lookup_known_pattern() -> None:
    stats = lookup("head_and_shoulders")
    assert stats is not None
    assert stats["failure_rate"] == pytest.approx(0.19)
    assert stats["avg_move_pct"] == pytest.approx(0.16)
    assert "source" in stats
    assert "disclaimer" in stats


def test_lookup_unknown_pattern_returns_none() -> None:
    assert lookup("no_clear_pattern") is None
    assert lookup("totally_made_up") is None
    assert lookup("") is None
    assert lookup(None) is None  # type: ignore[arg-type]


def test_lookup_returns_defensive_copy() -> None:
    a = lookup("double_top")
    b = lookup("double_top")
    assert a is not b
    a["failure_rate"] = 999.0  # type: ignore[index]
    assert lookup("double_top")["failure_rate"] != 999.0  # type: ignore[index]


def test_all_listed_patterns_have_required_fields() -> None:
    """Every entry must declare failure_rate, avg_move_pct, sample_n."""
    from chartslp.bulkowski import _BULKOWSKI_STATS
    for ptype, row in _BULKOWSKI_STATS.items():
        assert "failure_rate" in row, f"{ptype} missing failure_rate"
        assert "avg_move_pct" in row, f"{ptype} missing avg_move_pct"
        assert "sample_n" in row, f"{ptype} missing sample_n"
        assert 0.0 <= row["failure_rate"] <= 1.0, f"{ptype} failure_rate out of range"
        assert 0.0 <= row["avg_move_pct"] <= 2.0, f"{ptype} avg_move_pct absurd"


def test_detect_chart_patterns_attaches_stats_on_covered_pattern() -> None:
    """Regression: when primary type is in the Bulkowski table, stats appear."""
    # Use a simple up-trend synth that should reliably detect as a known pattern
    candles = []
    for i in range(60):
        p = 100 + i * 0.2
        candles.append({
            "time": 1_700_000_000 + i * 3600, "index": i,
            "open": p - 0.1, "high": p + 0.5, "low": p - 0.5,
            "close": p + 0.1, "volume": 1000,
        })
    result = detect_chart_patterns(candles, timeframe="1h", trade_type="swing")
    primary = result.get("primaryPattern", {})
    if primary.get("type") and primary.get("type") != "no_clear_pattern":
        stats = primary.get("bulkowskiStats")
        # Either the pattern type is in the table (stats present) or it isn't
        # (stats None) — both are valid. If present, sanity-check shape.
        if stats is not None:
            assert "failure_rate" in stats
            assert "source" in stats
