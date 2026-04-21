"""Parity test: output must remain stable across refactors.

Locks down top-level structure, pattern identity, trendline geometry, and
horizontal-level prices against a frozen golden fixture. Regenerate only when
intentionally changing detection behavior.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from chartslp import detect_chart_patterns

FIXTURES = Path(__file__).parent / "fixtures"


def _load_golden(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


def _trending_candles_1h() -> list[dict]:
    """Same generator used to build golden_trending_1h_swing.json."""
    rng = random.Random(42)
    candles = []
    for i in range(100):
        price = 100 + i * 0.15 + math.sin(i * 0.3) * 3 + rng.uniform(-0.4, 0.4)
        candles.append({
            "index": i,
            "time": 1_700_000_000 + i * 3600,
            "open": price - 0.2,
            "high": price + 1.2,
            "low": price - 1.2,
            "close": price + 0.2,
            "volume": 1000.0,
        })
    return candles


def test_trending_1h_swing_matches_golden() -> None:
    golden = _load_golden("golden_trending_1h_swing.json")
    result = detect_chart_patterns(_trending_candles_1h(), "1h", "swing")

    # Top-level keys identical
    assert set(result.keys()) == set(golden.keys())

    # Pattern identity
    assert result["primaryPattern"]["type"] == golden["primaryPattern"]["type"]
    assert result["primaryPattern"]["label"] == golden["primaryPattern"]["label"]

    # Stats counts identical
    assert result["stats"]["pivotCount"] == golden["stats"]["pivotCount"]
    assert abs(result["stats"]["atr"] - golden["stats"]["atr"]) < 1e-9
    assert abs(result["stats"]["volumeRatio"] - golden["stats"]["volumeRatio"]) < 1e-9

    # Shape / line counts stable
    assert len(result["trendLines"]) == len(golden["trendLines"])
    assert len(result["shapes"]) == len(golden["shapes"])
    assert len(result["horizontalLevels"]) == len(golden["horizontalLevels"])
    assert len(result["swingPoints"]) == len(golden["swingPoints"])

    # Timeframe / trade_type echo
    assert result["timeframe"] == golden["timeframe"]
    assert result["tradeType"] == golden["tradeType"]


def test_trending_1h_swing_trendlines_geometry_matches() -> None:
    """Every trendline point must match the golden within 1e-6 tolerance."""
    golden = _load_golden("golden_trending_1h_swing.json")
    result = detect_chart_patterns(_trending_candles_1h(), "1h", "swing")

    for i, (r_line, g_line) in enumerate(zip(result["trendLines"], golden["trendLines"])):
        assert len(r_line["points"]) == len(g_line["points"]), f"line {i} point count diverged"
        for j, (r_pt, g_pt) in enumerate(zip(r_line["points"], g_line["points"])):
            assert r_pt["time"] == g_pt["time"], f"line {i} pt {j} time diverged"
            assert abs(r_pt["price"] - g_pt["price"]) < 1e-6, (
                f"line {i} pt {j} price {r_pt['price']} vs {g_pt['price']}"
            )


def test_trending_1h_swing_horizontal_levels_prices_match() -> None:
    golden = _load_golden("golden_trending_1h_swing.json")
    result = detect_chart_patterns(_trending_candles_1h(), "1h", "swing")

    golden_prices = sorted(lvl["price"] for lvl in golden["horizontalLevels"])
    result_prices = sorted(lvl["price"] for lvl in result["horizontalLevels"])
    assert len(golden_prices) == len(result_prices)
    for r, g in zip(result_prices, golden_prices):
        assert abs(r - g) < 1e-6
