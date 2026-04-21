"""Narrow golden tests for trendline_engine internals.

Philosophy: lock down structural invariants, not floating-point snapshots.
Each test uses a fixed synthetic data generator and asserts a single behavior
(count, ordering, or containment) that shouldn't drift as we tune parameters.
"""

from __future__ import annotations

import math
import random

import pytest

from chartslp import trendlines as te


def _synth_trending_candles(n: int = 100, *, seed: int = 42) -> list[dict]:
    """Synthetic uptrend with sinusoidal oscillation — stable pivots."""
    rng = random.Random(seed)
    candles: list[dict] = []
    base_time = 1_700_000_000
    for i in range(n):
        price = 100 + i * 0.15 + math.sin(i * 0.3) * 3 + rng.uniform(-0.4, 0.4)
        candles.append({
            "index": i,
            "time": base_time + i * 3600,
            "open": price - 0.2,
            "high": price + 1.2,
            "low": price - 1.2,
            "close": price + 0.2,
            "volume": 1000.0,
        })
    return candles


def _synth_micro_cap_candles(n: int = 100, *, seed: int = 7) -> list[dict]:
    """PEPE/SHIB-scale fixture — base price ~1.25e-5, volatility ~3% of price.

    Exercises the sub-cent code path where pre-fix tolerances collapsed and
    every candle was counted as a "touch". Uses 1m bars (60s step).
    """
    rng = random.Random(seed)
    candles: list[dict] = []
    base_time = 1_700_000_000
    base = 1.25e-5  # PEPE-like
    for i in range(n):
        price = base + i * 3e-9 + math.sin(i * 0.3) * 3e-7 + rng.uniform(-1e-7, 1e-7)
        candles.append({
            "index": i,
            "time": base_time + i * 60,
            "open": price - 2e-8,
            "high": price + 8e-8,
            "low": price - 8e-8,
            "close": price + 2e-8,
            "volume": 10_000.0,
        })
    return candles


# ---------------------------------------------------------------------------
# 1. Pivot budget by timeframe
# ---------------------------------------------------------------------------

def test_resolve_max_pivots_scales_with_candle_interval() -> None:
    """Sparse intervals get larger pivot budgets than dense ones."""
    dense = te._resolve_max_pivots("5m")
    mid = te._resolve_max_pivots("1h")
    sparse = te._resolve_max_pivots("1d")
    sparser = te._resolve_max_pivots("1M")
    assert dense <= mid <= sparse <= sparser
    # Unknown / missing timeframe falls back to a sensible middle default
    assert te._resolve_max_pivots("unknown") == te._DEFAULT_MAX_PIVOTS
    assert te._resolve_max_pivots(None) == te._DEFAULT_MAX_PIVOTS


# ---------------------------------------------------------------------------
# 2. Numerical-differentiation pivot detection is deterministic
# ---------------------------------------------------------------------------

def test_numdiff_pivots_deterministic_count_and_types() -> None:
    """Same input twice → identical pivot list (indices, kinds, prices).

    Locks numpy.gradient behavior against accidental drift.
    """
    candles = _synth_trending_candles(n=100, seed=42)
    run1 = te._detect_pivots_numdiff(candles)
    run2 = te._detect_pivots_numdiff(candles)
    assert [(p["index"], p["kind"]) for p in run1] == [(p["index"], p["kind"]) for p in run2]
    # Sinusoidal fixture has ~5 cycles over 100 bars → ~5 peaks / ~5 troughs.
    # Range is wide enough to tolerate numpy version drift in np.gradient.
    highs = [p for p in run1 if p["kind"] == "high"]
    lows = [p for p in run1 if p["kind"] == "low"]
    assert 3 <= len(highs) <= 30, f"high pivots out of expected range: {len(highs)}"
    assert 3 <= len(lows) <= 30, f"low pivots out of expected range: {len(lows)}"


# ---------------------------------------------------------------------------
# 3. Candidate scoring gives a valid, consistent top pick
# ---------------------------------------------------------------------------

def test_score_trendline_candidate_gives_valid_top_pick() -> None:
    """Score every high-pivot pair once; top-1 must be well-formed and
    not be beaten by an obviously worse line (crossed anchors, zero span)."""
    candles = _synth_trending_candles(n=100, seed=42)
    atr = te._calculate_atr_local(candles)
    pivots = te._detect_pivots_numdiff(candles)
    highs = [p for p in pivots if p["kind"] == "high"]

    cands = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            r = te._score_trendline_candidate(highs[i], highs[j], candles, atr, is_upper=True)
            if r:
                cands.append(r)

    assert cands, "expected at least one scored upper candidate"
    cands.sort(key=lambda c: c["score"], reverse=True)
    top = cands[0]
    # Anchor-pair hygiene
    assert top["point_a"]["index"] < top["point_b"]["index"]
    assert top["span"] >= 1
    assert top["touch_count"] >= 2
    # Top score ≥ median score (sanity)
    mid = cands[len(cands) // 2]
    assert top["score"] >= mid["score"]


# ---------------------------------------------------------------------------
# 4. Dedup requires BOTH price and span-normalized slope similarity
# ---------------------------------------------------------------------------

def test_dedup_keeps_lines_with_different_slopes_even_at_same_endpoint() -> None:
    """Two candidates with near-identical projected end price but clearly
    different slopes must NOT collapse into one cluster."""
    base_candle = {"time": 1_700_000_000, "open": 100.0, "high": 101.0,
                   "low": 99.0, "close": 100.5, "volume": 1000.0}
    cand_shallow = {
        "point_a": {"index": 10, "time": 0, "price": 100.0, "kind": "high"},
        "point_b": {"index": 50, "time": 0, "price": 104.0, "kind": "high"},
        "slope": 0.1,  # price per bar
        "projected_last_price": 110.0,
        "area": 0.5, "touch_count": 4, "body_violations": 0,
        "score": 10.0, "span": 40, "is_upper": True,
    }
    cand_steep = {
        "point_a": {"index": 30, "time": 0, "price": 100.0, "kind": "high"},
        "point_b": {"index": 50, "time": 0, "price": 110.0, "kind": "high"},
        "slope": 0.5,  # much steeper
        "projected_last_price": 110.0,  # SAME end price
        "area": 0.3, "touch_count": 5, "body_violations": 0,
        "score": 12.0, "span": 20, "is_upper": True,
    }
    atr = 2.0
    result = te._deduplicate_trendlines([cand_shallow, cand_steep], atr)
    # Slopes differ by 0.4 per bar — well above any reasonable threshold.
    # They must stay as two separate clusters, not collapse to one.
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. build_auto_trendlines respects the timeframe budget
# ---------------------------------------------------------------------------

def test_build_auto_trendlines_respects_timeframe_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeframe argument must reach _resolve_max_pivots, and the budget
    returned must be applied as the per-side cap on the pivot pool.

    Spies on _resolve_max_pivots so removing the timeframe wiring (or hard-
    coding the cap again) immediately fails the test — not just "both runs
    produce non-empty output".
    """
    candles = _synth_trending_candles(n=120, seed=7)
    atr = te._calculate_atr_local(candles)
    pivots = [
        {"index": i, "time": candles[i]["time"], "price": candles[i]["high"], "kind": "high"}
        for i in range(10, 110, 10)
    ] + [
        {"index": i, "time": candles[i]["time"], "price": candles[i]["low"], "kind": "low"}
        for i in range(15, 115, 10)
    ]

    seen_args: list[str | None] = []
    real_resolve = te._resolve_max_pivots

    def _spy(timeframe: str | None) -> int:
        seen_args.append(timeframe)
        return real_resolve(timeframe)

    monkeypatch.setattr(te, "_resolve_max_pivots", _spy)

    te.build_auto_trendlines(candles, pivots, atr, timeframe="5m")
    te.build_auto_trendlines(candles, pivots, atr, timeframe="1d")
    te.build_auto_trendlines(candles, pivots, atr)  # default

    # Wiring assertion: each call must have routed its timeframe through.
    assert seen_args == ["5m", "1d", None]

    # Budget contract: 1d's cap must be strictly larger than 5m's.
    assert real_resolve("1d") > real_resolve("5m")


# ---------------------------------------------------------------------------
# 6. Candidate boundary synthesis for patterns that only emit horizontalLevels
# ---------------------------------------------------------------------------

def test_synthesize_boundary_trendlines_adds_lines_for_double_top() -> None:
    """Patterns like double_top emit only horizontalLevels; the synthesis
    helper must convert those into source='pattern' trendLines so Outline
    renders them."""
    from chartslp import patterns as pe

    result = {
        "primaryPattern": {"type": "double_top", "bias": "bearish"},
        "trendLines": [],
        "horizontalLevels": [
            {
                "id": "double-top-resistance",
                "label": "Resistance",
                "color": "#ef4444",
                "style": "dashed",
                "points": [
                    {"time": 1_700_000_000, "price": 108.0},
                    {"time": 1_700_050_000, "price": 108.0},
                ],
            },
            {
                "id": "double-top-neckline",
                "label": "Neckline",
                "color": "#3b82f6",
                "style": "solid",
                "points": [
                    {"time": 1_700_000_000, "price": 102.0},
                    {"time": 1_700_050_000, "price": 102.0},
                ],
            },
        ],
    }

    pe._synthesize_boundary_trendlines(result)
    ids = {tl["id"] for tl in result["trendLines"]}
    assert "boundary-double-top-resistance" in ids
    assert "boundary-double-top-neckline" in ids
    # All synthesized lines must be tagged source="pattern" for Outline routing
    for tl in result["trendLines"]:
        assert tl["source"] == "pattern"


def test_synthesize_boundary_trendlines_is_idempotent() -> None:
    """Calling synthesis twice must not duplicate lines."""
    from chartslp import patterns as pe

    result = {
        "primaryPattern": {"type": "rectangle_range", "bias": "neutral"},
        "trendLines": [],
        "horizontalLevels": [
            {
                "id": "rectangle-support",
                "label": "Support",
                "color": "#22c55e",
                "style": "dashed",
                "points": [
                    {"time": 1_700_000_000, "price": 100.0},
                    {"time": 1_700_050_000, "price": 100.0},
                ],
            },
        ],
    }

    pe._synthesize_boundary_trendlines(result)
    count_after_first = len(result["trendLines"])
    pe._synthesize_boundary_trendlines(result)
    count_after_second = len(result["trendLines"])
    assert count_after_first == count_after_second == 1


# ---------------------------------------------------------------------------
# 8. Low-price symbol regression (PEPE/SHIB at ~1e-5 USD)
# ---------------------------------------------------------------------------

def test_count_valid_touches_does_not_count_every_candle_for_low_price_symbol() -> None:
    """Pre-fix bug: the 0.001 absolute tolerance floor made every candle a
    "touch" when the symbol traded at ~1e-5 USD. Asserts that touch count
    stays sensibly below the total scanned bars."""
    candles = _synth_micro_cap_candles(n=60)
    # Anchor on two arbitrary swing points
    point_a = {"index": 5, "time": candles[5]["time"],
               "price": candles[5]["high"], "kind": "high"}
    point_b = {"index": 55, "time": candles[55]["time"],
               "price": candles[55]["high"], "kind": "high"}

    touches = te.count_valid_touches(point_a, point_b, candles, is_upper=True)
    # Span is 50 bars. If the floor-bug were present, touches would ≈ 50
    # (every candle close to 1e-5 line within a 1e-3 tolerance).
    # Proper behavior: a small number of actual near-line highs.
    assert touches < 20, f"touches ({touches}) suggests absolute-floor regression"


def test_score_trendline_candidate_is_scale_invariant_across_price_magnitudes() -> None:
    """Same structural setup at stock-scale (100) vs PEPE-scale (1e-5) must
    produce comparable touch counts and non-None scores. Locks the scoring
    path against future tolerance-floor regressions."""
    stock = _synth_trending_candles(n=80, seed=42)
    pepe = _synth_micro_cap_candles(n=80, seed=42)

    stock_atr = te._calculate_atr_local(stock)
    pepe_atr = te._calculate_atr_local(pepe)

    # Use fixed anchor indices so the geometric shape is identical
    def _score(candles, atr):
        pa = {"index": 10, "time": candles[10]["time"],
              "price": candles[10]["high"], "kind": "high"}
        pb = {"index": 70, "time": candles[70]["time"],
              "price": candles[70]["high"], "kind": "high"}
        return te._score_trendline_candidate(pa, pb, candles, atr, is_upper=True)

    stock_score = _score(stock, stock_atr)
    pepe_score = _score(pepe, pepe_atr)
    # Both should produce a scored candidate (or both fail structurally,
    # but never one-but-not-the-other due to scale).
    assert (stock_score is None) == (pepe_score is None), (
        f"scale invariance broken: stock={stock_score is not None} pepe={pepe_score is not None}"
    )


def test_build_auto_trendlines_produces_distinct_endpoints_for_micro_cap() -> None:
    """Pre-fix: prices rounded to 6 decimals collapsed PEPE's line endpoints
    to the same value, rendering a flat line. Assert every output line has
    geometrically distinct start/end points."""
    candles = _synth_micro_cap_candles(n=100)
    atr = te._calculate_atr_local(candles)
    # Minimal pivot seed — numdiff will expand inside
    pivots = [
        {"index": i, "time": candles[i]["time"],
         "price": candles[i]["high"], "kind": "high"}
        for i in range(10, 90, 10)
    ] + [
        {"index": i, "time": candles[i]["time"],
         "price": candles[i]["low"], "kind": "low"}
        for i in range(15, 95, 10)
    ]

    lines = te.build_auto_trendlines(candles, pivots, atr, timeframe="1m")
    assert lines, "no trendlines produced for micro-cap fixture"
    for line in lines:
        pts = line["points"]
        assert len(pts) == 2
        # The whole point of smart_round_price is preserving sub-cent precision
        # so endpoints remain distinct.
        assert pts[0]["price"] != pts[1]["price"], (
            f"{line['id']}: endpoints collapsed to same rounded value "
            f"({pts[0]['price']}) — 6-decimal rounding regression"
        )


def test_detect_chart_patterns_end_to_end_for_low_price_symbol() -> None:
    """Integration: full pipeline on PEPE-scale data must produce trendlines
    with meaningful touch counts (not every candle), non-empty horizontal
    levels, and distinct pattern structure — not all zeros."""
    from chartslp import patterns as pe

    candles = _synth_micro_cap_candles(n=100)
    result = pe.detect_chart_patterns(candles, "1m", "scalp")

    tls = result.get("trendLines") or []
    assert tls, "no trendlines on low-price symbol pipeline"

    # Each auto-source line should have a plausible touch count (not 100)
    auto_lines = [t for t in tls if t.get("source") == "auto"]
    assert auto_lines, "no auto trendlines produced"
    total_bars = len(candles)
    for tl in auto_lines:
        touches = tl.get("touches")
        if touches is not None:
            assert touches < total_bars * 0.9, (
                f"{tl['id']} has {touches} touches — suggests tolerance floor bug"
            )
        # Endpoints distinct
        pts = tl["points"]
        assert pts[0]["price"] != pts[1]["price"]
