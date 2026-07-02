"""Tests for the ICT/SMC structural detection engine (chartslp.ict_structures).

Each detector is a pure function pinned by a synthetic fixture with a known
structure plus boundary values. ``detect_ict_structures`` is the public entry
point. The engine is additive — these tests never touch detect_chart_patterns.
"""

from chartslp.ict_structures import (
    detect_dealing_range,
    detect_fvg,
    detect_ict_structures,
    detect_liquidity,
    detect_market_structure,
    detect_order_blocks,
    format_ict_engine_context,
)


def _flat_candles(n=60, base=100.0, step=3600):
    """Doji-flat candles: no pivots, no gaps — the empty-structure baseline."""
    return [
        {
            "time": i * step,
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base,
            "volume": 1.0,
        }
        for i in range(n)
    ]


# --- Task 1: scaffold + entry point -----------------------------------------


def test_short_input_returns_empty_shape():
    out = detect_ict_structures(_flat_candles(5))
    assert out["marketStructure"]["trend"] == "ranging"
    assert out["marketStructure"]["lastEvent"] is None
    assert out["orderBlocks"] == []
    assert out["fairValueGaps"] == []
    assert out["liquidity"]["buySide"] == []
    assert out["liquidity"]["sellSide"] == []
    assert out["liquidity"]["recentSweep"] is None


def test_output_has_all_top_level_keys():
    out = detect_ict_structures(_flat_candles(60), timeframe="1h", trade_type="swing")
    for key in (
        "version",
        "timeframe",
        "tradeType",
        "marketStructure",
        "orderBlocks",
        "fairValueGaps",
        "liquidity",
        "dealingRange",
        "summary",
    ):
        assert key in out
    assert out["version"] == 1
    assert out["timeframe"] == "1h"
    assert out["tradeType"] == "swing"


def test_summary_reflects_ranging_flat_series():
    out = detect_ict_structures(_flat_candles(60))
    assert out["summary"] == "Market structure ranging."


# --- Task 2: fair value gaps -------------------------------------------------


def _bullish_fvg_candles():
    """3-candle bullish imbalance at i=20: prev.high(101) < next.low(104).

    Last 14 candles stay flat so _calculate_atr == 2.0 exactly, making the
    min-gap threshold deterministic for the boundary test (gap = 3.0).
    """
    c = _flat_candles(40)
    c[19].update(high=101)
    c[20].update(open=102, high=106, low=102, close=105)
    c[21].update(open=104, high=107, low=104, close=106)
    return c


def test_fvg_detects_bullish_gap():
    fvgs = [f for f in detect_fvg(_bullish_fvg_candles()) if f["kind"] == "bullish"]
    assert any(abs(f["bottom"] - 101) < 0.5 and abs(f["top"] - 104) < 0.5 for f in fvgs)


def test_fvg_no_gap_returns_empty():
    assert detect_fvg(_flat_candles(40)) == []


def test_fvg_min_gap_threshold_boundary():
    c = _bullish_fvg_candles()  # gap = 3.0, atr = 2.0 -> threshold = 2.0 * min_gap_atr
    # gap exactly at threshold is inclusive (kept)
    assert any(f["kind"] == "bullish" for f in detect_fvg(c, min_gap_atr=1.5))
    # just above threshold rejects
    assert not any(f["kind"] == "bullish" for f in detect_fvg(c, min_gap_atr=1.51))


def test_fvg_detects_bearish_gap():
    c = _flat_candles(40)
    c[19].update(low=99)
    c[20].update(open=98, high=98, low=94, close=95)  # bearish: close < open
    c[21].update(open=96, high=96, low=93, close=94)
    # prev.low(99) > next.high(96) -> bearish FVG, top=99, bottom=96
    bear = [f for f in detect_fvg(c) if f["kind"] == "bearish"]
    assert any(abs(f["top"] - 99) < 0.5 and abs(f["bottom"] - 96) < 0.5 for f in bear)


# --- Task 3: market structure (BOS / CHoCH) ----------------------------------


def _pivot(index, price, kind, step=3600):
    return {"index": index, "time": index * step, "price": float(price), "kind": kind}


def test_market_structure_bullish_bos():
    # L,H,L,H with L1 < L2 < H1 < H2 -> higher low + higher high broken upward.
    pivots = [
        _pivot(2, 100, "low"),
        _pivot(6, 110, "high"),
        _pivot(10, 105, "low"),
        _pivot(14, 115, "high"),
    ]
    ms = detect_market_structure(_flat_candles(30), pivots)
    assert ms["lastEvent"] == "BOS"
    assert ms["trend"] == "bullish"
    assert ms["events"][-1]["direction"] == "bullish"


def test_market_structure_bearish_choch():
    # H,L,H,L with L2 < L1 < H1 < H2 -> higher high (uptrend) then prior low broken.
    pivots = [
        _pivot(2, 110, "high"),
        _pivot(6, 105, "low"),
        _pivot(10, 115, "high"),
        _pivot(14, 100, "low"),
    ]
    ms = detect_market_structure(_flat_candles(30), pivots)
    assert ms["lastEvent"] == "CHoCH"
    assert ms["trend"] == "bearish"


def test_market_structure_insufficient_pivots_is_ranging():
    pivots = [_pivot(2, 100, "low"), _pivot(6, 110, "high")]
    ms = detect_market_structure(_flat_candles(30), pivots)
    assert ms["trend"] == "ranging"
    assert ms["lastEvent"] is None
    assert ms["events"] == []


# --- Task 4: liquidity pools + sweep -----------------------------------------


def test_liquidity_buyside_pool_and_sweep():
    candles = _flat_candles(30)
    # candle 20 wicks above the ~110.15 pool then closes back below it.
    candles[20].update(open=100, high=111, low=99, close=109)
    highs = [_pivot(6, 110.0, "high"), _pivot(10, 110.3, "high")]
    out = detect_liquidity(candles, highs, atr=2.0)  # tol = 2.0 * 0.25 = 0.5
    assert len(out["buySide"]) == 1
    assert out["buySide"][0]["swept"] is True
    assert out["recentSweep"]["side"] == "buy"


def test_liquidity_sellside_pool_unswept():
    candles = _flat_candles(30)  # flat lows at 99, never dip below a 90 pool
    lows = [_pivot(6, 90.0, "low"), _pivot(10, 90.2, "low")]
    out = detect_liquidity(candles, lows, atr=2.0)
    assert len(out["sellSide"]) == 1
    assert out["sellSide"][0]["swept"] is False
    assert out["recentSweep"] is None


def test_liquidity_equal_tol_boundary():
    candles = _flat_candles(30)
    atr = 2.0  # tol = 0.5
    clustered = detect_liquidity(
        candles, [_pivot(6, 110.0, "high"), _pivot(10, 110.5, "high")], atr
    )
    apart = detect_liquidity(
        candles, [_pivot(6, 110.0, "high"), _pivot(10, 110.51, "high")], atr
    )
    assert len(clustered["buySide"]) == 1  # diff 0.5 == tol -> clustered (inclusive)
    assert apart["buySide"] == []  # diff 0.51 > tol -> not clustered


# --- Task 5: order blocks ----------------------------------------------------


def test_order_block_bullish_impulse():
    candles = _flat_candles(40)
    candles[15].update(open=101, high=101, low=98, close=99)  # down candle
    candles[16].update(open=99, high=108, low=99, close=107)  # up impulse
    ob = [o for o in detect_order_blocks(candles, [], 2.0) if o["kind"] == "bullish"]
    assert len(ob) >= 1
    assert abs(ob[0]["top"] - 101) < 0.5 and abs(ob[0]["bottom"] - 98) < 0.5
    assert isinstance(ob[0]["mitigated"], bool)


def test_order_block_bearish_impulse():
    candles = _flat_candles(40)
    candles[15].update(open=99, high=102, low=99, close=101)  # up candle
    candles[16].update(open=101, high=101, low=92, close=93)  # down impulse
    ob = [o for o in detect_order_blocks(candles, [], 2.0) if o["kind"] == "bearish"]
    assert len(ob) >= 1
    assert abs(ob[0]["top"] - 102) < 0.5 and abs(ob[0]["bottom"] - 99) < 0.5


def test_order_block_impulse_threshold_boundary():
    def build(impulse_high):
        c = _flat_candles(40)
        c[15].update(open=101, high=101, low=98, close=99)  # down candle, high=101
        c[16].update(open=99, high=impulse_high, low=99, close=impulse_high - 1)  # up
        return c

    # impulse = impulse_high - 101; threshold at impulse_atr=1.0, atr=2.0 -> 2.0
    just_meets = detect_order_blocks(build(103.0), [], 2.0, impulse_atr=1.0)  # impulse 2.0 == thr
    just_below = detect_order_blocks(build(102.99), [], 2.0, impulse_atr=1.0)  # impulse 1.99 < thr
    assert any(o["kind"] == "bullish" for o in just_meets)
    assert not any(o["kind"] == "bullish" for o in just_below)


def test_order_block_breaker_when_closed_through():
    # A bullish OB [98,101] that a later candle CLOSES below (95.x) has FAILED →
    # it flips polarity into a breaker block (broken=True).
    candles = _flat_candles(40)
    candles[15].update(open=101, high=101, low=98, close=99)  # bullish OB origin
    candles[16].update(open=99, high=108, low=99, close=107)  # up impulse
    candles[25].update(open=99, high=99, low=95, close=96)  # closes below OB bottom 98
    ob = [o for o in detect_order_blocks(candles, [], 2.0) if o["kind"] == "bullish"]
    assert ob and ob[0]["broken"] is True


def test_order_block_not_broken_when_only_mitigated():
    # Price returns into the zone (flat candles at 100, inside [98,101]) but never
    # closes below the bottom → mitigated, not broken.
    candles = _flat_candles(40)
    candles[15].update(open=101, high=101, low=98, close=99)  # bullish OB origin
    candles[16].update(open=99, high=108, low=99, close=107)  # up impulse
    ob = [o for o in detect_order_blocks(candles, [], 2.0) if o["kind"] == "bullish"]
    assert ob and ob[0]["mitigated"] is True and ob[0]["broken"] is False


# --- Task 6: dealing range (equilibrium / premium-discount / OTE) -------------


def test_dealing_range_zones_and_ote():
    pivots = [_pivot(4, 100, "low"), _pivot(10, 120, "high")]  # range [100,120], eq 110
    assert detect_dealing_range(pivots, 104)["zone"] == "discount"
    assert detect_dealing_range(pivots, 116)["zone"] == "premium"
    dr = detect_dealing_range(pivots, 110)
    assert dr["zone"] == "equilibrium"
    assert dr["equilibrium"] == 110
    assert dr["high"] == 120 and dr["low"] == 100
    assert 100 <= dr["ote"]["bottom"] <= dr["ote"]["top"] <= 120


def test_dealing_range_equilibrium_band_boundary():
    pivots = [_pivot(4, 100, "low"), _pivot(10, 120, "high")]  # band = 0.02 * 20 = 0.4
    assert detect_dealing_range(pivots, 110.4)["zone"] == "equilibrium"  # == eq+band, inclusive
    assert detect_dealing_range(pivots, 110.5)["zone"] == "premium"  # > eq+band


def test_dealing_range_insufficient_pivots_returns_none():
    assert detect_dealing_range([_pivot(4, 100, "low")], 105) is None
    assert detect_dealing_range([], 105) is None


# --- Task 7: prompt context formatter ----------------------------------------


def _sample_ict_data():
    return {
        "version": 1,
        "timeframe": "1h",
        "tradeType": "swing",
        "marketStructure": {"trend": "bullish", "lastEvent": "BOS", "events": []},
        "orderBlocks": [
            {
                "id": "ob_bull_15",
                "kind": "bullish",
                "top": 101,
                "bottom": 98,
                "mid": 99.5,
                "time": 54000,
                "mitigated": False,
            }
        ],
        "fairValueGaps": [],
        "liquidity": {
            "buySide": [],
            "sellSide": [{"price": 90.1, "time": 36000, "swept": True}],
            "recentSweep": {"side": "sell", "price": 90.1, "time": 72000},
        },
        "dealingRange": {
            "high": 120,
            "low": 100,
            "equilibrium": 110,
            "zone": "discount",
            "ote": {"top": 107.6, "bottom": 104.2},
        },
        "summary": "Market structure bullish (last BOS); price in discount.",
    }


def test_format_ict_context_mentions_structures():
    text = format_ict_engine_context(_sample_ict_data(), current_price=104, current_time=72000)
    low = text.lower()
    assert "ict structure" in low
    assert "order block" in low
    assert "liquidity" in low
    assert "discount" in low
    assert "bullish" in low


def test_format_ict_context_empty_returns_blank():
    assert format_ict_engine_context({}) == ""
    assert format_ict_engine_context(None) == ""
