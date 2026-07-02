"""ICT / SMC structural detection engine (additive, zero-impact).

A new, independent entry point built over chartslp's existing normalized
candles, swing pivots and ATR tolerances. It detects the ICT structure set —
market structure (BOS / CHoCH), liquidity pools + sweeps, order blocks, fair
value gaps, and the dealing range (equilibrium / premium-discount / OTE) — for
the ``smc`` analysis strategy.

It does **not** touch ``detect_chart_patterns`` or any existing public surface;
it is purely additive. Detection rules are ported natively (numpy / dict / list,
no pandas) from the reference algorithms in joshyattridge/smart-money-concepts
(``smc.py``). See docs/superpowers/specs/2026-06-24-ict-engine-design.md.
"""

from __future__ import annotations

from chartslp.patterns import (
    _TIMEFRAME_MAX_SCAN,
    _calculate_atr,
    _calculate_pattern_tolerances,
    _detect_pivots,
    _format_pivot_time,
    normalize_klines,
)
from chartslp.precision import smart_round_price

# Per-timeframe swing window — identical to detect_chart_patterns so ICT swing
# structure matches the pattern engine's pivots.
_PIVOT_LR = {"1m": 2, "3m": 2, "5m": 2, "15m": 3, "30m": 3, "1h": 4, "4h": 4}

# Minimum candles for stable structure (mirrors the pattern engine's guard).
_MIN_CANDLES = 20


def _ict_empty(timeframe: str, trade_type: str) -> dict:
    """The neutral / insufficient-data shape (all top-level keys present)."""
    return {
        "version": 1,
        "timeframe": timeframe,
        "tradeType": trade_type,
        "marketStructure": {"trend": "ranging", "lastEvent": None, "events": []},
        "orderBlocks": [],
        "fairValueGaps": [],
        "liquidity": {"buySide": [], "sellSide": [], "recentSweep": None},
        "dealingRange": None,
        "summary": "",
    }


def detect_ict_structures(
    klines,
    *,
    timeframe: str = "1h",
    trade_type: str = "swing",
    current_price: float | None = None,
) -> dict:
    """Detect the ICT structure set over ``klines``.

    Reuses chartslp's normalized candles + per-timeframe swing pivots + ATR
    tolerances. Sync (like detect_chart_patterns); the async api path calls it
    via ``asyncio.to_thread``.
    """
    candles = normalize_klines(klines)
    max_scan = _TIMEFRAME_MAX_SCAN.get(timeframe, 100)
    if len(candles) > max_scan:
        candles = candles[-max_scan:]
        for idx, candle in enumerate(candles):
            candle["index"] = idx

    if len(candles) < _MIN_CANDLES:
        return _ict_empty(timeframe, trade_type)

    atr, _flat_tol, min_move, _breakout = _calculate_pattern_tolerances(candles, trade_type)
    pivot_lr = _PIVOT_LR.get(timeframe, 4)
    pivots = _detect_pivots(candles, left=pivot_lr, right=pivot_lr, min_move=min_move * 0.5)

    if current_price is None:
        current_price = candles[-1]["close"]

    market_structure = detect_market_structure(candles, pivots)
    fair_value_gaps = detect_fvg(candles)
    liquidity = detect_liquidity(candles, pivots, atr)
    order_blocks = detect_order_blocks(candles, pivots, atr)
    dealing_range = detect_dealing_range(pivots, current_price)

    return {
        "version": 1,
        "timeframe": timeframe,
        "tradeType": trade_type,
        "marketStructure": market_structure,
        "orderBlocks": order_blocks,
        "fairValueGaps": fair_value_gaps,
        "liquidity": liquidity,
        "dealingRange": dealing_range,
        "summary": _build_summary(
            market_structure, order_blocks, fair_value_gaps, liquidity, dealing_range
        ),
    }


# --- Detection units (filled in by Tasks 2-6; stubs return empty structures) -


def _ms_event(event_type: str, direction: str, level: float, breaking_pivot: dict) -> dict:
    return {
        "type": event_type,
        "direction": direction,
        "price": smart_round_price(level),
        "time": breaking_pivot["time"],
    }


def detect_market_structure(candles: list[dict], pivots: list[dict]) -> dict:
    """Break of Structure / Change of Character (native port of smc.py::bos_choch).

    Operates on OUR alternating swing pivots. Over each window of 4 consecutive
    swings the four ICT patterns are matched (smc's level-ordering conditions):

    - bullish BOS  : L,H,L,H with L1 < L2 < H1 < H2 (higher low + higher high)
    - bullish CHoCH: L,H,L,H with H2 > H1 > L1 > L2 (downtrend, then high broken up)
    - bearish BOS  : H,L,H,L with H1 > H2 > L1 > L2 (lower high + lower low)
    - bearish CHoCH: H,L,H,L with L2 < L1 < H1 < H2 (uptrend, then low broken down)

    The broken level is the prior same-side swing (smc's level_order[-3]). Trend
    is read from the most recent event; no events -> "ranging".
    """
    swings = [p for p in pivots if p.get("kind") in ("high", "low")]
    events: list[dict] = []
    hl_order: list[int] = []
    level_order: list[float] = []
    positions: list[dict] = []

    for piv in swings:
        hl_order.append(1 if piv["kind"] == "high" else -1)
        level_order.append(piv["price"])
        positions.append(piv)
        if len(level_order) < 4:
            continue

        last4 = hl_order[-4:]
        a, b, c, d = level_order[-4], level_order[-3], level_order[-2], level_order[-1]
        breaker = positions[-1]

        if last4 == [-1, 1, -1, 1]:
            # a=L1, b=H1, c=L2, d=H2 ; broken level = H1 (b)
            if a < c < b < d:
                events.append(_ms_event("BOS", "bullish", b, breaker))
            elif d > b > a > c:
                events.append(_ms_event("CHoCH", "bullish", b, breaker))
        elif last4 == [1, -1, 1, -1]:
            # a=H1, b=L1, c=H2, d=L2 ; broken level = L1 (b)
            if a > c > b > d:
                events.append(_ms_event("BOS", "bearish", b, breaker))
            elif d < b < a < c:
                events.append(_ms_event("CHoCH", "bearish", b, breaker))

    if not events:
        return {"trend": "ranging", "lastEvent": None, "events": []}

    last = events[-1]
    trend = "bullish" if last["direction"] == "bullish" else "bearish"
    return {"trend": trend, "lastEvent": last["type"], "events": events}


def detect_fvg(candles: list[dict], *, min_gap_atr: float = 0.1) -> list[dict]:
    """Fair Value Gaps — 3-candle imbalances (native port of smc.py::fvg).

    Bullish FVG when ``high[i-1] < low[i+1]`` on a bullish middle candle; the
    gap band is ``[high[i-1], low[i+1]]``. Bearish FVG when ``low[i-1] >
    high[i+1]`` on a bearish middle candle; band ``[high[i+1], low[i-1]]``.
    Gaps narrower than ``atr * min_gap_atr`` are noise and dropped. ``filled``
    is True once a later candle trades back into the band.
    """
    n = len(candles)
    if n < 3:
        return []

    atr = _calculate_atr(candles)
    min_gap = atr * min_gap_atr
    out: list[dict] = []

    for i in range(1, n - 1):
        prev_c, cur, nxt = candles[i - 1], candles[i], candles[i + 1]
        bullish = prev_c["high"] < nxt["low"] and cur["close"] > cur["open"]
        bearish = prev_c["low"] > nxt["high"] and cur["close"] < cur["open"]
        if bullish:
            top, bottom, kind = nxt["low"], prev_c["high"], "bullish"
        elif bearish:
            top, bottom, kind = prev_c["low"], nxt["high"], "bearish"
        else:
            continue

        if (top - bottom) < min_gap:
            continue

        filled = False
        for later in candles[i + 2:]:
            if kind == "bullish" and later["low"] <= top:
                filled = True
                break
            if kind == "bearish" and later["high"] >= bottom:
                filled = True
                break

        out.append({
            "id": f"fvg_{i}",
            "kind": kind,
            "top": smart_round_price(top),
            "bottom": smart_round_price(bottom),
            "mid": smart_round_price((top + bottom) / 2),
            "time": cur["time"],
            "filled": filled,
        })

    return out


def _detect_sweep(
    candles: list[dict], level: float, after_index: int, side: str
) -> tuple[bool, int | None]:
    """First candle after ``after_index`` that wicks beyond ``level`` then
    closes back through it. Returns (swept, swept_time)."""
    for idx in range(after_index + 1, len(candles)):
        cdl = candles[idx]
        if side == "buy" and cdl["high"] > level and cdl["close"] < level:
            return True, cdl["time"]
        if side == "sell" and cdl["low"] < level and cdl["close"] > level:
            return True, cdl["time"]
    return False, None


def _cluster_liquidity(
    side_pivots: list[dict], tol: float, candles: list[dict], side: str
) -> list[dict]:
    """Group equal highs/lows (within ``tol``) into pools (smc.py::liquidity)."""
    pools: list[dict] = []
    used = [False] * len(side_pivots)
    for i, seed in enumerate(side_pivots):
        if used[i]:
            continue
        group = [seed]
        for j in range(i + 1, len(side_pivots)):
            if not used[j] and abs(side_pivots[j]["price"] - seed["price"]) <= tol:
                group.append(side_pivots[j])
                used[j] = True
        if len(group) < 2:
            continue
        level = sum(g["price"] for g in group) / len(group)
        last_piv = max(group, key=lambda g: g["index"])
        swept, swept_time = _detect_sweep(candles, level, last_piv["index"], side)
        pools.append({
            "side": side,
            "price": smart_round_price(level),
            "time": last_piv["time"],
            "swept": swept,
            "sweptTime": swept_time,
        })
    return pools


def detect_liquidity(
    candles: list[dict], pivots: list[dict], atr: float, *, equal_tol_atr: float = 0.25
) -> dict:
    """Buy-side (equal highs) / sell-side (equal lows) liquidity pools + sweep.

    Equal-level tolerance is ``atr * equal_tol_atr``; a pool needs >= 2 swing
    points. ``recentSweep`` is the most recently swept pool across both sides.
    """
    tol = atr * equal_tol_atr
    highs = [p for p in pivots if p.get("kind") == "high"]
    lows = [p for p in pivots if p.get("kind") == "low"]
    buy_pools = _cluster_liquidity(highs, tol, candles, "buy")
    sell_pools = _cluster_liquidity(lows, tol, candles, "sell")

    swept_pools = [p for p in buy_pools + sell_pools if p["swept"]]
    recent_sweep = None
    if swept_pools:
        latest = max(swept_pools, key=lambda p: p["sweptTime"])
        recent_sweep = {
            "side": latest["side"],
            "price": latest["price"],
            "time": latest["sweptTime"],
        }

    def _project(pools):
        return [{"price": p["price"], "time": p["time"], "swept": p["swept"]} for p in pools]

    return {
        "buySide": _project(buy_pools),
        "sellSide": _project(sell_pools),
        "recentSweep": recent_sweep,
    }


def _make_ob(index: int, candle: dict, candles: list[dict], kind: str) -> dict:
    top, bottom = candle["high"], candle["low"]
    # Scan later candles (past the impulse leg) for two states:
    #  - mitigated: price returns into the zone (tested, may still hold).
    #  - broken: price CLOSES through the far side -> the OB has FAILED and flips
    #    polarity into a breaker block (a broken bullish OB becomes bearish
    #    resistance; a broken bearish OB becomes bullish support).
    mitigated = False
    broken = False
    for later in candles[index + 2:]:
        if kind == "bullish":
            if later["low"] <= top:
                mitigated = True
            if later["close"] < bottom:
                broken = True
        else:
            if later["high"] >= bottom:
                mitigated = True
            if later["close"] > top:
                broken = True
        if broken:
            break
    return {
        "id": f"ob_{kind[:4]}_{index}",
        "kind": kind,
        "top": smart_round_price(top),
        "bottom": smart_round_price(bottom),
        "mid": smart_round_price((top + bottom) / 2),
        "time": candle["time"],
        "mitigated": mitigated,
        "broken": broken,
    }


def detect_order_blocks(
    candles: list[dict], pivots: list[dict], atr: float, *,
    impulse_atr: float = 1.0, lookahead: int = 3,
) -> list[dict]:
    """Order blocks — the last opposing candle before a displacement impulse.

    Bullish OB = a down candle immediately followed by an up candle whose
    leg displaces >= ``impulse_atr * atr`` above the block; bearish OB is the
    mirror. Zone = the opposing candle's high/low. ``mitigated`` once price
    later returns into the zone. (Native port of smc.py::ob, impulse-anchored;
    ``pivots`` is accepted for signature parity and reserved for future
    swing-anchoring.)
    """
    n = len(candles)
    if n < 3 or atr <= 0:
        return []

    threshold = impulse_atr * atr
    out: list[dict] = []
    for i in range(n - 2):
        cur, nxt = candles[i], candles[i + 1]
        window = candles[i + 1:min(i + 1 + lookahead, n)]
        if cur["close"] < cur["open"] and nxt["close"] > nxt["open"]:
            impulse = max(w["high"] for w in window) - cur["high"]
            if impulse >= threshold:
                out.append(_make_ob(i, cur, candles, "bullish"))
        elif cur["close"] > cur["open"] and nxt["close"] < nxt["open"]:
            impulse = cur["low"] - min(w["low"] for w in window)
            if impulse >= threshold:
                out.append(_make_ob(i, cur, candles, "bearish"))

    return out


def detect_dealing_range(
    pivots: list[dict], current_price: float | None, *, eq_band_pct: float = 0.02
) -> dict | None:
    """Dealing range -> equilibrium / premium-discount / OTE (smc.py::retracements).

    Range = most recent swing high <-> swing low. Equilibrium = 50%; ``zone`` is
    ``current_price`` vs equilibrium with a small band. OTE = the 62-79%
    retracement band, sided by the most recent leg's direction (discount OTE for
    a recent up-leg, premium OTE for a recent down-leg).
    """
    highs = [p for p in pivots if p.get("kind") == "high"]
    lows = [p for p in pivots if p.get("kind") == "low"]
    if not highs or not lows or current_price is None:
        return None

    recent_high, recent_low = highs[-1], lows[-1]
    high = max(recent_high["price"], recent_low["price"])
    low = min(recent_high["price"], recent_low["price"])
    rng = high - low
    if rng <= 0:
        return None

    eq = (high + low) / 2.0
    band = rng * eq_band_pct
    eq_eps = max(1.0, abs(eq)) * 1e-9  # don't flip the band on float dust at the boundary
    if abs(current_price - eq) <= band + eq_eps:
        zone = "equilibrium"
    elif current_price > eq:
        zone = "premium"
    else:
        zone = "discount"

    if recent_high["index"] >= recent_low["index"]:
        # recent up-leg -> long OTE sits in discount (62-79% retrace from the high)
        ote_bottom, ote_top = low + 0.21 * rng, low + 0.38 * rng
    else:
        # recent down-leg -> short OTE sits in premium
        ote_bottom, ote_top = low + 0.62 * rng, low + 0.79 * rng

    return {
        "high": smart_round_price(high),
        "low": smart_round_price(low),
        "equilibrium": smart_round_price(eq),
        "zone": zone,
        "ote": {
            "top": smart_round_price(ote_top),
            "bottom": smart_round_price(ote_bottom),
        },
    }


def _build_summary(
    market_structure: dict,
    order_blocks: list[dict],
    fair_value_gaps: list[dict],
    liquidity: dict,
    dealing_range: dict | None,
) -> str:
    trend = market_structure.get("trend", "ranging")
    last_event = market_structure.get("lastEvent")
    parts = [f"Market structure {trend}" + (f" (last {last_event})" if last_event else "")]

    if order_blocks:
        unmitigated = sum(1 for o in order_blocks if not o["mitigated"])
        breakers = sum(1 for o in order_blocks if o.get("broken"))
        seg = f"{len(order_blocks)} order block(s), {unmitigated} unmitigated"
        if breakers:
            seg += f", {breakers} breaker(s)"
        parts.append(seg)
    if fair_value_gaps:
        unfilled = sum(1 for f in fair_value_gaps if not f["filled"])
        parts.append(f"{len(fair_value_gaps)} FVG(s), {unfilled} unfilled")
    sweep = liquidity.get("recentSweep")
    if sweep:
        parts.append(f"{sweep['side']}-side liquidity swept")
    if dealing_range:
        parts.append(f"price in {dealing_range['zone']}")

    return "; ".join(parts) + "."


def format_ict_engine_context(
    ict_data: dict | None, *, current_price: float | None = None, current_time: int | None = None
) -> str:
    """Render detected ICT structures as a prompt context block.

    Sibling of ``format_pattern_engine_context`` — same bullet-line style. For
    the ``smc`` strategy this block replaces the PATTERN ENGINE block. Empty /
    falsy ``ict_data`` -> ``""``.
    """
    if not ict_data:
        return ""

    market_structure = ict_data.get("marketStructure") or {}
    timeframe = ict_data.get("timeframe") or ""

    lines = ["\n**ICT STRUCTURE (deterministic smart-money structure from raw OHLCV):**"]

    header_parts = []
    if timeframe:
        header_parts.append(f"Timeframe: {timeframe}")
    if current_time is not None:
        header_parts.append(f"Current time: {_format_pivot_time(current_time)}")
    if current_price is not None:
        header_parts.append(f"Current price: {current_price}")
    if header_parts:
        lines.append(f"- {' | '.join(header_parts)}")

    trend = market_structure.get("trend", "ranging")
    last_event = market_structure.get("lastEvent")
    ms_line = f"- Market structure: {trend}"
    if last_event:
        ms_line += f" | last structural event: {last_event}"
    lines.append(ms_line)

    for ob in (ict_data.get("orderBlocks") or [])[:4]:
        kind = ob.get("kind")
        if ob.get("broken"):
            flip = "bearish" if kind == "bullish" else "bullish"
            state = f"broken -> {flip} breaker"
        elif ob.get("mitigated"):
            state = "mitigated"
        else:
            state = "unmitigated"
        lines.append(
            f"- Order block [{kind}]: zone {ob.get('bottom')}-{ob.get('top')} "
            f"(mid {ob.get('mid')}) | {state} | @{_format_pivot_time(ob.get('time'))}"
        )

    for fvg in (ict_data.get("fairValueGaps") or [])[:4]:
        state = "filled" if fvg.get("filled") else "open"
        lines.append(
            f"- Fair value gap [{fvg.get('kind')}]: {fvg.get('bottom')}-{fvg.get('top')} "
            f"(mid {fvg.get('mid')}) | {state}"
        )

    liquidity = ict_data.get("liquidity") or {}
    buy_side = liquidity.get("buySide") or []
    sell_side = liquidity.get("sellSide") or []
    if buy_side:
        levels = ", ".join(str(p["price"]) for p in buy_side[:3])
        lines.append(f"- Buy-side liquidity (equal highs): {levels}")
    if sell_side:
        levels = ", ".join(str(p["price"]) for p in sell_side[:3])
        lines.append(f"- Sell-side liquidity (equal lows): {levels}")
    sweep = liquidity.get("recentSweep")
    if sweep:
        lines.append(
            f"- Recent liquidity sweep: {sweep.get('side')}-side @ {sweep.get('price')} "
            f"({_format_pivot_time(sweep.get('time'))})"
        )

    dealing_range = ict_data.get("dealingRange")
    if dealing_range:
        ote = dealing_range.get("ote") or {}
        lines.append(
            f"- Dealing range: {dealing_range.get('low')}-{dealing_range.get('high')} "
            f"| equilibrium {dealing_range.get('equilibrium')} | price in {dealing_range.get('zone')} "
            f"| OTE {ote.get('bottom')}-{ote.get('top')}"
        )

    summary = ict_data.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")

    # These ICT concepts are NOT pre-detected by the engine — the methodology
    # references them, so derive them yourself from the candle timestamps + series.
    lines.append(
        "- Not pre-detected (read from the candle timestamps + series yourself): "
        "killzone / session time, previous-day & previous-week highs/lows (PDH/PDL)."
    )

    return "\n".join(lines)
