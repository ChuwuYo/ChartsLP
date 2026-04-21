"""Trendline detection and rendering engine.

Extracted from pattern_engine.py — handles building, scoring, and filtering
of support/resistance trendlines. Pattern detection logic remains in
pattern_engine.py; this module is concerned only with *lines*.

Public API used by pattern_engine:
    build_trend_line          — construct a single trendline with validation
    build_pattern_boundary_lines — upper/lower boundary pair from shape pivots
    build_auto_trendlines     — fallback trendlines when no pattern detected

Utility helpers (also used by pattern_engine):
    project_price             — linear projection along a two-point line
    count_valid_touches       — how many candles touch a given line
"""

from __future__ import annotations

import numpy as np

from chartslp.precision import smart_round_price

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def project_price(point_a: dict, point_b: dict, target_index: int) -> float:
    """Project the price at *target_index* along the line through *point_a* → *point_b*."""
    index_delta = point_b["index"] - point_a["index"]
    if index_delta == 0:
        return point_b["price"]
    slope = (point_b["price"] - point_a["price"]) / index_delta
    return point_a["price"] + slope * (target_index - point_a["index"])


def count_valid_touches(
    point_a: dict,
    point_b: dict,
    candles: list[dict],
    is_upper: bool,
    tolerance_factor: float = 0.5,
) -> int:
    """Count how many candles have valid touches on the trendline.

    A valid touch = high/low is within *tolerance_factor* × ATR of the projected price.
    """
    touches = 0
    atr = _calculate_atr_local(candles) if candles else 0
    # Floor must scale with price — a fixed 0.001 absolute floor makes every
    # candle a "touch" for ultra-low-priced symbols (PEPE, SHIB at ~1e-5 USD)
    # where ATR itself is ~1e-7. Using last_price × 1e-5 keeps the floor
    # meaningful on $100 stocks (~0.001) AND on sub-cent memecoins.
    last_price = candles[-1]["close"] if candles else 0
    price_floor = abs(last_price) * 1e-5
    tolerance = max(atr * tolerance_factor, price_floor, 1e-12)

    for idx in range(point_a["index"], min(point_b["index"] + 1, len(candles))):
        c = candles[idx]
        proj = project_price(point_a, point_b, idx)
        diff = abs((c["high"] if is_upper else c["low"]) - proj)
        if diff <= tolerance:
            touches += 1

    return touches


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calculate_atr_local(candles: list[dict], period: int = 14) -> float:
    """ATR calculation — local copy to avoid circular import.

    Kept intentionally identical to pattern_engine._calculate_atr so the two
    can be used interchangeably.  When pattern_engine calls *this* module it
    passes ATR as a parameter where possible; this function is a fallback for
    code paths that need ATR but don't receive it.
    """
    if len(candles) < 2:
        return 0.0

    true_ranges: list[float] = []
    previous_close = candles[0]["close"]
    for candle in candles[1:]:
        high = candle["high"]
        low = candle["low"]
        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(tr)
        previous_close = candle["close"]

    if not true_ranges:
        return 0.0

    period = min(period, len(true_ranges))
    return sum(true_ranges[-period:]) / period


# ---------------------------------------------------------------------------
# Numerical-differentiation pivot detection (ref: trendln METHOD_NUMDIFF)
# ---------------------------------------------------------------------------

def _detect_pivots_numdiff(
    candles: list[dict],
    accuracy: int = 2,
) -> list[dict]:
    """Detect swing high/low pivots using numerical differentiation.

    Uses findiff to compute 1st/2nd derivatives of the high and low price
    series separately, then finds zero-crossings of the 1st derivative
    confirmed by the sign of the 2nd derivative.

    This finds more pivots than the fractal-window method in pattern_engine
    because it doesn't require a point to be the absolute max/min of N
    neighbors — it only needs a local slope reversal.

    Reference: trendln ``get_extrema`` with ``METHOD_NUMDIFF``, reimplemented
    using ``numpy.gradient`` instead of findiff to avoid the sympy dependency chain.
    """
    if len(candles) < 5:
        return []

    highs = np.asarray([c["high"] for c in candles], dtype=np.float64)
    lows = np.asarray([c["low"] for c in candles], dtype=np.float64)

    def _find_extrema(series: np.ndarray, find_max: bool) -> list[int]:
        """Find indices where the series has local maxima (find_max=True) or minima.

        Uses numpy.gradient for 1st derivative (central differences, same as
        findiff acc=2) and applies it twice for the 2nd derivative.
        """
        mom = np.gradient(series)        # 1st derivative
        momacc = np.gradient(mom)        # 2nd derivative
        n = len(mom)
        indices: list[int] = []

        for x in range(n):
            # 2nd derivative check: maxima need momacc < 0, minima need momacc > 0
            if find_max and momacc[x] >= 0:
                continue
            if not find_max and momacc[x] <= 0:
                continue

            # 1st derivative zero-crossing check
            is_zero_cross = False
            if mom[x] == 0:
                is_zero_cross = True
            elif x < n - 1:
                if find_max and mom[x] > 0 and mom[x + 1] < 0 and series[x] >= series[x + 1]:
                    is_zero_cross = True
                elif not find_max and mom[x] < 0 and mom[x + 1] > 0 and series[x] <= series[x + 1]:
                    is_zero_cross = True
            if not is_zero_cross and x > 0:
                if find_max and mom[x - 1] > 0 and mom[x] < 0 and series[x - 1] < series[x]:
                    is_zero_cross = True
                elif not find_max and mom[x - 1] < 0 and mom[x] > 0 and series[x - 1] > series[x]:
                    is_zero_cross = True

            if is_zero_cross:
                indices.append(x)

        return indices

    max_idxs = _find_extrema(highs, find_max=True)
    min_idxs = _find_extrema(lows, find_max=False)

    pivots: list[dict] = []
    for idx in max_idxs:
        pivots.append({
            "index": idx,
            "time": candles[idx]["time"],
            "price": float(highs[idx]),
            "kind": "high",
        })
    for idx in min_idxs:
        pivots.append({
            "index": idx,
            "time": candles[idx]["time"],
            "price": float(lows[idx]),
            "kind": "low",
        })

    pivots.sort(key=lambda p: p["index"])
    return pivots


def _merge_pivot_sources(
    fractal_pivots: list[dict],
    numdiff_pivots: list[dict],
    atr: float,
) -> list[dict]:
    """Merge pivots from fractal-window and numdiff, removing duplicates.

    Fractal pivots are authoritative (they come from pattern_engine); numdiff
    pivots fill in the gaps where the fractal window was too strict.
    """
    merged = list(fractal_pivots)
    existing_indices = {p["index"] for p in merged}
    tolerance = max(atr * 0.3, 1e-9)

    for np_pivot in numdiff_pivots:
        # Skip if there's already a pivot at this index
        if np_pivot["index"] in existing_indices:
            continue
        # Skip if there's a very close existing pivot (within 2 bars and similar price)
        too_close = False
        for ep in merged:
            if ep["kind"] != np_pivot["kind"]:
                continue
            if abs(ep["index"] - np_pivot["index"]) <= 2 and abs(ep["price"] - np_pivot["price"]) < tolerance:
                too_close = True
                break
        if not too_close:
            merged.append(np_pivot)
            existing_indices.add(np_pivot["index"])

    merged.sort(key=lambda p: p["index"])
    return merged


# ---------------------------------------------------------------------------
# Single trendline builder
# ---------------------------------------------------------------------------

def build_trend_line(
    line_id: str,
    label: str,
    color: str,
    style: str,
    point_a: dict,
    point_b: dict,
    candles: list[dict],
    partner_line: dict | None = None,
) -> dict | None:
    """Build a boundary trendline following common technical-analysis best practices.

    Improvements over naive projection:
    1. Extension capped at 1.5× pattern span (prevents runaway lines)
    2. Body-penetration validation during extension (stops at sustained breach)
    3. Intersection check with partner line (truncates before crossing)
    4. Touch-point validation (ensures line respects wick boundaries)
    5. ATR-based price clamping (prevents extreme projections)
    """
    if not candles:
        return None
    last_idx = len(candles) - 1

    is_upper = (
        point_a.get("kind") == "high"
        or "upper" in (line_id or "").lower()
        or "resist" in (label or "").lower()
    )

    # Calculate ATR and tolerances early
    atr = _calculate_atr_local(candles) if len(candles) >= 14 else 0
    pattern_span = abs(point_b["index"] - point_a["index"])
    if pattern_span < 1:
        pattern_span = 1

    # --- IMPROVEMENT 1: Keep the first pivot anchored and adjust slope via point_b ---
    anchor_a = dict(point_a)
    anchor_span = point_b["index"] - anchor_a["index"]
    if anchor_span != 0:
        original_slope = (point_b["price"] - anchor_a["price"]) / anchor_span
        adjusted_slope = original_slope
        margin = atr * 0.05

        for idx in range(anchor_a["index"] + 1, min(point_b["index"] + 1, len(candles))):
            c = candles[idx]
            steps = idx - anchor_a["index"]
            if steps <= 0:
                continue

            if is_upper:
                required_price = max(c["open"], c["close"]) + margin
                required_slope = (required_price - anchor_a["price"]) / steps
                adjusted_slope = max(adjusted_slope, required_slope)
            else:
                required_price = min(c["open"], c["close"]) - margin
                required_slope = (required_price - anchor_a["price"]) / steps
                adjusted_slope = min(adjusted_slope, required_slope)

        # Limit how far we can bend the line from the original anchor pair.
        max_anchor_shift = atr * 2.0 if atr > 0 else abs(point_b["price"] - anchor_a["price"]) * 0.5
        adjusted_price_b = anchor_a["price"] + adjusted_slope * anchor_span
        if abs(adjusted_price_b - point_b["price"]) > max_anchor_shift:
            if adjusted_price_b > point_b["price"]:
                adjusted_price_b = point_b["price"] + max_anchor_shift
            else:
                adjusted_price_b = point_b["price"] - max_anchor_shift

        point_a = anchor_a
        point_b = {**point_b, "price": adjusted_price_b}

    # --- IMPROVEMENT 2: Extension cap based on pattern span ---
    max_extension = int(pattern_span * 1.5)
    max_extend_idx = min(point_b["index"] + max_extension, last_idx)

    # --- IMPROVEMENT 3: Body-penetration check during extension ---
    breakout_idx = max_extend_idx
    consecutive_body_breach = 0
    body_breach_threshold = 2

    for idx in range(point_b.get("index", 0) + 1, max_extend_idx + 1):
        if idx >= len(candles):
            break
        lp = project_price(point_a, point_b, idx)
        c = candles[idx]
        body_top = max(c["open"], c["close"])
        body_bot = min(c["open"], c["close"])

        if is_upper:
            breach = body_top > lp + atr * 0.3
        else:
            breach = body_bot < lp - atr * 0.3

        if breach:
            consecutive_body_breach += 1
        else:
            consecutive_body_breach = 0

        if consecutive_body_breach >= body_breach_threshold:
            breakout_idx = idx - 1
            break

    extend_idx = max(point_b["index"], min(breakout_idx, max_extend_idx))

    # --- IMPROVEMENT 4: Intersection check with partner line ---
    if partner_line and len(partner_line.get("points", [])) >= 2:
        partner_pts = partner_line["points"]
        partner_slope = 0
        partner_start_idx = 0
        if len(candles) > 1:
            p_start = next((c for c in candles if c["time"] == partner_pts[0]["time"]), None)
            p_end = next((c for c in candles if c["time"] == partner_pts[-1]["time"]), None)
            if p_start and p_end and p_end["index"] != p_start["index"]:
                partner_slope = (partner_pts[-1]["price"] - partner_pts[0]["price"]) / (p_end["index"] - p_start["index"])
                partner_start_idx = p_start["index"]

        for idx in range(point_b["index"] + 1, extend_idx + 1):
            my_price = project_price(point_a, point_b, idx)
            partner_price = partner_pts[0]["price"] + partner_slope * (idx - partner_start_idx)

            if is_upper and my_price < partner_price:
                extend_idx = max(point_b["index"], idx - 1)
                break
            elif not is_upper and my_price > partner_price:
                extend_idx = max(point_b["index"], idx - 1)
                break

    extend_price = project_price(point_a, point_b, extend_idx)

    # --- IMPROVEMENT 5: Smart price clamping based on recent range ---
    recent_lookback = min(30, len(candles))
    recent = candles[-recent_lookback:]
    price_high = max(c["high"] for c in recent)
    price_low = min(c["low"] for c in recent)
    price_range = price_high - price_low

    margin = max(atr * 1.2, price_range * 0.25)
    clamped_high = price_high + margin
    clamped_low = price_low - margin

    extend_price = max(clamped_low, min(extend_price, clamped_high))

    # --- IMPROVEMENT 6: Validate minimum touches (quality check) ---
    touch_count = count_valid_touches(point_a, point_b, candles, is_upper)
    if touch_count < 2:
        best_extend_idx = extend_idx
        best_touches = touch_count
        for try_idx in range(point_b["index"], min(extend_idx + 5, last_idx + 1)):
            test_b = {**point_b, "index": try_idx, "time": candles[try_idx]["time"]}
            test_touches = count_valid_touches(point_a, test_b, candles, is_upper)
            if test_touches > best_touches:
                best_touches = test_touches
                best_extend_idx = try_idx
        extend_idx = best_extend_idx
        extend_price = project_price(point_a, point_b, extend_idx)
        extend_price = max(clamped_low, min(extend_price, clamped_high))

    return {
        "id": line_id,
        "label": label,
        "color": color,
        "style": style,
        "points": [
            {"time": point_a["time"], "price": smart_round_price(point_a["price"])},
            {
                "time": candles[extend_idx]["time"],
                "price": smart_round_price(extend_price),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Pattern boundary lines (upper + lower from shape pivots)
# ---------------------------------------------------------------------------

def build_pattern_boundary_lines(
    shape_pivots: list[dict],
    candles: list[dict],
    atr: float,
) -> list[dict]:
    """Build upper and lower boundary trendlines from a pattern's shape pivots.

    Uses two-pass approach to enable intersection detection:
    1. Build both lines without partner info
    2. Rebuild with partner info to prevent crossing
    """
    if not shape_pivots or len(shape_pivots) < 3:
        return []

    highs = [p for p in shape_pivots if p.get("kind") == "high"]
    lows = [p for p in shape_pivots if p.get("kind") == "low"]
    lines: list[dict] = []

    upper_line = None
    lower_line = None

    # First pass: build initial lines
    if len(highs) >= 2:
        first_h, last_h = highs[0], highs[-1]
        upper_line = build_trend_line("pattern-upper", "Upper", "#ef4444", "solid", first_h, last_h, candles)

    if len(lows) >= 2:
        first_l, last_l = lows[0], lows[-1]
        lower_line = build_trend_line("pattern-lower", "Lower", "#3b82f6", "solid", first_l, last_l, candles)

    # Second pass: rebuild with partner info to detect intersections
    if upper_line and lower_line:
        if len(highs) >= 2:
            first_h, last_h = highs[0], highs[-1]
            upper_line = build_trend_line(
                "pattern-upper", "Upper", "#ef4444", "solid",
                first_h, last_h, candles, partner_line=lower_line
            )

        if len(lows) >= 2:
            first_l, last_l = lows[0], lows[-1]
            lower_line = build_trend_line(
                "pattern-lower", "Lower", "#3b82f6", "solid",
                first_l, last_l, candles, partner_line=upper_line
            )

    # Validate lines don't cross at end points
    if upper_line and lower_line:
        upper_pts = upper_line.get("points", [])
        lower_pts = lower_line.get("points", [])
        if len(upper_pts) >= 2 and len(lower_pts) >= 2:
            upper_end = upper_pts[-1]["price"]
            lower_end = lower_pts[-1]["price"]
            if upper_end < lower_end:
                upper_end_time = upper_pts[-1]["time"]
                lower_end_time = lower_pts[-1]["time"]
                if upper_end_time != lower_end_time:
                    common_end_time = min(upper_end_time, lower_end_time)
                    common_idx = next(
                        (c["index"] for c in candles if c["time"] == common_end_time),
                        len(candles) - 1
                    )
                    if len(highs) >= 2 and len(lows) >= 2:
                        upper_end_price = project_price(highs[0], highs[-1], common_idx)
                        lower_end_price = project_price(lows[0], lows[-1], common_idx)
                        if upper_end_price >= lower_end_price:
                            common_candle = candles[common_idx] if common_idx < len(candles) else candles[-1]
                            upper_pts[-1] = {"time": common_candle["time"], "price": smart_round_price(upper_end_price)}
                            lower_pts[-1] = {"time": common_candle["time"], "price": smart_round_price(lower_end_price)}

    if upper_line:
        lines.append(upper_line)
    if lower_line:
        lines.append(lower_line)

    return lines


# ---------------------------------------------------------------------------
# Auto trendline candidate scoring (ref: trendln get_bestfit + measure_area)
# ---------------------------------------------------------------------------

def _score_trendline_candidate(
    point_a: dict,
    point_b: dict,
    candles: list[dict],
    atr: float,
    is_upper: bool,
) -> dict | None:
    """Score a single pivot pair as a trendline candidate.

    Three-phase evaluation:
    1. **Anchor residual gate** (ref: trendln ``get_bestfit`` on pivot points only):
       Check that the two anchor pivots actually sit near the line they define.
       This is trivially true for a 2-point line, so also check that the line
       doesn't massively violate price structure (wrong-side body penetration).
    2. **Touch counting + Riemann area** (ref: trendln ``measure_area``):
       Scan all candles between anchors. Count how many highs/lows touch the
       line (within ATR tolerance). Accumulate wrong-side penetration area.
    3. **Composite score**: touches (primary), area penalty, span bonus.

    Returns a candidate dict with score, or None if the pair fails gating.
    """
    idx_a = point_a["index"]
    idx_b = point_b["index"]
    span = idx_b - idx_a
    if span < 1:
        return None

    total_bars = len(candles)
    if total_bars < 2:
        return None

    slope = (point_b["price"] - point_a["price"]) / span
    intercept = point_a["price"]  # price at idx_a
    # Price-relative floor keeps tolerances meaningful on sub-cent symbols
    # (PEPE/SHIB). Without this, a 1e-9 absolute floor would force
    # touch_tolerance to 1e-9 for symbols priced below ~2e-9, lumping every
    # candle into "touch" territory.
    last_price = abs(candles[-1]["close"]) if candles else 0.0
    safe_atr = max(atr, last_price * 1e-6, 1e-12)

    start_idx = max(idx_a, 0)
    end_idx = min(idx_b + 1, total_bars)

    # -- Phase 1: Structural gate --
    # A valid resistance line should sit ABOVE most candle bodies;
    # a valid support line should sit BELOW most candle bodies.
    # Count how many candle bodies the line cuts through on the wrong side.
    body_violations = 0
    for idx in range(start_idx, end_idx):
        line_price = intercept + slope * (idx - idx_a)
        c = candles[idx]
        body_top = max(c["open"], c["close"])
        body_bot = min(c["open"], c["close"])
        if is_upper and line_price < body_bot - safe_atr * 0.1:
            # Resistance line below candle body — structurally wrong
            body_violations += 1
        elif not is_upper and line_price > body_top + safe_atr * 0.1:
            # Support line above candle body — structurally wrong
            body_violations += 1

    # Gate: if >60% of candle bodies are on the wrong side, this isn't a valid line
    violation_ratio = body_violations / max(end_idx - start_idx, 1)
    if violation_ratio > 0.6:
        return None

    # -- Phase 2: Touch counting + Riemann area --
    touch_count = 0
    # Same price-relative floor as the gate above — keep touch_tolerance
    # meaningful even when atr collapses to zero (identical candles).
    touch_tolerance = max(safe_atr * 0.5, last_price * 1e-7, 1e-12)
    area = 0.0

    for idx in range(start_idx, end_idx):
        line_price = intercept + slope * (idx - idx_a)
        c = candles[idx]
        candle_edge = c["high"] if is_upper else c["low"]

        # Touch: candle high/low is within tolerance of the line
        if abs(candle_edge - line_price) <= touch_tolerance:
            touch_count += 1

        # Area: wrong-side penetration (ref: trendln measure_area)
        if is_upper:
            overshoot = candle_edge - line_price  # high above resistance
        else:
            overshoot = line_price - candle_edge  # low below support
        if overshoot > 0:
            area += overshoot

    area_normalized = area / max(span, 1)

    # Must have at least 2 touches (the anchors themselves should count)
    if touch_count < 2:
        return None

    # -- Phase 3: Composite score --
    last_idx = total_bars - 1
    projected_last_price = intercept + slope * (last_idx - idx_a)

    score = (
        touch_count * 2.0                              # Primary: more touches = more valid
        - area_normalized / safe_atr * 0.5             # Penalty: wrong-side penetration
        + span / max(total_bars, 1) * 0.5              # Bonus: longer lines more structural
        - violation_ratio * 2.0                        # Penalty: body violations
    )

    return {
        "point_a": point_a,
        "point_b": point_b,
        "slope": slope,
        "projected_last_price": projected_last_price,
        "area": area_normalized,
        "touch_count": touch_count,
        "body_violations": body_violations,
        "score": score,
        "span": span,
        "is_upper": is_upper,
    }


# ---------------------------------------------------------------------------
# Deduplication (ref: pytrendline _mark_duplicates — 2D clustering)
# ---------------------------------------------------------------------------

def _deduplicate_trendlines(candidates: list[dict], atr: float) -> list[dict]:
    """Remove near-duplicate trendlines via 2D clustering on (projected_last_price, slope).

    Inspired by pytrendline ``_mark_duplicates`` (414-505): groups candidates
    whose projected last price and per-bar slope are within ATR-based thresholds,
    then keeps only the highest-scoring member of each group.
    """
    if len(candidates) <= 1:
        return list(candidates)

    safe_atr = max(atr, 1e-9)
    price_threshold = safe_atr * 0.3
    # Slope threshold: normalize by typical span. Two lines are "same slope"
    # if their per-bar slope differs by less than ATR / typical_span.
    # With typical span ~40 bars and ATR ~3, threshold ≈ 0.075 per bar.
    avg_span = sum(c["span"] for c in candidates) / max(len(candidates), 1)
    slope_threshold = safe_atr * 0.02 / max(avg_span / 40.0, 0.5)

    # Assign group IDs via greedy single-link clustering
    group_ids: list[int] = [-1] * len(candidates)
    next_group = 0

    for i, cand_i in enumerate(candidates):
        if group_ids[i] >= 0:
            continue
        # Start a new group
        group_ids[i] = next_group
        # Find all candidates that cluster with this one
        for j in range(i + 1, len(candidates)):
            if group_ids[j] >= 0:
                continue
            price_diff = abs(cand_i["projected_last_price"] - candidates[j]["projected_last_price"])
            # Normalize slope difference by span for comparability
            span_i = max(cand_i["span"], 1)
            span_j = max(candidates[j]["span"], 1)
            slope_per_bar_i = cand_i["slope"]
            slope_per_bar_j = candidates[j]["slope"]
            slope_diff = abs(slope_per_bar_i - slope_per_bar_j)

            if price_diff < price_threshold and slope_diff < slope_threshold:
                group_ids[j] = next_group
        next_group += 1

    # Keep best scorer per group
    best_per_group: dict[int, dict] = {}
    for idx, gid in enumerate(group_ids):
        cand = candidates[idx]
        if gid not in best_per_group or cand["score"] > best_per_group[gid]["score"]:
            best_per_group[gid] = cand

    return list(best_per_group.values())


# ---------------------------------------------------------------------------
# Auto trendlines — enumerate / score / deduplicate / filter pipeline
# ---------------------------------------------------------------------------

# Pivot budget per side, keyed by candle interval. Higher-density charts
# (1m/5m) don't need more structure than the most recent ~12 swings; sparse
# daily/weekly bars carry more per-bar information so we scan more of them
# to catch longer-term lines. N² scoring is tractable up to ~25 per side.
_MAX_PIVOTS_BY_INTERVAL: dict[str, int] = {
    "1m": 12, "3m": 12, "5m": 12, "15m": 12,
    "30m": 14, "1h": 15, "4h": 16,
    "1d": 20, "1w": 22, "1M": 24,
}
_DEFAULT_MAX_PIVOTS = 15


def _resolve_max_pivots(timeframe: str | None) -> int:
    """Return the pivot-scan budget for the given candle interval."""
    if not timeframe:
        return _DEFAULT_MAX_PIVOTS
    return _MAX_PIVOTS_BY_INTERVAL.get(timeframe, _DEFAULT_MAX_PIVOTS)


def build_auto_trendlines(
    candles: list[dict],
    pivots: list[dict],
    atr: float,
    max_pairs: int = 3,
    timeframe: str | None = None,
) -> list[dict]:
    """Generate multiple paired trendlines (upper + lower) ranked by score.

    Each pair is identified by a ``pair`` index (1-based) so the frontend can
    group and toggle them together.  Lines within a pair share the same
    ``pair`` value and are ordered upper-first.

    Pipeline (ref: trendln + pytrendline — score independently, pair naturally):
    1. Enhance pivots with numdiff
    2. Score all upper / lower candidates independently
    3. Deduplicate each side
    4. Zip top-N upper with top-N lower → N pairs
    5. Finalize each pair via build_trend_line with partner awareness
    """
    if not candles or not pivots:
        return []

    # -- Step 1: Enhance pivots --
    numdiff_pivots = _detect_pivots_numdiff(candles)
    enhanced_pivots = _merge_pivot_sources(pivots, numdiff_pivots, atr)

    # Pivot budget scales with candle interval — sparse TFs scan more.
    max_pivots = _resolve_max_pivots(timeframe)
    high_pivots = [p for p in enhanced_pivots if p["kind"] == "high"][-max_pivots:]
    low_pivots = [p for p in enhanced_pivots if p["kind"] == "low"][-max_pivots:]

    if len(high_pivots) < 2 and len(low_pivots) < 2:
        return []

    # -- Step 2: Score all pivot pairs per side --
    upper_candidates: list[dict] = []
    for i in range(len(high_pivots)):
        for j in range(i + 1, len(high_pivots)):
            r = _score_trendline_candidate(high_pivots[i], high_pivots[j], candles, atr, True)
            if r:
                upper_candidates.append(r)

    lower_candidates: list[dict] = []
    for i in range(len(low_pivots)):
        for j in range(i + 1, len(low_pivots)):
            r = _score_trendline_candidate(low_pivots[i], low_pivots[j], candles, atr, False)
            if r:
                lower_candidates.append(r)

    # -- Step 3: Deduplicate --
    upper_candidates = _deduplicate_trendlines(upper_candidates, atr)
    lower_candidates = _deduplicate_trendlines(lower_candidates, atr)

    # -- Step 4: Sort and take top-N per side --
    upper_candidates.sort(key=lambda c: c["score"], reverse=True)
    lower_candidates.sort(key=lambda c: c["score"], reverse=True)

    top_uppers = upper_candidates[:max_pairs]
    top_lowers = lower_candidates[:max_pairs]

    # -- Step 5: Build pairs by zipping top-N, finalize with partner awareness --
    lines: list[dict] = []
    # top_uppers/top_lowers are already sliced to max_pairs, so max(len, len) ≤ max_pairs.
    n_pairs = max(len(top_uppers), len(top_lowers))

    for pair_idx in range(n_pairs):
        pair_num = pair_idx + 1
        u_cand = top_uppers[pair_idx] if pair_idx < len(top_uppers) else None
        l_cand = top_lowers[pair_idx] if pair_idx < len(top_lowers) else None

        suffix = f"-{pair_num}" if pair_num > 1 else ""

        # Symmetric two-pass (matches build_pattern_boundary_lines):
        # 1. build both without partners
        # 2. rebuild both with the other as partner so neither uses stale info
        lower_line = None
        upper_line = None

        if l_cand:
            lower_line = build_trend_line(
                f"auto-lower{suffix}", "Support", "#B8E600", "dashed",
                l_cand["point_a"], l_cand["point_b"], candles,
            )
        if u_cand:
            upper_line = build_trend_line(
                f"auto-upper{suffix}", "Resistance", "#B8E600", "dashed",
                u_cand["point_a"], u_cand["point_b"], candles,
            )
        # Pass 2: rebuild both with the partner line established above
        if u_cand and lower_line:
            upper_line = build_trend_line(
                f"auto-upper{suffix}", "Resistance", "#B8E600", "dashed",
                u_cand["point_a"], u_cand["point_b"], candles,
                partner_line=lower_line,
            ) or upper_line
        if l_cand and upper_line:
            lower_line = build_trend_line(
                f"auto-lower{suffix}", "Support", "#B8E600", "dashed",
                l_cand["point_a"], l_cand["point_b"], candles,
                partner_line=upper_line,
            ) or lower_line

        if upper_line and u_cand:
            upper_line["score"] = round(u_cand["score"], 2)
            upper_line["touches"] = u_cand["touch_count"]
            upper_line["pair"] = pair_num
            lines.append(upper_line)
        if lower_line and l_cand:
            lower_line["score"] = round(l_cand["score"], 2)
            lower_line["touches"] = l_cand["touch_count"]
            lower_line["pair"] = pair_num
            lines.append(lower_line)

    return lines
