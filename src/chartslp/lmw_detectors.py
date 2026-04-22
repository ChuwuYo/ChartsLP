"""LMW-style geometric pattern detectors.

Formal geometric pattern judgments based on:

    Lo, A. W., Mamaysky, H., & Wang, J. (2000).
    "Foundations of Technical Analysis: Computational Algorithms, Statistical
    Inference, and Empirical Implementation." Journal of Finance 55(4).

Each detector enumerates pivot sub-sequences of a fixed arity, then applies
LMW's geometric inequalities plus Bulkowski-flavoured symmetry tolerances
(for H&S top/bottom specifically).

Signature matches the existing ``_detect_*`` helpers in ``patterns.py`` so
these can be slotted into the same detector pipeline::

    fn(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio)
      -> dict | None

Imports from ``patterns`` are resolved lazily inside each function to avoid
any load-time circular reference.

Patterns covered in this module (Step 5):
  - head_and_shoulders, inverse_head_and_shoulders
  - ascending_triangle, descending_triangle, symmetrical_triangle
  - rising_wedge, falling_wedge

Flag/pennant, channel, rectangle and the remaining 21 patterns stay on the
legacy heuristic path for now (Step 6).
"""
from __future__ import annotations

from statistics import mean

# ---------------------------------------------------------------------------
# Small helpers (self-contained, no patterns imports)
# ---------------------------------------------------------------------------

def _fit_line(pivots: list[dict]) -> tuple[float, float, float]:
    """Least-squares line through *pivots* on (index, price).

    Returns ``(slope, intercept, rmse)``. rmse is sqrt(mean squared residual);
    callers can divide by avg_range for a dimensionless fit-quality score.
    """
    if len(pivots) < 2:
        return (0.0, pivots[0]["price"] if pivots else 0.0, 0.0)
    xs = [p["index"] for p in pivots]
    ys = [p["price"] for p in pivots]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    residuals = [ys[i] - (slope * xs[i] + intercept) for i in range(n)]
    rmse = (sum(r * r for r in residuals) / n) ** 0.5
    return (slope, intercept, rmse)


# ---------------------------------------------------------------------------
# Head and Shoulders (5-extrema structure)
# ---------------------------------------------------------------------------

def detect_head_and_shoulders_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    """LMW-strict head & shoulders top.

    Requires 5 extrema in order high-low-high-low-high where:
      - E3 (head) is strictly higher than E1 and E5 (shoulders)
      - |E1 - avg_shoulder| / avg_shoulder < SYM_TOL_H_S_TOP
      - |E2 - avg_trough|  / avg_trough  < NECK_TOL
      - Bulkowski finding: unsymmetrical H&S top outperforms symmetric
        (23.8% vs 22.8% avg move) → widen SYM_TOL_H_S_TOP to 3%.

    Returns the canonical pattern dict consumed by detect_chart_patterns.
    """
    return _hs_core(
        pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
        expected_kinds=["high", "low", "high", "low", "high"],
        is_top=True,
        pattern_type="head_and_shoulders",
        sym_tol_pct=0.030,       # Bulkowski-widened
        neck_tol_pct=0.020,
    )


def detect_inverse_head_and_shoulders_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    """LMW-strict inverse head & shoulders bottom.

    5 extrema low-high-low-high-low, head (E3) strictly lower than both
    shoulders. Symmetric bottoms historically outperform (Bulkowski) so
    the symmetry tolerance is tighter at 1.5%.
    """
    return _hs_core(
        pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
        expected_kinds=["low", "high", "low", "high", "low"],
        is_top=False,
        pattern_type="inverse_head_and_shoulders",
        sym_tol_pct=0.015,
        neck_tol_pct=0.020,
    )


def _hs_core(
    pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
    *, expected_kinds, is_top, pattern_type, sym_tol_pct, neck_tol_pct,
):
    from chartslp.patterns import (
        PATTERN_LABELS,
        _build_horizontal_level,
        _build_marker,
        _pattern_confidence,
    )
    from chartslp.precision import smart_round_price

    if len(pivots) < 5 or not candles:
        return None

    best = None
    best_score = -1.0

    # Scan recent sub-sequences first (most recent wins ties)
    for start in range(len(pivots) - 5, -1, -1):
        seq = pivots[start:start + 5]
        if [p["kind"] for p in seq] != expected_kinds:
            continue

        ls, trough_a, head, trough_b, rs = seq

        if is_top:
            # Head strictly higher than both shoulders
            if head["price"] <= ls["price"] or head["price"] <= rs["price"]:
                continue
            # Shoulders must be above the troughs
            if ls["price"] <= max(trough_a["price"], trough_b["price"]):
                continue
            if rs["price"] <= max(trough_a["price"], trough_b["price"]):
                continue
        else:
            if head["price"] >= ls["price"] or head["price"] >= rs["price"]:
                continue
            if ls["price"] >= min(trough_a["price"], trough_b["price"]):
                continue
            if rs["price"] >= min(trough_a["price"], trough_b["price"]):
                continue

        avg_shoulder = mean([ls["price"], rs["price"]])
        avg_trough = mean([trough_a["price"], trough_b["price"]])
        if abs(avg_shoulder) < 1e-12 or abs(avg_trough) < 1e-12:
            continue

        sym_ls = abs(ls["price"] - avg_shoulder) / abs(avg_shoulder)
        sym_rs = abs(rs["price"] - avg_shoulder) / abs(avg_shoulder)
        if sym_ls > sym_tol_pct or sym_rs > sym_tol_pct:
            continue

        neck_err_a = abs(trough_a["price"] - avg_trough) / abs(avg_trough)
        neck_err_b = abs(trough_b["price"] - avg_trough) / abs(avg_trough)
        if neck_err_a > neck_tol_pct or neck_err_b > neck_tol_pct:
            continue

        # Head prominence relative to shoulders
        prominence = (head["price"] - avg_shoulder) if is_top else (avg_shoulder - head["price"])
        if prominence < min_move * 0.5:
            continue

        # State: breakout vs forming vs retest
        current_price = candles[-1]["close"]
        neckline = avg_trough
        if is_top:
            if current_price < neckline - breakout_buffer:
                state = "breakdown"
            elif current_price < avg_shoulder - breakout_buffer:
                state = "retest"
            else:
                state = "forming"
        else:
            if current_price > neckline + breakout_buffer:
                state = "breakout"
            elif current_price > avg_shoulder + breakout_buffer:
                state = "retest"
            else:
                state = "forming"

        # Score components (LMW: stricter geometry → higher base confidence)
        shoulder_sim = 1.0 - max(sym_ls, sym_rs) / max(sym_tol_pct, 1e-9)
        neck_flat = 1.0 - max(neck_err_a, neck_err_b) / max(neck_tol_pct, 1e-9)
        head_score = min(1.0, prominence / max(min_move * 2.0, 1e-9))
        volume_score = min(1.0, max(volume_ratio - 1.0, 0.0))

        confidence = _pattern_confidence(
            0.52,  # LMW strict base
            shoulder_sim * 0.22,
            neck_flat * 0.14,
            head_score * 0.18,
            volume_score * 0.08,
        )

        score = confidence * (1.0 + (start / max(len(pivots) - 5, 1)) * 0.05)
        if score <= best_score:
            continue

        bias = "bearish" if is_top else "bullish"
        marker_color = "#ef4444" if is_top else "#22c55e"
        label = PATTERN_LABELS[pattern_type]
        description = (
            "LMW-strict three-peak reversal; middle peak is the head."
            if is_top else
            "LMW-strict three-trough reversal; middle trough is the head."
        )

        markers = []
        if state in ("breakdown", "breakout"):
            markers.append(_build_marker(
                f"{pattern_type}-break",
                "Breakdown" if is_top else "Breakout",
                candles[-1]["time"], current_price, marker_color, bias,
            ))

        result = {
            "primaryPattern": {
                "type": pattern_type,
                "label": label,
                "bias": bias,
                "state": state,
                "confidence": confidence,
                "description": description,
            },
            "shapePivots": list(seq),
            "trendLines": [],
            "horizontalLevels": [
                _build_horizontal_level(
                    f"{pattern_type}-neckline", "Neckline", "#3b82f6", "dashed",
                    neckline, trough_a["time"], candles[-1]["time"], candles=candles,
                ),
                _build_horizontal_level(
                    f"{pattern_type}-shoulder",
                    "Resistance" if is_top else "Support",
                    marker_color, "dashed",
                    avg_shoulder, ls["time"], candles[-1]["time"], candles=candles,
                ),
            ],
            "markers": markers,
            "levels": {
                "support" if is_top else "resistance": smart_round_price(neckline),
                "resistance" if is_top else "support": smart_round_price(head["price"]),
            },
        }
        best = result
        best_score = score

    return best


# ---------------------------------------------------------------------------
# Triangles (ascending, descending, symmetrical) — slope-regression based
# ---------------------------------------------------------------------------

def _triangle_core(
    pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
    *, variant,
):
    """LMW-strict triangle detection using least-squares slope regression on
    pivot highs and lows independently.

    variant: "ascending" / "descending" / "symmetrical"
      - ascending:   flat top, rising bottom    (bullish bias)
      - descending:  falling top, flat bottom   (bearish bias)
      - symmetrical: falling top, rising bottom (neutral — bias follows prior trend)

    Needs at least 2 highs + 2 lows within the last 4-8 pivots.
    """
    from chartslp.patterns import (
        PATTERN_LABELS,
        _avg_bar_length,
        _build_horizontal_level,
        _pattern_confidence,
    )
    from chartslp.precision import smart_round_price
    from chartslp.trendlines import build_trend_line

    if len(pivots) < 4 or not candles:
        return None

    # Take the last window of pivots likely to form the triangle
    scan = pivots[-8:]
    highs = [p for p in scan if p["kind"] == "high"]
    lows = [p for p in scan if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    top_slope, top_int, top_rmse = _fit_line(highs[-4:] if len(highs) >= 4 else highs)
    bot_slope, bot_int, bot_rmse = _fit_line(lows[-4:] if len(lows) >= 4 else lows)

    avg_bar = _avg_bar_length(candles, scan[0]["index"], scan[-1]["index"] + 1)
    if avg_bar <= 0:
        return None

    # Slope tolerance in price/bar: a "flat" line has |slope| < flat_tol.
    # Per-bar only — mixing ``min_move`` inflated this on high-priced assets
    # (XAU/USD ~4800) and misclassified sloped lines as flat.
    flat_tol = max(avg_bar * 0.05, 1e-9)
    # Convergence check for sym triangle: top must descend, bottom ascend
    # and they should converge (not diverge)

    def _is_flat(s: float) -> bool:
        return abs(s) <= flat_tol

    def _is_rising(s: float) -> bool:
        return s > flat_tol

    def _is_falling(s: float) -> bool:
        return s < -flat_tol

    if variant == "ascending":
        if not (_is_flat(top_slope) and _is_rising(bot_slope)):
            return None
        pattern_type = "ascending_triangle"
        bias = "bullish"
    elif variant == "descending":
        if not (_is_falling(top_slope) and _is_flat(bot_slope)):
            return None
        pattern_type = "descending_triangle"
        bias = "bearish"
    else:  # symmetrical
        if not (_is_falling(top_slope) and _is_rising(bot_slope)):
            return None
        pattern_type = "symmetrical_triangle"
        bias = "neutral"

    # Reject if the two lines diverge (broadening, not triangle)
    if variant == "symmetrical" and (top_slope - bot_slope) >= 0:
        # top falling (negative) should be less than bottom rising (positive)
        # so top_slope - bot_slope must be < 0 for convergence
        return None

    # Geometric fit quality (lower rmse / avg_bar = better)
    top_fit = max(0.0, 1.0 - top_rmse / max(avg_bar, 1e-9))
    bot_fit = max(0.0, 1.0 - bot_rmse / max(avg_bar, 1e-9))

    # Minimum price range requirement — avoid flat-ish consolidations
    price_range = max(p["price"] for p in scan) - min(p["price"] for p in scan)
    if price_range < min_move * 1.0:
        return None

    # Build boundary lines using existing helper (keeps contract with frontend)
    top_pivots = sorted(highs[-4:] if len(highs) >= 4 else highs, key=lambda p: p["index"])
    bot_pivots = sorted(lows[-4:] if len(lows) >= 4 else lows, key=lambda p: p["index"])
    upper_line = build_trend_line(
        f"{pattern_type}-upper", "Resistance", "#ef4444", "solid",
        top_pivots[0], top_pivots[-1], candles,
    )
    lower_line = build_trend_line(
        f"{pattern_type}-lower", "Support", "#22c55e", "solid",
        bot_pivots[0], bot_pivots[-1], candles,
    )

    # Breakout state
    current_price = candles[-1]["close"]
    last_idx = candles[-1]["index"]
    top_projected = top_slope * last_idx + top_int
    bot_projected = bot_slope * last_idx + bot_int

    if current_price > top_projected + breakout_buffer:
        state = "breakout"
    elif current_price < bot_projected - breakout_buffer:
        state = "breakdown"
    else:
        state = "forming"

    volume_score = min(1.0, max(volume_ratio - 1.0, 0.0))
    confidence = _pattern_confidence(
        0.52,  # LMW strict base — geometry passed, raise over heuristic overlap
        top_fit * 0.18,
        bot_fit * 0.18,
        min(1.0, price_range / max(min_move * 3.0, 1e-9)) * 0.12,
        volume_score * 0.06,
    )

    shape_pivots = sorted(top_pivots + bot_pivots, key=lambda p: p["index"])

    trend_lines = [ln for ln in (upper_line, lower_line) if ln]
    for ln in trend_lines:
        ln["source"] = "pattern"

    return {
        "primaryPattern": {
            "type": pattern_type,
            "label": PATTERN_LABELS[pattern_type],
            "bias": bias,
            "state": state,
            "confidence": confidence,
            "description": f"LMW-style {pattern_type.replace('_', ' ')} via slope regression.",
        },
        "shapePivots": shape_pivots,
        "trendLines": trend_lines,
        "horizontalLevels": [
            _build_horizontal_level(
                f"{pattern_type}-apex", "Apex", "#64748b", "dotted",
                (top_projected + bot_projected) / 2.0,
                scan[0]["time"], candles[-1]["time"], candles=candles,
            ),
        ],
        "markers": [],
        "levels": {
            "support": smart_round_price(bot_projected),
            "resistance": smart_round_price(top_projected),
        },
    }


def detect_ascending_triangle_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _triangle_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                          variant="ascending")


def detect_descending_triangle_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _triangle_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                          variant="descending")


def detect_symmetrical_triangle_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _triangle_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                          variant="symmetrical")


# ---------------------------------------------------------------------------
# Channels (ascending, descending) — parallel slope regression
# ---------------------------------------------------------------------------

def _channel_core(
    pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
    *, direction,
):
    """LMW-style channel detection via least-squares slope regression.

    Unlike legacy ``_detect_channel`` which demands exactly 4 pivots in strict
    high-low-high-low order, this fits independent lines to the high-series
    and low-series pivots, tolerating missing swings the fractal detector
    failed to catch on noisy data.

    direction="ascending":   both slopes positive, near-parallel → bullish continuation
    direction="descending":  both slopes negative, near-parallel → bearish continuation
    """
    from chartslp.patterns import (
        PATTERN_LABELS,
        _avg_bar_length,
        _pattern_confidence,
    )
    from chartslp.precision import smart_round_price
    from chartslp.trendlines import build_trend_line

    if len(pivots) < 4 or not candles:
        return None

    # Widen the scan window vs triangle/wedge — channels routinely span 40-80
    # bars on intraday TFs and can have 6+ pivots the engine is allowed to see.
    scan = pivots[-10:]
    highs = [p for p in scan if p["kind"] == "high"]
    lows = [p for p in scan if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    top_slope, top_int, top_rmse = _fit_line(highs[-4:] if len(highs) >= 4 else highs)
    bot_slope, bot_int, bot_rmse = _fit_line(lows[-4:] if len(lows) >= 4 else lows)

    avg_bar = _avg_bar_length(candles, scan[0]["index"], scan[-1]["index"] + 1)
    if avg_bar <= 0:
        return None

    # flat_tol is a per-bar price-change cutoff. Previously mixed in
    # ``min_move * 0.05`` which can be huge on high-priced assets (XAU/USD
    # last_price ~ 4800 × 0.0021 ≈ 10 → flat_tol ≈ 0.5) and wrongly flagged
    # genuinely sloped lines as "flat". Use only the per-bar geometric
    # range, same style as pytrendline_ref ``avg_candle_range * 0.05``.
    flat_tol = max(avg_bar * 0.05, 1e-9)

    if direction == "ascending":
        if not (top_slope > flat_tol and bot_slope > flat_tol):
            return None
        pattern_type = "ascending_channel"
        bias = "bullish"
    else:  # descending
        if not (top_slope < -flat_tol and bot_slope < -flat_tol):
            return None
        pattern_type = "descending_channel"
        bias = "bearish"

    # Near-parallel: slope ratio close to 1.0. Threshold 0.45 chosen by
    # reference-library benchmarking — pytrendline's "duplicate slope" /
    # "max error" pair spans a ~3.3× loose/strict range; a noisy real channel
    # (BTC 1m / XAU/USD 1m) fits 0.45-0.70 post-noise while true wedges
    # consistently fall below 0.30. Tightening again will miss fresh channels
    # with < 6 touches per side.
    sign_aligned = top_slope * bot_slope > 0
    if not sign_aligned:
        return None
    ratio = min(abs(top_slope), abs(bot_slope)) / max(abs(top_slope), abs(bot_slope))
    if ratio < 0.45:
        return None  # diverging or converging enough to be wedge/triangle

    # Channel width must stay roughly constant from start to end of the scan
    last_idx = candles[-1]["index"]
    x_start = scan[0]["index"]
    x_end = last_idx
    top_at_start = top_slope * x_start + top_int
    bot_at_start = bot_slope * x_start + bot_int
    top_at_end = top_slope * x_end + top_int
    bot_at_end = bot_slope * x_end + bot_int
    width_start = top_at_start - bot_at_start
    width_end = top_at_end - bot_at_end
    if width_start <= 0 or width_end <= 0:
        return None
    width_change = abs(width_end - width_start) / max(width_start, 1e-9)
    if width_change > 0.55:
        return None  # widened from 0.40 — noisy real channels drift more

    # Minimum amplitude — don't fire on micro-channels
    price_range = max(p["price"] for p in scan) - min(p["price"] for p in scan)
    if price_range < min_move * 1.0:
        return None

    top_fit = max(0.0, 1.0 - top_rmse / max(avg_bar, 1e-9))
    bot_fit = max(0.0, 1.0 - bot_rmse / max(avg_bar, 1e-9))

    top_pivots = sorted(highs[-4:] if len(highs) >= 4 else highs, key=lambda p: p["index"])
    bot_pivots = sorted(lows[-4:] if len(lows) >= 4 else lows, key=lambda p: p["index"])
    upper = build_trend_line(
        f"{pattern_type}-upper", "Upper", "#ef4444", "solid",
        top_pivots[0], top_pivots[-1], candles,
    )
    lower = build_trend_line(
        f"{pattern_type}-lower", "Lower", "#22c55e", "solid",
        bot_pivots[0], bot_pivots[-1], candles,
    )

    current_price = candles[-1]["close"]
    top_projected = top_at_end
    bot_projected = bot_at_end

    if current_price > top_projected + breakout_buffer:
        state = "breakout"
    elif current_price < bot_projected - breakout_buffer:
        state = "breakdown"
    else:
        state = "forming"

    volume_score = min(1.0, max(volume_ratio - 1.0, 0.0))
    confidence = _pattern_confidence(
        0.52,
        top_fit * 0.18,
        bot_fit * 0.18,
        ratio * 0.10,
        (1.0 - width_change) * 0.08,
        volume_score * 0.04,
    )

    shape_pivots = sorted(top_pivots + bot_pivots, key=lambda p: p["index"])

    # Align shape tail with the newest pivot in scan so recency-based sorts
    # treat this channel on par with triangle/wedge detectors (which naturally
    # include the freshest pivot). Without this, highs[-4:]/lows[-4:] can chop
    # off the last swing and leave lastIdx 3-5 bars behind competing detectors.
    newest_in_scan = max(scan, key=lambda p: p["index"])
    if shape_pivots[-1]["index"] < newest_in_scan["index"]:
        shape_pivots = sorted(shape_pivots + [newest_in_scan], key=lambda p: p["index"])

    trend_lines = [ln for ln in (upper, lower) if ln]
    for ln in trend_lines:
        ln["source"] = "pattern"

    return {
        "primaryPattern": {
            "type": pattern_type,
            "label": PATTERN_LABELS[pattern_type],
            "bias": bias,
            "state": state,
            "confidence": confidence,
            "description": f"LMW-style {pattern_type.replace('_', ' ')} via parallel slope regression.",
        },
        "shapePivots": shape_pivots,
        "trendLines": trend_lines,
        "horizontalLevels": [],
        "markers": [],
        "levels": {
            "support": smart_round_price(bot_projected),
            "resistance": smart_round_price(top_projected),
        },
    }


def detect_ascending_channel_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _channel_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                         direction="ascending")


def detect_descending_channel_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _channel_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                         direction="descending")


# ---------------------------------------------------------------------------
# Wedges (rising, falling) — both lines slope same direction, converge
# ---------------------------------------------------------------------------

def _wedge_core(
    pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
    *, direction,
):
    """LMW-strict wedge.

    - Rising wedge (bearish):  top rising, bottom rising, top slope < bottom slope
      (converging as they climb). Lines converge upward.
    - Falling wedge (bullish): top falling, bottom falling, top slope < bottom slope
      (top falls faster). Converging downward.
    """
    from chartslp.patterns import (
        PATTERN_LABELS,
        _avg_bar_length,
        _pattern_confidence,
    )
    from chartslp.precision import smart_round_price
    from chartslp.trendlines import build_trend_line

    if len(pivots) < 4 or not candles:
        return None

    scan = pivots[-8:]
    highs = [p for p in scan if p["kind"] == "high"]
    lows = [p for p in scan if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    top_slope, top_int, top_rmse = _fit_line(highs[-4:] if len(highs) >= 4 else highs)
    bot_slope, bot_int, bot_rmse = _fit_line(lows[-4:] if len(lows) >= 4 else lows)

    avg_bar = _avg_bar_length(candles, scan[0]["index"], scan[-1]["index"] + 1)
    if avg_bar <= 0:
        return None
    # Per-bar flat threshold only (see channel_core note on min_move inflation).
    flat_tol = max(avg_bar * 0.05, 1e-9)

    # Meaningful convergence: the two slopes must differ substantially.
    # Plan appendix A suggests slope_ratio ∈ (0.3, 0.9); widened lower bound
    # to 0.15 based on observed data — noisy fits on aggressive wedges can
    # produce ratios below 0.3, while parallel channels consistently fall
    # above 0.92. Upper bound 0.90 keeps channel-vs-wedge separation clean.
    #
    # For rising wedge: both slopes positive, top < bot; ratio = top/bot.
    # For falling wedge: both slopes negative, top < bot (top more negative);
    #                    ratio = bot/top so ratio stays in (0, 1).
    SLOPE_RATIO_MIN = 0.15
    SLOPE_RATIO_MAX = 0.90

    if direction == "rising":
        if not (top_slope > flat_tol and bot_slope > flat_tol):
            return None
        if top_slope >= bot_slope:
            return None  # parallel / diverging — not a wedge
        ratio = top_slope / bot_slope
        if not (SLOPE_RATIO_MIN < ratio < SLOPE_RATIO_MAX):
            return None  # too parallel (channel) or too conical
        pattern_type = "rising_wedge"
        bias = "bearish"
    else:
        if not (top_slope < -flat_tol and bot_slope < -flat_tol):
            return None
        if top_slope >= bot_slope:
            return None  # top not falling faster → not a wedge
        ratio = bot_slope / top_slope
        if not (SLOPE_RATIO_MIN < ratio < SLOPE_RATIO_MAX):
            return None
        pattern_type = "falling_wedge"
        bias = "bullish"

    top_fit = max(0.0, 1.0 - top_rmse / max(avg_bar, 1e-9))
    bot_fit = max(0.0, 1.0 - bot_rmse / max(avg_bar, 1e-9))

    # Minimum vertical amplitude
    price_range = max(p["price"] for p in scan) - min(p["price"] for p in scan)
    if price_range < min_move * 1.0:
        return None

    top_pivots = sorted(highs[-4:] if len(highs) >= 4 else highs, key=lambda p: p["index"])
    bot_pivots = sorted(lows[-4:] if len(lows) >= 4 else lows, key=lambda p: p["index"])
    upper = build_trend_line(
        f"{pattern_type}-upper", "Upper", "#ef4444", "solid",
        top_pivots[0], top_pivots[-1], candles,
    )
    lower = build_trend_line(
        f"{pattern_type}-lower", "Lower", "#22c55e", "solid",
        bot_pivots[0], bot_pivots[-1], candles,
    )

    current_price = candles[-1]["close"]
    last_idx = candles[-1]["index"]
    top_projected = top_slope * last_idx + top_int
    bot_projected = bot_slope * last_idx + bot_int

    if current_price > top_projected + breakout_buffer:
        state = "breakout"
    elif current_price < bot_projected - breakout_buffer:
        state = "breakdown"
    else:
        state = "forming"

    confidence = _pattern_confidence(
        0.52,  # LMW strict base — two converging slopes same direction is rare
        top_fit * 0.18,
        bot_fit * 0.18,
        min(1.0, price_range / max(min_move * 3.0, 1e-9)) * 0.10,
    )

    shape_pivots = sorted(top_pivots + bot_pivots, key=lambda p: p["index"])
    trend_lines = [ln for ln in (upper, lower) if ln]
    for ln in trend_lines:
        ln["source"] = "pattern"

    return {
        "primaryPattern": {
            "type": pattern_type,
            "label": PATTERN_LABELS[pattern_type],
            "bias": bias,
            "state": state,
            "confidence": confidence,
            "description": f"LMW-style {pattern_type.replace('_', ' ')} — converging slopes.",
        },
        "shapePivots": shape_pivots,
        "trendLines": trend_lines,
        "horizontalLevels": [],
        "markers": [],
        "levels": {
            "support": smart_round_price(bot_projected),
            "resistance": smart_round_price(top_projected),
        },
    }


def detect_rising_wedge_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _wedge_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                       direction="rising")


def detect_falling_wedge_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _wedge_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                       direction="falling")


# ---------------------------------------------------------------------------
# Flags — pole + counter-trend consolidation
# ---------------------------------------------------------------------------

def _flag_core(
    pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
    *, direction,
):
    """LMW-style flag: a directional pole followed by a weak counter-trend drift.

    direction = "bull":  up pole, slight down-drift flag   → bullish continuation
    direction = "bear":  down pole, slight up-drift flag   → bearish continuation

    Geometric definition:
      - Pole: first third of the scanned window. Close-price slope must be
        strong-directional (|slope| ≥ pole_min_slope).
      - Flag: remaining two thirds. Close-price slope must be counter-trend
        (opposite sign) AND weaker than ~40% of the pole slope.
      - Price must still be near the flag (bounded within flag range ± breakout_buffer).
    """
    from chartslp.patterns import PATTERN_LABELS, _avg_bar_length, _pattern_confidence
    from chartslp.precision import smart_round_price

    if len(candles) < 25:
        return None

    # Scan the last ~50 bars (or all if shorter)
    scan_len = min(len(candles), 50)
    scan = candles[-scan_len:]
    pole_split = scan_len // 3
    if pole_split < 5 or scan_len - pole_split < 8:
        return None

    pole_candles = scan[:pole_split]
    flag_candles = scan[pole_split:]

    # Linear regression on close prices for each segment
    def _slope_of(bars: list[dict]) -> float:
        n = len(bars)
        if n < 2:
            return 0.0
        xs = [b["index"] for b in bars]
        ys = [b["close"] for b in bars]
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        return num / den if den else 0.0

    pole_slope = _slope_of(pole_candles)
    flag_slope = _slope_of(flag_candles)

    avg_bar = _avg_bar_length(candles, scan[0]["index"], scan[-1]["index"] + 1)
    if avg_bar <= 0:
        return None

    # Pole must be a strong directional move; plan appendix A suggests
    # |slope| > 2 × ATR/bar. ``min_move`` is swing-profile ATR-ish, so:
    pole_min_slope = max(min_move * 0.25, avg_bar * 0.15)

    if direction == "bull":
        if pole_slope < pole_min_slope:
            return None
        # Flag is counter-trend (downward) but weaker
        if flag_slope > 0:
            return None
        if abs(flag_slope) > abs(pole_slope) * 0.4:
            return None
        pattern_type = "bull_flag"
        bias = "bullish"
    else:
        if pole_slope > -pole_min_slope:
            return None
        if flag_slope < 0:
            return None
        if abs(flag_slope) > abs(pole_slope) * 0.4:
            return None
        pattern_type = "bear_flag"
        bias = "bearish"

    # Pole magnitude (vertical extent)
    pole_magnitude = abs(pole_candles[-1]["close"] - pole_candles[0]["close"])
    if pole_magnitude < min_move * 2.0:
        return None

    # Flag consolidation range
    flag_high = max(c["high"] for c in flag_candles)
    flag_low = min(c["low"] for c in flag_candles)
    flag_range = flag_high - flag_low
    # Flag should be tight relative to the pole — under 60% pole magnitude
    if flag_range > pole_magnitude * 0.6:
        return None

    # State
    current = candles[-1]["close"]
    if direction == "bull":
        if current > flag_high + breakout_buffer:
            state = "breakout"
        elif current > flag_high - breakout_buffer * 0.5:
            state = "retest"
        else:
            state = "forming"
    else:
        if current < flag_low - breakout_buffer:
            state = "breakdown"
        elif current < flag_low + breakout_buffer * 0.5:
            state = "retest"
        else:
            state = "forming"

    # Confidence
    pole_score = min(1.0, pole_magnitude / max(min_move * 4.0, 1e-9))
    tightness_score = max(0.0, 1.0 - flag_range / max(pole_magnitude, 1e-9))
    volume_score = min(1.0, max(volume_ratio - 1.0, 0.0))
    confidence = _pattern_confidence(
        0.55,  # bumped from 0.50: flag geometry is information-dense (pole + counter)
        pole_score * 0.22,
        tightness_score * 0.15,
        volume_score * 0.08,
    )

    # Shape pivots: pole start/end + flag endpoints
    shape_pivots = [
        {"index": pole_candles[0]["index"], "time": pole_candles[0]["time"],
         "price": pole_candles[0]["close"], "kind": "low" if direction == "bull" else "high"},
        {"index": pole_candles[-1]["index"], "time": pole_candles[-1]["time"],
         "price": pole_candles[-1]["close"], "kind": "high" if direction == "bull" else "low"},
        {"index": flag_candles[-1]["index"], "time": flag_candles[-1]["time"],
         "price": flag_candles[-1]["close"], "kind": "high" if direction == "bull" else "low"},
    ]

    return {
        "primaryPattern": {
            "type": pattern_type,
            "label": PATTERN_LABELS[pattern_type],
            "bias": bias,
            "state": state,
            "confidence": confidence,
            "description": f"LMW-style {pattern_type.replace('_', ' ')}: strong pole + weak counter-trend flag.",
        },
        "shapePivots": shape_pivots,
        "trendLines": [],
        "horizontalLevels": [],
        "markers": [],
        "levels": {
            "support": smart_round_price(flag_low),
            "resistance": smart_round_price(flag_high),
        },
    }


def detect_bull_flag_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _flag_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                      direction="bull")


def detect_bear_flag_lmw(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    return _flag_core(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio,
                      direction="bear")
