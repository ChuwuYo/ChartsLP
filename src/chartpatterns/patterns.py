from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from chartpatterns.precision import smart_round_price
from chartpatterns.trendlines import (
    build_auto_trendlines as _trendline_build_auto_trendlines,
)
from chartpatterns.trendlines import (
    build_pattern_boundary_lines as _trendline_build_pattern_boundary_lines,
)
from chartpatterns.trendlines import (
    build_trend_line as _trendline_build_trend_line,
)
from chartpatterns.trendlines import (
    count_valid_touches as _trendline_count_valid_touches,
)
from chartpatterns.trendlines import (
    project_price as _trendline_project_price,
)

PATTERN_LABELS = {
    "ascending_triangle": "Ascending Triangle",
    "descending_triangle": "Descending Triangle",
    "double_top": "Double Top",
    "double_bottom": "Double Bottom",
    "rising_wedge": "Rising Wedge",
    "falling_wedge": "Falling Wedge",
    "head_and_shoulders": "Head and Shoulders",
    "inverse_head_and_shoulders": "Inverse Head and Shoulders",
    "ascending_channel": "Ascending Channel",
    "descending_channel": "Descending Channel",
    "triple_top": "Triple Top",
    "triple_bottom": "Triple Bottom",
    "bull_flag": "Bull Flag",
    "bear_flag": "Bear Flag",
    "bull_pennant": "Bull Pennant",
    "bear_pennant": "Bear Pennant",
    "rectangle_range": "Rectangle Range",
    "cup_and_handle": "Cup and Handle",
    "broadening_formation": "Broadening Formation",
    "rounding_bottom": "Rounding Bottom",
    "rounding_top": "Rounding Top",
    "symmetrical_triangle": "Symmetrical Triangle",
    "v_reversal_bottom": "V-Reversal Bottom",
    "v_reversal_top": "V-Reversal Top",
    "bullish_vcp": "Bullish VCP",
    "bearish_vcp": "Bearish VCP",
    "bullish_trend_structure": "Bullish Trend (HH-HL)",
    "bearish_trend_structure": "Bearish Trend (LH-LL)",
    "no_clear_pattern": "No Clear Pattern",
}

TRENDSPIDER_PATTERN_COLORS_DARK = {
    "bearish": "#FFA813",
    "bullish": "#5DCC61",
    "neutral": "#56AEE0",
}

TRENDSPIDER_PATTERN_CLASS = {
    "ascending_triangle": "bearish",
    "descending_triangle": "bullish",
    "symmetrical_triangle": "neutral",
    "ascending_channel": "bearish",
    "descending_channel": "bullish",
    "rectangle_range": "neutral",
    "broadening_formation": "neutral",
    "rising_wedge": "bearish",
    "falling_wedge": "bullish",
    "double_top": "bearish",
    "double_bottom": "bullish",
    "triple_top": "bearish",
    "triple_bottom": "bullish",
    "head_and_shoulders": "bearish",
    "inverse_head_and_shoulders": "bullish",
    "bull_flag": "bullish",
    "bear_flag": "bearish",
    "bull_pennant": "bullish",
    "bear_pennant": "bearish",
    "cup_and_handle": "bullish",
    "rounding_bottom": "bullish",
    "rounding_top": "bearish",
    "v_reversal_bottom": "bullish",
    "v_reversal_top": "bearish",
    "bullish_vcp": "bullish",
    "bearish_vcp": "bearish",
    "bullish_trend_structure": "bullish",
    "bearish_trend_structure": "bearish",
}


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_time(time_value):
    if isinstance(time_value, (int, float)):
        time_value = int(time_value)
        return time_value // 1000 if time_value > 10**11 else time_value

    if isinstance(time_value, str) and time_value.strip():
        try:
            parsed = time_value.replace("Z", "+00:00")
            from datetime import datetime
            return int(datetime.fromisoformat(parsed).timestamp())
        except Exception:
            try:
                from datetime import datetime
                return int(datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                return 0
    return 0


def normalize_klines(klines):
    """Normalize Binance-style list candles and dict candles into one shape."""
    normalized = []

    for idx, item in enumerate(klines or []):
        if isinstance(item, dict):
            time_value = item.get("time") or item.get("timestamp") or item.get("datetime")
            open_price = _to_float(item.get("open", item.get("o")))
            high_price = _to_float(item.get("high", item.get("h")))
            low_price = _to_float(item.get("low", item.get("l")))
            close_price = _to_float(item.get("close", item.get("c")))
            volume = _to_float(item.get("volume", item.get("v")))
        elif isinstance(item, (list, tuple)) and len(item) >= 5:
            time_value = item[0]
            open_price = _to_float(item[1])
            high_price = _to_float(item[2])
            low_price = _to_float(item[3])
            close_price = _to_float(item[4])
            volume = _to_float(item[5] if len(item) > 5 else 0)
        else:
            continue

        if not all(math.isfinite(value) for value in [open_price, high_price, low_price, close_price]):
            continue

        if high_price < low_price:
            high_price, low_price = low_price, high_price

        timestamp = _normalize_time(time_value)
        if timestamp <= 0:
            continue

        normalized.append({
            "index": len(normalized),
            "time": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        })

    normalized.sort(key=lambda candle: candle["time"])
    for idx, candle in enumerate(normalized):
        candle["index"] = idx
    return normalized


def _calculate_atr(candles, period=14):
    if len(candles) < 2:
        return 0.0

    true_ranges = []
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


def _calculate_volume_ratio(candles, lookback=10):
    if len(candles) < 2:
        return 1.0

    recent = [candle["volume"] for candle in candles[-(lookback + 1):-1] if candle["volume"] > 0]
    current = max(candles[-1]["volume"], 0)
    if not recent or current <= 0:
        return 1.0
    baseline = sum(recent) / len(recent)
    return current / baseline if baseline > 0 else 1.0


def _avg_bar_length(candles, start_idx=0, end_idx=None):
    """Median candle range (high - low) over the given slice — localised volatility."""
    sl = candles[start_idx:end_idx]
    ranges = [c["high"] - c["low"] for c in sl if c["high"] > c["low"]]
    if not ranges:
        return 0.0
    ranges.sort()
    return ranges[len(ranges) // 2]


def _volume_at_index(candles, target_index):
    """Return volume of the candle at *target_index* (candle['index'] == target_index)."""
    if 0 <= target_index < len(candles) and candles[target_index]["index"] == target_index:
        return candles[target_index].get("volume", 0.0)
    for c in candles:
        if c["index"] == target_index:
            return c.get("volume", 0.0)
    return 0.0


def _is_level_breached(candles, level_price, after_index, direction="above"):
    """True if any close after *after_index* crossed *level_price* on a closing basis."""
    start = after_index + 1
    if start < 0:
        start = 0
    for c in candles[start:]:
        if c["index"] <= after_index:
            continue
        if direction == "above" and c["close"] > level_price:
            return True
        if direction == "below" and c["close"] < level_price:
            return True
    return False


def _project_price(point_a, point_b, target_index):
    return _trendline_project_price(point_a, point_b, target_index)



def _count_valid_touches(point_a, point_b, candles, is_upper, tolerance_factor=0.5):
    return _trendline_count_valid_touches(point_a, point_b, candles, is_upper, tolerance_factor)


def _build_trend_line(line_id, label, color, style, point_a, point_b, candles, partner_line=None):
    """Delegate to trendline_engine.build_trend_line."""
    return _trendline_build_trend_line(line_id, label, color, style, point_a, point_b, candles, partner_line)


def _build_pattern_boundary_lines(shape_pivots, candles, atr):
    """Delegate to trendline_engine.build_pattern_boundary_lines."""
    return _trendline_build_pattern_boundary_lines(shape_pivots, candles, atr)


def _count_level_touches(candles, price, touch_zone):
    """Count how many times price bounced off a horizontal level.

    Uses the same bounce-validation logic as ``_compute_clustered_levels``:
    the candle must be *near* the level AND show a directional reversal
    relative to the previous candle.
    """
    touches = 0
    for i in range(1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        near = abs(c["low"] - price) < touch_zone or abs(c["high"] - price) < touch_zone
        if not near:
            continue
        bounced_up = c["close"] > price and prev["close"] <= price + touch_zone
        bounced_down = c["close"] < price and prev["close"] >= price - touch_zone
        if bounced_up or bounced_down:
            touches += 1
    return touches


def _build_horizontal_level(level_id, label, color, style, price, start_time, end_time, candles=None):
    resolved_end_time = end_time
    line_type = _classify_shape_line_type(level_id, label)
    touches = 0

    touchable_types = {"support_line", "resistance_line", "neckline"}
    if candles and line_type in touchable_types:
        start_bound = min(start_time, end_time)
        end_bound = max(start_time, end_time)
        in_range = [c for c in candles if start_bound <= c["time"] <= end_bound]
        if in_range:
            atr = _calculate_atr(candles)
            touch_zone = max(atr * 0.3, 1e-9)

            # Determine breach direction for each line type.
            # Neckline in bullish patterns (ihs/double-bottom/triple-bottom/
            # round-bottom) sits ABOVE the lows → acts as resistance.
            # Neckline in bearish patterns (hs/double-top/triple-top/
            # round-top) sits BELOW the highs → acts as support.
            id_lower = (level_id or "").lower()
            if line_type == "neckline":
                if "ihs" in id_lower or "bottom" in id_lower:
                    breach_side = "resistance_line"
                else:
                    breach_side = "support_line"
            else:
                breach_side = line_type

            # First: find where the level gets breached to determine the
            # valid time window.  Only count touches within that window.
            consecutive_breach = 0
            breach_threshold = 1
            for candle in in_range[1:]:
                body_top = max(candle["open"], candle["close"])
                body_bot = min(candle["open"], candle["close"])
                if breach_side == "resistance_line":
                    breached = body_top > price
                else:
                    breached = body_bot < price

                if breached:
                    consecutive_breach += 1
                else:
                    consecutive_breach = 0

                if consecutive_breach >= breach_threshold:
                    resolved_end_time = in_range[max(0, in_range.index(candle) - 1)]["time"]
                    break

            # Count touches only up to the resolved end time so that
            # bounces after the line was invalidated are excluded.
            valid_candles = [c for c in in_range if c["time"] <= resolved_end_time]
            touches = _count_level_touches(valid_candles, price, touch_zone)

    level = {
        "id": level_id,
        "label": label,
        "color": color,
        "style": style,
        "price": smart_round_price(price),
        "points": [
            {"time": start_time, "price": smart_round_price(price)},
            {"time": resolved_end_time, "price": smart_round_price(price)},
        ],
    }
    if touches > 0:
        level["touches"] = touches
    return level


def _shape_outline_segments(points):
    segments = []
    for idx in range(1, len(points)):
        prev_point = points[idx - 1]
        current_point = points[idx]
        segments.append([
            {"time": prev_point["time"], "price": prev_point["price"]},
            {"time": current_point["time"], "price": current_point["price"]},
        ])
    return segments


def _classify_shape_line_type(line_id, label):
    text = f"{line_id or ''} {label or ''}".lower()
    if "neckline" in text:
        return "neckline"
    if "support" in text or "lower" in text:
        return "support_line"
    if "resistance" in text or "upper" in text:
        return "resistance_line"
    return "trendline"


def _find_shape_point(points, preferred_roles, fallback_idx=None):
    if not points:
        return None
    roles = {str(role).lower() for role in (preferred_roles or [])}
    for point in points:
        if str(point.get("role", "")).lower() in roles:
            return point
    if isinstance(fallback_idx, int) and 0 <= fallback_idx < len(points):
        return points[fallback_idx]
    return None


def _find_neighbor_pivot(pivots, anchor_index, kind, direction, max_bars=40):
    if not pivots:
        return None
    candidates = []
    for pivot in pivots:
        if pivot["kind"] != kind:
            continue
        delta = pivot["index"] - anchor_index
        if direction < 0 and delta >= 0:
            continue
        if direction > 0 and delta <= 0:
            continue
        if abs(delta) > max_bars:
            continue
        candidates.append(pivot)

    if not candidates:
        return None

    # Prefer nearest pivot in the chosen direction.
    candidates.sort(key=lambda p: abs(p["index"] - anchor_index))
    return candidates[0]


def _build_precise_double_lines(primary_type, points):
    if not points or len(points) < 3:
        return []

    if primary_type == "double_top":
        left = _find_shape_point(points, ["left_top", "high_1"], 0)
        mid = _find_shape_point(points, ["neckline", "mid_low", "low_2"], 1)
        right = _find_shape_point(points, ["right_top", "high_3"], 2)
        left_base = _find_shape_point(points, ["left_base"])
        confirm = _find_shape_point(points, ["breakdown_confirm", "right_break_confirm"])
        if not left or not mid or not right:
            return []

        top_price = (left["price"] + right["price"]) / 2.0
        zigzag_points = []
        if left_base:
            zigzag_points.append({"time": left_base["time"], "price": left_base["price"]})
        zigzag_points.extend([
            {"time": left["time"], "price": left["price"]},
            {"time": mid["time"], "price": mid["price"]},
            {"time": right["time"], "price": right["price"]},
        ])
        if confirm:
            zigzag_points.append({"time": confirm["time"], "price": confirm["price"]})

        return [
            {
                "id": "double-top-zigzag",
                "lineType": "structure_zigzag",
                "style": "solid",
                "color": "#f59e0b",
                "lineWidth": 2,
                "points": zigzag_points,
            },
            {
                "id": "double-top-resistance",
                "lineType": "resistance_line",
                "style": "dashed",
                "color": "#ef4444",
                "lineWidth": 1,
                "points": [
                    {"time": left["time"], "price": smart_round_price(top_price)},
                    {"time": right["time"], "price": smart_round_price(top_price)},
                ],
            },
            {
                "id": "double-top-neckline",
                "lineType": "neckline",
                "style": "dashed",
                "color": "#38bdf8",
                "lineWidth": 1,
                "points": [
                    {"time": mid["time"], "price": mid["price"]},
                    {"time": right["time"], "price": mid["price"]},
                ],
            },
        ]

    if primary_type == "double_bottom":
        left = _find_shape_point(points, ["left_bottom", "low_1"], 0)
        mid = _find_shape_point(points, ["neckline", "mid_high", "high_2"], 1)
        right = _find_shape_point(points, ["right_bottom", "low_3"], 2)
        left_base = _find_shape_point(points, ["left_base"])
        confirm = _find_shape_point(points, ["breakout_confirm", "right_break_confirm"])
        if not left or not mid or not right:
            return []

        bottom_price = (left["price"] + right["price"]) / 2.0
        zigzag_points = []
        if left_base:
            zigzag_points.append({"time": left_base["time"], "price": left_base["price"]})
        zigzag_points.extend([
            {"time": left["time"], "price": left["price"]},
            {"time": mid["time"], "price": mid["price"]},
            {"time": right["time"], "price": right["price"]},
        ])
        if confirm:
            zigzag_points.append({"time": confirm["time"], "price": confirm["price"]})

        return [
            {
                "id": "double-bottom-zigzag",
                "lineType": "structure_zigzag",
                "style": "solid",
                "color": "#f59e0b",
                "lineWidth": 2,
                "points": zigzag_points,
            },
            {
                "id": "double-bottom-support",
                "lineType": "support_line",
                "style": "dashed",
                "color": "#22c55e",
                "lineWidth": 1,
                "points": [
                    {"time": left["time"], "price": smart_round_price(bottom_price)},
                    {"time": right["time"], "price": smart_round_price(bottom_price)},
                ],
            },
            {
                "id": "double-bottom-neckline",
                "lineType": "neckline",
                "style": "dashed",
                "color": "#38bdf8",
                "lineWidth": 1,
                "points": [
                    {"time": mid["time"], "price": mid["price"]},
                    {"time": right["time"], "price": mid["price"]},
                ],
            },
        ]

    return []


def _build_shape_lines(primary_type, points, pattern_result):
    lines = []

    precise_double_lines = _build_precise_double_lines(primary_type, points)
    if precise_double_lines:
        lines.extend(precise_double_lines)
    elif points and len(points) >= 3:
        lines.append({
            "id": f"{primary_type}-structure-zigzag",
            "lineType": "structure_zigzag",
            "style": "solid",
            "color": "#f59e0b",
            "lineWidth": 2,
            "points": [{"time": p["time"], "price": p["price"]} for p in points],
        })

    # Only include lines that belong to the pattern shape itself.
    # Auto trendlines and cluster S/R levels are rendered separately and
    # must not be duplicated into shape.lines (would cause color mis-theming).
    for trend in (pattern_result or {}).get("trendLines", []):
        trend_points = trend.get("points") or []
        if len(trend_points) < 2:
            continue
        if trend.get("source") == "auto":
            continue
        tid = str(trend.get("id") or "")
        if tid.startswith("auto-"):
            continue
        lines.append({
            "id": trend.get("id", f"{primary_type}-trendline"),
            "lineType": _classify_shape_line_type(trend.get("id"), trend.get("label")),
            "style": trend.get("style", "solid"),
            "color": trend.get("color", "#4ade80"),
            "source": trend.get("source", "pattern"),
            "lineWidth": 2,
            "points": [{"time": p["time"], "price": p["price"]} for p in trend_points],
        })

    for level in (pattern_result or {}).get("horizontalLevels", []):
        level_points = level.get("points") or []
        if len(level_points) < 2:
            continue
        lid = str(level.get("id") or "")
        if lid.startswith("cluster-"):
            continue
        line_type = _classify_shape_line_type(level.get("id"), level.get("label"))
        lines.append({
            "id": level.get("id", f"{primary_type}-level"),
            "lineType": line_type,
            "style": level.get("style", "dashed"),
            "color": level.get("color", "#ef4444" if line_type == "resistance_line" else "#3b82f6"),
            "source": "pattern",
            "lineWidth": 1,
            "points": [{"time": p["time"], "price": p["price"]} for p in level_points],
        })

    return lines


def _find_recent_pattern_pivots(primary_type, pivots, tolerance, min_move):
    if not pivots:
        return []

    def _find_by_kinds(kinds):
        if not kinds or len(pivots) < len(kinds):
            return []
        for start in range(len(pivots) - len(kinds), -1, -1):
            sequence = pivots[start:start + len(kinds)]
            if [point["kind"] for point in sequence] == kinds:
                return sequence
        return []

    if primary_type == "double_top":
        candidates = []
        min_span_bars = 8
        for offset in range(len(pivots) - 2, 1, -1):
            seq = pivots[offset - 2:offset + 1]
            if [point["kind"] for point in seq] != ["high", "low", "high"]:
                continue
            high_a, low_mid, high_b = seq
            top_gap = abs(high_a["price"] - high_b["price"])
            valley_depth = min(high_a["price"], high_b["price"]) - low_mid["price"]
            span_bars = high_b["index"] - high_a["index"]
            if top_gap > tolerance or valley_depth < min_move:
                continue
            if span_bars < min_span_bars:
                continue
            score = (
                (valley_depth / max(min_move, 1e-9))
                + (span_bars / 10.0)
                - (top_gap / max(tolerance, 1e-9))
            )
            candidates.append((score, high_b["index"], seq))

        if not candidates:
            # Fallback without span constraint if market is very compressed.
            for offset in range(len(pivots) - 2, 1, -1):
                seq = pivots[offset - 2:offset + 1]
                if [point["kind"] for point in seq] != ["high", "low", "high"]:
                    continue
                high_a, low_mid, high_b = seq
                top_gap = abs(high_a["price"] - high_b["price"])
                valley_depth = min(high_a["price"], high_b["price"]) - low_mid["price"]
                if top_gap > tolerance or valley_depth < min_move:
                    continue
                span_bars = high_b["index"] - high_a["index"]
                score = (
                    (valley_depth / max(min_move, 1e-9))
                    + (span_bars / 10.0)
                    - (top_gap / max(tolerance, 1e-9))
                )
                candidates.append((score, high_b["index"], seq))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]
        return []

    if primary_type == "double_bottom":
        candidates = []
        min_span_bars = 8
        for offset in range(len(pivots) - 2, 1, -1):
            seq = pivots[offset - 2:offset + 1]
            if [point["kind"] for point in seq] != ["low", "high", "low"]:
                continue
            low_a, high_mid, low_b = seq
            bottom_gap = abs(low_a["price"] - low_b["price"])
            bounce_height = high_mid["price"] - max(low_a["price"], low_b["price"])
            span_bars = low_b["index"] - low_a["index"]
            if bottom_gap > tolerance or bounce_height < min_move:
                continue
            if span_bars < min_span_bars:
                continue
            score = (
                (bounce_height / max(min_move, 1e-9))
                + (span_bars / 10.0)
                - (bottom_gap / max(tolerance, 1e-9))
            )
            candidates.append((score, low_b["index"], seq))

        if not candidates:
            for offset in range(len(pivots) - 2, 1, -1):
                seq = pivots[offset - 2:offset + 1]
                if [point["kind"] for point in seq] != ["low", "high", "low"]:
                    continue
                low_a, high_mid, low_b = seq
                bottom_gap = abs(low_a["price"] - low_b["price"])
                bounce_height = high_mid["price"] - max(low_a["price"], low_b["price"])
                if bottom_gap > tolerance or bounce_height < min_move:
                    continue
                span_bars = low_b["index"] - low_a["index"]
                score = (
                    (bounce_height / max(min_move, 1e-9))
                    + (span_bars / 10.0)
                    - (bottom_gap / max(tolerance, 1e-9))
                )
                candidates.append((score, low_b["index"], seq))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]
        return []

    kind_pattern_map = {
        "triple_top": ["high", "low", "high", "low", "high"],
        "triple_bottom": ["low", "high", "low", "high", "low"],
        "head_and_shoulders": ["high", "low", "high", "low", "high"],
        "inverse_head_and_shoulders": ["low", "high", "low", "high", "low"],
        "cup_and_handle": ["high", "low", "high", "low"],
        "ascending_triangle": ["high", "low", "high", "low"],
        "descending_triangle": ["low", "high", "low", "high"],
        "falling_wedge": ["high", "low", "high", "low"],
        "rising_wedge": ["low", "high", "low", "high"],
        "ascending_channel": ["high", "low", "high", "low"],
        "descending_channel": ["low", "high", "low", "high"],
        "rectangle_range": ["high", "low", "high", "low"],
        "broadening_formation": ["high", "low", "high", "low"],
        "bullish_vcp": ["high", "low", "high", "low"],
        "bearish_vcp": ["low", "high", "low", "high"],
    }
    sequence = _find_by_kinds(kind_pattern_map.get(primary_type))
    if sequence:
        return sequence

    fallback_count_map = {
        "bull_flag": 6,
        "bear_flag": 6,
        "bull_pennant": 6,
        "bear_pennant": 6,
        "rounding_bottom": 8,
        "rounding_top": 8,
    }
    fallback_count = fallback_count_map.get(primary_type, 4)
    out = []
    for point in reversed(pivots):
        if not out or point["kind"] != out[-1]["kind"]:
            out.append(point)
        if len(out) >= fallback_count:
            break
    return list(reversed(out))


def _build_shapes_from_pattern(primary, pivots, tolerance, min_move, pattern_result=None):
    primary_type = (primary or {}).get("type")
    if not primary_type or primary_type == "no_clear_pattern":
        return []

    selected = _find_recent_pattern_pivots(primary_type, pivots, tolerance, min_move)
    if not selected or len(selected) < 3:
        return []

    points = []
    for idx, pivot in enumerate(selected, start=1):
        role = f"{pivot['kind']}_{idx}"
        points.append({
            "time": pivot["time"],
            "price": smart_round_price(pivot["price"]),
            "index": pivot["index"],
            "kind": pivot["kind"],
            "role": role,
        })

    return [{
        "id": f"{primary_type}-shape-1",
        "type": primary_type,
        "label": (primary or {}).get("label", PATTERN_LABELS.get(primary_type, primary_type)),
        "confidence": (primary or {}).get("confidence", 0.0),
        "state": (primary or {}).get("state", "monitor"),
        "anchors": points,
        "points": points,
        "outlineSegments": _shape_outline_segments(points),
        "lines": _build_shape_lines(primary_type, points, pattern_result),
        "zones": [],
    }]


def _build_shapes_from_pattern_result(pattern_result, pivots, tolerance, min_move, candles=None):
    if not pattern_result:
        return []

    primary = pattern_result.get("primaryPattern") or {}
    primary_type = primary.get("type")
    if not primary_type or primary_type == "no_clear_pattern":
        return []

    explicit_pivots = pattern_result.get("shapePivots") or []
    if explicit_pivots and len(explicit_pivots) >= 3:
        points = []
        for idx, pivot in enumerate(explicit_pivots, start=1):
            points.append({
                "time": pivot["time"],
                "price": smart_round_price(pivot["price"]),
                "index": pivot.get("index", idx - 1),
                "kind": pivot.get("kind", "point"),
                "role": pivot.get("role") or f"{pivot.get('kind', 'point')}_{idx}",
            })

        return [{
            "id": f"{primary_type}-shape-1",
            "type": primary_type,
            "label": primary.get("label", PATTERN_LABELS.get(primary_type, primary_type)),
            "confidence": primary.get("confidence", 0.0),
            "state": primary.get("state", "monitor"),
            "anchors": points,
            "points": points,
            "outlineSegments": _shape_outline_segments(points),
            "lines": _build_shape_lines(primary_type, points, pattern_result),
            "zones": [],
        }]

    return []


def _shape_signature_from_points(points):
    if not points:
        return tuple()
    indexes = []
    for point in points:
        index = point.get("index")
        if isinstance(index, (int, float)) and math.isfinite(index):
            indexes.append(int(index))
    return tuple(sorted(indexes))


def _shape_pivots_from_sequence(sequence):
    out = []
    high_count = 0
    low_count = 0
    for point in sequence or []:
        kind = point.get("kind")
        if kind == "high":
            high_count += 1
            role = f"high_{high_count}"
        elif kind == "low":
            low_count += 1
            role = f"low_{low_count}"
        else:
            role = "pivot"
        out.append({
            "index": point.get("index"),
            "time": point.get("time"),
            "price": point.get("price"),
            "kind": kind,
            "role": role,
        })
    return out


def _build_double_shape_from_sequence(pattern_type, sequence, seq_idx, score_hint=0.0):
    left, mid, right = sequence
    if pattern_type == "double_top":
        left_role, mid_role, right_role = "left_top", "neckline", "right_top"
        left_kind, mid_kind, right_kind = "high", "low", "high"
    else:
        left_role, mid_role, right_role = "left_bottom", "neckline", "right_bottom"
        left_kind, mid_kind, right_kind = "low", "high", "low"

    points = [
        {
            "index": left["index"],
            "time": left["time"],
            "price": smart_round_price(left["price"]),
            "kind": left_kind,
            "role": left_role,
        },
        {
            "index": mid["index"],
            "time": mid["time"],
            "price": smart_round_price(mid["price"]),
            "kind": mid_kind,
            "role": mid_role,
        },
        {
            "index": right["index"],
            "time": right["time"],
            "price": smart_round_price(right["price"]),
            "kind": right_kind,
            "role": right_role,
        },
    ]

    confidence = 0.35 + min(max(score_hint, 0.0), 2.2) * 0.10
    confidence = round(min(confidence, 0.78), 2)
    return {
        "id": f"{pattern_type}-shape-extra-{seq_idx}",
        "type": pattern_type,
        "label": PATTERN_LABELS.get(pattern_type, pattern_type),
        "confidence": confidence,
        "state": "forming",
        "anchors": points,
        "points": points,
        "outlineSegments": _shape_outline_segments(points),
        "lines": _build_shape_lines(pattern_type, points, None),
        "zones": [],
    }


def _collect_additional_double_shapes(pivots, tolerance, min_move, max_per_type=4):
    if not pivots or len(pivots) < 3:
        return []

    def _scan_one(pattern_type):
        latest_index = pivots[-1]["index"] if pivots else 0
        recent_floor = max(0, latest_index - 180)
        expected = ["high", "low", "high"] if pattern_type == "double_top" else ["low", "high", "low"]
        raw_candidates = []
        for i in range(0, len(pivots) - 2):
            seq = pivots[i:i + 3]
            if [point["kind"] for point in seq] != expected:
                continue

            left, mid, right = seq
            if right["index"] < recent_floor:
                continue
            span_bars = right["index"] - left["index"]
            left_leg = mid["index"] - left["index"]
            right_leg = right["index"] - mid["index"]
            if span_bars < 8 or min(left_leg, right_leg) < 2:
                continue
            symmetry_ratio = min(left_leg, right_leg) / max(max(left_leg, right_leg), 1)
            if symmetry_ratio < 0.35:
                continue

            if pattern_type == "double_top":
                outer_gap = abs(left["price"] - right["price"])
                mid_move = min(left["price"], right["price"]) - mid["price"]
                move_pct = mid_move / max(mean([left["price"], right["price"]]), 1e-9)
            else:
                outer_gap = abs(left["price"] - right["price"])
                mid_move = mid["price"] - max(left["price"], right["price"])
                move_pct = mid_move / max(mean([left["price"], right["price"]]), 1e-9)

            if outer_gap > tolerance or mid_move < min_move or move_pct < 0.0018:
                continue

            score = (
                (mid_move / max(min_move, 1e-9))
                + symmetry_ratio * 1.5
                + (span_bars / 10.0)
                - (outer_gap / max(tolerance, 1e-9))
            )
            raw_candidates.append({
                "start": left["index"],
                "end": right["index"],
                "score": score,
                "sequence": seq,
            })

        raw_candidates.sort(key=lambda item: (item["end"], item["score"]), reverse=True)
        selected = []
        occupied = []
        for item in raw_candidates:
            overlap = any(not (item["end"] < r0 or item["start"] > r1) for (r0, r1) in occupied)
            if overlap:
                continue
            selected.append(item)
            occupied.append((item["start"], item["end"]))
            if len(selected) >= max_per_type:
                break

        selected.reverse()
        return selected

    shapes = []
    seq_idx = 1
    for pattern_type in ("double_top", "double_bottom"):
        for item in _scan_one(pattern_type):
            shapes.append(
                _build_double_shape_from_sequence(
                    pattern_type,
                    item["sequence"],
                    seq_idx,
                    score_hint=item["score"],
                )
            )
            seq_idx += 1
    return shapes


def _compress_pivots(pivots, min_move):
    if not pivots:
        return []

    pivots = sorted(pivots, key=lambda pivot: pivot["index"])
    compressed = []

    for pivot in pivots:
        if not compressed:
            compressed.append(pivot)
            continue

        previous = compressed[-1]
        if pivot["kind"] == previous["kind"]:
            should_replace = (
                pivot["price"] >= previous["price"]
                if pivot["kind"] == "high"
                else pivot["price"] <= previous["price"]
            )
            if should_replace:
                compressed[-1] = pivot
            continue

        if abs(pivot["price"] - previous["price"]) < min_move:
            continue

        compressed.append(pivot)

    return compressed


def _detect_pivots(candles, left=2, right=2, min_move=0.0, include_tentative=False):
    if len(candles) < left + right + 1:
        return []

    raw_pivots = []
    for idx in range(left, len(candles) - right):
        candle = candles[idx]
        window_left = candles[idx - left:idx]
        window_right = candles[idx + 1:idx + right + 1]

        is_high = all(candle["high"] > item["high"] for item in window_left) and all(
            candle["high"] >= item["high"] for item in window_right
        )
        is_low = all(candle["low"] < item["low"] for item in window_left) and all(
            candle["low"] <= item["low"] for item in window_right
        )

        if is_high and is_low:
            high_gap = candle["high"] - max(item["high"] for item in window_left + window_right)
            low_gap = min(item["low"] for item in window_left + window_right) - candle["low"]
            if high_gap >= low_gap:
                is_low = False
            else:
                is_high = False

        if is_high:
            raw_pivots.append({
                "index": idx,
                "time": candle["time"],
                "price": candle["high"],
                "kind": "high",
            })
        if is_low:
            raw_pivots.append({
                "index": idx,
                "time": candle["time"],
                "price": candle["low"],
                "kind": "low",
            })

    compressed = _compress_pivots(raw_pivots, min_move)

    if include_tentative and len(candles) >= left + 1:
        tail = candles[-left - 1:]
        last = tail[-1]
        window = tail[:-1]
        if all(last["high"] > c["high"] for c in window):
            compressed.append({
                "index": len(candles) - 1,
                "time": last["time"],
                "price": last["high"],
                "kind": "high",
                "tentative": True,
            })
        elif all(last["low"] < c["low"] for c in window):
            compressed.append({
                "index": len(candles) - 1,
                "time": last["time"],
                "price": last["low"],
                "kind": "low",
                "tentative": True,
            })

    return compressed


def _calculate_pattern_tolerances(candles, trade_type):
    last_price = candles[-1]["close"] if candles else 0
    atr = _calculate_atr(candles)
    base_pct = 0.0015 if trade_type == "scalp" else 0.003
    breakout_pct = 0.0009 if trade_type == "scalp" else 0.0018
    flat_tolerance = max(atr * 0.55, last_price * base_pct)
    move_threshold = max(atr * 0.75, last_price * (base_pct * 1.4))
    breakout_buffer = max(atr * 0.25, last_price * breakout_pct)
    return atr, flat_tolerance, move_threshold, breakout_buffer


def _pattern_confidence(base_score, *components):
    raw = base_score + sum(max(0.0, min(component, 1.0)) for component in components)
    score = 0.30 + (raw - 0.30) * 0.65
    return round(max(0.30, min(score, 0.92)), 2)


def _trendspider_pattern_color(pattern_type, bias="neutral"):
    pattern_class = TRENDSPIDER_PATTERN_CLASS.get(pattern_type)
    if not pattern_class:
        pattern_class = {
            "bullish": "bullish",
            "bearish": "bearish",
        }.get(str(bias).lower(), "neutral")
    return TRENDSPIDER_PATTERN_COLORS_DARK[pattern_class]


def _apply_trendspider_pattern_theme(pattern_type, bias, pattern_result):
    """Apply pattern theme color to pattern shapes and their boundary lines only.

    Scope:
    - pattern shapes + shape.lines (pattern's visual outline)
    - trendLines tagged with source="pattern" (pattern boundary lines)

    NOT touched (have their own color semantics):
    - auto trendlines (bright yellow — set at creation)
    - horizontalLevels (candle red/green by support/resistance)
    - markers (bullish green / bearish red at creation)
    """
    if not pattern_result:
        return pattern_result

    themed_color = _trendspider_pattern_color(pattern_type, bias)

    for line in pattern_result.get("trendLines", []) or []:
        if line.get("source") == "pattern":
            line["color"] = themed_color

    for shape in pattern_result.get("shapes", []) or []:
        shape["color"] = themed_color
        for line in shape.get("lines", []) or []:
            if line.get("source") != "auto":
                line["color"] = themed_color

    for cand in pattern_result.get("allCandidates", []) or []:
        cand_type = cand.get("type", pattern_type)
        cand_bias = cand.get("bias", bias)
        cand_color = _trendspider_pattern_color(cand_type, cand_bias)
        for line in cand.get("trendLines", []) or []:
            # Candidate detector output = pattern boundary lines (never auto).
            # Skip only if explicitly tagged as auto to be safe.
            if line.get("source") == "auto":
                continue
            line["color"] = cand_color
            line.setdefault("source", "pattern")
        for shape in cand.get("shapes", []) or []:
            shape["color"] = cand_color
            for line in shape.get("lines", []) or []:
                if line.get("source") != "auto":
                    line["color"] = cand_color

    return pattern_result


def _pattern_selection_score(pattern_result, latest_index):
    primary = (pattern_result or {}).get("primaryPattern") or {}
    pattern_type = str(primary.get("type") or "")
    if not pattern_type or pattern_type == "no_clear_pattern":
        return -1e9

    confidence = float(primary.get("confidence") or 0.0)
    state = str(primary.get("state") or "monitor").lower()
    state_bonus_map = {
        "breakout": 0.10,
        "breakdown": 0.10,
        "confirmed": 0.10,
        "retest": 0.05,
        "forming": 0.02,
        "monitor": 0.0,
    }
    state_bonus = state_bonus_map.get(state, 0.0)

    recent_index = None
    for pivot in (pattern_result or {}).get("shapePivots", []):
        pivot_index = pivot.get("index")
        if isinstance(pivot_index, (int, float)) and math.isfinite(pivot_index):
            pivot_index = int(pivot_index)
            if recent_index is None or pivot_index > recent_index:
                recent_index = pivot_index

    recency_bonus = 0.0
    if recent_index is not None and isinstance(latest_index, (int, float)):
        age = max(0, int(latest_index) - recent_index)
        freshness = max(0.0, 1.0 - (age / 180.0))
        recency_bonus = freshness * 0.12

    specificity_bonus_map = {
        "head_and_shoulders": 0.08,
        "inverse_head_and_shoulders": 0.08,
        "triple_top": 0.06,
        "triple_bottom": 0.06,
        "cup_and_handle": 0.05,
        "double_top": 0.03,
        "double_bottom": 0.03,
        "ascending_triangle": 0.02,
        "descending_triangle": 0.02,
        "symmetrical_triangle": 0.02,
        "bullish_vcp": 0.04,
        "bearish_vcp": 0.04,
        "ascending_channel": 0.01,
        "descending_channel": 0.01,
    }
    specificity_bonus = specificity_bonus_map.get(pattern_type, 0.0)

    generic_penalty_types = (
        "falling_wedge", "rising_wedge",
        "bullish_trend_structure", "bearish_trend_structure",
    )
    generic_penalty = -0.04 if pattern_type in generic_penalty_types else 0.0

    return confidence + state_bonus + recency_bonus + specificity_bonus + generic_penalty


def _pattern_specificity_rank(pattern_type: str) -> int:
    """Higher = more specific label. Generic swing-structure fallbacks are lowest."""
    if not pattern_type or pattern_type == "no_clear_pattern":
        return 0
    tiers = {
        "head_and_shoulders": 100,
        "inverse_head_and_shoulders": 100,
        "triple_top": 92,
        "triple_bottom": 92,
        "cup_and_handle": 88,
        "double_top": 82,
        "double_bottom": 82,
        "ascending_triangle": 76,
        "descending_triangle": 76,
        "symmetrical_triangle": 74,
        "falling_wedge": 70,
        "rising_wedge": 70,
        "ascending_channel": 65,
        "descending_channel": 65,
        "bullish_vcp": 72,
        "bearish_vcp": 72,
        "bull_flag": 58,
        "bear_flag": 58,
        "bull_pennant": 58,
        "bear_pennant": 58,
        "rectangle_range": 52,
        "broadening_formation": 50,
        "rounding_bottom": 48,
        "rounding_top": 48,
        "bullish_trend_structure": 15,
        "bearish_trend_structure": 15,
    }
    return tiers.get(pattern_type, 40)


def _pattern_boundary_fingerprint(pattern_result):
    """Identity of the two boundary lines (first/last high & low), matching _build_pattern_boundary_lines."""
    sp = (pattern_result or {}).get("shapePivots") or []
    if not sp:
        return None
    highs = [p for p in sp if p.get("kind") == "high"]
    lows = [p for p in sp if p.get("kind") == "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        return (
            int(highs[0]["index"]),
            int(highs[-1]["index"]),
            int(lows[0]["index"]),
            int(lows[-1]["index"]),
        )
    return tuple(
        (int(p.get("index", 0)), str(p.get("kind", "")))
        for p in sorted(sp, key=lambda x: x.get("index", 0))
    )


def _dedupe_detector_candidates(detector_candidates, latest_index):
    """If two detectors draw the same upper/lower lines but use different names, keep the more specific label.

    When the winning label clearly beats a generic fallback (e.g. wedge vs trend structure), nudge confidence up.
    When two plausible labels tie on specificity (e.g. wedge vs channel), pull confidence down to reflect ambiguity.
    """
    groups = defaultdict(list)
    for result, score in detector_candidates:
        fp = _pattern_boundary_fingerprint(result)
        if fp is None:
            fp = ("__full__", tuple((p.get("index"), p.get("kind")) for p in (result.get("shapePivots") or [])))
        groups[fp].append((result, score))

    out = []
    for _fp, group in groups.items():
        group.sort(
            key=lambda item: (
                -_pattern_specificity_rank(((item[0] or {}).get("primaryPattern") or {}).get("type", "")),
                item[1],
                float(((item[0] or {}).get("primaryPattern") or {}).get("confidence") or 0.0),
            ),
        )
        winner_result, _winner_score = group[0]
        primary = dict((winner_result.get("primaryPattern") or {}))
        ptype = str(primary.get("type") or "")
        c1 = float(primary.get("confidence") or 0.0)

        if len(group) > 1:
            second_result = group[1][0]
            p2 = (second_result.get("primaryPattern") or {})
            sr1 = _pattern_specificity_rank(ptype)
            sr2 = _pattern_specificity_rank(str(p2.get("type") or ""))
            c2 = float(p2.get("confidence") or 0.0)
            specificity_gap = sr1 - sr2
            conf_gap = abs(c1 - c2)

            if specificity_gap >= 22:
                primary["confidence"] = round(min(0.92, c1 + 0.03), 2)
            elif specificity_gap <= 12 and conf_gap <= 0.11:
                merged = max(0.30, round(min(c1, c2) - 0.08, 2))
                primary["confidence"] = merged
                desc = str(primary.get("description") or "").strip()
                note = (
                    "Multiple pattern names fit this geometry; confidence is conservative."
                )
                primary["description"] = f"{desc} {note}".strip() if desc else note

        merged_result = {**winner_result, "primaryPattern": primary}
        out.append((merged_result, _pattern_selection_score(merged_result, latest_index)))

    out.sort(
        key=lambda item: (
            item[1],
            float(((item[0] or {}).get("primaryPattern") or {}).get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return out


def _build_marker(marker_id, label, time_value, price, color, direction):
    shape = "arrowUp" if direction == "bullish" else "arrowDown"
    position = "belowBar" if direction == "bullish" else "aboveBar"
    return {
        "id": marker_id,
        "text": label,
        "time": time_value,
        "price": smart_round_price(price),
        "color": color,
        "shape": shape,
        "position": position,
        "bias": direction,
    }


def _detect_double_top(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    candidates = []
    min_span_bars = 8
    latest_index = candles[-1]["index"] if candles else 0
    recent_floor = max(0, latest_index - 180)
    atr_local = _calculate_atr(candles)
    for offset in range(len(pivots) - 2, 1, -1):
        sequence = pivots[offset - 2:offset + 1]
        if [point["kind"] for point in sequence] != ["high", "low", "high"]:
            continue

        high_a, low_mid, high_b = sequence
        if high_b["index"] < recent_floor:
            continue
        top_gap = abs(high_a["price"] - high_b["price"])
        valley_depth = min(high_a["price"], high_b["price"]) - low_mid["price"]
        span_bars = high_b["index"] - high_a["index"]
        top_mean = mean([high_a["price"], high_b["price"]])
        valley_depth_pct = valley_depth / max(top_mean, 1e-9)
        left_leg = low_mid["index"] - high_a["index"]
        right_leg = high_b["index"] - low_mid["index"]
        symmetry_ratio = min(left_leg, right_leg) / max(max(left_leg, right_leg), 1)
        if top_gap > tolerance or valley_depth < min_move:
            continue
        if span_bars < min_span_bars:
            continue
        if min(left_leg, right_leg) < 2:
            continue
        if symmetry_ratio < 0.35:
            continue
        if valley_depth_pct < 0.002:
            continue
        if atr_local > 0 and valley_depth > atr_local * 6:
            continue
        avg_bar = _avg_bar_length(candles, high_a["index"], high_b["index"] + 1)
        if avg_bar > 0 and top_gap > avg_bar * 0.5:
            continue
        if _is_level_breached(candles, max(high_a["price"], high_b["price"]), high_b["index"], "above"):
            continue

        vol_a = _volume_at_index(candles, high_a["index"])
        vol_b = _volume_at_index(candles, high_b["index"])
        vol_bonus = 0.8 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
        age_bars = latest_index - high_b["index"]
        freshness = max(0.0, 1.0 - (age_bars / 180.0))
        score = (
            (valley_depth / max(min_move, 1e-9))
            + (span_bars / 10.0)
            - (top_gap / max(tolerance, 1e-9))
            + symmetry_ratio * 1.5
            + valley_depth_pct * 100
            + freshness * 2.0
            + vol_bonus
        )
        candidates.append((score, high_b["index"], sequence))

    if not candidates:
        for offset in range(len(pivots) - 2, 1, -1):
            sequence = pivots[offset - 2:offset + 1]
            if [point["kind"] for point in sequence] != ["high", "low", "high"]:
                continue
            high_a, low_mid, high_b = sequence
            top_gap = abs(high_a["price"] - high_b["price"])
            valley_depth = min(high_a["price"], high_b["price"]) - low_mid["price"]
            top_mean = mean([high_a["price"], high_b["price"]])
            valley_depth_pct = valley_depth / max(top_mean, 1e-9)
            left_leg = low_mid["index"] - high_a["index"]
            right_leg = high_b["index"] - low_mid["index"]
            symmetry_ratio = min(left_leg, right_leg) / max(max(left_leg, right_leg), 1)
            if top_gap > tolerance or valley_depth < min_move:
                continue
            span_bars = high_b["index"] - high_a["index"]
            if min(left_leg, right_leg) < 2:
                continue
            if symmetry_ratio < 0.30:
                continue
            if valley_depth_pct < 0.0015:
                continue
            if _is_level_breached(candles, max(high_a["price"], high_b["price"]), high_b["index"], "above"):
                continue
            vol_a = _volume_at_index(candles, high_a["index"])
            vol_b = _volume_at_index(candles, high_b["index"])
            vol_bonus = 0.6 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
            age_bars = latest_index - high_b["index"]
            freshness = max(0.0, 1.0 - (age_bars / 180.0))
            score = (
                (valley_depth / max(min_move, 1e-9))
                + (span_bars / 10.0)
                - (top_gap / max(tolerance, 1e-9))
                + symmetry_ratio * 1.2
                + valley_depth_pct * 100
                + freshness * 1.6
                + vol_bonus
            )
            candidates.append((score, high_b["index"], sequence))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, sequence = candidates[0]
    high_a, low_mid, high_b = sequence

    neckline = low_mid["price"]
    current_price = candles[-1]["close"]
    state = "forming"
    markers = []
    if current_price < neckline - breakout_buffer:
        state = "breakdown"
        markers.append(_build_marker("double-top-breakdown", "Breakdown", candles[-1]["time"], current_price, "#ef4444", "bearish"))
    elif current_price < max(high_a["price"], high_b["price"]) - breakout_buffer:
        state = "retest"

    top_gap = abs(high_a["price"] - high_b["price"])
    valley_depth = min(high_a["price"], high_b["price"]) - low_mid["price"]
    similarity = max(0.0, 1 - (top_gap / max(tolerance, 1e-9)))
    depth_score = min(1.0, valley_depth / max(min_move * 2.5, 1e-9))
    vol_a = _volume_at_index(candles, high_a["index"])
    vol_b = _volume_at_index(candles, high_b["index"])
    vol_confirmed = 1.0 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
    volume_score = min(1.0, max(volume_ratio - 1, 0))
    confidence = _pattern_confidence(0.48, similarity * 0.24, depth_score * 0.18, vol_confirmed * 0.08, volume_score * 0.06)

    left_base = _find_neighbor_pivot(pivots, high_a["index"], "low", -1, max_bars=40)
    confirm_break = _find_neighbor_pivot(pivots, high_b["index"], "low", 1, max_bars=50)
    if confirm_break and confirm_break["price"] > neckline:
        confirm_break = None

    shape_pivots = []
    if left_base:
        shape_pivots.append({
            "index": left_base["index"],
            "time": left_base["time"],
            "price": left_base["price"],
            "kind": "low",
            "role": "left_base",
        })
    shape_pivots.extend([
        {
            "index": high_a["index"],
            "time": high_a["time"],
            "price": high_a["price"],
            "kind": "high",
            "role": "left_top",
        },
        {
            "index": low_mid["index"],
            "time": low_mid["time"],
            "price": low_mid["price"],
            "kind": "low",
            "role": "neckline",
        },
        {
            "index": high_b["index"],
            "time": high_b["time"],
            "price": high_b["price"],
            "kind": "high",
            "role": "right_top",
        },
    ])
    if confirm_break:
        shape_pivots.append({
            "index": confirm_break["index"],
            "time": confirm_break["time"],
            "price": confirm_break["price"],
            "kind": "low",
            "role": "breakdown_confirm",
        })

    return {
        "primaryPattern": {
            "type": "double_top",
            "label": PATTERN_LABELS["double_top"],
            "bias": "bearish",
            "state": state,
            "confidence": confidence,
            "description": "Two similar highs with a neckline breakdown risk.",
        },
        "trendLines": [],
        "horizontalLevels": [
            _build_horizontal_level("double-top-resistance", "Resistance", "#ef4444", "dashed", mean([high_a["price"], high_b["price"]]), high_a["time"], candles[-1]["time"], candles=candles),
            _build_horizontal_level("double-top-neckline", "Neckline", "#3b82f6", "solid", neckline, low_mid["time"], candles[-1]["time"], candles=candles),
        ],
        "markers": markers,
        "shapePivots": shape_pivots,
        "levels": {
            "resistance": smart_round_price(mean([high_a["price"], high_b["price"]])),
            "support": smart_round_price(neckline),
        },
    }


def _detect_double_bottom(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    candidates = []
    min_span_bars = 8
    latest_index = candles[-1]["index"] if candles else 0
    recent_floor = max(0, latest_index - 180)
    atr_local = _calculate_atr(candles)
    for offset in range(len(pivots) - 2, 1, -1):
        sequence = pivots[offset - 2:offset + 1]
        if [point["kind"] for point in sequence] != ["low", "high", "low"]:
            continue

        low_a, high_mid, low_b = sequence
        if low_b["index"] < recent_floor:
            continue
        bottom_gap = abs(low_a["price"] - low_b["price"])
        bounce_height = high_mid["price"] - max(low_a["price"], low_b["price"])
        span_bars = low_b["index"] - low_a["index"]
        bottom_mean = mean([low_a["price"], low_b["price"]])
        bounce_height_pct = bounce_height / max(bottom_mean, 1e-9)
        left_leg = high_mid["index"] - low_a["index"]
        right_leg = low_b["index"] - high_mid["index"]
        symmetry_ratio = min(left_leg, right_leg) / max(max(left_leg, right_leg), 1)
        if bottom_gap > tolerance or bounce_height < min_move:
            continue
        if span_bars < min_span_bars:
            continue
        if min(left_leg, right_leg) < 2:
            continue
        if symmetry_ratio < 0.35:
            continue
        if bounce_height_pct < 0.002:
            continue
        if atr_local > 0 and bounce_height > atr_local * 6:
            continue
        avg_bar = _avg_bar_length(candles, low_a["index"], low_b["index"] + 1)
        if avg_bar > 0 and bottom_gap > avg_bar * 0.5:
            continue
        if _is_level_breached(candles, min(low_a["price"], low_b["price"]), low_b["index"], "below"):
            continue

        vol_a = _volume_at_index(candles, low_a["index"])
        vol_b = _volume_at_index(candles, low_b["index"])
        vol_bonus = 0.8 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
        age_bars = latest_index - low_b["index"]
        freshness = max(0.0, 1.0 - (age_bars / 180.0))
        score = (
            (bounce_height / max(min_move, 1e-9))
            + (span_bars / 10.0)
            - (bottom_gap / max(tolerance, 1e-9))
            + symmetry_ratio * 1.5
            + bounce_height_pct * 100
            + freshness * 2.0
            + vol_bonus
        )
        candidates.append((score, low_b["index"], sequence))

    if not candidates:
        for offset in range(len(pivots) - 2, 1, -1):
            sequence = pivots[offset - 2:offset + 1]
            if [point["kind"] for point in sequence] != ["low", "high", "low"]:
                continue
            low_a, high_mid, low_b = sequence
            bottom_gap = abs(low_a["price"] - low_b["price"])
            bounce_height = high_mid["price"] - max(low_a["price"], low_b["price"])
            bottom_mean = mean([low_a["price"], low_b["price"]])
            bounce_height_pct = bounce_height / max(bottom_mean, 1e-9)
            left_leg = high_mid["index"] - low_a["index"]
            right_leg = low_b["index"] - high_mid["index"]
            symmetry_ratio = min(left_leg, right_leg) / max(max(left_leg, right_leg), 1)
            if bottom_gap > tolerance or bounce_height < min_move:
                continue
            span_bars = low_b["index"] - low_a["index"]
            if min(left_leg, right_leg) < 2:
                continue
            if symmetry_ratio < 0.30:
                continue
            if bounce_height_pct < 0.0015:
                continue
            if _is_level_breached(candles, min(low_a["price"], low_b["price"]), low_b["index"], "below"):
                continue
            vol_a = _volume_at_index(candles, low_a["index"])
            vol_b = _volume_at_index(candles, low_b["index"])
            vol_bonus = 0.6 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
            age_bars = latest_index - low_b["index"]
            freshness = max(0.0, 1.0 - (age_bars / 180.0))
            score = (
                (bounce_height / max(min_move, 1e-9))
                + (span_bars / 10.0)
                - (bottom_gap / max(tolerance, 1e-9))
                + symmetry_ratio * 1.2
                + bounce_height_pct * 100
                + freshness * 1.6
                + vol_bonus
            )
            candidates.append((score, low_b["index"], sequence))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, sequence = candidates[0]
    low_a, high_mid, low_b = sequence

    neckline = high_mid["price"]
    current_price = candles[-1]["close"]
    state = "forming"
    markers = []
    if current_price > neckline + breakout_buffer:
        state = "breakout"
        markers.append(_build_marker("double-bottom-breakout", "Breakout", candles[-1]["time"], current_price, "#22c55e", "bullish"))
    elif current_price > min(low_a["price"], low_b["price"]) + breakout_buffer:
        state = "retest"

    bottom_gap = abs(low_a["price"] - low_b["price"])
    bounce_height = high_mid["price"] - max(low_a["price"], low_b["price"])
    similarity = max(0.0, 1 - (bottom_gap / max(tolerance, 1e-9)))
    bounce_score = min(1.0, bounce_height / max(min_move * 2.5, 1e-9))
    vol_a = _volume_at_index(candles, low_a["index"])
    vol_b = _volume_at_index(candles, low_b["index"])
    vol_confirmed = 1.0 if (vol_a > 0 and vol_b > 0 and vol_b < vol_a) else 0.0
    volume_score = min(1.0, max(volume_ratio - 1, 0))
    confidence = _pattern_confidence(0.48, similarity * 0.24, bounce_score * 0.18, vol_confirmed * 0.08, volume_score * 0.06)

    left_base = _find_neighbor_pivot(pivots, low_a["index"], "high", -1, max_bars=40)
    confirm_break = _find_neighbor_pivot(pivots, low_b["index"], "high", 1, max_bars=50)
    if confirm_break and confirm_break["price"] < neckline:
        confirm_break = None

    shape_pivots = []
    if left_base:
        shape_pivots.append({
            "index": left_base["index"],
            "time": left_base["time"],
            "price": left_base["price"],
            "kind": "high",
            "role": "left_base",
        })
    shape_pivots.extend([
        {
            "index": low_a["index"],
            "time": low_a["time"],
            "price": low_a["price"],
            "kind": "low",
            "role": "left_bottom",
        },
        {
            "index": high_mid["index"],
            "time": high_mid["time"],
            "price": high_mid["price"],
            "kind": "high",
            "role": "neckline",
        },
        {
            "index": low_b["index"],
            "time": low_b["time"],
            "price": low_b["price"],
            "kind": "low",
            "role": "right_bottom",
        },
    ])
    if confirm_break:
        shape_pivots.append({
            "index": confirm_break["index"],
            "time": confirm_break["time"],
            "price": confirm_break["price"],
            "kind": "high",
            "role": "breakout_confirm",
        })

    return {
        "primaryPattern": {
            "type": "double_bottom",
            "label": PATTERN_LABELS["double_bottom"],
            "bias": "bullish",
            "state": state,
            "confidence": confidence,
            "description": "Two similar lows with a neckline breakout setup.",
        },
        "trendLines": [],
        "horizontalLevels": [
            _build_horizontal_level("double-bottom-support", "Support", "#22c55e", "dashed", mean([low_a["price"], low_b["price"]]), low_a["time"], candles[-1]["time"], candles=candles),
            _build_horizontal_level("double-bottom-neckline", "Neckline", "#3b82f6", "solid", neckline, high_mid["time"], candles[-1]["time"], candles=candles),
        ],
        "markers": markers,
        "shapePivots": shape_pivots,
        "levels": {
            "support": smart_round_price(mean([low_a["price"], low_b["price"]])),
            "resistance": smart_round_price(neckline),
        },
    }


def _detect_triangle(pivots, candles, pattern_type, tolerance, min_move, breakout_buffer, volume_ratio):
    sequences = {
        "ascending_triangle": [["low", "high", "low", "high"], ["high", "low", "high", "low"]],
        "descending_triangle": [["high", "low", "high", "low"], ["low", "high", "low", "high"]],
    }

    for kinds in sequences[pattern_type]:
        for start in range(len(pivots) - len(kinds), -1, -1):
            sequence = pivots[start:start + len(kinds)]
            if [point["kind"] for point in sequence] != kinds:
                continue

            highs = [point for point in sequence if point["kind"] == "high"][-2:]
            lows = [point for point in sequence if point["kind"] == "low"][-2:]
            if len(highs) < 2 or len(lows) < 2:
                continue

            high_a, high_b = highs
            low_a, low_b = lows
            resistance = mean([high_a["price"], high_b["price"]])
            support_projection = _project_price(low_a, low_b, candles[-1]["index"])
            flat_gap = abs(high_a["price"] - high_b["price"])
            slope_gain = low_b["price"] - low_a["price"]
            span_start = min(high_a["index"], low_a["index"])
            span_end = max(high_b["index"], low_b["index"])
            avg_bar = _avg_bar_length(candles, span_start, span_end + 1)

            if pattern_type == "ascending_triangle":
                flat_ok = flat_gap <= tolerance and (avg_bar <= 0 or flat_gap <= avg_bar)
                if not flat_ok or slope_gain <= tolerance * 0.2:
                    continue
                state = "forming"
                markers = []
                if candles[-1]["close"] > resistance + breakout_buffer:
                    state = "breakout"
                    markers.append(_build_marker("ascending-triangle-breakout", "Breakout", candles[-1]["time"], candles[-1]["close"], "#22c55e", "bullish"))
                elif candles[-1]["close"] < support_projection - breakout_buffer:
                    state = "failed"
                compression_start = resistance - low_a["price"]
                compression_end = resistance - support_projection
                compression_score = 1 - (compression_end / max(compression_start, 1e-9))
                similarity = max(0.0, 1 - flat_gap / max(tolerance, 1e-9))
                ascent_score = min(1.0, slope_gain / max(min_move * 1.2, 1e-9))
                volume_score = min(1.0, max(volume_ratio - 1, 0))
                confidence = _pattern_confidence(0.44, similarity * 0.22, ascent_score * 0.16, compression_score * 0.12, volume_score * 0.04)
                return {
                    "primaryPattern": {
                        "type": "ascending_triangle",
                        "label": PATTERN_LABELS["ascending_triangle"],
                        "bias": "bullish",
                        "state": state,
                        "confidence": confidence,
                        "description": "Flat resistance with higher lows compressing into a bullish breakout setup.",
                    },
                    "trendLines": [
                        _build_trend_line("triangle-support", "Support", "#3b82f6", "solid", low_a, low_b, candles),
                    ],
                    "horizontalLevels": [
                        _build_horizontal_level("triangle-resistance", "Resistance", "#ef4444", "dashed", resistance, min(high_a["time"], high_b["time"]), candles[-1]["time"], candles=candles),
                    ],
                    "markers": markers,
                    "shapePivots": _shape_pivots_from_sequence(sequence),
                    "levels": {
                        "support": smart_round_price(support_projection),
                        "resistance": smart_round_price(resistance),
                    },
                }

            if pattern_type == "descending_triangle":
                support = mean([low_a["price"], low_b["price"]])
                resistance_projection = _project_price(high_a, high_b, candles[-1]["index"])
                slope_drop = high_b["price"] - high_a["price"]
                low_flat_gap = abs(low_a["price"] - low_b["price"])
                flat_ok = low_flat_gap <= tolerance and (avg_bar <= 0 or low_flat_gap <= avg_bar)
                if not flat_ok or slope_drop >= -(tolerance * 0.2):
                    continue
                state = "forming"
                markers = []
                if candles[-1]["close"] < support - breakout_buffer:
                    state = "breakdown"
                    markers.append(_build_marker("descending-triangle-breakdown", "Breakdown", candles[-1]["time"], candles[-1]["close"], "#ef4444", "bearish"))
                elif candles[-1]["close"] > resistance_projection + breakout_buffer:
                    state = "failed"
                compression_start = high_a["price"] - support
                compression_end = resistance_projection - support
                compression_score = 1 - (compression_end / max(compression_start, 1e-9))
                similarity = max(0.0, 1 - abs(low_a["price"] - low_b["price"]) / max(tolerance, 1e-9))
                descent_score = min(1.0, abs(slope_drop) / max(min_move * 1.2, 1e-9))
                volume_score = min(1.0, max(volume_ratio - 1, 0))
                confidence = _pattern_confidence(0.44, similarity * 0.22, descent_score * 0.16, compression_score * 0.12, volume_score * 0.04)
                return {
                    "primaryPattern": {
                        "type": "descending_triangle",
                        "label": PATTERN_LABELS["descending_triangle"],
                        "bias": "bearish",
                        "state": state,
                        "confidence": confidence,
                        "description": "Flat support with lower highs compressing into a bearish breakdown setup.",
                    },
                    "trendLines": [
                        _build_trend_line("triangle-resistance", "Resistance", "#ef4444", "solid", high_a, high_b, candles),
                    ],
                    "horizontalLevels": [
                        _build_horizontal_level("triangle-support", "Support", "#3b82f6", "dashed", support, min(low_a["time"], low_b["time"]), candles[-1]["time"], candles=candles),
                    ],
                    "markers": markers,
                    "shapePivots": _shape_pivots_from_sequence(sequence),
                    "levels": {
                        "support": smart_round_price(support),
                        "resistance": smart_round_price(resistance_projection),
                    },
                }
    return None


def _detect_wedge(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 4, -1, -1):
        sequence = pivots[start:start + 4]
        kinds = [point["kind"] for point in sequence]
        if kinds not in (["high", "low", "high", "low"], ["low", "high", "low", "high"]):
            continue

        highs = [point for point in sequence if point["kind"] == "high"][-2:]
        lows = [point for point in sequence if point["kind"] == "low"][-2:]
        if len(highs) < 2 or len(lows) < 2:
            continue

        high_a, high_b = highs
        low_a, low_b = lows
        high_slope = (high_b["price"] - high_a["price"]) / max(high_b["index"] - high_a["index"], 1)
        low_slope = (low_b["price"] - low_a["price"]) / max(low_b["index"] - low_a["index"], 1)
        distance_start = high_a["price"] - low_a["price"]
        distance_end = high_b["price"] - low_b["price"]
        if distance_start <= 0 or distance_end <= 0:
            continue

        converging = distance_end < distance_start * 0.75
        if not converging:
            continue

        span = max(high_b["index"], low_b["index"]) - min(high_a["index"], low_a["index"])
        if span < 6:
            continue
        atr_w = _calculate_atr(candles)
        if atr_w > 0 and distance_start < atr_w * 0.5:
            continue

        current_price = candles[-1]["close"]
        bullish = high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope) * 1.15
        bearish = high_slope > 0 and low_slope > 0 and low_slope > high_slope * 1.15
        if not bullish and not bearish:
            continue

        upper_projection = _project_price(high_a, high_b, candles[-1]["index"])
        lower_projection = _project_price(low_a, low_b, candles[-1]["index"])

        if bullish:
            state = "forming"
            markers = []
            if current_price > upper_projection + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("falling-wedge-breakout", "Breakout", candles[-1]["time"], current_price, "#22c55e", "bullish"))
            convergence_quality = min(1.0, (distance_start - distance_end) / max(distance_start, 1e-9))
            slope_diff = min(1.0, abs(high_slope - low_slope) / max(abs(low_slope), tolerance / max(candles[-1]["close"], 1e-9)))
            confidence = _pattern_confidence(
                0.36,
                slope_diff * 0.18,
                convergence_quality * 0.16,
                min(1.0, max(volume_ratio - 1, 0)) * 0.04,
            )
            sp = sorted(sequence, key=lambda p: p["index"])
            return {
                "primaryPattern": {
                    "type": "falling_wedge",
                    "label": PATTERN_LABELS["falling_wedge"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Downward converging structure with bullish squeeze potential.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                "horizontalLevels": [],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(lower_projection),
                    "resistance": smart_round_price(upper_projection),
                },
            }

        if bearish:
            state = "forming"
            markers = []
            if current_price < lower_projection - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("rising-wedge-breakdown", "Breakdown", candles[-1]["time"], current_price, "#ef4444", "bearish"))
            convergence_quality = min(1.0, (distance_start - distance_end) / max(distance_start, 1e-9))
            slope_diff = min(1.0, abs(low_slope - high_slope) / max(abs(high_slope), tolerance / max(candles[-1]["close"], 1e-9)))
            confidence = _pattern_confidence(
                0.36,
                slope_diff * 0.18,
                convergence_quality * 0.16,
                min(1.0, max(volume_ratio - 1, 0)) * 0.04,
            )
            sp = sorted(sequence, key=lambda p: p["index"])
            return {
                "primaryPattern": {
                    "type": "rising_wedge",
                    "label": PATTERN_LABELS["rising_wedge"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Upward converging structure with bearish squeeze risk.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                "horizontalLevels": [],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(lower_projection),
                    "resistance": smart_round_price(upper_projection),
                },
            }
    return None


def _detect_head_and_shoulders(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 4, -1, -1):
        sequence = pivots[start:start + 5]
        kinds = [point["kind"] for point in sequence]

        # Head and Shoulders Top
        if kinds == ["high", "low", "high", "low", "high"]:
            shoulder_left, low_a, head, low_b, shoulder_right = sequence
            shoulder_gap = abs(shoulder_left["price"] - shoulder_right["price"])
            head_prominence = head["price"] - max(shoulder_left["price"], shoulder_right["price"])
            avg_bar = _avg_bar_length(candles, shoulder_left["index"], shoulder_right["index"] + 1)
            shoulder_height_threshold = avg_bar * 0.6 if avg_bar > 0 else 0
            if shoulder_gap > tolerance * 1.2 or head_prominence < min_move * 0.6:
                continue
            if shoulder_height_threshold > 0 and head_prominence < shoulder_height_threshold:
                continue
            neckline_gap = abs(low_a["price"] - low_b["price"])
            if avg_bar > 0 and neckline_gap > avg_bar:
                continue
            if max(low_a["price"], low_b["price"]) >= min(shoulder_left["price"], shoulder_right["price"]):
                continue

            neckline = mean([low_a["price"], low_b["price"]])
            current_price = candles[-1]["close"]
            state = "forming"
            markers = []
            if current_price < neckline - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("hs-breakdown", "Breakdown", candles[-1]["time"], current_price, "#ef4444", "bearish"))
            elif current_price < max(shoulder_left["price"], shoulder_right["price"]) - breakout_buffer:
                state = "retest"

            shoulder_similarity = max(0.0, 1 - (shoulder_gap / max(tolerance, 1e-9)))
            head_score = min(1.0, head_prominence / max(min_move * 2.0, 1e-9))
            neckline_flat_score = max(0.0, 1 - neckline_gap / max(avg_bar if avg_bar > 0 else tolerance, 1e-9))
            volume_score = min(1.0, max(volume_ratio - 1, 0))
            confidence = _pattern_confidence(0.46, shoulder_similarity * 0.2, head_score * 0.2, neckline_flat_score * 0.08, volume_score * 0.06)

            return {
                "primaryPattern": {
                    "type": "head_and_shoulders",
                    "label": PATTERN_LABELS["head_and_shoulders"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Three-peak reversal structure where the middle peak forms the head.",
                },
                "shapePivots": list(sequence),
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("hs-neckline", "Neckline", "#3b82f6", "dashed", neckline, low_a["time"], candles[-1]["time"], candles=candles),
                    _build_horizontal_level("hs-shoulder-resistance", "Resistance", "#ef4444", "dashed", mean([shoulder_left["price"], shoulder_right["price"]]), shoulder_left["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(neckline),
                    "resistance": smart_round_price(head["price"]),
                },
            }

        # Inverse Head and Shoulders Bottom
        if kinds == ["low", "high", "low", "high", "low"]:
            shoulder_left, high_a, head, high_b, shoulder_right = sequence
            shoulder_gap = abs(shoulder_left["price"] - shoulder_right["price"])
            head_depth = min(shoulder_left["price"], shoulder_right["price"]) - head["price"]
            avg_bar_inv = _avg_bar_length(candles, shoulder_left["index"], shoulder_right["index"] + 1)
            shoulder_depth_threshold = avg_bar_inv * 0.6 if avg_bar_inv > 0 else 0
            if shoulder_gap > tolerance * 1.2 or head_depth < min_move * 0.6:
                continue
            if shoulder_depth_threshold > 0 and head_depth < shoulder_depth_threshold:
                continue
            neckline_gap_inv = abs(high_a["price"] - high_b["price"])
            if avg_bar_inv > 0 and neckline_gap_inv > avg_bar_inv:
                continue
            if min(high_a["price"], high_b["price"]) <= max(shoulder_left["price"], shoulder_right["price"]):
                continue

            neckline = mean([high_a["price"], high_b["price"]])
            current_price = candles[-1]["close"]
            state = "forming"
            markers = []
            if current_price > neckline + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("ihs-breakout", "Breakout", candles[-1]["time"], current_price, "#22c55e", "bullish"))
            elif current_price > min(high_a["price"], high_b["price"]) - breakout_buffer:
                state = "retest"

            shoulder_similarity = max(0.0, 1 - (shoulder_gap / max(tolerance, 1e-9)))
            head_score = min(1.0, head_depth / max(min_move * 2.0, 1e-9))
            neckline_flat_score_inv = max(0.0, 1 - neckline_gap_inv / max(avg_bar_inv if avg_bar_inv > 0 else tolerance, 1e-9))
            volume_score = min(1.0, max(volume_ratio - 1, 0))
            confidence = _pattern_confidence(0.46, shoulder_similarity * 0.2, head_score * 0.2, neckline_flat_score_inv * 0.08, volume_score * 0.06)

            return {
                "primaryPattern": {
                    "type": "inverse_head_and_shoulders",
                    "label": PATTERN_LABELS["inverse_head_and_shoulders"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Three-trough reversal structure where the middle trough forms the head.",
                },
                "shapePivots": list(sequence),
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("ihs-neckline", "Neckline", "#3b82f6", "dashed", neckline, high_a["time"], candles[-1]["time"], candles=candles),
                    _build_horizontal_level("ihs-shoulder-support", "Support", "#22c55e", "dashed", mean([shoulder_left["price"], shoulder_right["price"]]), shoulder_left["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(head["price"]),
                    "resistance": smart_round_price(neckline),
                },
            }
    return None


def _detect_channel(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 3, -1, -1):
        sequence = pivots[start:start + 4]
        kinds = [point["kind"] for point in sequence]
        if kinds not in (["high", "low", "high", "low"], ["low", "high", "low", "high"]):
            continue

        highs = [point for point in sequence if point["kind"] == "high"][-2:]
        lows = [point for point in sequence if point["kind"] == "low"][-2:]
        if len(highs) < 2 or len(lows) < 2:
            continue

        high_a, high_b = highs
        low_a, low_b = lows
        high_idx_delta = max(high_b["index"] - high_a["index"], 1)
        low_idx_delta = max(low_b["index"] - low_a["index"], 1)
        high_slope = (high_b["price"] - high_a["price"]) / high_idx_delta
        low_slope = (low_b["price"] - low_a["price"]) / low_idx_delta
        if high_slope == 0 or low_slope == 0:
            continue

        same_direction = (high_slope > 0 and low_slope > 0) or (high_slope < 0 and low_slope < 0)
        if not same_direction:
            continue

        parallel_gap = abs(high_slope - low_slope) / max(abs(high_slope), abs(low_slope), 1e-9)
        parallel_score = max(0.0, 1 - parallel_gap)
        if parallel_score < 0.45:
            continue

        distance_start = high_a["price"] - low_a["price"]
        distance_end = high_b["price"] - low_b["price"]
        if distance_start <= 0 or distance_end <= 0:
            continue

        width_change = abs(distance_end - distance_start) / max(distance_start, 1e-9)
        if width_change > 0.35:
            # Likely wedge/triangle rather than channel.
            continue

        upper_projection = _project_price(high_a, high_b, candles[-1]["index"])
        lower_projection = _project_price(low_a, low_b, candles[-1]["index"])
        current_price = candles[-1]["close"]
        volume_score = min(1.0, max(volume_ratio - 1, 0))

        if high_slope > 0 and low_slope > 0:
            state = "forming"
            markers = []
            if current_price > upper_projection + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("ascending-channel-breakout", "Breakout", candles[-1]["time"], current_price, "#22c55e", "bullish"))
            elif current_price < lower_projection - breakout_buffer:
                state = "failed"

            confidence = _pattern_confidence(0.4, parallel_score * 0.28, (1 - width_change) * 0.18, volume_score * 0.04)
            sp = sorted(sequence, key=lambda p: p["index"])
            return {
                "primaryPattern": {
                    "type": "ascending_channel",
                    "label": PATTERN_LABELS["ascending_channel"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Parallel upward channel containing price swings with trend continuation bias.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                "horizontalLevels": [],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(lower_projection),
                    "resistance": smart_round_price(upper_projection),
                },
            }

        if high_slope < 0 and low_slope < 0:
            state = "forming"
            markers = []
            if current_price < lower_projection - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("descending-channel-breakdown", "Breakdown", candles[-1]["time"], current_price, "#ef4444", "bearish"))
            elif current_price > upper_projection + breakout_buffer:
                state = "failed"

            confidence = _pattern_confidence(0.4, parallel_score * 0.28, (1 - width_change) * 0.18, volume_score * 0.04)
            sp = sorted(sequence, key=lambda p: p["index"])
            return {
                "primaryPattern": {
                    "type": "descending_channel",
                    "label": PATTERN_LABELS["descending_channel"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Parallel downward channel containing price swings with trend continuation bias.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                "horizontalLevels": [],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(lower_projection),
                    "resistance": smart_round_price(upper_projection),
                },
            }
    return None


def _linear_slope(values):
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _detect_triple_top_bottom(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 4, -1, -1):
        seq = pivots[start:start + 5]
        kinds = [point["kind"] for point in seq]

        # Triple Top: high-low-high-low-high
        if kinds == ["high", "low", "high", "low", "high"]:
            h1, l1, h2, l2, h3 = seq
            tops = [h1["price"], h2["price"], h3["price"]]
            top_band = max(tops) - min(tops)
            if top_band > tolerance * 1.6:
                continue
            neckline = mean([l1["price"], l2["price"]])
            depth = min(tops) - neckline
            if depth < min_move:
                continue

            current = candles[-1]["close"]
            state = "forming"
            markers = []
            if current < neckline - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("triple-top-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
            elif current < max(tops) - breakout_buffer:
                state = "retest"

            symmetry = max(0.0, 1 - top_band / max(tolerance * 1.6, 1e-9))
            depth_score = min(1.0, depth / max(min_move * 2.0, 1e-9))
            volume_score = min(1.0, max(volume_ratio - 1, 0))
            confidence = _pattern_confidence(0.44, symmetry * 0.22, depth_score * 0.18, volume_score * 0.04)
            resistance = mean(tops)
            return {
                "primaryPattern": {
                    "type": "triple_top",
                    "label": PATTERN_LABELS["triple_top"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Three similar highs with repeated rejection before neckline risk.",
                },
                "shapePivots": list(seq),
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("triple-top-resistance", "Resistance", "#ef4444", "dashed", resistance, h1["time"], candles[-1]["time"], candles=candles),
                    _build_horizontal_level("triple-top-neckline", "Neckline", "#3b82f6", "solid", neckline, l1["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(neckline),
                    "resistance": smart_round_price(resistance),
                },
            }

        # Triple Bottom: low-high-low-high-low
        if kinds == ["low", "high", "low", "high", "low"]:
            l1, h1, l2, h2, l3 = seq
            bottoms = [l1["price"], l2["price"], l3["price"]]
            bottom_band = max(bottoms) - min(bottoms)
            if bottom_band > tolerance * 1.6:
                continue
            neckline = mean([h1["price"], h2["price"]])
            bounce = neckline - max(bottoms)
            if bounce < min_move:
                continue

            current = candles[-1]["close"]
            state = "forming"
            markers = []
            if current > neckline + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("triple-bottom-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
            elif current > min(bottoms) + breakout_buffer:
                state = "retest"

            symmetry = max(0.0, 1 - bottom_band / max(tolerance * 1.6, 1e-9))
            bounce_score = min(1.0, bounce / max(min_move * 2.0, 1e-9))
            volume_score = min(1.0, max(volume_ratio - 1, 0))
            confidence = _pattern_confidence(0.44, symmetry * 0.22, bounce_score * 0.18, volume_score * 0.04)
            support = mean(bottoms)
            return {
                "primaryPattern": {
                    "type": "triple_bottom",
                    "label": PATTERN_LABELS["triple_bottom"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Three similar lows with repeated demand before neckline breakout.",
                },
                "shapePivots": list(seq),
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("triple-bottom-support", "Support", "#22c55e", "dashed", support, l1["time"], candles[-1]["time"], candles=candles),
                    _build_horizontal_level("triple-bottom-neckline", "Neckline", "#3b82f6", "solid", neckline, h1["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(support),
                    "resistance": smart_round_price(neckline),
                },
            }
    return None


def _detect_flag_and_pennant(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    if len(candles) < 24:
        return None
    window = candles[-24:]
    pole = window[:10]
    consolidation = window[10:]
    if len(consolidation) < 8:
        return None

    pole_start = pole[0]["close"]
    pole_end = pole[-1]["close"]
    if pole_start <= 0:
        return None
    pole_return = (pole_end - pole_start) / pole_start
    cons_high = max(c["high"] for c in consolidation)
    cons_low = min(c["low"] for c in consolidation)
    cons_range = cons_high - cons_low
    if cons_range <= 0:
        return None
    cons_closes = [c["close"] for c in consolidation]
    slope = _linear_slope(cons_closes)
    slope_pct = slope / max(cons_closes[-1], 1e-9)
    threshold = max(0.0035, min_move / max(candles[-1]["close"], 1e-9))
    current = candles[-1]["close"]

    # Bull flag / pennant
    if pole_return > threshold * 2.0:
        pullback = (pole_end - cons_low) / max(abs(pole_end - pole_start), 1e-9)
        if 0.08 <= pullback <= 0.65:
            highs = [c["high"] for c in consolidation]
            lows = [c["low"] for c in consolidation]
            high_slope = _linear_slope(highs)
            low_slope = _linear_slope(lows)
            converging = high_slope < 0 and low_slope > 0
            narrowing = (max(highs[:3]) - min(lows[:3])) > (max(highs[-3:]) - min(lows[-3:]))
            is_pennant = converging and narrowing
            is_flag = slope_pct < 0 and abs(high_slope - low_slope) / max(abs(high_slope), abs(low_slope), 1e-9) < 0.6
            if is_pennant or is_flag:
                top_line = max(c["high"] for c in consolidation[:-1])
                state = "forming"
                markers = []
                if current > top_line + breakout_buffer:
                    state = "breakout"
                    markers.append(_build_marker("bull-cont-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
                confidence = _pattern_confidence(
                    0.42,
                    min(1.0, abs(pole_return) / (threshold * 4.0)) * 0.22,
                    min(1.0, (0.65 - pullback) / 0.65) * 0.14,
                    min(1.0, max(volume_ratio - 1, 0)) * 0.04,
                )
                kind = "bull_pennant" if is_pennant else "bull_flag"
                sp = [
                    {"time": consolidation[0]["time"], "price": highs[0], "index": consolidation[0]["index"], "kind": "high"},
                    {"time": consolidation[0]["time"], "price": lows[0], "index": consolidation[0]["index"], "kind": "low"},
                    {"time": consolidation[-1]["time"], "price": highs[-1], "index": consolidation[-1]["index"], "kind": "high"},
                    {"time": consolidation[-1]["time"], "price": lows[-1], "index": consolidation[-1]["index"], "kind": "low"},
                ]
                sp.sort(key=lambda p: (p["index"], p["kind"]))
                return {
                    "primaryPattern": {
                        "type": kind,
                        "label": PATTERN_LABELS[kind],
                        "bias": "bullish",
                        "state": state,
                        "confidence": confidence,
                        "description": "Strong impulse followed by a tight consolidation continuation setup.",
                    },
                    "shapePivots": sp,
                    "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                    "horizontalLevels": [],
                    "markers": markers,
                    "levels": {
                        "support": smart_round_price(cons_low),
                        "resistance": smart_round_price(cons_high),
                    },
                }

    # Bear flag / pennant
    if pole_return < -(threshold * 2.0):
        bounce = (cons_high - pole_end) / max(abs(pole_end - pole_start), 1e-9)
        if 0.08 <= bounce <= 0.65:
            highs = [c["high"] for c in consolidation]
            lows = [c["low"] for c in consolidation]
            high_slope = _linear_slope(highs)
            low_slope = _linear_slope(lows)
            converging = high_slope < 0 and low_slope > 0
            narrowing = (max(highs[:3]) - min(lows[:3])) > (max(highs[-3:]) - min(lows[-3:]))
            is_pennant = converging and narrowing
            is_flag = slope_pct > 0 and abs(high_slope - low_slope) / max(abs(high_slope), abs(low_slope), 1e-9) < 0.6
            if is_pennant or is_flag:
                bottom_line = min(c["low"] for c in consolidation[:-1])
                state = "forming"
                markers = []
                if current < bottom_line - breakout_buffer:
                    state = "breakdown"
                    markers.append(_build_marker("bear-cont-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
                confidence = _pattern_confidence(
                    0.42,
                    min(1.0, abs(pole_return) / (threshold * 4.0)) * 0.22,
                    min(1.0, (0.65 - bounce) / 0.65) * 0.14,
                    min(1.0, max(volume_ratio - 1, 0)) * 0.04,
                )
                kind = "bear_pennant" if is_pennant else "bear_flag"
                sp = [
                    {"time": consolidation[0]["time"], "price": highs[0], "index": consolidation[0]["index"], "kind": "high"},
                    {"time": consolidation[0]["time"], "price": lows[0], "index": consolidation[0]["index"], "kind": "low"},
                    {"time": consolidation[-1]["time"], "price": highs[-1], "index": consolidation[-1]["index"], "kind": "high"},
                    {"time": consolidation[-1]["time"], "price": lows[-1], "index": consolidation[-1]["index"], "kind": "low"},
                ]
                sp.sort(key=lambda p: (p["index"], p["kind"]))
                return {
                    "primaryPattern": {
                        "type": kind,
                        "label": PATTERN_LABELS[kind],
                        "bias": "bearish",
                        "state": state,
                        "confidence": confidence,
                        "description": "Strong downside impulse followed by a tight consolidation continuation setup.",
                    },
                    "shapePivots": sp,
                    "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
                    "horizontalLevels": [],
                    "markers": markers,
                    "levels": {
                        "support": smart_round_price(cons_low),
                        "resistance": smart_round_price(cons_high),
                    },
                }
    return None


def _detect_rectangle_range(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    if len(candles) < 20:
        return None
    recent = candles[-20:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    closes = [c["close"] for c in recent]
    resistance = mean(sorted(highs)[-4:])
    support = mean(sorted(lows)[:4])
    width = resistance - support
    if width < min_move * 1.2:
        return None
    drift = abs(_linear_slope(closes))
    if drift > width / max(len(closes), 1) * 0.35:
        return None
    current = closes[-1]
    state = "forming"
    markers = []
    bias = "neutral"
    if current > resistance + breakout_buffer:
        state = "breakout"
        bias = "bullish"
        markers.append(_build_marker("range-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
    elif current < support - breakout_buffer:
        state = "breakdown"
        bias = "bearish"
        markers.append(_build_marker("range-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
    confidence = _pattern_confidence(
        0.36,
        min(1.0, width / max(min_move * 4.0, 1e-9)) * 0.12,
        max(0.0, 1 - (drift / max(width / len(closes), 1e-9))) * 0.18,
    )
    return {
        "primaryPattern": {
            "type": "rectangle_range",
            "label": PATTERN_LABELS["rectangle_range"],
            "bias": bias,
            "state": state,
            "confidence": confidence,
            "description": "Price oscillates inside a horizontal support-resistance box.",
        },
        "trendLines": [],
        "horizontalLevels": [
            _build_horizontal_level("rectangle-support", "Support", "#3b82f6", "dashed", support, recent[0]["time"], candles[-1]["time"], candles=candles),
            _build_horizontal_level("rectangle-resistance", "Resistance", "#ef4444", "dashed", resistance, recent[0]["time"], candles[-1]["time"], candles=candles),
        ],
        "markers": markers,
        "levels": {"support": smart_round_price(support), "resistance": smart_round_price(resistance)},
    }


def _detect_cup_and_handle(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 3, -1, -1):
        seq = pivots[start:start + 4]
        kinds = [p["kind"] for p in seq]
        if kinds != ["high", "low", "high", "low"]:
            continue
        left_rim, cup_low, right_rim, handle_low = seq
        rim_gap = abs(left_rim["price"] - right_rim["price"])
        cup_depth = min(left_rim["price"], right_rim["price"]) - cup_low["price"]
        handle_depth = right_rim["price"] - handle_low["price"]
        if rim_gap > tolerance * 1.4:
            continue
        if cup_depth < min_move * 1.5:
            continue
        if handle_depth <= 0 or handle_depth > cup_depth * 0.6:
            continue
        resistance = mean([left_rim["price"], right_rim["price"]])
        current = candles[-1]["close"]
        state = "forming"
        markers = []
        if current > resistance + breakout_buffer:
            state = "breakout"
            markers.append(_build_marker("cup-handle-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
        elif current > handle_low["price"] + breakout_buffer:
            state = "retest"
        confidence = _pattern_confidence(
            0.42,
            max(0.0, 1 - rim_gap / max(tolerance * 1.4, 1e-9)) * 0.18,
            min(1.0, cup_depth / max(min_move * 3.0, 1e-9)) * 0.16,
            min(1.0, max(volume_ratio - 1, 0)) * 0.04,
        )
        return {
            "primaryPattern": {
                "type": "cup_and_handle",
                "label": PATTERN_LABELS["cup_and_handle"],
                "bias": "bullish",
                "state": state,
                "confidence": confidence,
                "description": "Rounded base with a shallow handle pullback below resistance.",
            },
            "shapePivots": list(seq),
            "trendLines": [
                _build_trend_line("cup-handle-handle", "Handle", "#3b82f6", "solid", right_rim, handle_low, candles),
            ],
            "horizontalLevels": [
                _build_horizontal_level("cup-handle-resistance", "Resistance", "#ef4444", "dashed", resistance, left_rim["time"], candles[-1]["time"], candles=candles),
            ],
            "markers": markers,
            "levels": {"support": smart_round_price(handle_low["price"]), "resistance": smart_round_price(resistance)},
        }
    return None


def _detect_broadening_formation(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 3, -1, -1):
        seq = pivots[start:start + 4]
        kinds = [p["kind"] for p in seq]
        if kinds not in (["high", "low", "high", "low"], ["low", "high", "low", "high"]):
            continue
        highs = [p for p in seq if p["kind"] == "high"]
        lows = [p for p in seq if p["kind"] == "low"]
        if len(highs) < 2 or len(lows) < 2:
            continue
        h1, h2 = highs[-2:]
        l1, l2 = lows[-2:]
        width_start = h1["price"] - l1["price"]
        width_end = h2["price"] - l2["price"]
        if width_start <= 0 or width_end <= 0:
            continue
        expanding = width_end > width_start * 1.18
        if not expanding:
            continue
        upper_projection = _project_price(h1, h2, candles[-1]["index"])
        lower_projection = _project_price(l1, l2, candles[-1]["index"])
        current = candles[-1]["close"]
        state = "forming"
        bias = "volatile"
        markers = []
        if current > upper_projection + breakout_buffer:
            state = "breakout"
            bias = "bullish"
            markers.append(_build_marker("broadening-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
        elif current < lower_projection - breakout_buffer:
            state = "breakdown"
            bias = "bearish"
            markers.append(_build_marker("broadening-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
        confidence = _pattern_confidence(
            0.38,
            min(1.0, (width_end - width_start) / max(width_start, 1e-9)) * 0.22,
            min(1.0, max(volume_ratio - 1, 0)) * 0.04,
        )
        sp = sorted(seq, key=lambda p: p["index"])
        return {
            "primaryPattern": {
                "type": "broadening_formation",
                "label": PATTERN_LABELS["broadening_formation"],
                "bias": bias,
                "state": state,
                "confidence": confidence,
                "description": "Expanding swing amplitude with diverging support and resistance boundaries.",
            },
            "shapePivots": sp,
            "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
            "horizontalLevels": [],
            "markers": markers,
            "levels": {"support": smart_round_price(lower_projection), "resistance": smart_round_price(upper_projection)},
        }
    return None


def _detect_rounding_bottom_top(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    if len(candles) < 28:
        return None
    recent = candles[-28:]
    closes = [c["close"] for c in recent]
    mid_idx = len(closes) // 2
    left = closes[:mid_idx]
    right = closes[mid_idx:]
    if len(left) < 6 or len(right) < 6:
        return None
    current = closes[-1]

    # Rounding bottom: left slope down, right slope up, center forms trough.
    left_slope = _linear_slope(left)
    right_slope = _linear_slope(right)
    min_price = min(closes)
    max_price = max(closes)
    trough_idx = closes.index(min_price)
    if left_slope < 0 and right_slope > 0 and abs(trough_idx - mid_idx) <= len(closes) * 0.25:
        neckline = mean(sorted(closes[-10:])[-3:])
        depth = neckline - min_price
        if depth > min_move:
            state = "forming"
            markers = []
            if current > neckline + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("round-bottom-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
            confidence = _pattern_confidence(
                0.36,
                min(1.0, abs(left_slope) / max(min_move / max(current, 1e-9), 1e-9)) * 0.12,
                min(1.0, abs(right_slope) / max(min_move / max(current, 1e-9), 1e-9)) * 0.12,
                min(1.0, depth / max(min_move * 3.0, 1e-9)) * 0.14,
            )
            return {
                "primaryPattern": {
                    "type": "rounding_bottom",
                    "label": PATTERN_LABELS["rounding_bottom"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Gradual U-shaped base transitioning from selling pressure to accumulation.",
                },
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("round-bottom-neckline", "Neckline", "#ef4444", "dashed", neckline, recent[0]["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {"support": smart_round_price(min_price), "resistance": smart_round_price(neckline)},
            }

    # Rounding top: left slope up, right slope down, center forms crest.
    crest_idx = closes.index(max_price)
    if left_slope > 0 and right_slope < 0 and abs(crest_idx - mid_idx) <= len(closes) * 0.25:
        neckline = mean(sorted(closes[-10:])[:3])
        height = max_price - neckline
        if height > min_move:
            state = "forming"
            markers = []
            if current < neckline - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("round-top-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
            confidence = _pattern_confidence(
                0.36,
                min(1.0, abs(left_slope) / max(min_move / max(current, 1e-9), 1e-9)) * 0.12,
                min(1.0, abs(right_slope) / max(min_move / max(current, 1e-9), 1e-9)) * 0.12,
                min(1.0, height / max(min_move * 3.0, 1e-9)) * 0.14,
            )
            return {
                "primaryPattern": {
                    "type": "rounding_top",
                    "label": PATTERN_LABELS["rounding_top"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Gradual dome-like top transitioning from buying pressure to distribution.",
                },
                "trendLines": [],
                "horizontalLevels": [
                    _build_horizontal_level("round-top-neckline", "Neckline", "#3b82f6", "dashed", neckline, recent[0]["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {"support": smart_round_price(neckline), "resistance": smart_round_price(max_price)},
            }
    return None


def _detect_symmetrical_triangle(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    for start in range(len(pivots) - 3, -1, -1):
        sequence = pivots[start:start + 4]
        kinds = [p["kind"] for p in sequence]
        if kinds not in (["high", "low", "high", "low"], ["low", "high", "low", "high"]):
            continue
        highs = sorted([p for p in sequence if p["kind"] == "high"], key=lambda p: p["index"])
        lows = sorted([p for p in sequence if p["kind"] == "low"], key=lambda p: p["index"])
        if len(highs) < 2 or len(lows) < 2:
            continue
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]
        lower_highs = h2["price"] < h1["price"]
        higher_lows = l2["price"] > l1["price"]
        if not (lower_highs and higher_lows):
            continue
        high_drop = h1["price"] - h2["price"]
        low_rise = l2["price"] - l1["price"]
        if high_drop < min_move * 0.3 or low_rise < min_move * 0.3:
            continue
        current = candles[-1]["close"]
        upper_proj = _project_price(h1, h2, candles[-1]["index"])
        lower_proj = _project_price(l1, l2, candles[-1]["index"])
        state = "forming"
        bias = "neutral"
        markers = []
        if current > upper_proj + breakout_buffer:
            state = "breakout"
            bias = "bullish"
            markers.append(_build_marker("sym-tri-breakout", "Breakout", candles[-1]["time"], current, "#22c55e", "bullish"))
        elif current < lower_proj - breakout_buffer:
            state = "breakdown"
            bias = "bearish"
            markers.append(_build_marker("sym-tri-breakdown", "Breakdown", candles[-1]["time"], current, "#ef4444", "bearish"))
        convergence = 1 - (abs(upper_proj - lower_proj) / max(h1["price"] - l1["price"], 1e-9))
        confidence = _pattern_confidence(0.40, min(1.0, convergence) * 0.22, min(1.0, max(volume_ratio - 1, 0)) * 0.04)
        sp = sorted(sequence, key=lambda p: p["index"])
        return {
            "primaryPattern": {
                "type": "symmetrical_triangle",
                "label": PATTERN_LABELS["symmetrical_triangle"],
                "bias": bias,
                "state": state,
                "confidence": confidence,
                "description": "Converging highs and lows forming a symmetrical squeeze pattern.",
            },
            "shapePivots": sp,
            "trendLines": _build_pattern_boundary_lines(sp, candles, _calculate_atr(candles)),
            "horizontalLevels": [],
            "markers": markers,
            "levels": {"support": smart_round_price(lower_proj), "resistance": smart_round_price(upper_proj)},
        }
    return None


def _detect_v_reversal(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    if len(candles) < 10:
        return None
    window_size = min(20, len(candles))
    window = candles[-window_size:]
    atr = _calculate_atr(candles) if len(candles) >= 14 else 0
    if atr <= 0:
        return None
    threshold = atr * 1.5

    low_idx = min(range(len(window)), key=lambda i: window[i]["low"])
    high_idx = max(range(len(window)), key=lambda i: window[i]["high"])

    low_c = window[low_idx]
    high_c = window[high_idx]
    first_c = window[0]
    last_c = window[-1]

    drop_to_low = first_c["close"] - low_c["low"]
    rise_from_low = last_c["close"] - low_c["low"]
    if drop_to_low > threshold and rise_from_low > threshold and low_idx > 1 and low_idx < len(window) - 2:
        sp = [
            {"time": first_c["time"], "price": first_c["close"], "index": first_c["index"], "kind": "high"},
            {"time": low_c["time"], "price": low_c["low"], "index": low_c["index"], "kind": "low"},
            {"time": last_c["time"], "price": last_c["close"], "index": last_c["index"], "kind": "high"},
        ]
        confidence = _pattern_confidence(0.38, min(1.0, drop_to_low / (threshold * 2)) * 0.2, min(1.0, rise_from_low / (threshold * 2)) * 0.2)
        return {
            "primaryPattern": {
                "type": "v_reversal_bottom",
                "label": PATTERN_LABELS["v_reversal_bottom"],
                "bias": "bullish",
                "state": "confirmed",
                "confidence": confidence,
                "description": "Sharp decline followed by immediate sharp recovery.",
            },
            "shapePivots": sp,
            "trendLines": _build_pattern_boundary_lines(sp, candles, atr),
            "horizontalLevels": [],
            "markers": [],
            "levels": {"support": smart_round_price(low_c["low"]), "resistance": smart_round_price(max(first_c["close"], last_c["close"]))},
        }

    rise_to_high = high_c["high"] - first_c["close"]
    drop_from_high = high_c["high"] - last_c["close"]
    if rise_to_high > threshold and drop_from_high > threshold and high_idx > 1 and high_idx < len(window) - 2:
        sp = [
            {"time": first_c["time"], "price": first_c["close"], "index": first_c["index"], "kind": "low"},
            {"time": high_c["time"], "price": high_c["high"], "index": high_c["index"], "kind": "high"},
            {"time": last_c["time"], "price": last_c["close"], "index": last_c["index"], "kind": "low"},
        ]
        confidence = _pattern_confidence(0.38, min(1.0, rise_to_high / (threshold * 2)) * 0.2, min(1.0, drop_from_high / (threshold * 2)) * 0.2)
        return {
            "primaryPattern": {
                "type": "v_reversal_top",
                "label": PATTERN_LABELS["v_reversal_top"],
                "bias": "bearish",
                "state": "confirmed",
                "confidence": confidence,
                "description": "Sharp rise followed by immediate sharp decline.",
            },
            "shapePivots": sp,
            "trendLines": _build_pattern_boundary_lines(sp, candles, atr),
            "horizontalLevels": [],
            "markers": [],
            "levels": {"support": smart_round_price(min(first_c["close"], last_c["close"])), "resistance": smart_round_price(high_c["high"])},
        }
    return None


def _detect_vcp(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    """Detect Volatility Contraction Pattern (VCP) — bullish or bearish.

    Bullish VCP: A(high) → B(low) → C(high) → D(low), where volatility
    contracts (C-D range < A-B range), A≈C on a straight line, B is the
    absolute low, and D is a higher low.

    Bearish VCP is the mirror image.
    """
    if len(pivots) < 4 or len(candles) < 20:
        return None

    atr_v = _calculate_atr(candles)
    if atr_v <= 0:
        return None
    current_price = candles[-1]["close"]

    for start in range(len(pivots) - 3, -1, -1):
        seq = pivots[start:start + 4]
        kinds = [p["kind"] for p in seq]

        if kinds == ["high", "low", "high", "low"]:
            a, b, c, d = seq
            avg_bar = _avg_bar_length(candles, a["index"], d["index"] + 1)
            if avg_bar <= 0:
                continue
            contraction_ab = a["price"] - b["price"]
            contraction_cd = c["price"] - d["price"]
            if contraction_ab <= 0 or contraction_cd <= 0:
                continue
            if not (
                abs(a["price"] - c["price"]) <= avg_bar
                and b["price"] < min(a["price"], c["price"], d["price"])
                and d["price"] < min(a["price"], c["price"])
                and d["price"] > b["price"]
                and abs(b["price"] - d["price"]) >= avg_bar * 0.8
                and contraction_cd < contraction_ab * 0.85
            ):
                continue
            if _is_level_breached(candles, c["price"], c["index"], "above"):
                continue
            if _is_level_breached(candles, d["price"], d["index"], "below"):
                continue
            if current_price > c["price"]:
                continue

            contraction_ratio = contraction_cd / max(contraction_ab, 1e-9)
            tightness = max(0.0, 1.0 - contraction_ratio)
            span_bars = d["index"] - a["index"]
            confidence = _pattern_confidence(
                0.44,
                tightness * 0.22,
                min(1.0, span_bars / 30.0) * 0.14,
                min(1.0, max(volume_ratio - 1, 0)) * 0.04,
            )
            state = "forming"
            markers = []
            if current_price > c["price"] + breakout_buffer:
                state = "breakout"
                markers.append(_build_marker("vcp-bull-breakout", "Breakout", candles[-1]["time"], current_price, "#22c55e", "bullish"))

            resistance = mean([a["price"], c["price"]])
            sp = list(seq)
            return {
                "primaryPattern": {
                    "type": "bullish_vcp",
                    "label": PATTERN_LABELS["bullish_vcp"],
                    "bias": "bullish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Volatility contracting with higher lows — bullish base building.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, atr_v),
                "horizontalLevels": [
                    _build_horizontal_level("vcp-resistance", "Resistance", "#ef4444", "dashed", resistance, a["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(d["price"]),
                    "resistance": smart_round_price(resistance),
                },
            }

        if kinds == ["low", "high", "low", "high"]:
            a, b, c, d = seq
            avg_bar = _avg_bar_length(candles, a["index"], d["index"] + 1)
            if avg_bar <= 0:
                continue
            contraction_ab = b["price"] - a["price"]
            contraction_cd = d["price"] - c["price"]
            if contraction_ab <= 0 or contraction_cd <= 0:
                continue
            if not (
                abs(a["price"] - c["price"]) <= avg_bar
                and b["price"] > max(a["price"], c["price"], d["price"])
                and d["price"] > max(a["price"], c["price"])
                and d["price"] < b["price"]
                and abs(b["price"] - d["price"]) >= avg_bar * 0.8
                and contraction_cd < contraction_ab * 0.85
            ):
                continue
            if _is_level_breached(candles, c["price"], c["index"], "below"):
                continue
            if _is_level_breached(candles, d["price"], d["index"], "above"):
                continue
            if current_price < c["price"]:
                continue

            contraction_ratio = contraction_cd / max(contraction_ab, 1e-9)
            tightness = max(0.0, 1.0 - contraction_ratio)
            span_bars = d["index"] - a["index"]
            confidence = _pattern_confidence(
                0.44,
                tightness * 0.22,
                min(1.0, span_bars / 30.0) * 0.14,
                min(1.0, max(volume_ratio - 1, 0)) * 0.04,
            )
            state = "forming"
            markers = []
            if current_price < c["price"] - breakout_buffer:
                state = "breakdown"
                markers.append(_build_marker("vcp-bear-breakdown", "Breakdown", candles[-1]["time"], current_price, "#ef4444", "bearish"))

            support = mean([a["price"], c["price"]])
            sp = list(seq)
            return {
                "primaryPattern": {
                    "type": "bearish_vcp",
                    "label": PATTERN_LABELS["bearish_vcp"],
                    "bias": "bearish",
                    "state": state,
                    "confidence": confidence,
                    "description": "Volatility contracting with lower highs — bearish distribution.",
                },
                "shapePivots": sp,
                "trendLines": _build_pattern_boundary_lines(sp, candles, atr_v),
                "horizontalLevels": [
                    _build_horizontal_level("vcp-support", "Support", "#22c55e", "dashed", support, a["time"], candles[-1]["time"], candles=candles),
                ],
                "markers": markers,
                "levels": {
                    "support": smart_round_price(support),
                    "resistance": smart_round_price(d["price"]),
                },
            }
    return None


def _detect_trend_structure(pivots, candles, tolerance, min_move, breakout_buffer, volume_ratio):
    """Fallback: detect HH-HL (bullish) or LH-LL (bearish) from the most recent pivots."""
    recent = pivots[-4:] if len(pivots) >= 4 else pivots[-3:] if len(pivots) >= 3 else []
    if len(recent) < 3:
        return None
    highs = [p for p in recent if p["kind"] == "high"]
    lows = [p for p in recent if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        if len(recent) >= 3:
            highs = highs or [max(recent, key=lambda p: p["price"])]
            lows = lows or [min(recent, key=lambda p: p["price"])]
        if len(highs) < 2 or len(lows) < 2:
            return None

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    hh_hl = h2["price"] > h1["price"] and l2["price"] > l1["price"]
    lh_ll = h2["price"] < h1["price"] and l2["price"] < l1["price"]

    if not hh_hl and not lh_ll:
        return None

    sp = sorted(recent, key=lambda p: p["index"])
    atr = _calculate_atr(candles) if len(candles) >= 14 else 0
    if hh_hl:
        ptype = "bullish_trend_structure"
        bias = "bullish"
        desc = "Higher highs and higher lows forming a bullish trend structure."
    else:
        ptype = "bearish_trend_structure"
        bias = "bearish"
        desc = "Lower highs and lower lows forming a bearish trend structure."

    confidence = _pattern_confidence(0.34, 0.10, min(1.0, max(volume_ratio - 1, 0)) * 0.04)
    return {
        "primaryPattern": {
            "type": ptype,
            "label": PATTERN_LABELS[ptype],
            "bias": bias,
            "state": "forming",
            "confidence": confidence,
            "description": desc,
        },
        "shapePivots": sp,
        "trendLines": _build_pattern_boundary_lines(sp, candles, atr),
        "horizontalLevels": [],
        "markers": [],
        "levels": {"support": smart_round_price(l2["price"]), "resistance": smart_round_price(h2["price"])},
    }


def _compute_clustered_levels(candles, pivots, atr):
    """Volume-weighted agglomerative pivot clustering with touch validation and polarity flip.

    Returns (support_clusters, resistance_clusters) where each cluster is a dict:
    {center, first_time, last_time, touches, volume_score, flipped, kind}
    """
    if not pivots or not candles:
        recent = candles[-20:] if candles else []
        s = min((c["low"] for c in recent), default=0)
        r = max((c["high"] for c in recent), default=0)
        t0 = recent[0]["time"] if recent else 0
        return (
            [{"center": smart_round_price(s), "first_time": t0, "last_time": t0, "touches": 1, "volume_score": 1.0, "flipped": False, "kind": "support"}],
            [{"center": smart_round_price(r), "first_time": t0, "last_time": t0, "touches": 1, "volume_score": 1.0, "flipped": False, "kind": "resistance"}],
        )

    cluster_tolerance = max(atr * 0.6, 1e-9)

    enriched = []
    for p in pivots:
        vol = _volume_at_index(candles, p["index"])
        enriched.append({
            "price": p["price"],
            "volume": max(vol, 1.0),
            "time": p["time"],
            "index": p["index"],
            "kind": p.get("kind", "point"),
        })
    enriched.sort(key=lambda x: x["price"])

    clusters = []
    for ep in enriched:
        merged = False
        best_dist = cluster_tolerance
        best_cl = None
        for cl in clusters:
            dist = abs(ep["price"] - cl["center"])
            if dist < best_dist:
                best_dist = dist
                best_cl = cl
        if best_cl is not None:
            total_vol = best_cl["total_volume"] + ep["volume"]
            best_cl["center"] = (
                best_cl["center"] * best_cl["total_volume"] + ep["price"] * ep["volume"]
            ) / total_vol
            best_cl["total_volume"] = total_vol
            best_cl["pivots"].append(ep)
            best_cl["first_time"] = min(best_cl["first_time"], ep["time"])
            best_cl["last_time"] = max(best_cl["last_time"], ep["time"])
            best_cl["first_index"] = min(best_cl["first_index"], ep["index"])
            best_cl["last_index"] = max(best_cl["last_index"], ep["index"])
            merged = True
        if not merged:
            clusters.append({
                "center": ep["price"],
                "total_volume": ep["volume"],
                "pivots": [ep],
                "first_time": ep["time"],
                "last_time": ep["time"],
                "first_index": ep["index"],
                "last_index": ep["index"],
            })

    touch_zone = max(atr * 0.3, 1e-9)
    current = candles[-1]["close"]
    latest_idx = candles[-1]["index"]

    results = []
    for cl in clusters:
        if len(cl["pivots"]) < 1:
            continue

        valid_touches = _count_level_touches(candles, cl["center"], touch_zone)
        valid_touches = max(valid_touches, len(cl["pivots"]))

        high_pivots = sum(1 for p in cl["pivots"] if p["kind"] == "high")
        low_pivots = sum(1 for p in cl["pivots"] if p["kind"] == "low")
        total_pivots = max(high_pivots + low_pivots, 1)
        if high_pivots / total_pivots > 0.6:
            natural_kind = "resistance"
        elif low_pivots / total_pivots > 0.6:
            natural_kind = "support"
        else:
            natural_kind = "resistance" if cl["center"] >= current else "support"

        flipped = False
        if natural_kind == "support" and current < cl["center"] - touch_zone:
            natural_kind = "resistance"
            flipped = True
        elif natural_kind == "resistance" and current > cl["center"] + touch_zone:
            natural_kind = "support"
            flipped = True

        recency = max(0.3, 1.0 - (latest_idx - cl["last_index"]) / max(latest_idx, 1))
        avg_vol = cl["total_volume"] / max(len(cl["pivots"]), 1)
        vol_baseline = mean([_volume_at_index(candles, c["index"]) for c in candles[-20:]]) if len(candles) >= 20 else 1.0
        vol_score = avg_vol / max(vol_baseline, 1.0)

        score = valid_touches * (0.5 + vol_score * 0.5) * recency

        results.append({
            "center": smart_round_price(cl["center"]),
            "first_time": cl["first_time"],
            "last_time": cl["last_time"],
            "touches": valid_touches,
            "volume_score": round(vol_score, 3),
            "flipped": flipped,
            "kind": natural_kind,
            "score": score,
        })

    supports = sorted(
        [r for r in results if r["kind"] == "support"],
        key=lambda r: r["score"], reverse=True,
    )[:3]
    resistances = sorted(
        [r for r in results if r["kind"] == "resistance"],
        key=lambda r: r["score"], reverse=True,
    )[:3]

    supports = sorted(supports, key=lambda r: r["center"], reverse=True)
    resistances = sorted(resistances, key=lambda r: r["center"])

    if not supports:
        recent = candles[-20:]
        t0 = recent[0]["time"]
        supports = [{"center": smart_round_price(min(c["low"] for c in recent)), "first_time": t0, "last_time": t0, "touches": 1, "volume_score": 1.0, "flipped": False, "kind": "support", "score": 0}]
    if not resistances:
        recent = candles[-20:]
        t0 = recent[0]["time"]
        resistances = [{"center": smart_round_price(max(c["high"] for c in recent)), "first_time": t0, "last_time": t0, "touches": 1, "volume_score": 1.0, "flipped": False, "kind": "resistance", "score": 0}]

    return supports, resistances



_TIMEFRAME_MAX_SCAN = {
    "1m": 120, "3m": 120, "5m": 120, "15m": 120,
    "30m": 100, "1h": 100, "4h": 100,
    "1d": 80, "1w": 80, "1M": 60, "6M": 80, "1Y": 80, "max": 80,
}


def _build_auto_trendlines(candles, pivots, atr, timeframe=None):
    """Delegate to trendline_engine.build_auto_trendlines."""
    return _trendline_build_auto_trendlines(candles, pivots, atr, timeframe=timeframe)


_BOUNDARY_PATTERN_TYPES = frozenset({
    "ascending_triangle", "descending_triangle",
    "head_and_shoulders", "inverse_head_and_shoulders",
    "triple_top", "triple_bottom",
    "rectangle_range", "cup_and_handle",
    # Patterns whose detector outputs only horizontalLevels (neckline/support/resistance):
    "double_top", "double_bottom",
    "rounding_top", "rounding_bottom",
})


def _synthesize_boundary_trendlines(result):
    """Convert a pattern's horizontalLevels into ``source="pattern"`` trendLines.

    Used so that the Outline toggle can draw the neckline / flat edge / etc.
    for every pattern that describes its structure via horizontal levels.
    Mutates *result* in place and returns it.
    """
    if not result:
        return result
    primary = (result.get("primaryPattern") or {})
    if primary.get("type") not in _BOUNDARY_PATTERN_TYPES:
        return result

    existing_tl = list(result.get("trendLines") or [])
    existing_ids = {t.get("id") for t in existing_tl}
    for h_level in (result.get("horizontalLevels") or []):
        lid = h_level.get("id") or ""
        if lid.startswith("cluster-"):
            continue
        pts = h_level.get("points") or []
        if len(pts) != 2:
            continue
        tl_id = f"boundary-{lid}"
        if tl_id in existing_ids:
            continue
        existing_tl.append({
            "id": tl_id,
            "label": h_level.get("label", ""),
            "color": h_level.get("color", "#888"),
            "style": "solid",
            "source": "pattern",
            "points": pts,
        })
        existing_ids.add(tl_id)
    result["trendLines"] = existing_tl
    return result


def detect_chart_patterns(klines, timeframe="1h", trade_type="swing"):
    candles = normalize_klines(klines)
    max_scan = _TIMEFRAME_MAX_SCAN.get(timeframe, 100)
    if len(candles) > max_scan:
        candles = candles[-max_scan:]
        for idx, candle in enumerate(candles):
            candle["index"] = idx

    if len(candles) < 20:
        return {
            "version": 1,
            "generatedFrom": "ohlcv",
            "timeframe": timeframe,
            "tradeType": trade_type,
            "primaryPattern": {
                "type": "no_clear_pattern",
                "label": PATTERN_LABELS["no_clear_pattern"],
                "bias": "neutral",
                "state": "insufficient_data",
                "confidence": 0.0,
                "description": "Not enough candles to detect a stable chart structure.",
            },
            "swingPoints": [],
            "trendLines": [],
            "horizontalLevels": [],
            "markers": [],
            "levels": {"support": None, "resistance": None},
            "stats": {"pivotCount": 0, "atr": 0.0, "volumeRatio": 1.0},
        }

    atr, tolerance, min_move, breakout_buffer = _calculate_pattern_tolerances(candles, trade_type)

    pivot_lr = {"1m": 2, "3m": 2, "5m": 2, "15m": 3, "30m": 3, "1h": 4, "4h": 4}.get(timeframe, 4)
    pivots = _detect_pivots(candles, left=pivot_lr, right=pivot_lr, min_move=min_move * 0.5, include_tentative=True)
    display_pivots = pivots
    volume_ratio = _calculate_volume_ratio(candles)

    detectors = [
        _detect_double_top,
        _detect_double_bottom,
        _detect_triple_top_bottom,
        _detect_head_and_shoulders,
        _detect_cup_and_handle,
        _detect_vcp,
        lambda points, bars, tol, move, break_buf, vol_ratio: _detect_triangle(points, bars, "ascending_triangle", tol, move, break_buf, vol_ratio),
        lambda points, bars, tol, move, break_buf, vol_ratio: _detect_triangle(points, bars, "descending_triangle", tol, move, break_buf, vol_ratio),
        _detect_symmetrical_triangle,
        _detect_wedge,
        _detect_channel,
        _detect_flag_and_pennant,
        _detect_broadening_formation,
        _detect_rectangle_range,
        _detect_rounding_bottom_top,
        _detect_v_reversal,
        _detect_trend_structure,
    ]

    recent_pivots_for_detection = pivots[-12:] if len(pivots) > 12 else pivots

    pattern_result = None
    detector_candidates = []
    latest_index = candles[-1]["index"] if candles else 0
    for detector in detectors:
        result = detector(recent_pivots_for_detection, candles, tolerance, min_move, breakout_buffer, volume_ratio)
        if not result:
            continue
        detector_candidates.append((result, _pattern_selection_score(result, latest_index)))

    detector_candidates = _dedupe_detector_candidates(detector_candidates, latest_index)

    if detector_candidates:
        def _last_shape_index(cand):
            sp = (cand or {}).get("shapePivots") or []
            return sp[-1]["index"] if sp else 0

        detector_candidates.sort(
            key=lambda item: (
                item[1],
                float(((item[0] or {}).get("primaryPattern") or {}).get("confidence") or 0.0),
            ),
            reverse=True,
        )
        recency_cutoff = len(candles) * 0.5
        recent_candidates = [
            (cand, _score) for cand, _score in detector_candidates
            if _last_shape_index(cand) >= recency_cutoff
        ]
        if recent_candidates:
            recent_candidates.sort(
                key=lambda item: (_last_shape_index(item[0]), item[1]),
                reverse=True,
            )
            pattern_result = recent_candidates[0][0]
        elif detector_candidates:
            pattern_result = detector_candidates[0][0]

    s_clusters, r_clusters = _compute_clustered_levels(candles, pivots, atr)

    # --- Decoupled trendline detection: always run independently ---
    auto_trends = _build_auto_trendlines(candles, pivots, atr, timeframe=timeframe)
    for tl in auto_trends:
        tl["source"] = "auto"

    if not pattern_result:
        pattern_result = {
            "primaryPattern": {
                "type": "no_clear_pattern",
                "label": PATTERN_LABELS["no_clear_pattern"],
                "bias": "neutral",
                "state": "monitor",
                "confidence": 0.0,
                "description": "Recent swings do not form a clean high-confidence pattern yet.",
            },
            "trendLines": [],
            "horizontalLevels": [],
            "markers": [],
            "levels": {
                "support": None,
                "resistance": None,
            },
        }

    # Tag pattern-sourced trendlines
    for tl in (pattern_result.get("trendLines") or []):
        if "source" not in tl:
            tl["source"] = "pattern"

    # Merge: auto trendlines + pattern trendlines, dedup by endpoint proximity.
    # Both price AND time must match — two lines at different bar ranges that
    # happen to land on the same price are NOT duplicates.
    pattern_tls = list(pattern_result.get("trendLines") or [])
    merged_tls = list(pattern_tls)  # pattern lines first (higher priority)
    price_threshold = max(atr * 0.3, 1e-9)
    # Time threshold: ~1 bar in seconds. Use candle interval if known.
    if len(candles) >= 2:
        bar_seconds = max(candles[1]["time"] - candles[0]["time"], 1)
    else:
        bar_seconds = 1
    time_threshold = bar_seconds * 2  # 2 bars of tolerance
    for atl in auto_trends:
        atl_pts = atl.get("points") or []
        if len(atl_pts) < 2:
            continue
        # Check if an auto line overlaps with any pattern line on both axes
        duplicate = False
        for ptl in pattern_tls:
            ptl_pts = ptl.get("points") or []
            if len(ptl_pts) < 2:
                continue
            price_close = (
                abs(atl_pts[0]["price"] - ptl_pts[0]["price"]) < price_threshold
                and abs(atl_pts[-1]["price"] - ptl_pts[-1]["price"]) < price_threshold
            )
            time_close = (
                abs(atl_pts[0]["time"] - ptl_pts[0]["time"]) < time_threshold
                and abs(atl_pts[-1]["time"] - ptl_pts[-1]["time"]) < time_threshold
            )
            if price_close and time_close:
                duplicate = True
                break
        if not duplicate:
            merged_tls.append(atl)
    pattern_result["trendLines"] = merged_tls

    end_time = candles[-1]["time"]
    hl = list(pattern_result.get("horizontalLevels") or [])
    for idx, cl in enumerate(s_clusters):
        sid = f"cluster-s{idx + 1}"
        if not any(abs(h.get("price", 0) - cl["center"]) < atr * 0.3 for h in hl):
            label = f"S{idx + 1}"
            if cl["flipped"]:
                label += " ↕"
            level = _build_horizontal_level(sid, label, "#22c55e", "dashed", cl["center"], cl["first_time"], end_time, candles=candles)
            level["touches"] = cl["touches"]
            level["flipped"] = cl["flipped"]
            level["startTime"] = cl["first_time"]
            hl.append(level)
    for idx, cl in enumerate(r_clusters):
        rid = f"cluster-r{idx + 1}"
        if not any(abs(h.get("price", 0) - cl["center"]) < atr * 0.3 for h in hl):
            label = f"R{idx + 1}"
            if cl["flipped"]:
                label += " ↕"
            level = _build_horizontal_level(rid, label, "#ef4444", "dashed", cl["center"], cl["first_time"], end_time, candles=candles)
            level["touches"] = cl["touches"]
            level["flipped"] = cl["flipped"]
            level["startTime"] = cl["first_time"]
            hl.append(level)
    pattern_result["horizontalLevels"] = hl

    # Derive levels.support/resistance from the final horizontalLevels so that
    # every consumer (AI text, chart "S"/"R" lines, frontend buttons) sees the
    # same values.  Pick the strongest (most touches) support/resistance from
    # the levels that are actually drawn on the chart.
    current = candles[-1]["close"]
    best_support = None
    best_resistance = None
    for h in hl:
        hp = h.get("price", 0)
        ht = h.get("touches", 0)
        h_label = (h.get("label") or "").lower()
        # Classify by label: S*, Support, Neckline in bearish patterns → support
        # R*, Resistance, Neckline in bullish patterns → resistance
        is_support = h_label.startswith("s") or "support" in h_label
        is_resistance = h_label.startswith("r") or "resistance" in h_label
        if "neckline" in h_label:
            h_id = (h.get("id") or "").lower()
            if "ihs" in h_id or "bottom" in h_id:
                is_resistance = True
            else:
                is_support = True
        if is_support and hp <= current:
            if best_support is None or ht > best_support[1] or (ht == best_support[1] and hp > best_support[0]):
                best_support = (hp, ht)
        if is_resistance and hp >= current:
            if best_resistance is None or ht > best_resistance[1] or (ht == best_resistance[1] and hp < best_resistance[0]):
                best_resistance = (hp, ht)
    # Only populate levels from lines that are actually visible on the chart.
    # If no same-side horizontalLevel exists, set to None rather than falling
    # back to _recent_levels (which could produce a value not drawn anywhere).
    pattern_result["levels"]["support"] = smart_round_price(best_support[0]) if best_support else None
    pattern_result["levels"]["resistance"] = smart_round_price(best_resistance[0]) if best_resistance else None

    # Ensure pattern-specific horizontal boundaries (e.g. triangle flat edge,
    # neckline) appear as trendLines so the Outline toggle shows them.
    _synthesize_boundary_trendlines(pattern_result)

    swing_points = []
    lookback_bars = min(300, len(candles))
    min_index = max(len(candles) - lookback_bars, 0)
    recent_display = [pivot for pivot in display_pivots if pivot["index"] >= min_index]
    recent_pivots = [pivot for pivot in pivots if pivot["index"] >= min_index]
    for idx, pivot in enumerate(recent_display, start=1):
        entry = {
            "index": pivot["index"],
            "time": pivot["time"],
            "price": smart_round_price(pivot["price"]),
            "kind": pivot["kind"],
            "label": f"{'H' if pivot['kind'] == 'high' else 'L'}{idx}",
        }
        if pivot.get("tentative"):
            entry["tentative"] = True
        swing_points.append(entry)

    primary = pattern_result["primaryPattern"]
    summary = primary["label"]
    if primary["type"] != "no_clear_pattern":
        summary = f"{primary['label']} ({primary['state']}, {int(primary['confidence'] * 100)}%)"

    shapes = _build_shapes_from_pattern_result(pattern_result, recent_pivots, tolerance, min_move, candles=candles)

    all_candidates = []
    primary_candidate_entry = None
    for cand, score in detector_candidates:
        cp = (cand.get("primaryPattern") or {})
        ctype = cp.get("type", "")
        if not ctype or ctype == "no_clear_pattern":
            continue
        # Synthesize boundary trendlines for each candidate so Outline works
        # regardless of which candidate the user selects (idempotent — safe on primary).
        _synthesize_boundary_trendlines(cand)
        cand_shapes = _build_shapes_from_pattern_result(cand, recent_pivots, tolerance, min_move, candles=candles)
        cand_trends = cand.get("trendLines", [])
        if not cand_shapes and not cand_trends:
            continue
        sp = (cand.get("shapePivots") or [])
        last_idx = sp[-1]["index"] if sp else 0
        entry = {
            "type": ctype,
            "label": cp.get("label", PATTERN_LABELS.get(ctype, ctype)),
            "bias": cp.get("bias", "neutral"),
            "confidence": cp.get("confidence", 0.0),
            "score": round(score, 4),
            "lastIndex": last_idx,
            "shapes": cand_shapes,
            "trendLines": cand_trends,
        }
        if cand is pattern_result:
            primary_candidate_entry = entry
        else:
            all_candidates.append(entry)
    all_candidates.sort(key=lambda c: c.get("lastIndex", 0), reverse=True)
    if primary_candidate_entry:
        all_candidates.insert(0, primary_candidate_entry)

    pattern_result["shapes"] = shapes
    pattern_result["allCandidates"] = all_candidates
    pattern_result = _apply_trendspider_pattern_theme(primary.get("type", ""), primary.get("bias", "neutral"), pattern_result)

    return {
        "version": 1,
        "generatedFrom": "ohlcv",
        "timeframe": timeframe,
        "tradeType": trade_type,
        "summary": summary,
        "primaryPattern": primary,
        "shapes": pattern_result.get("shapes", []),
        "swingPoints": swing_points,
        "trendLines": pattern_result.get("trendLines", []),
        "horizontalLevels": pattern_result.get("horizontalLevels", []),
        "markers": pattern_result.get("markers", []),
        "levels": pattern_result.get("levels", {"support": None, "resistance": None}),
        "stats": {
            "pivotCount": len(pivots),
            "atr": smart_round_price(atr),
            "volumeRatio": round(volume_ratio, 2),
        },
        "allCandidates": pattern_result.get("allCandidates", []),
    }


def _format_pivot_time(epoch):
    from datetime import datetime, timezone
    try:
        dt = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return str(epoch)


def format_pattern_engine_context(pattern_engine, *, current_price=None, current_time=None, all_candidates=None):
    if not pattern_engine:
        return ""

    primary = pattern_engine.get("primaryPattern") or {}
    stats = pattern_engine.get("stats") or {}
    timeframe = pattern_engine.get("timeframe") or ""

    lines = ["\n**PATTERN ENGINE (deterministic structure from raw OHLCV):**"]

    header_parts = []
    if timeframe:
        header_parts.append(f"Timeframe: {timeframe}")
    if current_time is not None:
        header_parts.append(f"Current time: {_format_pivot_time(current_time)}")
    if current_price is not None:
        header_parts.append(f"Current price: {current_price}")
    if header_parts:
        lines.append(f"- {' | '.join(header_parts)}")

    label = primary.get("label") or PATTERN_LABELS.get(primary.get("type"), "No Clear Pattern")
    bias = primary.get("bias", "neutral")
    state = primary.get("state", "monitor")
    lines.append(f"- Primary pattern: {label} | bias={bias} | state={state}")

    shapes = pattern_engine.get("shapes") or []
    for shape in shapes:
        shape_type = shape.get("type") or shape.get("label") or "shape"
        pts = shape.get("points") or shape.get("anchors") or []
        if pts:
            anchors = " → ".join(
                f"{p.get('role') or p.get('kind', '?')}={p['price']}(@{_format_pivot_time(p['time'])})"
                for p in pts
            )
            lines.append(f"- Shape [{shape_type}]: {anchors}")

    # Alternative candidates (skip primary at index 0)
    if all_candidates:
        for ci, cand in enumerate(all_candidates[1:4], start=1):
            c_label = cand.get("label", cand.get("type", "?"))
            c_bias = cand.get("bias", "neutral")
            c_state = cand.get("state", "forming")
            alt_line = f"- Alternative pattern [{ci}]: {c_label} | bias={c_bias} | state={c_state}"
            c_shapes = cand.get("shapes") or []
            if c_shapes:
                for cs in c_shapes[:1]:
                    pts = cs.get("points") or []
                    if pts:
                        anchors = " → ".join(
                            f"{p.get('role') or p.get('kind', '?')}={p['price']}(@{_format_pivot_time(p['time'])})"
                            for p in pts
                        )
                        alt_line += f"\n  Shape: {anchors}"
            lines.append(alt_line)

    # Trendline analysis — explicitly find one upper + one lower per source.
    # Pattern-source lines (if present) describe the detected formation's
    # structure; auto-source lines describe the best-scoring channel regardless
    # of whether a pattern was detected.
    trend_lines = pattern_engine.get("trendLines") or []

    def _classify_side(tl):
        """Return 'upper' / 'lower' / None based on id + label."""
        idl = (tl.get("id") or "").lower()
        lbl = (tl.get("label") or "").lower()
        text = f"{idl} {lbl}"
        if "upper" in text or "resist" in text:
            return "upper"
        if "lower" in text or "support" in text:
            return "lower"
        return None

    def _fmt_line(tl, side_hint):
        pts = tl.get("points") or []
        if len(pts) < 2:
            return None, None
        p1, p2 = pts[0], pts[-1]
        idx_diff = p2.get("time", 0) - p1.get("time", 0)
        if idx_diff == 0:
            return None, None
        slope = (p2["price"] - p1["price"]) / (idx_diff / 3600)
        label = side_hint.capitalize()
        touches = tl.get("touches")
        extra = f", {touches} touches" if touches else ""
        return f"{label}(slope={slope:+.2f}/hr, from {p1['price']} to {p2['price']}{extra})", slope

    def _find_pair(candidates):
        """Pick first upper + first lower from a list; returns (upper_tl, lower_tl)."""
        upper = next((t for t in candidates if _classify_side(t) == "upper"), None)
        lower = next((t for t in candidates if _classify_side(t) == "lower"), None)
        return upper, lower

    pattern_lines = [t for t in trend_lines if t.get("source") == "pattern"]
    pattern_upper, pattern_lower = _find_pair(pattern_lines)

    # Group all auto-source lines by pair (already ranked: pair 1 = best score).
    auto_by_pair: dict[int, list[dict]] = {}
    for t in trend_lines:
        if t.get("source") != "auto":
            continue
        pair = int(t.get("pair", 1))
        auto_by_pair.setdefault(pair, []).append(t)

    def _emit(upper, lower, kind):
        parts = []
        slopes = []
        if upper:
            part, slope = _fmt_line(upper, "upper")
            if part:
                parts.append(part)
                slopes.append(slope)
        if lower:
            part, slope = _fmt_line(lower, "lower")
            if part:
                parts.append(part)
                slopes.append(slope)
        if not parts:
            return None
        direction = ""
        if len(slopes) == 2:
            if slopes[0] * slopes[1] < 0 or abs(slopes[0]) > abs(slopes[1]) * 1.3:
                direction = " → converging"
            elif abs(slopes[0]) < abs(slopes[1]) * 0.7:
                direction = " → diverging"
            else:
                direction = " → parallel"
        return f"- {kind}: {' '.join(parts)}{direction}"

    # Naming matches UI: "轮廓" (Outline) = pattern boundary, "趋势线" (Trendline) = auto channels.
    # Emit outline + every auto pair (max 3). Multi-pair confluence is a strong
    # structural signal — AI should see all channels, not just the top one.
    outline_line = _emit(pattern_upper, pattern_lower, "Outline (pattern boundary)")
    if outline_line:
        lines.append(outline_line)
    for pair_num in sorted(auto_by_pair):
        auto_u, auto_l = _find_pair(auto_by_pair[pair_num])
        pair_line = _emit(auto_u, auto_l, f"Trendline (channel pair {pair_num})")
        if pair_line:
            lines.append(pair_line)

    # NOTE: levels.support/resistance is now derived from horizontalLevels
    # (the same data shown on the chart).  We omit the redundant "Engine levels"
    # line here because "Key price levels" below already exposes the same values
    # with richer context (touch counts, labels).  This prevents the AI from
    # citing a price that doesn't match any visible chart line.

    atr = stats.get("atr")
    vol_ratio = stats.get("volumeRatio")
    if atr:
        stat_parts = [f"ATR(14): {round(atr, 4)}"]
        if vol_ratio is not None:
            stat_parts.append(f"Volume ratio: {round(vol_ratio, 2)}")
        lines.append(f"- {' | '.join(stat_parts)}")

    swing_points = pattern_engine.get("swingPoints") or []
    if swing_points:
        recent = ", ".join(
            f"{sp['label']}={sp['price']}(@{_format_pivot_time(sp['time'])})"
            for sp in swing_points[-6:]
        )
        lines.append(f"- Recent swing pivots: {recent}")

    # Horizontal levels (support/resistance clusters with touch counts)
    h_levels = pattern_engine.get("horizontalLevels") or []
    if h_levels:
        level_parts = []
        for hl in h_levels[:6]:
            label = hl.get("label", "?")
            price = hl.get("price", 0)
            touches = hl.get("touches")
            part = f"{label}={price}"
            if touches:
                part += f"({touches} touches)"
            level_parts.append(part)
        lines.append(f"- Key price levels: {', '.join(level_parts)}")

    # Markers (breakout/breakdown signals) — expose price & time so AI can
    # measure distance between currentPrice and the trigger level (see
    # DISTANCE & FRESHNESS GUARD in prompt).
    markers = pattern_engine.get("markers") or []
    if markers:
        marker_parts = []
        for m in markers[:3]:
            label = m.get("text") or m.get("label") or ""
            bias = m.get("bias", "")
            price = m.get("price")
            time_val = m.get("time")
            part = f"{label}({bias})"
            if price is not None:
                part += f" @ price={price}"
            if time_val is not None:
                part += f", t={_format_pivot_time(time_val)}"
            marker_parts.append(part)
        lines.append(f"- Active signals: {'; '.join(marker_parts)}")

    lines.append("- Use this engine output as the primary chart-structure anchor for patternAnalysis and tradeReasoning.")
    return "\n".join(lines)
