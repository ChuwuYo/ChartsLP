"""ATR-adaptive tolerance resolver for LMW-style pattern detectors.

Provides a single source of truth for the threshold knobs that every
geometric pattern check uses (pivot grouping, symmetry, breakout buffer,
retest window). All values are derived from the average candle range so a
single set of multipliers works across 1m crypto and 1d equities without
hand-tuning per timeframe or asset class.

Design notes:
 - Coexists with the existing ``patterns._calculate_pattern_tolerances``
   (tuned for ``trade_type`` scalp/swing). Do NOT replace that in P3 — it
   is still the canonical path for the pre-LMW detectors. P1 detectors
   will opt into ``Tolerances`` here when rewritten.
 - ``asset_class`` is a stub parameter (defaults ``None``). P4 wires real
   multipliers for crypto / stock / forex / etf / commodity here.

Reference: pytrendline-ref's ``DEFAULT_CONFIG`` (``Reference/pytrendline-ref/
pytrendline/detect.py:8-44``) demonstrates the ``lambda candles: avg_range
* mult`` pattern. We take the same idea but add explicit per-TF and
per-asset-class overrides for surgical tuning.
"""
from __future__ import annotations

from typing import Literal


AssetClass = Literal["crypto", "stock", "forex", "etf", "commodity"]


# ---------------------------------------------------------------------------
# Base multipliers — each knob is `avg_range * multiplier`
# Values chosen to match roughly what existing detectors use today, giving
# a smooth migration path when P1 detectors opt in.
# ---------------------------------------------------------------------------

_BASE_MULTIPLIERS: dict[str, float] = {
    "pivot_symmetry_tol":   0.020,   # two peaks/troughs considered "level"
    "pivot_grouping":       0.010,   # pivots at same price cluster
    "breakout_buffer":      0.080,   # how far past neckline is a valid breakout
    "duplicate_price_tol":  0.020,   # same-level trendline dedupe distance
    "retest_tol":           0.050,   # pullback-to-neckline retest window
}

# Per-timeframe multiplier adjustments. Keys are (timeframe, knob) tuples.
# A value here REPLACES the base multiplier; missing keys fall back to base.
# Used sparingly — most tuning should stay in the base table. Reach for a
# per-TF override only when a specific TF genuinely needs to deviate.
_TF_OVERRIDES: dict[tuple[str, str], float] = {
    ("1m", "breakout_buffer"): 0.120,  # 1m noise; widen breakout confirmation
    ("3m", "breakout_buffer"): 0.110,
    ("1d", "pivot_symmetry_tol"): 0.015,  # daily pivots are cleaner, be stricter
    ("1w", "pivot_symmetry_tol"): 0.012,
}

# Per-asset-class multiplicative scaling on top of the resolved multiplier.
# TODO(P4): fill real values — crypto looser (higher vol), forex tighter.
_ASSET_CLASS_SCALE: dict[AssetClass, dict[str, float]] = {
    "crypto":    {"pivot_symmetry_tol": 1.0, "breakout_buffer": 1.0,  # TODO(P4)
                  "pivot_grouping": 1.0, "duplicate_price_tol": 1.0,
                  "retest_tol": 1.0},
    "stock":     {"pivot_symmetry_tol": 1.0, "breakout_buffer": 1.0,  # TODO(P4)
                  "pivot_grouping": 1.0, "duplicate_price_tol": 1.0,
                  "retest_tol": 1.0},
    "forex":     {"pivot_symmetry_tol": 1.0, "breakout_buffer": 1.0,  # TODO(P4)
                  "pivot_grouping": 1.0, "duplicate_price_tol": 1.0,
                  "retest_tol": 1.0},
    "etf":       {"pivot_symmetry_tol": 1.0, "breakout_buffer": 1.0,  # TODO(P4)
                  "pivot_grouping": 1.0, "duplicate_price_tol": 1.0,
                  "retest_tol": 1.0},
    "commodity": {"pivot_symmetry_tol": 1.0, "breakout_buffer": 1.0,  # TODO(P4)
                  "pivot_grouping": 1.0, "duplicate_price_tol": 1.0,
                  "retest_tol": 1.0},
}


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def avg_candle_range(candles: list[dict]) -> float:
    """Mean of ``high - low`` over the input candles.

    Used as the base unit for all adaptive thresholds. Guaranteed > 0
    (floor of 1e-9) so callers can divide by it safely.

    Note: uses bar range (high - low) rather than true range (which
    includes overnight gaps). On gap-bearing markets (stock / ETF) this
    under-estimates volatility by roughly 5-15%; acceptable for pattern
    tolerance scaling. Switch to true-range only if gap-aware thresholds
    become necessary.
    """
    if not candles:
        return 1e-9
    ranges = [c["high"] - c["low"] for c in candles if c["high"] >= c["low"]]
    if not ranges:
        return 1e-9
    mean = sum(ranges) / len(ranges)
    return max(mean, 1e-9)


def resolve_multiplier(
    knob: str,
    *,
    timeframe: str | None = None,
    asset_class: AssetClass | None = None,
) -> float:
    """Resolve the effective multiplier for ``knob`` under the given context.

    Precedence (last wins): base → per-TF override → per-asset-class scale.
    """
    base = _BASE_MULTIPLIERS.get(knob)
    if base is None:
        raise KeyError(f"unknown tolerance knob: {knob!r}")

    mult = base
    if timeframe is not None:
        mult = _TF_OVERRIDES.get((timeframe, knob), mult)

    if asset_class is not None:
        scale_map = _ASSET_CLASS_SCALE.get(asset_class)
        if scale_map is not None:
            mult *= scale_map.get(knob, 1.0)

    return mult


# ---------------------------------------------------------------------------
# Tolerances bundle — the handle every detector takes
# ---------------------------------------------------------------------------

class Tolerances:
    """Bundle of ATR-scaled thresholds for a single detect_chart_patterns call.

    Instantiate once per call (cheap — one pass over candles), then ask for
    whichever knob a detector needs::

        tol = Tolerances.for_candles(candles, timeframe="1h", asset_class="crypto")
        if abs(peak_a - peak_b) < tol.pivot_symmetry_tol:
            ...

    Keeping the object immutable means detectors can share it safely.
    """

    __slots__ = ("_avg_range", "_timeframe", "_asset_class")

    def __init__(
        self,
        avg_range: float,
        *,
        timeframe: str | None = None,
        asset_class: AssetClass | None = None,
    ) -> None:
        self._avg_range = max(float(avg_range), 1e-9)
        self._timeframe = timeframe
        self._asset_class = asset_class

    @classmethod
    def for_candles(
        cls,
        candles: list[dict],
        *,
        timeframe: str | None = None,
        asset_class: AssetClass | None = None,
    ) -> "Tolerances":
        return cls(
            avg_candle_range(candles),
            timeframe=timeframe,
            asset_class=asset_class,
        )

    def _scaled(self, knob: str) -> float:
        return self._avg_range * resolve_multiplier(
            knob, timeframe=self._timeframe, asset_class=self._asset_class,
        )

    @property
    def avg_range(self) -> float:
        return self._avg_range

    @property
    def pivot_symmetry_tol(self) -> float:
        return self._scaled("pivot_symmetry_tol")

    @property
    def pivot_grouping(self) -> float:
        return self._scaled("pivot_grouping")

    @property
    def breakout_buffer(self) -> float:
        return self._scaled("breakout_buffer")

    @property
    def duplicate_price_tol(self) -> float:
        return self._scaled("duplicate_price_tol")

    @property
    def retest_tol(self) -> float:
        return self._scaled("retest_tol")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Tolerances(avg_range={self._avg_range:.4g}, "
            f"tf={self._timeframe!r}, asset={self._asset_class!r})"
        )
