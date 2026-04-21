"""Perceptually Important Points (PIP) pivot detection.

Ports the PIP algorithm from Fu, Chung, Luk & Ng (2008):

  Fu, T., Chung, F., Luk, R. & Ng, V. (2008).
  "Stock time series pattern matching: Template-based vs. rule-based approaches."
  Engineering Applications of AI 20(3).

Reference implementation: neurotrader888/TechnicalAnalysisAutomation
(``Reference/taa-ref/perceptually_important.py``, MIT). The original uses
pandas + matplotlib for demos; this port keeps only the core algorithm and
strips all non-numpy dependencies.

Algorithm (greedy top-down)::

    1. pips = [first, last]
    2. while len(pips) < n_pips:
         for each adjacent pair (pips[k], pips[k+1]):
             find the single point in [pips[k], pips[k+1]] farthest from
             the chord connecting them (by PD / ED / VD)
         insert the point with the global max distance
    3. return pips

Complexity: O(n * n_pips). For our usage (n ≤ 500, n_pips ≤ 32) this is
well under 5 ms.

Bug fix vs reference: the neurotrader888 original initialises ``md_i = -1``
and ``insert_index = -1`` and unconditionally calls ``pips_x.insert(-1, -1)``
when no interior point remains (e.g. every segment is already linear). That
produces a garbage sentinel index in the output. This port breaks out of
the outer loop instead, returning a clean sorted list.

Public API::

    find_pips(series, n_pips, dist_measure)
    detect_pivots_pip_pd(candles, n_pips=None, timeframe=None)
    detect_pivots_hybrid(candles, fractal_pivots, timeframe=None)

The hybrid detector merges fractal-window pivots (from patterns._detect_pivots)
with PIP-PD results so short TFs get noise-robust extrema coverage without
losing the strict fractal pivots that downstream detectors depend on.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# PIP budget per TF.  Short TFs get fewer pips (noise-dense, fewer meaningful
# turning points); sparse daily/weekly get more.  Values chosen to mirror the
# trendline pivot budget in trendlines.py:_MAX_PIVOTS_BY_INTERVAL.
# ---------------------------------------------------------------------------

_N_PIPS_BY_TF: dict[str, int] = {
    "1m": 20, "3m": 20, "5m": 20, "15m": 24,
    "30m": 24, "1h": 28, "4h": 28,
    "1d": 32, "1w": 32, "1M": 32,
}
_DEFAULT_N_PIPS = 24

DIST_ED = 1  # Euclidean distance (sum of distances to both segment endpoints)
DIST_PD = 2  # Perpendicular distance (point-to-line); default for this port
DIST_VD = 3  # Vertical distance (|y - line_y|); fastest but time-axis sensitive

DistMeasure = Literal[1, 2, 3]


# ---------------------------------------------------------------------------
# Core PIP algorithm (numpy-only port)
# ---------------------------------------------------------------------------

def find_pips(
    series: np.ndarray,
    n_pips: int,
    dist_measure: DistMeasure = DIST_PD,
) -> tuple[list[int], list[float]]:
    """Select the ``n_pips`` most perceptually important points from ``series``.

    Returns ``(indices, values)`` — two parallel lists sorted by index.  The
    first and last samples are always included (they anchor the algorithm).
    """
    n = len(series)
    if n < 2:
        return (list(range(n)), [float(v) for v in series])

    if n_pips < 2:
        n_pips = 2
    n_pips = min(n_pips, n)

    pips_x: list[int] = [0, n - 1]
    pips_y: list[float] = [float(series[0]), float(series[-1])]

    # NOTE: the reference loop runs range(2, n_pips), which inserts n_pips - 2
    # additional points (giving a final length of n_pips). Keep identical bounds.
    for _ in range(2, n_pips):
        md = 0.0
        md_i = -1
        insert_index = -1

        # Scan each segment (pips_x[k], pips_x[k+1]) for the point whose
        # distance from the segment chord is maximal.
        for k in range(len(pips_x) - 1):
            left_x = pips_x[k]
            right_x = pips_x[k + 1]
            if right_x - left_x < 2:
                continue  # no interior points in this segment

            left_y = pips_y[k]
            right_y = pips_y[k + 1]
            time_diff = right_x - left_x
            price_diff = right_y - left_y
            slope = price_diff / time_diff
            intercept = left_y - left_x * slope
            denom_pd = (slope * slope + 1.0) ** 0.5

            for i in range(left_x + 1, right_x):
                y_i = float(series[i])
                if dist_measure == DIST_ED:
                    d1 = ((left_x - i) ** 2 + (left_y - y_i) ** 2) ** 0.5
                    d2 = ((right_x - i) ** 2 + (right_y - y_i) ** 2) ** 0.5
                    d = d1 + d2
                elif dist_measure == DIST_PD:
                    d = abs(slope * i + intercept - y_i) / denom_pd
                else:  # DIST_VD
                    d = abs(slope * i + intercept - y_i)

                if d > md:
                    md = d
                    md_i = i
                    insert_index = k + 1

        if md_i < 0 or insert_index < 0:
            break  # no more interior points available

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, float(series[md_i]))

    return pips_x, pips_y


# ---------------------------------------------------------------------------
# Pivot detector (wraps find_pips; returns the dict shape used throughout chartslp)
# ---------------------------------------------------------------------------

def _resolve_n_pips(timeframe: str | None, override: int | None) -> int:
    if override is not None:
        return max(2, override)
    if timeframe and timeframe in _N_PIPS_BY_TF:
        return _N_PIPS_BY_TF[timeframe]
    return _DEFAULT_N_PIPS


def detect_pivots_pip_pd(
    candles: list[dict],
    *,
    n_pips: int | None = None,
    timeframe: str | None = None,
) -> list[dict]:
    """Return swing pivots via PIP-PD on the high and low series.

    Running PIP independently on highs and lows (matching the approach in
    trendlines._detect_pivots_numdiff) preserves the high/low kind label that
    downstream pattern detectors rely on.

    When the resolved ``n_pips`` is ≥ len(candles), returns all bars as pivots
    (guaranteed no data loss).  When len(candles) < 5, returns [] to match the
    existing detector behaviour.
    """
    if not candles or len(candles) < 5:
        return []

    highs = np.asarray([c["high"] for c in candles], dtype=np.float64)
    lows = np.asarray([c["low"] for c in candles], dtype=np.float64)
    limit = _resolve_n_pips(timeframe, n_pips)
    limit = min(limit, len(candles))
    if limit < 3:
        return []

    high_idx, _ = find_pips(highs, limit, DIST_PD)
    low_idx, _ = find_pips(lows, limit, DIST_PD)

    pivots: list[dict] = []
    for idx in high_idx:
        pivots.append({
            "index": int(idx),
            "time": candles[idx]["time"],
            "price": float(highs[idx]),
            "kind": "high",
        })
    for idx in low_idx:
        pivots.append({
            "index": int(idx),
            "time": candles[idx]["time"],
            "price": float(lows[idx]),
            "kind": "low",
        })

    pivots.sort(key=lambda p: (p["index"], p["kind"]))
    return pivots


def detect_pivots_hybrid(
    candles: list[dict],
    fractal_pivots: list[dict],
    *,
    n_pips: int | None = None,
    timeframe: str | None = None,
    price_tolerance: float | None = None,
) -> list[dict]:
    """Merge fractal-window pivots with PIP-PD pivots.

    Fractal pivots are authoritative — they satisfy the strict "N-bar local
    extremum" definition that downstream pattern detectors rely on.  PIP-PD
    pivots are additive: they fill in turning points the fractal window was
    too narrow to catch, *without* displacing any fractal pivot.

    A PIP pivot is dropped when a fractal pivot of the same kind is within 2
    bars and within ``price_tolerance`` of it.  ``price_tolerance`` defaults
    to 0.05 × average candle range (≈ ATR) when None.
    """
    if not candles:
        return list(fractal_pivots)

    pip_pivots = detect_pivots_pip_pd(candles, n_pips=n_pips, timeframe=timeframe)
    if not pip_pivots:
        return list(fractal_pivots)

    if price_tolerance is None:
        ranges = [c["high"] - c["low"] for c in candles if c["high"] > c["low"]]
        avg_range = sum(ranges) / len(ranges) if ranges else 0.0
        price_tolerance = max(avg_range * 0.05, 1e-9)

    merged = list(fractal_pivots)
    # Dedup key is (index, kind) — a single bar can legitimately serve as both
    # a high-series PIP and a low-series PIP (e.g. a long-wicked shooting star
    # where the same bar tops the high curve and troughs the low curve).
    existing_keys = {(p["index"], p["kind"]) for p in merged}

    for pip in pip_pivots:
        key = (pip["index"], pip["kind"])
        if key in existing_keys:
            continue
        too_close = False
        for fr in fractal_pivots:
            if fr["kind"] != pip["kind"]:
                continue
            if abs(fr["index"] - pip["index"]) <= 2 and abs(fr["price"] - pip["price"]) < price_tolerance:
                too_close = True
                break
        if not too_close:
            merged.append(pip)
            existing_keys.add(key)

    merged.sort(key=lambda p: (p["index"], p["kind"]))
    return merged
