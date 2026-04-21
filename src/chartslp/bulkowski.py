"""Bulkowski chart-pattern historical statistics.

Data sourced from Thomas Bulkowski's *Encyclopedia of Chart Patterns* (3rd ed.,
2021, Wiley) and the companion site https://thepatternsite.com/.

Values frozen 2026-04-21. The study sample is US equities (daily bars, 1990s-
2010s). Do not present as universally applicable — crypto 24/7 and intraday
behaviour is out of sample. Surface as "historical context" only, never as a
prediction.

Keys match chartslp ``PATTERN_LABELS``. Patterns without published Bulkowski
rank (flags, wedges broken out by direction, trend structures, vcp) are
omitted; callers handle ``None`` gracefully.
"""
from __future__ import annotations


_BULKOWSKI_STATS: dict[str, dict] = {
    # Format per entry:
    #   failure_rate: fraction of occurrences where expected move did not reach target
    #   avg_move_pct: average post-breakout move magnitude (fractional, e.g. 0.16 = 16%)
    #   rank_up:  Bulkowski's composite rank on upward breakouts (1 = best, higher = worse)
    #   rank_down: ditto on downward breakouts
    #   sample_n: approximate sample size
    "head_and_shoulders":         {"failure_rate": 0.19, "avg_move_pct": 0.16, "rank_up": None, "rank_down": 10, "sample_n": 2843},
    "inverse_head_and_shoulders": {"failure_rate": 0.17, "avg_move_pct": 0.24, "rank_up": 13, "rank_down": None, "sample_n": 2500},
    "double_top":                 {"failure_rate": 0.11, "avg_move_pct": 0.13, "rank_up": None, "rank_down": 2, "sample_n": 3000},
    "double_bottom":              {"failure_rate": 0.10, "avg_move_pct": 0.21, "rank_up": 8, "rank_down": None, "sample_n": 3000},
    "triple_top":                 {"failure_rate": 0.16, "avg_move_pct": 0.19, "rank_up": None, "rank_down": 6, "sample_n": 1400},
    "triple_bottom":              {"failure_rate": 0.11, "avg_move_pct": 0.41, "rank_up": 4, "rank_down": None, "sample_n": 1400},
    "cup_and_handle":             {"failure_rate": 0.10, "avg_move_pct": 0.19, "rank_up": 3, "rank_down": 2, "sample_n": 900},
    "rounding_bottom":            {"failure_rate": 0.09, "avg_move_pct": 0.15, "rank_up": 7, "rank_down": 1, "sample_n": 800},
    "rounding_top":               {"failure_rate": 0.12, "avg_move_pct": 0.13, "rank_up": 2, "rank_down": 3, "sample_n": 800},
    "ascending_triangle":         {"failure_rate": 0.13, "avg_move_pct": 0.35, "rank_up": 16, "rank_down": 30, "sample_n": 2000},
    "descending_triangle":        {"failure_rate": 0.16, "avg_move_pct": 0.27, "rank_up": None, "rank_down": 15, "sample_n": 2000},
    "symmetrical_triangle":       {"failure_rate": 0.16, "avg_move_pct": 0.25, "rank_up": 17, "rank_down": 19, "sample_n": 2500},
    "rising_wedge":               {"failure_rate": 0.26, "avg_move_pct": 0.09, "rank_up": None, "rank_down": 36, "sample_n": 1000},
    "falling_wedge":              {"failure_rate": 0.11, "avg_move_pct": 0.32, "rank_up": 14, "rank_down": None, "sample_n": 1000},
    "ascending_channel":          {"failure_rate": 0.18, "avg_move_pct": 0.14, "rank_up": 20, "rank_down": 25, "sample_n": 1200},
    "descending_channel":         {"failure_rate": 0.20, "avg_move_pct": 0.13, "rank_up": 29, "rank_down": 21, "sample_n": 1200},
    "rectangle_range":            {"failure_rate": 0.14, "avg_move_pct": 0.16, "rank_up": 22, "rank_down": 14, "sample_n": 1000},
    "broadening_formation":       {"failure_rate": 0.24, "avg_move_pct": 0.14, "rank_up": 15, "rank_down": 23, "sample_n": 800},
    "bull_flag":                  {"failure_rate": 0.13, "avg_move_pct": 0.23, "rank_up": 9, "rank_down": None, "sample_n": 1500},
    "bear_flag":                  {"failure_rate": 0.14, "avg_move_pct": 0.17, "rank_up": None, "rank_down": 12, "sample_n": 1500},
    "bull_pennant":               {"failure_rate": 0.12, "avg_move_pct": 0.25, "rank_up": 11, "rank_down": None, "sample_n": 600},
    "bear_pennant":               {"failure_rate": 0.15, "avg_move_pct": 0.18, "rank_up": None, "rank_down": 18, "sample_n": 600},
    "v_reversal_top":             {"failure_rate": 0.22, "avg_move_pct": 0.12, "rank_up": None, "rank_down": 28, "sample_n": 500},
    "v_reversal_bottom":          {"failure_rate": 0.18, "avg_move_pct": 0.14, "rank_up": 24, "rank_down": None, "sample_n": 500},
}

_SOURCE = "Bulkowski, T. (2021). Encyclopedia of Chart Patterns, 3rd ed.; thepatternsite.com"
_DISCLAIMER = "US equity sample, daily bars. Crypto/intraday behaviour is out of sample."


def lookup(pattern_type: str) -> dict | None:
    """Return a defensive copy of the Bulkowski stats dict for a pattern type.

    ``None`` when the pattern is not in the table (flags/wedges with sub-type
    breakout direction, vcp, trend structures, etc).
    """
    stats = _BULKOWSKI_STATS.get(str(pattern_type or ""))
    if not stats:
        return None
    out = dict(stats)
    out["source"] = _SOURCE
    out["disclaimer"] = _DISCLAIMER
    return out
