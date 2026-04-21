"""Adaptive numeric rounding driven by significant-figure preservation.

The data layer should not throw away information the UI might still want to show.
Instead of a hardcoded decimal ladder, we derive decimals from the value's
magnitude so micro-cap prices (e.g. 0.00000358) and tiny percentage moves keep
their meaningful digits, while large-magnitude values stay at a sane minimum.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# How many significant figures to preserve for each kind of quantity.
_SIG_FIGURES_PRICE = 6
_SIG_FIGURES_PERCENT = 4

# Floors keep normal cases at conventional display precision; ceiling is set
# to 15 — the practical float64 significant-digit limit — so 6 sig figs
# survive down to ~1e-10 prices (trillion-supply memecoins).
_MIN_PRICE_DECIMALS = 2
_MIN_PERCENT_DECIMALS = 4
_MAX_DECIMALS = 15


def _adaptive_decimals(value: float, sig_figures: int, min_decimals: int) -> int:
    abs_v = abs(value)
    if abs_v == 0:
        return min_decimals
    magnitude = math.floor(math.log10(abs_v))
    return max(min_decimals, min(_MAX_DECIMALS, sig_figures - 1 - magnitude))


def _degrade_to_zero(value: float, kind: str) -> float:
    """Defensive downgrade for NaN/Inf coming from upstream math.

    Raising would take down the realtime endpoint on a bad tick; we swallow to
    0.0 but emit a warning so the bug is observable.
    """
    logger.warning("number_precision: %s received non-finite value %r; returning 0.0", kind, value)
    return 0.0


def smart_round_price(value: float) -> float:
    if not math.isfinite(value):
        return _degrade_to_zero(value, "smart_round_price")
    return round(value, _adaptive_decimals(value, _SIG_FIGURES_PRICE, _MIN_PRICE_DECIMALS))


def smart_round_percent(value: float) -> float:
    if not math.isfinite(value):
        return _degrade_to_zero(value, "smart_round_percent")
    return round(value, _adaptive_decimals(value, _SIG_FIGURES_PERCENT, _MIN_PERCENT_DECIMALS))
