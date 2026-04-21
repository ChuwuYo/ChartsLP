"""Synthetic pattern generators for baseline detection recall testing.

Each generator produces a list of OHLC+V candle dicts matching the chartslp
input contract: ``{time, open, high, low, close, volume}``.

Patterns are parameterised by a seed for reproducibility. Noise is injected
so no two variants are identical, but the underlying geometric shape must
remain recognisable to the detectors in ``chartslp.patterns``.

Top-10 coverage for P6 layer 1 baseline: head_and_shoulders (+inverse),
double_top/bottom, triangle (asc/desc/sym), wedge (rising/falling), flag
(bull/bear). Remaining 18 patterns follow in P2.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

BAR_SECONDS = 3600  # 1h bars
BASE_TIME = 1_700_000_000


def _make_candle(
    time: int,
    open_: float,
    close: float,
    *,
    wick_pct: float = 0.002,
    volume: float = 1_000.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Build a valid candle with wicks derived from open/close + small random jitter."""
    rng = rng or np.random.default_rng(0)
    mid = (open_ + close) / 2.0
    wick = max(abs(mid) * wick_pct, 0.01)
    hi = max(open_, close) + abs(rng.normal(0, wick))
    lo = min(open_, close) - abs(rng.normal(0, wick))
    return {
        "time": int(time),
        "open": float(open_),
        "high": float(hi),
        "low": float(lo),
        "close": float(close),
        "volume": float(volume),
    }


def _ramp(start: float, end: float, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Linear ramp start→end over n points with Gaussian noise proportional to range."""
    base = np.linspace(start, end, n)
    delta = abs(end - start) if end != start else 1.0
    return base + rng.normal(0, delta * noise, n)


def _candles_from_series(
    series: np.ndarray,
    *,
    volume_pattern: Callable[[int, int], float] | None = None,
    rng: np.random.Generator,
) -> list[dict]:
    """Turn a close-price series into candles. Open = prev close. Volume optional."""
    candles: list[dict] = []
    n = len(series)
    for i in range(n):
        open_ = float(series[i - 1]) if i > 0 else float(series[0])
        close = float(series[i])
        vol = volume_pattern(i, n) if volume_pattern else 1_000.0
        candles.append(_make_candle(
            time=BASE_TIME + i * BAR_SECONDS,
            open_=open_,
            close=close,
            rng=rng,
            volume=vol,
        ))
    return candles


def _breakout_volume(spike_idx: int, spike_mult: float = 3.0) -> Callable[[int, int], float]:
    def _vol(i: int, n: int) -> float:
        base = 1_000.0
        return base * spike_mult if i == spike_idx else base + (i % 7) * 30
    return _vol


# ---------------------------------------------------------------------------
# Head and Shoulders (top) — 5 extrema: LS, trough, head, trough, RS
# ---------------------------------------------------------------------------

def synth_head_and_shoulders(
    *,
    seed: int = 0,
    base: float = 100.0,
    shoulder_h: float = 8.0,
    head_h: float = 14.0,
    n_bars: int = 60,
    noise: float = 0.01,
    include_breakout: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Waypoints at relative positions
    wp = [
        (0,           base),
        (n_bars // 6, base + shoulder_h),   # LS
        (n_bars * 2 // 6, base),            # trough
        (n_bars * 3 // 6, base + head_h),   # Head
        (n_bars * 4 // 6, base),            # trough
        (n_bars * 5 // 6, base + shoulder_h), # RS
        (n_bars - 6,  base),                # neckline
    ]
    if include_breakout:
        wp.append((n_bars - 1, base - shoulder_h * 0.6))  # breakdown
    else:
        wp.append((n_bars - 1, base - 1.0))

    xs = np.array([w[0] for w in wp])
    ys = np.array([w[1] for w in wp])
    series = np.interp(np.arange(n_bars), xs, ys)
    series = series + rng.normal(0, head_h * noise, n_bars)

    breakout_idx = wp[-1][0] if include_breakout else -1
    vol_pat = _breakout_volume(breakout_idx) if include_breakout else None
    return _candles_from_series(series, volume_pattern=vol_pat, rng=rng)


def synth_inverse_head_and_shoulders(
    *,
    seed: int = 0,
    base: float = 100.0,
    shoulder_d: float = 8.0,
    head_d: float = 14.0,
    n_bars: int = 60,
    noise: float = 0.01,
    include_breakout: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    wp = [
        (0,           base),
        (n_bars // 6, base - shoulder_d),
        (n_bars * 2 // 6, base),
        (n_bars * 3 // 6, base - head_d),
        (n_bars * 4 // 6, base),
        (n_bars * 5 // 6, base - shoulder_d),
        (n_bars - 6,  base),
    ]
    if include_breakout:
        wp.append((n_bars - 1, base + shoulder_d * 0.6))
    else:
        wp.append((n_bars - 1, base + 1.0))

    xs = np.array([w[0] for w in wp])
    ys = np.array([w[1] for w in wp])
    series = np.interp(np.arange(n_bars), xs, ys)
    series = series + rng.normal(0, head_d * noise, n_bars)

    breakout_idx = wp[-1][0] if include_breakout else -1
    vol_pat = _breakout_volume(breakout_idx) if include_breakout else None
    return _candles_from_series(series, volume_pattern=vol_pat, rng=rng)


# ---------------------------------------------------------------------------
# Double Top / Bottom
# ---------------------------------------------------------------------------

def synth_double_top(
    *,
    seed: int = 0,
    base: float = 100.0,
    peak_h: float = 10.0,
    sep_bars: int = 20,
    trough_d: float = 5.0,
    n_bars: int = 50,
    noise: float = 0.008,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    p1 = n_bars // 4
    p2 = p1 + sep_bars
    p2 = min(p2, n_bars - 8)
    trough = (p1 + p2) // 2
    wp = [
        (0, base),
        (p1, base + peak_h),
        (trough, base + peak_h - trough_d),
        (p2, base + peak_h - 0.3),
        (n_bars - 3, base),
        (n_bars - 1, base - peak_h * 0.3),
    ]
    xs = np.array([w[0] for w in wp])
    ys = np.array([w[1] for w in wp])
    series = np.interp(np.arange(n_bars), xs, ys)
    series = series + rng.normal(0, peak_h * noise, n_bars)
    return _candles_from_series(series, volume_pattern=_breakout_volume(n_bars - 1), rng=rng)


def synth_double_bottom(
    *,
    seed: int = 0,
    base: float = 100.0,
    trough_d: float = 10.0,
    sep_bars: int = 20,
    peak_h: float = 5.0,
    n_bars: int = 50,
    noise: float = 0.008,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    b1 = n_bars // 4
    b2 = b1 + sep_bars
    b2 = min(b2, n_bars - 8)
    peak = (b1 + b2) // 2
    wp = [
        (0, base),
        (b1, base - trough_d),
        (peak, base - trough_d + peak_h),
        (b2, base - trough_d + 0.3),
        (n_bars - 3, base),
        (n_bars - 1, base + trough_d * 0.3),
    ]
    xs = np.array([w[0] for w in wp])
    ys = np.array([w[1] for w in wp])
    series = np.interp(np.arange(n_bars), xs, ys)
    series = series + rng.normal(0, trough_d * noise, n_bars)
    return _candles_from_series(series, volume_pattern=_breakout_volume(n_bars - 1), rng=rng)


# ---------------------------------------------------------------------------
# Triangles (ascending, descending, symmetrical)
# ---------------------------------------------------------------------------

def _synth_triangle(
    *,
    seed: int,
    base: float,
    top_slope: float,
    bot_slope: float,
    n_bars: int,
    noise: float,
    amplitude: float,
    periods: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = np.arange(n_bars)
    top_line = base + amplitude + top_slope * x
    bot_line = base - amplitude + bot_slope * x
    # oscillate between top and bottom with `periods` swings
    phase = np.sin(np.linspace(0, periods * np.pi, n_bars))
    series = (top_line + bot_line) / 2 + phase * (top_line - bot_line) / 2
    series = series + rng.normal(0, amplitude * noise, n_bars)
    return _candles_from_series(series, volume_pattern=_breakout_volume(n_bars - 1), rng=rng)


def synth_ascending_triangle(*, seed: int = 0, n_bars: int = 50, noise: float = 0.01) -> list[dict]:
    # Flat top, rising bottom
    return _synth_triangle(seed=seed, base=100.0, top_slope=0.0, bot_slope=0.25,
                           n_bars=n_bars, noise=noise, amplitude=8.0, periods=4)


def synth_descending_triangle(*, seed: int = 0, n_bars: int = 50, noise: float = 0.01) -> list[dict]:
    # Falling top, flat bottom
    return _synth_triangle(seed=seed, base=100.0, top_slope=-0.25, bot_slope=0.0,
                           n_bars=n_bars, noise=noise, amplitude=8.0, periods=4)


def synth_symmetrical_triangle(*, seed: int = 0, n_bars: int = 50, noise: float = 0.01) -> list[dict]:
    # Converging top & bottom
    return _synth_triangle(seed=seed, base=100.0, top_slope=-0.2, bot_slope=0.2,
                           n_bars=n_bars, noise=noise, amplitude=10.0, periods=4)


# ---------------------------------------------------------------------------
# Wedges (rising, falling)
# ---------------------------------------------------------------------------

def synth_rising_wedge(*, seed: int = 0, n_bars: int = 50, noise: float = 0.01) -> list[dict]:
    # Both lines rising, top slower (converging upward)
    return _synth_triangle(seed=seed, base=100.0, top_slope=0.15, bot_slope=0.35,
                           n_bars=n_bars, noise=noise, amplitude=8.0, periods=4)


def synth_falling_wedge(*, seed: int = 0, n_bars: int = 50, noise: float = 0.01) -> list[dict]:
    # Both lines falling, top faster (converging downward)
    return _synth_triangle(seed=seed, base=100.0, top_slope=-0.35, bot_slope=-0.15,
                           n_bars=n_bars, noise=noise, amplitude=8.0, periods=4)


# ---------------------------------------------------------------------------
# Flags (bull / bear) — impulsive pole followed by slight counter-trend channel
# ---------------------------------------------------------------------------

def _synth_flag(*, seed: int, direction: int, n_bars: int, noise: float) -> list[dict]:
    """direction=+1 bull flag (up pole, down drift), -1 bear flag (down pole, up drift)"""
    rng = np.random.default_rng(seed)
    pole_len = max(8, n_bars // 4)
    flag_len = n_bars - pole_len
    base = 100.0
    pole_height = 20.0 * direction
    # Pole
    pole_series = np.linspace(base, base + pole_height, pole_len)
    # Flag: slight counter-trend oscillation around a slowly drifting line
    flag_top = base + pole_height - 1.0 * direction
    flag_bot = base + pole_height - 4.0 * direction
    drift = np.linspace(0, -1.5 * direction, flag_len)
    phase = np.sin(np.linspace(0, 3 * np.pi, flag_len))
    flag_series = (flag_top + flag_bot) / 2 + drift + phase * 1.5
    series = np.concatenate([pole_series, flag_series])
    series = series + rng.normal(0, abs(pole_height) * noise, n_bars)

    def _vol(i: int, _n: int) -> float:
        if i < pole_len - 1:
            return 1_500.0 + i * 50  # rising volume on pole
        if i == n_bars - 1:
            return 3_500.0
        return 700.0 + (i % 5) * 30

    return _candles_from_series(series, volume_pattern=_vol, rng=rng)


def synth_bull_flag(*, seed: int = 0, n_bars: int = 50, noise: float = 0.008) -> list[dict]:
    return _synth_flag(seed=seed, direction=+1, n_bars=n_bars, noise=noise)


def synth_bear_flag(*, seed: int = 0, n_bars: int = 50, noise: float = 0.008) -> list[dict]:
    return _synth_flag(seed=seed, direction=-1, n_bars=n_bars, noise=noise)


# ---------------------------------------------------------------------------
# Registry — pattern_type -> (generator, acceptable primary labels)
# ---------------------------------------------------------------------------

# Some patterns legitimately overlap (e.g. asc_triangle resembles asc_channel on
# noisy data). The registry accepts a set of "close relatives" as valid detections.

SYNTHETIC_REGISTRY: dict[str, tuple[Callable[..., list[dict]], frozenset[str]]] = {
    "head_and_shoulders": (
        synth_head_and_shoulders,
        frozenset({"head_and_shoulders"}),
    ),
    "inverse_head_and_shoulders": (
        synth_inverse_head_and_shoulders,
        frozenset({"inverse_head_and_shoulders"}),
    ),
    "double_top": (
        synth_double_top,
        frozenset({"double_top", "triple_top"}),
    ),
    "double_bottom": (
        synth_double_bottom,
        frozenset({"double_bottom", "triple_bottom"}),
    ),
    "ascending_triangle": (
        synth_ascending_triangle,
        frozenset({"ascending_triangle", "ascending_channel", "rectangle_range"}),
    ),
    "descending_triangle": (
        synth_descending_triangle,
        frozenset({"descending_triangle", "descending_channel", "rectangle_range"}),
    ),
    "symmetrical_triangle": (
        synth_symmetrical_triangle,
        frozenset({"symmetrical_triangle"}),
    ),
    "rising_wedge": (
        synth_rising_wedge,
        frozenset({"rising_wedge", "ascending_channel"}),
    ),
    "falling_wedge": (
        synth_falling_wedge,
        frozenset({"falling_wedge", "descending_channel"}),
    ),
    "bull_flag": (
        synth_bull_flag,
        frozenset({"bull_flag", "bull_pennant", "bullish_trend_structure"}),
    ),
    "bear_flag": (
        synth_bear_flag,
        frozenset({"bear_flag", "bear_pennant", "bearish_trend_structure"}),
    ),
}
