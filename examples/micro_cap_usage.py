"""Micro-cap example: PEPE/SHIB-scale prices (1e-5 to 1e-9 range).

Demonstrates the sub-cent code path where fixed tolerances would collapse
every candle into a "touch". chartpatterns scales tolerance to ATR.

Run:
    python examples/micro_cap_usage.py
"""

from __future__ import annotations

import math
import random

from chartpatterns import detect_chart_patterns


def synth_pepe_like(n: int = 100, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    candles: list[dict] = []
    base = 1.25e-5
    for i in range(n):
        price = base + i * 3e-9 + math.sin(i * 0.3) * 3e-7 + rng.uniform(-1e-7, 1e-7)
        candles.append({
            "index": i,
            "time": 1_700_000_000 + i * 60,
            "open": price - 2e-8,
            "high": price + 8e-8,
            "low": price - 8e-8,
            "close": price + 2e-8,
            "volume": 10_000.0,
        })
    return candles


def main() -> None:
    candles = synth_pepe_like()
    result = detect_chart_patterns(candles, timeframe="1m", trade_type="scalp")

    print("Pattern:", result["primaryPattern"]["label"])
    print("ATR:", result["stats"]["atr"])
    print("Trendlines:", len(result["trendLines"]))

    # Touch counts should be < total candles (no fake "all touch" bug)
    for line in result["trendLines"][:3]:
        touches = line.get("touches", "?")
        print(f"  line touches: {touches} (out of {len(candles)} candles)")


if __name__ == "__main__":
    main()
