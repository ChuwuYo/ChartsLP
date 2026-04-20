"""Minimal example: run pattern detection on synthetic uptrend.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import math
import random

from chartpatterns import detect_chart_patterns


def synth_uptrend(n: int = 100, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    candles: list[dict] = []
    for i in range(n):
        price = 100 + i * 0.15 + math.sin(i * 0.3) * 3 + rng.uniform(-0.4, 0.4)
        candles.append({
            "index": i,
            "time": 1_700_000_000 + i * 3600,
            "open": price - 0.2,
            "high": price + 1.2,
            "low": price - 1.2,
            "close": price + 0.2,
            "volume": 1000.0,
        })
    return candles


def main() -> None:
    candles = synth_uptrend()
    result = detect_chart_patterns(candles, timeframe="1h", trade_type="swing")

    print("Pattern:", result["primaryPattern"]["label"])
    print("Bias:", result["primaryPattern"]["bias"])
    print("Confidence:", result["primaryPattern"].get("confidence"))
    print("ATR:", result["stats"]["atr"])
    print("Pivots:", result["stats"]["pivotCount"])
    print("Trendlines:", len(result["trendLines"]))
    print("Horizontal levels:", len(result["horizontalLevels"]))
    print()
    for line in result["trendLines"][:3]:
        print("  line:", line["points"], "touches:", line.get("touches"))


if __name__ == "__main__":
    main()
