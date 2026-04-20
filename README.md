# chartpatterns

Candlestick chart pattern and trendline detection for Python.

29 built-in pattern detectors. ATR-scaled tolerances that hold up on micro-cap
prices down to 1e-9. Zero framework dependencies — just `numpy`.

## Install

Editable (for local development):

```bash
pip install -e /path/to/chartpatterns
```

From a git URL:

```bash
pip install "chartpatterns @ git+https://github.com/YOUR_USER/chartpatterns.git@v0.1.0"
```

## Quick start

```python
from chartpatterns import detect_chart_patterns

candles = [
    {"index": 0, "time": 1700000000, "open": 100.0, "high": 101.0,
     "low": 99.0, "close": 100.5, "volume": 1000.0},
    # ... 20+ candles
]

result = detect_chart_patterns(candles, timeframe="1h", trade_type="swing")

print(result["primaryPattern"]["label"])   # e.g. "Ascending Triangle"
print(result["stats"]["atr"])
for line in result["trendLines"]:
    print(line["points"], line.get("score"))
```

## Supported patterns

29 pattern types, including:

- Triangles: ascending, descending, symmetrical, broadening
- Wedges: rising, falling
- Reversals: double/triple top & bottom, head-and-shoulders (+ inverse),
  V-reversal top/bottom, rounding top/bottom
- Continuation: bull/bear flag, bull/bear pennant, rectangle range
- Channels: ascending, descending
- Complex: cup-and-handle, VCP (volatility contraction pattern)
- Structural: bullish/bearish trend structures (HH-HL / LH-LL)

## Public API

```python
from chartpatterns import (
    detect_chart_patterns,
    format_pattern_engine_context,
    build_auto_trendlines,
    build_pattern_boundary_lines,
    build_trend_line,
    project_price,
    count_valid_touches,
    PATTERN_LABELS,
)
```

Type contracts (TypedDict):

```python
from chartpatterns.types import (
    Candle, Pivot, TrendLine, TrendLinePoint,
    HorizontalLevel, PatternShape, PrimaryPattern,
    PatternStats, PatternResult, TradeType, Timeframe,
)
```

See `examples/` for runnable snippets.

## Requirements

- Python 3.12+
- numpy 2.0+

## Development

```bash
uv sync --extra dev
uv run pytest            # 26 tests
uv run ruff check src tests
```

## License

MIT
