# Live Snapshot Fixtures

P6 layer 2 of the pattern engine improvement plan. These JSON files capture
real market data from the production data sources (Binance / TwelveData) to
provide regression protection beyond the synthetic fixtures in layer 1.

## Why

Synthetic fixtures guarantee geometric correctness but miss real-market
noise (volume spikes, gap-ups, synthetic TD last candles, thin tape). Live
snapshots guard against subtle regressions like "my LMW rewrite now misses
this specific H&S that worked before."

## Schema

Each snapshot is a JSON document with:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "trade_type": "swing",
  "asset_class": "crypto",
  "captured_at": "2026-04-21T12:34:56Z",
  "candles": [ { "time": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ... }, ... ],
  "blessed_result": { ...output of detect_chart_patterns(...) at capture time }
}
```

## How to capture

The capture helper lives at `scripts/capture_chartslp_snapshot.py` (created
in step 2 of the plan). Usage:

```bash
cd /Users/chuwuyo/Documents/trendiq-react
api/.venv/bin/python scripts/capture_chartslp_snapshot.py \
  --symbol BTCUSDT --timeframe 1h --trade-type swing \
  --out packages/chartslp/tests/fixtures/live_snapshots/btcusdt_1h_swing.json
```

The helper calls `get_cached_chart_payload()` (see `.claude/rules/backend.md`
— must use cache, not direct provider) and then runs `detect_chart_patterns`
to record the blessed result.

## How to bless after a legitimate behavior change

When a P1 detector rewrite intentionally changes results on some snapshots:

1. Run `pytest packages/chartslp/tests/test_pattern_snapshots.py` — it fails
   with a diff against the blessed result
2. Manually inspect the diff in dev chart (P6 layer 3)
3. If the new output is better, re-run the capture helper with
   `--rebless <path>` to update `blessed_result` in place
4. Commit the updated fixture

## Recommended initial coverage (to be captured by user)

- `btcusdt_1h_swing.json` — crypto major, mid TF
- `btcusdt_5m_scalp.json` — crypto major, short TF
- `aapl_1d_swing.json` — US equity, daily
- `aapl_5m_scalp.json` — US equity, intraday
- `eurusd_1h_swing.json` — forex major
- `eurusd_15m_swing.json` — forex intraday

Start with 3 (1 per asset class) and grow.
