# Changelog

## [0.1.1] - 2026-05-23

### Changed

- Drop the redundant `↕` suffix appended to flipped S/R cluster labels.
  The `flipped` boolean field on the level remains the canonical signal;
  downstream consumers should branch on that instead of parsing the label.

## [0.1.0] - 2026-04-20

### Added

- `detect_chart_patterns(candles, timeframe, trade_type)` — main detection
  entry point. Returns a dict with `primaryPattern`, `trendLines`,
  `horizontalLevels`, `swingPoints`, `shapes`, `stats`, and `allCandidates`.
- `build_auto_trendlines()`, `build_trend_line()`,
  `build_pattern_boundary_lines()` — standalone trendline construction.
- `project_price()`, `count_valid_touches()` — trendline geometry helpers.
- `format_pattern_engine_context()` — helper for passing pattern state into
  downstream LLM prompts.
- 29 pattern types: triangles, wedges, H&S (+ inverse), channels, flags,
  pennants, cup-and-handle, VCP, broadening, rounding, V-reversals, trend
  structures, rectangle ranges.
- `smart_round_price()` utility for micro-cap precision (prices down to 1e-9
  keep meaningful significant digits).
- TypedDict contracts in `chartslp.types`: `Candle`, `Pivot`,
  `TrendLine`, `PatternResult`, etc.
- Test suite: 26 tests — unit coverage, end-to-end micro-cap scenarios,
  golden parity fixtures.
