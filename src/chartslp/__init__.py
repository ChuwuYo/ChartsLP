"""chartslp — candlestick pattern and trendline detection engine."""

from chartslp._version import __version__
from chartslp.patterns import (
    PATTERN_LABELS,
    detect_chart_patterns,
    format_pattern_engine_context,
)
from chartslp.trendlines import (
    build_auto_trendlines,
    build_pattern_boundary_lines,
    build_trend_line,
    count_valid_touches,
    project_price,
)

__all__ = [
    "__version__",
    "PATTERN_LABELS",
    "detect_chart_patterns",
    "format_pattern_engine_context",
    "build_auto_trendlines",
    "build_pattern_boundary_lines",
    "build_trend_line",
    "count_valid_touches",
    "project_price",
]
