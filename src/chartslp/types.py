"""Public type contracts for chartslp.

Uses TypedDict for dict-based candle/pivot compatibility with existing
consumers. No pydantic runtime dependency.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class Candle(TypedDict):
    index: int
    time: int  # Unix epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


class Pivot(TypedDict):
    index: int
    time: int
    price: float
    kind: Literal["high", "low"]
    label: NotRequired[str]
    tentative: NotRequired[bool]


class TrendLinePoint(TypedDict):
    time: int
    price: float


class TrendLine(TypedDict):
    points: list[TrendLinePoint]
    id: NotRequired[str]
    label: NotRequired[str]
    color: NotRequired[str]
    style: NotRequired[str]
    source: NotRequired[Literal["auto", "pattern"]]
    score: NotRequired[float]
    touches: NotRequired[int]
    pair: NotRequired[int]


class HorizontalLevel(TypedDict):
    price: float
    label: str
    color: NotRequired[str]
    style: NotRequired[str]
    startTime: NotRequired[int]
    touches: NotRequired[int]
    flipped: NotRequired[bool]


class PatternShape(TypedDict):
    kind: str
    points: list[TrendLinePoint]
    color: NotRequired[str]


class PrimaryPattern(TypedDict):
    type: str
    label: str
    bias: Literal["bullish", "bearish", "neutral"]
    state: NotRequired[str]
    confidence: NotRequired[float]
    neckline: NotRequired[float]


class PatternStats(TypedDict):
    pivotCount: int
    atr: float
    volumeRatio: float


class PatternResult(TypedDict):
    version: int
    generatedFrom: str
    timeframe: str
    tradeType: str
    summary: str
    primaryPattern: PrimaryPattern
    shapes: list[PatternShape]
    swingPoints: list[Pivot]
    trendLines: list[TrendLine]
    horizontalLevels: list[HorizontalLevel]
    markers: list[dict]
    levels: dict[str, float | None]
    stats: PatternStats
    allCandidates: NotRequired[list[dict]]
    themeColor: NotRequired[str]


TradeType = Literal["swing", "scalp"]
# Casing is significant: "1m" = 1 minute, "1M" = 1 month, "1w" = 1 week.
# Keep these lowercase-w to match the data layer (market_data emits "1w"/"1M").
Timeframe = Literal[
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"
]
