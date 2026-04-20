from chartpatterns.types import Candle, Pivot, TrendLine


def test_candle_required_fields():
    candle: Candle = {
        "index": 0,
        "time": 1700000000,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
    }
    assert candle["close"] == 100.5


def test_pivot_required_fields():
    pivot: Pivot = {
        "index": 10,
        "time": 1700001000,
        "price": 105.0,
        "kind": "high",
    }
    assert pivot["kind"] == "high"


def test_trendline_minimal():
    line: TrendLine = {
        "points": [
            {"time": 1700000000, "price": 100.0},
            {"time": 1700001000, "price": 105.0},
        ],
    }
    assert len(line["points"]) == 2
