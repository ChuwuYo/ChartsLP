from __future__ import annotations

import pytest

from chartpatterns import patterns as pattern_engine

_DETECTOR_NAMES = (
    "_detect_double_top",
    "_detect_double_bottom",
    "_detect_triple_top_bottom",
    "_detect_head_and_shoulders",
    "_detect_cup_and_handle",
    "_detect_vcp",
    "_detect_triangle",
    "_detect_symmetrical_triangle",
    "_detect_wedge",
    "_detect_channel",
    "_detect_flag_and_pennant",
    "_detect_broadening_formation",
    "_detect_rectangle_range",
    "_detect_rounding_bottom_top",
    "_detect_v_reversal",
    "_detect_trend_structure",
)


def _build_klines(count: int = 30, *, start_time: int = 1_700_000_000, step: int = 60) -> list[dict]:
    klines: list[dict] = []
    for idx in range(count):
        base = 100.0 + idx * 0.2
        klines.append({
            "time": start_time + idx * step,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.4,
            "volume": 1_000.0 + idx,
        })
    return klines


def _patch_pattern_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    detector_overrides: dict[str, object],
    auto_trends: list[dict] | None = None,
) -> None:
    def _noop_detector(*args, **kwargs):
        return None

    for name in _DETECTOR_NAMES:
        monkeypatch.setattr(pattern_engine, name, detector_overrides.get(name, _noop_detector))

    monkeypatch.setattr(pattern_engine, "_calculate_pattern_tolerances", lambda candles, trade_type: (1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(pattern_engine, "_detect_pivots", lambda *args, **kwargs: [])
    monkeypatch.setattr(pattern_engine, "_calculate_volume_ratio", lambda candles: 1.0)
    monkeypatch.setattr(pattern_engine, "_compute_clustered_levels", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(pattern_engine, "_dedupe_detector_candidates", lambda detector_candidates, latest_index: detector_candidates)
    monkeypatch.setattr(pattern_engine, "_build_shapes_from_pattern_result", lambda *args, **kwargs: [])
    monkeypatch.setattr(pattern_engine, "_apply_trendspider_pattern_theme", lambda pattern_type, bias, result: result)
    monkeypatch.setattr(pattern_engine, "_pattern_selection_score", lambda result, latest_index: float(result.get("_test_score", 0.0)))
    monkeypatch.setattr(pattern_engine, "_build_auto_trendlines", lambda *args, **kwargs: list(auto_trends or []))


def test_synthesize_boundary_trendlines_applies_to_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    klines = _build_klines()

    primary_result = {
        "_test_score": 5.0,
        "primaryPattern": {
            "type": "bullish_trend_structure",
            "label": "Bullish Trend (HH-HL)",
            "bias": "bullish",
            "state": "confirmed",
            "confidence": 0.82,
        },
        "trendLines": [
            {
                "id": "pattern-uptrend",
                "label": "Lower",
                "style": "solid",
                "color": "#5DCC61",
                "points": [
                    {"time": klines[20]["time"], "price": 103.0},
                    {"time": klines[28]["time"], "price": 105.0},
                ],
            }
        ],
        "horizontalLevels": [],
        "markers": [],
        "levels": {"support": None, "resistance": None},
        "shapePivots": [
            {"index": 20, "time": klines[20]["time"], "price": 103.0, "kind": "low"},
            {"index": 24, "time": klines[24]["time"], "price": 106.0, "kind": "high"},
            {"index": 28, "time": klines[28]["time"], "price": 105.0, "kind": "low"},
        ],
    }
    candidate_result = {
        "_test_score": 9.0,
        "primaryPattern": {
            "type": "double_top",
            "label": "Double Top",
            "bias": "bearish",
            "state": "monitor",
            "confidence": 0.74,
        },
        "trendLines": [],
        "horizontalLevels": [
            {
                "id": "double-top-neckline",
                "label": "Neckline",
                "color": "#3b82f6",
                "points": [
                    {"time": klines[8]["time"], "price": 99.5},
                    {"time": klines[12]["time"], "price": 99.5},
                ],
            },
            {
                "id": "cluster-r1",
                "label": "R1",
                "color": "#ef4444",
                "points": [
                    {"time": klines[8]["time"], "price": 103.5},
                    {"time": klines[12]["time"], "price": 103.5},
                ],
            },
        ],
        "markers": [],
        "levels": {"support": None, "resistance": None},
        "shapePivots": [
            {"index": 8, "time": klines[8]["time"], "price": 103.0, "kind": "high"},
            {"index": 10, "time": klines[10]["time"], "price": 99.5, "kind": "low"},
            {"index": 12, "time": klines[12]["time"], "price": 103.1, "kind": "high"},
        ],
    }

    _patch_pattern_engine(
        monkeypatch,
        detector_overrides={
            "_detect_double_top": lambda *args, **kwargs: candidate_result,
            "_detect_trend_structure": lambda *args, **kwargs: primary_result,
        },
    )

    result = pattern_engine.detect_chart_patterns(klines)

    assert result["allCandidates"][0]["type"] == "bullish_trend_structure"
    candidate = next(c for c in result["allCandidates"] if c["type"] == "double_top")
    assert {line["id"] for line in candidate["trendLines"]} == {"boundary-double-top-neckline"}
    assert candidate["trendLines"][0]["source"] == "pattern"


def test_auto_pattern_dedup_uses_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    klines = _build_klines()
    pattern_line = {
        "id": "pattern-lower",
        "label": "Lower",
        "style": "solid",
        "color": "#5DCC61",
        "points": [
            {"time": klines[5]["time"], "price": 100.0},
            {"time": klines[10]["time"], "price": 101.0},
        ],
    }
    auto_line = {
        "id": "auto-lower",
        "label": "Support",
        "style": "dashed",
        "color": "#B8E600",
        "points": [
            {"time": klines[15]["time"], "price": 100.0},
            {"time": klines[20]["time"], "price": 101.0},
        ],
    }
    primary_result = {
        "_test_score": 3.0,
        "primaryPattern": {
            "type": "bullish_trend_structure",
            "label": "Bullish Trend (HH-HL)",
            "bias": "bullish",
            "state": "confirmed",
            "confidence": 0.8,
        },
        "trendLines": [pattern_line],
        "horizontalLevels": [],
        "markers": [],
        "levels": {"support": None, "resistance": None},
        "shapePivots": [
            {"index": 20, "time": klines[20]["time"], "price": 103.0, "kind": "low"},
            {"index": 24, "time": klines[24]["time"], "price": 106.0, "kind": "high"},
            {"index": 28, "time": klines[28]["time"], "price": 105.0, "kind": "low"},
        ],
    }

    _patch_pattern_engine(
        monkeypatch,
        detector_overrides={
            "_detect_trend_structure": lambda *args, **kwargs: primary_result,
        },
        auto_trends=[auto_line],
    )

    result = pattern_engine.detect_chart_patterns(klines)

    trend_lines = {line["id"]: line for line in result["trendLines"]}
    assert set(trend_lines) == {"pattern-lower", "auto-lower"}
    assert trend_lines["pattern-lower"]["source"] == "pattern"
    assert trend_lines["auto-lower"]["source"] == "auto"