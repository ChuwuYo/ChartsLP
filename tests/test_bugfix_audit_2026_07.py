"""Regression tests for the 2026-07 chartslp calculation-audit bugs.

Each test encodes the audit's concrete failing input and asserts the CORRECT
behaviour, so it fails on the buggy code and passes after the fix.
"""
from __future__ import annotations

import pytest

from chartslp import patterns

# --- Bug #1 / #2: double-top / double-bottom off-by-one drops the freshest
# window (offset upper bound was len(pivots)-2, so the last pivot could never be
# the second peak/trough; with exactly 3 pivots the loop was empty). The same
# idiom lives in _find_recent_pattern_pivots — tested here (no candles needed).

def test_find_recent_pivots_double_top_includes_last_window():
    pivots = [
        {"kind": "high", "index": 0, "price": 100.0, "time": 0},
        {"kind": "low", "index": 10, "price": 90.0, "time": 600},
        {"kind": "high", "index": 20, "price": 100.0, "time": 1200},  # freshest peak
    ]
    seq = patterns._find_recent_pattern_pivots("double_top", pivots, tolerance=1.0, min_move=1.0)
    assert [p["kind"] for p in seq] == ["high", "low", "high"]
    assert seq[-1]["index"] == 20  # the just-formed second top IS detected


def test_find_recent_pivots_double_bottom_includes_last_window():
    pivots = [
        {"kind": "low", "index": 0, "price": 100.0, "time": 0},
        {"kind": "high", "index": 10, "price": 110.0, "time": 600},
        {"kind": "low", "index": 20, "price": 100.0, "time": 1200},  # freshest trough
    ]
    seq = patterns._find_recent_pattern_pivots("double_bottom", pivots, tolerance=1.0, min_move=1.0)
    assert [p["kind"] for p in seq] == ["low", "high", "low"]
    assert seq[-1]["index"] == 20


# --- Bug #3: _dedupe_detector_candidates group.sort lacked reverse=True on the
# quality tiebreakers, so on a specificity tie it KEPT the lower score/confidence
# candidate. After the fix the higher-quality one survives.

def _cand(confidence, score):
    result = {
        "primaryPattern": {"type": "double_top", "confidence": confidence},
        "shapePivots": [
            {"index": 1, "kind": "high"},
            {"index": 5, "kind": "low"},
            {"index": 9, "kind": "high"},
        ],
    }
    return (result, score)


@pytest.mark.xfail(
    reason="Confirmed bug (dedup group.sort keeps lower-quality on specificity tie), "
    "but the naive fix (reverse=True) regresses bull_flag recall 5/5 -> 1/5: "
    "_pattern_selection_score ranks symmetrical_triangle above bull_flag on shared "
    "geometry, so the buggy 'keep lower score' accidentally preserves the correct "
    "flag label. Fix DEFERRED until the score function is re-tuned so 'higher score' "
    "== 'better label'. See docs / audit 2026-07.",
    strict=True,
)
def test_dedupe_keeps_higher_quality_candidate_on_tie():
    weak = _cand(0.40, 1.0)
    strong = _cand(0.85, 3.0)
    # Same type + same shapePivots → same fingerprint → one survivor.
    out = patterns._dedupe_detector_candidates([weak, strong], latest_index=10, trade_type="scalp")
    assert len(out) == 1
    survivor = out[0][0]["primaryPattern"]
    assert survivor["confidence"] == 0.85  # the stronger candidate, not the weaker
