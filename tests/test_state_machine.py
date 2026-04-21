"""Tests for the P2 three-state machine (FORMING / CONFIRMED / RETESTED).

trade_type="scalp" → FORMING wins ties; trade_type="swing" → RETESTED wins.
"""
from __future__ import annotations

import pytest

from chartslp.patterns import (
    _STATE_FAMILY,
    _STATE_WEIGHT_BY_PROFILE,
    _pattern_selection_score,
    _state_bonus,
)


def _mk(pattern_type: str, state: str, confidence: float = 0.7, last_idx: int = 40) -> dict:
    return {
        "primaryPattern": {
            "type": pattern_type,
            "state": state,
            "confidence": confidence,
        },
        "shapePivots": [{"index": last_idx, "kind": "high"}],
    }


def test_state_family_mapping() -> None:
    assert _STATE_FAMILY["forming"] == "FORMING"
    assert _STATE_FAMILY["monitor"] == "FORMING"
    assert _STATE_FAMILY["breakout"] == "CONFIRMED"
    assert _STATE_FAMILY["breakdown"] == "CONFIRMED"
    assert _STATE_FAMILY["confirmed"] == "CONFIRMED"
    assert _STATE_FAMILY["retest"] == "RETESTED"


def test_state_bonus_scalp_prefers_forming() -> None:
    scalp_forming = _state_bonus("forming", "scalp")
    scalp_confirmed = _state_bonus("breakout", "scalp")
    scalp_retested = _state_bonus("retest", "scalp")
    assert scalp_forming > scalp_confirmed
    assert scalp_retested > scalp_confirmed  # retested second
    assert scalp_forming == pytest.approx(0.15)


def test_state_bonus_swing_prefers_retested() -> None:
    swing_forming = _state_bonus("forming", "swing")
    swing_confirmed = _state_bonus("breakout", "swing")
    swing_retested = _state_bonus("retest", "swing")
    assert swing_retested > swing_confirmed > swing_forming
    assert swing_retested == pytest.approx(0.15)


def test_state_bonus_position_same_as_swing() -> None:
    for state in ("forming", "breakout", "retest"):
        assert _state_bonus(state, "position") == _state_bonus(state, "swing")


def test_state_bonus_unknown_profile_defaults_to_swing() -> None:
    assert _state_bonus("breakout", "exotic") == _state_bonus("breakout", "swing")


def test_state_bonus_unknown_state_returns_zero() -> None:
    assert _state_bonus("totally_made_up", "swing") == 0.0


def test_pattern_selection_score_scalp_flips_preference() -> None:
    """Same confidence, two states — scalp picks forming, swing picks retest."""
    forming_result = _mk("head_and_shoulders", "forming", confidence=0.70)
    retested_result = _mk("head_and_shoulders", "retest", confidence=0.70)

    # Scalp: forming should outrank retest (forming=0.15 vs retest=0.12)
    score_forming_scalp = _pattern_selection_score(forming_result, 50, "scalp")
    score_retested_scalp = _pattern_selection_score(retested_result, 50, "scalp")
    assert score_forming_scalp > score_retested_scalp

    # Swing: retest should outrank forming (retest=0.15 vs forming=0.02)
    score_forming_swing = _pattern_selection_score(forming_result, 50, "swing")
    score_retested_swing = _pattern_selection_score(retested_result, 50, "swing")
    assert score_retested_swing > score_forming_swing


def test_pattern_selection_score_trade_type_none_defaults_to_swing() -> None:
    result = _mk("double_top", "retest")
    assert _pattern_selection_score(result, 50, None) == _pattern_selection_score(result, 50, "swing")


def test_state_weight_tables_complete() -> None:
    """Every profile must assign weights to all three state families."""
    for profile, table in _STATE_WEIGHT_BY_PROFILE.items():
        for family in ("FORMING", "CONFIRMED", "RETESTED"):
            assert family in table, f"profile {profile} missing {family}"
            assert 0.0 <= table[family] <= 1.0, f"weight out of range"
