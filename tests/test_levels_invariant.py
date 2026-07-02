"""Invariant: a detected level pair must never invert (support <= resistance).

Guards audit bugs #4/#5 — converging triangle/wedge/channel detectors projected
both boundary lines to the current bar and wrote support=bottom, resistance=top
with no ordering check, so a crossed apex produced support > resistance.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from chartslp import detect_chart_patterns

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_patterns.py"
_spec = importlib.util.spec_from_file_location("_synthetic_patterns", _FIXTURE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_synthetic_patterns"] = _module
_spec.loader.exec_module(_module)
SYNTHETIC_REGISTRY = _module.SYNTHETIC_REGISTRY


@pytest.mark.parametrize("pattern_name", sorted(SYNTHETIC_REGISTRY.keys()))
def test_levels_never_invert(pattern_name):
    generator = SYNTHETIC_REGISTRY[pattern_name][0]
    for seed in range(5):
        candles = generator(seed=seed)
        result = detect_chart_patterns(candles, timeframe="1h", trade_type="swing")
        levels = (result or {}).get("levels") or {}
        support, resistance = levels.get("support"), levels.get("resistance")
        if support is not None and resistance is not None:
            assert support <= resistance, (
                f"{pattern_name} seed {seed}: inverted levels "
                f"support={support} > resistance={resistance}"
            )
