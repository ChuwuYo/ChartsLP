"""Live snapshot regression test for detect_chart_patterns.

P6 layer 2 of the pattern engine improvement plan. Captures real market data
+ blessed detection output, and guards against behavioural drift.

Skips silently if the snapshots directory is empty (initial state). Users
populate it via ``scripts/capture_chartslp_snapshot.py`` (see fixtures
README).

Diff mode: when a snapshot exists but blessed_result differs from current
output, the test reports a structured diff rather than a raw equality fail,
so the human reviewer can decide whether to re-bless.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartslp import detect_chart_patterns

_SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "live_snapshots"

_REQUIRED_KEYS = ("symbol", "timeframe", "trade_type", "candles", "blessed_result")


def _collect_snapshots() -> list[Path]:
    if not _SNAPSHOT_DIR.exists():
        return []
    return sorted(p for p in _SNAPSHOT_DIR.glob("*.json") if p.is_file())


def _summary(result: dict | None) -> dict:
    """Project full detection output to the fields that matter for regression."""
    if not result:
        return {"primaryType": None, "primaryState": None, "secondaryCount": 0}
    primary = (result.get("primaryPattern") or {})
    secondaries = result.get("secondaryPatterns") or result.get("alternatives") or []
    return {
        "primaryType": primary.get("type"),
        "primaryState": primary.get("state"),
        "primaryConfidence": round(float(primary.get("confidence") or 0.0), 3),
        "secondaryCount": len(secondaries) if isinstance(secondaries, list) else 0,
    }


_SNAPSHOTS = _collect_snapshots()


@pytest.mark.skipif(not _SNAPSHOTS, reason="no live snapshots captured yet; see fixtures README")
@pytest.mark.parametrize("snapshot_path", _SNAPSHOTS, ids=lambda p: p.stem)
def test_live_snapshot_regression(snapshot_path: Path) -> None:
    data = json.loads(snapshot_path.read_text())
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    assert not missing, f"{snapshot_path.name} missing keys: {missing}"

    candles = data["candles"]
    assert candles, f"{snapshot_path.name} has empty candles"

    actual = detect_chart_patterns(
        candles,
        timeframe=data["timeframe"],
        trade_type=data["trade_type"],
    )
    blessed = data["blessed_result"]

    actual_summary = _summary(actual)
    blessed_summary = _summary(blessed)

    assert actual_summary == blessed_summary, (
        f"Snapshot {snapshot_path.name} drifted.\n"
        f"  blessed:  {blessed_summary}\n"
        f"  current:  {actual_summary}\n"
        f"If the new behaviour is intentional, re-bless via "
        f"scripts/capture_chartslp_snapshot.py --rebless {snapshot_path}"
    )
