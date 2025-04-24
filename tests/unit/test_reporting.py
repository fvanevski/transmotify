# test/unit/test_reporting.py

"""Unit & integration tests for reporting layer (summaries + generator)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pytest

# ---------------------------------------------------------------------------
# Stub‑out matplotlib so the plotting code runs headless in CI environments.
# ---------------------------------------------------------------------------
import sys
from types import ModuleType

sys.modules["matplotlib"] = ModuleType("matplotlib")
plt_stub = ModuleType("matplotlib.pyplot")
plt_stub.figure = lambda *a, **k: None
plt_stub.savefig = lambda *a, **k: None
plt_stub.close = lambda *a, **k: None
sys.modules["matplotlib.pyplot"] = plt_stub

# ---------------------------------------------------------------------------
# Imports after stubbing matplotlib
# ---------------------------------------------------------------------------
from core.config import Config
from reporting import summaries as summ
from reporting import generator as gen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_segments() -> List[Dict]:
    """Return a minimal list of annotated segments across two speakers."""
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello",
            "speaker": "SPEAKER_00",
            "emotion": "joy",
            "fused_emotion": "joy",
        },
        {
            "start": 2.0,
            "end": 5.0,
            "text": "sad day",
            "speaker": "SPEAKER_01",
            "emotion": "sadness",
            "fused_emotion": "sadness",
        },
        {
            "start": 6.0,
            "end": 8.0,
            "text": "again",
            "speaker": "SPEAKER_00",
            "emotion": "joy",
            "fused_emotion": "joy",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_summary_basic():
    segs = _dummy_segments()
    summary = summ.build_summary(segs)

    # Per‑speaker keys present
    assert set(summary["per_speaker"].keys()) == {"SPEAKER_00", "SPEAKER_01"}

    # Total time computed correctly (00 has 4s)
    speaker_00 = summary["per_speaker"]["SPEAKER_00"]
    assert abs(speaker_00["total_time"] - 4.0) < 1e-6

    # Global dominant emotion should be joy (2 vs 1)
    assert summary["global"]["dominant_emotion"] == "joy"


def test_generate_all(tmp_path: Path):
    segs = _dummy_segments()

    # Create a throw‑away config with plots & json enabled
    cfg_path = tmp_path / "cfg.json"
    cfg = Config(json_path=cfg_path)
    for key, value in {
        "include_json_summary": True,
        "include_plots": True,
        "include_csv_summary": False,
        "include_script_transcript": False,
        "include_source_audio": False,
        "cleanup_temp_on_success": False,
    }.items():
        cfg.set(key, value)

    art_root = tmp_path / "artifacts"
    manifest = gen.generate_all(segs, cfg, art_root)

    # Manifest should contain summary_json and plots dict with at least trajectory
    summary_json = manifest.get("summary_json")
    assert summary_json and summary_json.exists()

    plots = manifest.get("plots", {})
    assert "trajectory" in plots

    # JSON structure sanity
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "per_speaker" in data and "global" in data
