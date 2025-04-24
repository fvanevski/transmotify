 """Unit tests for speech_analysis.labeling.selector utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pytest

from speech_analysis.labeling import selector as sel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_segments() -> List[Dict[str, Any]]:
    """Return a small, deterministic set of whisperx‑style segments."""
    return [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 5.0, "end": 20.0, "speaker": "SPEAKER_01", "text": "Long talk"},
        {"start": 21.0, "end": 25.0, "speaker": "SPEAKER_00", "text": "Short"},
        {"start": 30.0, "end": 50.0, "speaker": "SPEAKER_01", "text": "Another long"},
        {"start": 55.0, "end": 60.0, "speaker": "SPEAKER_02", "text": "Tiny"},
    ]


# ---------------------------------------------------------------------------
# identify_eligible_speakers
# ---------------------------------------------------------------------------

def test_identify_eligible_speakers(synthetic_segments):
    eligible = sel.identify_eligible_speakers(
        synthetic_segments, min_total_time=15.0, min_block_time=10.0
    )
    assert eligible == ["SPEAKER_01"]  # only speaker 01 meets thresholds


# ---------------------------------------------------------------------------
# select_preview_time_segments
# ---------------------------------------------------------------------------

def test_select_preview_time_segments_primary(synthetic_segments):
    starts = sel.select_preview_time_segments(
        speaker_id="SPEAKER_01",
        segments=synthetic_segments,
        preview_duration=5.0,
        min_block_time=10.0,
    )
    # Should pick start of the two long blocks: 5 and 30 seconds
    assert starts[:2] == [5, 30]


def test_select_preview_time_segments_fallback(synthetic_segments):
    # Speaker 00 has only short blocks (<10s) but longest is 4s; fallback should give at least one.
    starts = sel.select_preview_time_segments(
        speaker_id="SPEAKER_00",
        segments=synthetic_segments,
        preview_duration=3.0,
        min_block_time=10.0,
    )
    # Should return non‑empty list with ints
    assert starts and all(isinstance(x, int) for x in starts)


# ---------------------------------------------------------------------------
# match_snippets_to_speakers (happy path)
# ---------------------------------------------------------------------------

def test_match_snippets_to_speakers(synthetic_segments):
    snippets = {"Alice": "long talk another long"}
    mapping = sel.match_snippets_to_speakers(
        synthetic_segments, speaker_snippet_map=snippets, threshold=0.5
    )
    assert mapping == {"SPEAKER_01": "Alice"}
