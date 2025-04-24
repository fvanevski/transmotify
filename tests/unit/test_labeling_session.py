 """Tests for speech_analysis.labeling.session.LabelingSession."""

from __future__ import annotations

from typing import List, Dict

import pytest

from speech_analysis.labeling.session import LabelingSession, _ItemState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_segments() -> List[Dict]:
    """Create a tiny synthetic transcript with two speakers."""
    return [
        {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00", "text": "hello"},
        {"start": 9.0, "end": 22.0, "speaker": "SPEAKER_01", "text": "world"},
    ]


# Monkey‑patch selector functions so tests are deterministic
@pytest.fixture(autouse=True)
def _patch_selector(monkeypatch):
    from speech_analysis.labeling import selector as sel

    monkeypatch.setattr(
        sel,
        "identify_eligible_speakers",
        lambda segments, *_: ["SPEAKER_00", "SPEAKER_01"],
    )
    monkeypatch.setattr(
        sel,
        "select_preview_time_segments",
        lambda speaker_id, *_: [0, 5],
    )
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_flow():
    sess = LabelingSession(preview_duration=5.0, min_block_time=4.0)
    segs = _mini_segments()
    sess.add_item("item1", "https://youtu.be/dummy", segs)

    # Start should give first speaker
    spk, url, starts = sess.start("item1")
    assert spk == "SPEAKER_00"
    assert starts == [0, 5]

    # Store label for first speaker and advance
    sess.store_label("item1", spk, "Alice")
    nxt = sess.next("item1")
    assert nxt[0] == "SPEAKER_01"

    # Finalize relabels segments
    sess.store_label("item1", "SPEAKER_01", "Bob")
    relabeled_segs, label_map = sess.finalize_item("item1")
    assert label_map == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    assert relabeled_segs[0]["speaker"] == "Alice"
    assert relabeled_segs[1]["speaker"] == "Bob"

    # Session now has no pending items
    assert sess.pending_items() == []
