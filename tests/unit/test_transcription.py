# test/unit/test_transcription.py

"""Unit tests for transcription helpers (whisperx wrapper + segments).
These tests mock out heavy dependencies so they run in <100 ms and require
no external binaries.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from transcription.segments import (
    load_segments,
    SegmentLoadError,
)
from transcription.whisperx_wrapper import (
    transcribe,
    WhisperXError,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_json(tmp_path: Path) -> Path:
    """Creates a minimal WhisperX‑style JSON file."""
    content = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.2,
                "text": "hello world",
                "speaker": "SPEAKER_00",
                "words": [],
            },
            {
                "start": 1.3,
                "end": 2.0,
                "text": "second sentence",
                "speaker": "SPEAKER_01",
                "words": [],
            },
        ]
    }
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_segments
# ---------------------------------------------------------------------------


def test_load_segments_basic(tmp_json: Path):
    segs = load_segments(tmp_json)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "SPEAKER_00"


def test_load_segments_bad_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(SegmentLoadError):
        load_segments(bad)


# ---------------------------------------------------------------------------
# transcribe – path discovery logic mocked
# ---------------------------------------------------------------------------


def _mock_run(cmd, **kwargs):  # noqa: D401  (simple helper)
    """Pretend to run whisperx by touching an output file."""
    out_dir = Path(cmd[cmd.index("--output_dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    # create a "fallback" JSON not matching <stem>.json
    (out_dir / "fallback.json").write_text("{}", encoding="utf-8")
    return None  # Capture_output branch unused here


def test_transcribe_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Patch utils.subprocess.run used inside whisperx_wrapper
    from speech_analysis import utils as _u  # dynamic import for patch path

    monkeypatch.setattr("transcription.whisperx_wrapper._run", _mock_run, raising=True)

    cfg = SimpleNamespace(
        whisper_model_size="tiny",
        device="cpu",
        hf_token=None,
        whisper_batch_size=1,
        whisper_language="auto",
        whisper_compute_type="int8",
    )

    audio = tmp_path / "audio.wav"
    audio.touch()

    out_dir = tmp_path / "out"

    json_path = transcribe(audio, cfg, out_dir)
    assert json_path.name == "fallback.json"


def test_transcribe_no_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _run_no_files(cmd, **kwargs):
        Path(cmd[cmd.index("--output_dir") + 1]).mkdir(parents=True, exist_ok=True)
        # no json produced
        return None

    monkeypatch.setattr(
        "transcription.whisperx_wrapper._run",
        _run_no_files,
        raising=True,
    )

    cfg = SimpleNamespace(
        whisper_model_size="tiny",
        device="cpu",
        hf_token=None,
        whisper_batch_size=1,
        whisper_language="auto",
        whisper_compute_type="int8",
    )

    audio = tmp_path / "audio.wav"
    audio.touch()

    with pytest.raises(WhisperXError):
        transcribe(audio, cfg, tmp_path / "out")
