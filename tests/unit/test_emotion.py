# test/unit/test_emotion.py

"""Unit tests for emotion stack (fusion + analyzer)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict

import numpy as np
import pytest
import torch
import torchaudio

from emotion import fusion as fus
from emotion.analyzer import MultimodalAnalyzer


# ---------------------------------------------------------------------------
# Fusion helper test
# ---------------------------------------------------------------------------


def test_fuse_weighted_average():
    text_probs = {"joy": 0.7, "anger": 0.2, "sadness": 0.1}
    audio_probs = {"joy": 0.1, "anger": 0.8, "sadness": 0.1}

    label, conf = fus.fuse_weighted_average(
        text_probs, audio_probs, text_weight=0.6, audio_weight=0.4
    )

    assert label == "joy"  # 0.7*0.6 + 0.1*0.4 = 0.46 vs anger 0.2*0.6 + 0.8*0.4 = 0.44
    assert pytest.approx(conf, rel=1e-3) == 0.46


# ---------------------------------------------------------------------------
# Analyzer end‑to‑end with mocked models
# ---------------------------------------------------------------------------


class _StubText:
    def predict(self, text: str) -> Dict[str, float]:
        return {"joy": 0.8, "anger": 0.1, "sadness": 0.1}


class _StubAudio:
    def predict(self, waveform, sample_rate):
        return {"joy": 0.2, "anger": 0.7, "sadness": 0.1}


def test_analyzer_integration(monkeypatch, tmp_path):
    # ------------------------------------------------------------------
    # Patch models before instantiating analyzer
    # ------------------------------------------------------------------
    from emotion import text_model as tmod
    from emotion import audio_model as amod

    monkeypatch.setattr(tmod, "TextEmotionModel", lambda *_a, **_k: _StubText())
    monkeypatch.setattr(amod, "AudioEmotionModel", lambda *_a, **_k: _StubAudio())

    # ------------------------------------------------------------------
    # Create dummy audio file (~1 s silence) for two segments
    # ------------------------------------------------------------------
    sr = 16000
    waveform = torch.zeros(sr)  # 1 second mono
    audio_path = tmp_path / "dummy.wav"
    torchaudio.save(str(audio_path), waveform.unsqueeze(0), sr)

    # ------------------------------------------------------------------
    # Prepare segments
    # ------------------------------------------------------------------
    segments: List[Dict] = [
        {"start": 0.0, "end": 0.5, "text": "Hello world", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "text": "Another line", "speaker": "SPEAKER_00"},
    ]

    cfg = SimpleNamespace(
        device="cpu",
        text_fusion_weight=0.6,
        audio_fusion_weight=0.4,
        visual_frame_rate=1,
        deepface_detector_backend="opencv",
        deepface_models=["Emotion"],
    )

    analyzer = MultimodalAnalyzer(cfg)
    out = analyzer.run(segments, audio_path=str(audio_path), video_path="")

    assert len(out) == 2
    for seg in out:
        assert seg["fused_emotion"] in {"joy", "anger", "sadness", "neutral"}
        assert 0 <= seg["fused_emotion_confidence"] <= 1
        assert "emotion" in seg
        assert seg["emotion"] == seg["fused_emotion"]
