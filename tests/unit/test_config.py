# test/unit/test_config.py

"""Unit tests for core.config and config.schema"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "user_config.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_defaults_load():
    cfg = Config()
    # basic defaults
    assert cfg.output_dir.name == "output"
    assert cfg.temp_dir.name == "temp"
    assert cfg.device in ("cpu", "cuda")  # auto‑detect may flip
    # fusion weights default to sane values adding ≈ 1
    assert abs(cfg.text_fusion_weight + cfg.audio_fusion_weight - 1.0) < 1e-6


def test_json_override(tmp_path: Path):
    user_cfg = {
        "log_level": "DEBUG",
        "text_fusion_weight": 0.7,
        "audio_fusion_weight": 0.3,
    }
    json_path = _write_json(tmp_path, user_cfg)

    cfg = Config(json_path=json_path)

    assert cfg.log_level == "DEBUG"
    assert abs(cfg.text_fusion_weight + cfg.audio_fusion_weight - 1.0) < 1e-6


def test_save_roundtrip(tmp_path: Path):
    cfg = Config()
    cfg.set("log_level", "WARNING")

    save_path = tmp_path / "roundtrip.json"
    cfg.save(save_path)

    loaded = json.loads(save_path.read_text())
    assert loaded["log_level"] == "WARNING"


@pytest.mark.parametrize("bad_weights", [(0.8, 0.1), (0.8, 0.8)])
def test_bad_weights_normalised(tmp_path: Path, bad_weights):
    text_w, audio_w = bad_weights
    json_path = _write_json(
        tmp_path, {"text_fusion_weight": text_w, "audio_fusion_weight": audio_w}
    )
    cfg = Config(json_path=json_path)
    # internal validator should renormalise to sum≈1
    assert abs(cfg.text_fusion_weight + cfg.audio_fusion_weight - 1.0) < 1e-6
