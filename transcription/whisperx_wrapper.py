# transcription/whisperx_wrapper.py

"""transcription.whisperx_wrapper
------------------------------------------------
Thin wrapper around the *whisperx* command‑line interface. Handles masking of
HF tokens in logs, device selection, and automatic discovery of the primary
JSON transcript file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Final
from uuid import uuid4

from core.logging import get_logger
from core.config import Config
from utils.subprocess import run as _run
from constants import (
    INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
    FINAL_STRUCTURED_TRANSCRIPT_NAME,
    EMOTION_SUMMARY_JSON_NAME,
    EMOTION_SUMMARY_CSV_NAME,
    SCRIPT_TRANSCRIPT_NAME,
)

logger = get_logger(__name__)

__all__: Final = ["WhisperXError", "transcribe"]


class WhisperXError(RuntimeError):
    """Raised when the whisperx subprocess fails or JSON output can’t be found."""


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _mask_hf_token(cmd: List[str]) -> str:
    """Return a string version of *cmd* with the `--hf_token` value replaced."""
    masked: list[str] = []
    skip_next = False
    for arg in cmd:
        if skip_next:
            skip_next = False  # don't append real token
            continue
        if arg == "--hf_token":
            masked.extend([arg, "*****"])
            skip_next = True
        else:
            masked.append(arg)
    return " ".join(masked)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def transcribe(audio: Path, cfg: Config, out_dir: Path) -> Path:
    """Run *whisperx* on *audio* and return the path to the primary JSON file."""

    session = uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "whisperx",
        str(audio),
        "--model",
        cfg.whisper_model_size,
        "--diarize",
        "--output_dir",
        str(out_dir),
        "--output_format",
        "json",
        "--device",
        cfg.device,
    ]

    if cfg.hf_token:
        cmd.extend(["--hf_token", cfg.hf_token])
    else:
        logger.warning("%s | running diarisation without HF token; may fail", session)

    # optional params
    if cfg.whisper_language and cfg.whisper_language != "auto":
        cmd.extend(["--language", cfg.whisper_language])
    if cfg.whisper_batch_size:
        cmd.extend(["--batch_size", str(cfg.whisper_batch_size)])
    if cfg.whisper_compute_type:
        cmd.extend(["--compute_type", cfg.whisper_compute_type])

    # min/max speakers (may be None)
    if hasattr(cfg, "diarization_min_speakers") and cfg.diarization_min_speakers:
        cmd.extend(["--min_speakers", str(cfg.diarization_min_speakers)])
    if hasattr(cfg, "diarization_max_speakers") and cfg.diarization_max_speakers:
        cmd.extend(["--max_speakers", str(cfg.diarization_max_speakers)])

    logger.info("%s | executing whisperx: %s", session, _mask_hf_token(cmd))

    try:
        _run(cmd, stream_callback=lambda l: _log_progress(l, session))
    except Exception as exc:
        raise WhisperXError(f"whisperx failed: {exc}") from exc

    # primary JSON is usually <stem>.json. fall back to search otherwise.
    primary = out_dir / f"{audio.stem}.json"
    if primary.exists():
        return primary

    ignore = {
        INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
        FINAL_STRUCTURED_TRANSCRIPT_NAME,
        EMOTION_SUMMARY_JSON_NAME,
        EMOTION_SUMMARY_CSV_NAME,
        SCRIPT_TRANSCRIPT_NAME,
    }

    candidates = [f for f in out_dir.glob("*.json") if f.name not in ignore]
    if not candidates:
        raise WhisperXError("could not locate whisperx JSON output")

    if len(candidates) > 1:
        logger.warning(
            "%s | multiple JSON outputs found; choosing %s", session, candidates[0].name
        )
    return candidates[0]


def _log_progress(line: str, session: str) -> None:
    """Heuristic parsing of whisperx stdout lines for informative logs."""
    line_low = line.lower()
    if "loading model" in line_low:
        logger.info("%s | whisperx: loading model", session)
    elif "detected language:" in line_low:
        logger.info("%s | whisperx: %s", session, line.strip())
    elif re.search(r"transcribing \d+ segments", line_low):
        logger.info("%s | whisperx progress: %s", session, line.strip())
    elif "performing diarization" in line_low:
        logger.info("%s | whisperx: diarisation started", session)
    elif "diarization complete" in line_low:
        logger.info("%s | whisperx: diarisation finished", session)
    elif any(tok in line_low for tok in ("error", "warning")):
        logger.warning("%s | whisperx: %s", session, line.strip())
