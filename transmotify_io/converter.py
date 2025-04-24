# io/converter.py

"""io.converter
--------------------------------
Audio conversion helpers built on ffmpeg/ffprobe.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final, Optional

from core.logging import get_logger
from utils.subprocess import run as _run

logger = get_logger(__name__)

__all__: Final = [
    "ConverterError",
    "convert_to_wav",
    "duration_ok",
]


class ConverterError(RuntimeError):
    """Raised when ffmpeg/ffprobe operations fail."""




def duration_ok(audio: Path, *, min_sec: float = 5.0) -> bool:
    """Return *True* if *audio* duration >= *min_sec* (or duration cannot be read).

    Uses *ffprobe* so requires it to be on PATH.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio),
    ]
    try:
        out = _run(cmd, capture_output=True)
        dur = float(out.strip()) if out and out.strip() != "N/A" else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffprobe failed (%s); skipping duration check", exc)
        return True

    if dur is None:
        logger.warning("Duration unknown for %s – continuing", audio.name)
        return True

    if dur < min_sec:
        logger.warning("Audio %s too short (%.2fs < %.2fs)", audio.name, dur, min_sec)
        return False
    return True


# ---------------------------------------------------------------------------
# ffmpeg conversion
# ---------------------------------------------------------------------------


def _ffmpeg_to_wav(src: Path, dst: Path, *, channels: int, rate: int) -> None:
    """Invoke ffmpeg to convert *src* → *dst* WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        str(channels),
        "-ar",
        str(rate),
        "-vn",
        str(dst),
    ]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        raise ConverterError(f"ffmpeg failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_to_wav(
    src: Path,
    *,
    cfg,  # core.config.Config – imported via forward ref to avoid cycle
    dst: Path
) -> Path:
    """Convert *src* audio file to 16‑kHz mono WAV and return the new path.

    """
    channels: int = int(getattr(cfg, "ffmpeg_audio_channels", 1))
    rate: int = int(getattr(cfg, "ffmpeg_audio_samplerate", 16000))

    logger.info("Converting %s → %s (ac=%d, ar=%d)", src.name, dst.name, channels, rate)

    _ffmpeg_to_wav(src, dst, channels=channels, rate=rate)
    if not dst.exists():
        raise ConverterError(f"ffmpeg reported success but {dst} not found")

    return dst
