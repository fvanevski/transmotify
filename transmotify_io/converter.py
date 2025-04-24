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

# Updated __all__ to include extract_audio_segment
__all__: Final = [
    "ConverterError",
    "convert_to_wav",
    "duration_ok",
    "extract_audio_segment",
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
# ffmpeg conversion and extraction
# ---------------------------------------------------------------------------

def _ffmpeg_to_wav(src: Path, dst: Path, *, channels: int, rate: int) -> None:
    """Invoke ffmpeg to convert *src* → *dst* WAV."""
    cmd = [
        "ffmpeg",
        "-y", # Overwrite output without asking
        "-i",
        str(src),
        "-ac", # Audio channels
        str(channels),
        "-ar", # Audio sample rate
        str(rate),
        "-vn", # No video
        str(dst),
    ]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        raise ConverterError(f"ffmpeg WAV conversion failed: {exc}") from exc


def _ffmpeg_extract_segment(
    src: Path, dst: Path, *, start_sec: float, duration_sec: float
) -> None:
    """Invoke ffmpeg to extract a segment from *src* → *dst*."""
    cmd = [
        "ffmpeg",
        "-y", # Overwrite output without asking
        "-i",
        str(src),
        "-ss", # Start time
        str(start_sec),
        "-t", # Duration
        str(duration_sec),
        "-c", # Codec - copy to avoid re-encoding if possible
        "copy",
        "-vn", # No video
        str(dst),
    ]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        # If codec copy fails (e.g., format change), try re-encoding
        logger.warning("ffmpeg segment extraction with 'copy' codec failed (%s), retrying with re-encoding...", exc)
        # Use standard WAV codec (signed 16-bit PCM little-endian)
        # Also specify channels/rate to match expected WAV format if re-encoding
        cmd = [
            "ffmpeg", "-y", "-i", str(src), "-ss", str(start_sec), "-t", str(duration_sec),
            "-ac", "1", "-ar", "16000", "-vn", str(dst)
        ]
        try:
            _run(cmd)
        except Exception as exc_reencode:
             raise ConverterError(f"ffmpeg segment extraction failed even with re-encoding: {exc_reencode}") from exc_reencode


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


def extract_audio_segment(
    src: Path,
    dst: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> Path:
    """Extracts an audio segment from *src* to *dst* between the given times.

    Args:
        src: Path to the source audio file.
        dst: Path where the extracted segment will be saved.
        start_sec: Start time of the segment in seconds.
        end_sec: End time of the segment in seconds.

    Returns:
        The path to the created segment file (*dst*).

    Raises:
        ConverterError: If ffmpeg fails.
        ValueError: If end_sec <= start_sec.
    """
    if end_sec <= start_sec:
        raise ValueError(f"End time ({end_sec}) must be after start time ({start_sec})")

    duration_sec = end_sec - start_sec
    logger.info("Extracting audio segment from %s [%.3f - %.3f] (%.3fs) -> %s",
                src.name, start_sec, end_sec, duration_sec, dst.name)

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    _ffmpeg_extract_segment(src, dst, start_sec=start_sec, duration_sec=duration_sec)

    if not dst.exists():
        raise ConverterError(f"ffmpeg reported success but extracted segment {dst} not found")

    return dst
