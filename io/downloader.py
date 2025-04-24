# io/downloader.py

"""io.downloader
--------------------------------
Utilities for acquiring audio assets from remote sources (currently YouTube)
and converting them into pipeline‑ready WAV files.

The public entry point is :pyfunc:`download_youtube` which orchestrates:

1. **yt‑dlp** download → WebM/Opus (or whatever bestaudio) into a temp dir.
2. Metadata fetch via yt‑dlp JSON dump.
3. Conversion to 16‑kHz mono WAV via :pymod:`io.converter`.

The actual subprocess execution is delegated to
:pyfunc:`utils.subprocess.run` so that we inherit logging and
error handling semantics.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from core.config import Config
from core.logging import get_logger
from utils.paths import ensure_dir
from utils.subprocess import run, SubprocessError

# This module depends on converter but converter does not depend on downloader
from io import converter as _converter  # type: ignore  # will be created in next step

logger = get_logger(__name__)

__all__ = [
    "download_youtube",
]


_DEFAULT_YTDLP_FORMAT = "bestaudio/best"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _progress_cb(session: str):
    """Return a callback that parses yt‑dlp download progress lines."""

    prog_rx = re.compile(
        r"\[download\]\s+(?P<pct>[\d.]+%) of\s+~?(?P<size>[\d.]+\s*\w+)\s+at\s+(?P<speed>[\d.]+\s*\w+/s)\s+ETA\s+(?P<eta>[\d:]+)"
    )

    def _cb(line: str) -> None:  # noqa: D401
        m = prog_rx.search(line)
        if m:
            logger.debug(
                "%s yt-dlp %s %s ETA %s",
                session,
                m.group("pct"),
                m.group("speed"),
                m.group("eta"),
            )
        elif "[download] Destination:" in line:
            logger.info("%s destination %s", session, line.split(":", 1)[1].strip())

    return _cb


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def download_youtube(
    url: str,
    cfg: Config,
    *,
    tmp: Path | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Download *url* with **yt-dlp** and return a 16‑kHz mono WAV plus metadata.

    Parameters
    ----------
    url:
        The YouTube watch URL.
    cfg:
        Validated :class:`core.config.Config` object.
    tmp:
        Optional directory for intermediate files; defaults to ``cfg.temp_dir``.

    Returns
    -------
    (wav_path, metadata)
        ``wav_path`` is a Path to the converted WAV file; ``metadata`` is a dict
        with keys like ``video_title``, ``upload_date`` etc.
    """

    session = uuid.uuid4().hex[:8]
    tmp_dir = ensure_dir(tmp or Path(cfg.temp_dir) / "youtube")

    webm_path = tmp_dir / f"audio_{session}.webm"
    wav_path = tmp_dir / f"audio_{session}.wav"

    yt_format = getattr(cfg, "youtube_dl_format", _DEFAULT_YTDLP_FORMAT)

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    cmd = [
        "yt-dlp",
        "-f",
        yt_format,
        "-o",
        str(webm_path),
        "--",
        url,
    ]

    logger.info("%s yt-dlp starting", session)
    try:
        run(cmd, stream_callback=_progress_cb(session))
    except SubprocessError as e:
        raise RuntimeError(f"yt-dlp failed ({e})") from e

    if not webm_path.exists():
        raise FileNotFoundError(f"yt-dlp claims success but {webm_path} is missing")

    # ------------------------------------------------------------------
    # 2. Convert → WAV (delegated)
    # ------------------------------------------------------------------
    _converter.convert_to_wav(webm_path, wav_path, cfg)

    if not wav_path.exists():
        raise FileNotFoundError("ffmpeg conversion did not produce WAV file")

    # ------------------------------------------------------------------
    # 3. Metadata fetch
    # ------------------------------------------------------------------
    meta_cmd = ["yt-dlp", "-j", "--", url]
    metadata: Dict[str, Any] = {"youtube_url": url}
    try:
        out = run(meta_cmd, capture_output=True)
        video_info = json.loads(out) if out else {}
        metadata.update(
            {
                "video_title": video_info.get("fulltitle"),
                "video_description": video_info.get("description"),
                "video_uploader": video_info.get("uploader"),
                "video_creators": video_info.get("creators"),
                "upload_date": video_info.get("upload_date"),
                "release_date": video_info.get("release_date"),
            }
        )
    except (SubprocessError, json.JSONDecodeError):
        logger.warning("%s metadata fetch failed", session)

    # Clean‑up intermediate file but ignore errors
    webm_path.unlink(missing_ok=True)

    logger.info("%s download complete → %s", session, wav_path.name)
    return wav_path, metadata
