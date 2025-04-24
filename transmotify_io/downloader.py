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
# import re # No longer needed for progress parsing
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from core.config import Config
from core.logging import get_logger
from utils.paths import ensure_dir
from utils.subprocess import run, SubprocessError

import transmotify_io.converter as _converter  # type: ignore

logger = get_logger(__name__)

__all__ = [
    "download_youtube",
]


_DEFAULT_YTDLP_FORMAT = "bestaudio/best"


# ---------------------------------------------------------------------------
# helpers - Progress callback removed as yt-dlp will run quietly
# ---------------------------------------------------------------------------

# def _progress_cb(session: str):
#     """Return a callback that parses yt‑dlp download progress lines."""
#
#     prog_rx = re.compile(
#         r"\[download\]\s+(?P<pct>[\d.]+) of\s+~?(?P<size>[\d.]+\s*\w+)\s+at\s+(?P<speed>[\d.]+\s*\w+/s)\s+ETA\s+(?P<eta>[\d:]+)"
#     )
#
#     def _cb(line: str) -> None:  # noqa: D401
#         m = prog_rx.search(line)
#         if m:
#             logger.debug(
#                 "%s yt-dlp %s %s ETA %s",
#                 session,
#                 m.group("pct"),
#                 m.group("speed"),
#                 m.group("eta"),
#             )
#         elif "[download] Destination:" in line:
#             logger.info("%s destination %s", session, line.split(":", 1)[1].strip())
#
#     return _cb


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

    # Use a generic extension first, yt-dlp might save as .m4a etc.
    download_path_pattern = tmp_dir / f"audio_{session}.%(ext)s"
    wav_path = tmp_dir / f"audio_{session}.wav"

    yt_format = getattr(cfg, "youtube_dl_format", _DEFAULT_YTDLP_FORMAT)

    # ------------------------------------------------------------------
    # 1. Download (quietly)
    # ------------------------------------------------------------------
    cmd = [
        "yt-dlp",
        "--quiet",  # Suppress non-error output
        "-f",
        yt_format,
        "-o",
        str(download_path_pattern), # Use pattern to get actual extension
        "--",
        url,
    ]

    logger.info("%s yt-dlp starting download for %s", session, url)
    downloaded_file_path = None
    try:
        # Run without stream callback, capture output to find filename
        output = run(cmd, capture_output=True)
        # Find the actual downloaded file name (yt-dlp doesn't make this easy when quiet)
        # We'll determine it after conversion, or rely on converter finding it.

        # Simple way: Find the first file matching the pattern
        # This assumes only one file is downloaded per session, which should be true
        potential_files = list(tmp_dir.glob(f"audio_{session}.*"))
        # Exclude the target WAV file if it somehow exists already
        potential_files = [f for f in potential_files if f.suffix != '.wav']
        if not potential_files:
             raise FileNotFoundError(f"Could not find downloaded audio file matching pattern audio_{session}.* in {tmp_dir}")
        downloaded_file_path = potential_files[0]
        logger.info("%s yt-dlp downloaded %s", session, downloaded_file_path.name)

    except SubprocessError as e:
        raise RuntimeError(f"yt-dlp download failed ({e})") from e
    except FileNotFoundError as e:
         raise e # Re-raise specific error
    except Exception as e:
        # Catch other potential errors during file finding
        raise RuntimeError(f"Error finding downloaded file for {session}: {e}") from e


    if not downloaded_file_path or not downloaded_file_path.exists():
        raise FileNotFoundError(f"yt-dlp claims success but downloaded file {downloaded_file_path} is missing")

    # ------------------------------------------------------------------
    # 2. Convert → WAV (delegated)
    # ------------------------------------------------------------------
    # Pass the actual downloaded file path
    wav_path = _converter.convert_to_wav(downloaded_file_path, dst=wav_path, cfg=cfg)

    if not wav_path.exists():
        raise FileNotFoundError(f"ffmpeg conversion from {downloaded_file_path.name} did not produce WAV file {wav_path}")

    # ------------------------------------------------------------------
    # 3. Metadata fetch
    # ------------------------------------------------------------------
    # Use --print-json for clean JSON output, --no-warnings to suppress other messages
    meta_cmd = ["yt-dlp", "--print-json", "--no-warnings", "--", url]
    metadata: Dict[str, Any] = {"youtube_url": url}
    logger.info("%s Fetching metadata for %s", session, url)
    try:
        out = run(meta_cmd, capture_output=True)
        if out:
            video_info = json.loads(out)
            metadata.update(
                {
                    "video_title": video_info.get("fulltitle"),
                    "video_description": video_info.get("description"),
                    "video_uploader": video_info.get("uploader"),
                    "video_creators": video_info.get("creators"),
                    "upload_date": video_info.get("upload_date"), # Format YYYYMMDD
                    "release_date": video_info.get("release_date"), # Format YYYYMMDD
                    "duration_string": video_info.get("duration_string"),
                    "channel": video_info.get("channel"),
                    "channel_url": video_info.get("channel_url"),
                    "thumbnail": video_info.get("thumbnail"),
                }
            )
            logger.info("%s Metadata fetch successful", session)
        else:
            logger.warning("%s metadata fetch returned empty output", session)

    except SubprocessError as e:
        logger.warning("%s metadata fetch failed (yt-dlp error: %s)", session, e)
    except json.JSONDecodeError as e:
        logger.warning("%s metadata fetch failed (JSON decode error: %s)", session, e)
        logger.debug("Faulty JSON string was: %s", out) # Log faulty output only on debug

    # Clean‑up intermediate downloaded file but ignore errors
    if downloaded_file_path and downloaded_file_path.exists():
        downloaded_file_path.unlink(missing_ok=True)

    logger.info("%s Processing complete for %s -> %s", session, url, wav_path.name)
    return wav_path, metadata
