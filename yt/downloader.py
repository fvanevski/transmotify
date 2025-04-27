# yt/downloader.py
"""
Handles downloading audio/video streams from YouTube using yt-dlp.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, TextIO, Union

logger = logging.getLogger(__name__)

# Assuming utils.wrapper is available from previous phases
try:
    from utils.wrapper import safe_run
except ImportError:
    # Fallback dummy safe_run if wrapper is missing
    def safe_run(*args, **kwargs):
        raise RuntimeError("utils.wrapper.safe_run not available")


def download_youtube_stream(
    youtube_url: str,
    output_path: Union[str, Path],
    youtube_dl_format: str = "bestaudio/best",
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[YT DL]",
) -> Optional[Path]:
    """
    Downloads the best audio stream from a YouTube URL using yt-dlp.

    Args:
        youtube_url: The URL of the YouTube video.
        output_path: The desired path for the downloaded stream file (e.g., .webm, .m4a).
        youtube_dl_format: The format string for yt-dlp (default: 'bestaudio/best').
        log_file_handle: Optional file handle for logging subprocess output.
        log_prefix: Prefix for log messages.

    Returns:
        The Path object to the downloaded file if successful, None otherwise.
    """
    output_path_obj = Path(output_path)
    logger.info(f"{log_prefix} Starting download for {youtube_url} to {output_path_obj}")

    # Ensure output directory exists (using file_manager utility eventually)
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"{log_prefix} Failed to create parent directory {output_path_obj.parent}: {e}"
        )
        return None

    command: List[str] = [
        "yt-dlp",
        "-f",
        youtube_dl_format,
        "--no-playlist",  # Ensure only single video is downloaded if URL is part of playlist
        "--no-abort-on-error",  # Try to continue if parts fail (e.g. subtitles)
        "-o",
        str(output_path_obj),  # Specify exact output path
        "--",  # End of options, ensures URL is treated as positional argument
        youtube_url,
    ]

    def yt_dlp_output_callback(line: str):
        """Parses yt-dlp output for logging."""
        progress_match = re.search(
            r"\[download\]\s+([\d.]+%) of\s+~?([\d.]+\s*\w+)\s+at\s+([\d.]+\s*\w+/s)\s+ETA\s+([\d:]+)",
            line,
        )
        if progress_match:
            logger.info(
                f"{log_prefix} DL Progress: {progress_match.group(1)} at {progress_match.group(3)} ETA {progress_match.group(4)}"
            )
        elif "[download] Destination:" in line:
            # Logged outside if needed, this line is less informative with -o
            pass
        elif "[info]" in line and "Downloading" not in line:
            logger.info(f"{log_prefix} Info: {line.split(':', 1)[-1].strip()}")
        elif "[ExtractAudio]" in line:
            logger.info(
                f"{log_prefix} Extracting audio..."
            )  # yt-dlp might do internal conversion
        elif "ERROR:" in line:
            # Log error, but don't raise immediately, let safe_run handle exit code
            logger.error(f"{log_prefix} yt-dlp Error Logged: {line.strip()}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=yt_dlp_output_callback,
        )

        if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
            logger.error(
                f"{log_prefix} Download finished, but output file is missing or empty: {output_path_obj}"
            )
            raise FileNotFoundError(
                f"yt-dlp failed to create a valid output file at {output_path_obj}"
            )

        logger.info(f"{log_prefix} Download completed successfully: {output_path_obj}")
        return output_path_obj

    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"{log_prefix} Download failed for {youtube_url}: {e}")
        # Attempt cleanup of potentially incomplete file
        output_path_obj.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.error(
            f"{log_prefix} Unexpected error during download for {youtube_url}: {e}"
        )
        output_path_obj.unlink(missing_ok=True)
        return None
