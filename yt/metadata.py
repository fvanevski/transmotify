# yt/metadata.py
"""
Handles fetching metadata (title, description, etc.) for YouTube videos using yt-dlp.
"""

import json
import traceback
from typing import Dict, Any, Optional, List
import logging

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str, **kwargs):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str, **kwargs):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str, **kwargs):
        print(f"INFO (logging unavailable): {message}")

    # Dummy safe_run if wrapper is missing
    def safe_run(*args, **kwargs):
        raise RuntimeError("utils.wrapper.safe_run not available")

logger = logging.getLogger(__name__)


def fetch_youtube_metadata(
    youtube_url: str, log_prefix: str = "[YT Meta]"
) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for a YouTube video as a JSON object using yt-dlp.

    Args:
        youtube_url: The URL of the YouTube video.
        log_prefix: Prefix for log messages.

    Returns:
        A dictionary containing video metadata if successful, None otherwise.
        Includes standard yt-dlp fields like 'title', 'description', 'uploader', etc.
    """
    logger.info(f"{log_prefix} Fetching metadata for {youtube_url}...")
    command: List[str] = [
        "yt-dlp",
        "-j",  # Output JSON
        "--no-playlist",  # Ensure metadata for single video
        "--",
        youtube_url,
    ]

    try:
        # Use safe_run to capture the JSON output string
        # Do not pass log_file_handle here to avoid flooding log with JSON
        metadata_output = safe_run(
            command=command,
            log_file_handle=None,  # Avoid writing large JSON to main log
            log_prefix=log_prefix,
            capture_output=True,
            output_callback=lambda line: None,  # Suppress callback logging for this
        )

        if not metadata_output:
            logger.warning(
                f"{log_prefix} No metadata output captured from yt-dlp for {youtube_url}."
            )
            return None  # Indicate failure if no output

        try:
            # Log that we received *some* output before parsing
            # logger.info(f"{log_prefix} Received metadata output (length: {len(metadata_output)}). Parsing...") # Too verbose
            metadata = json.loads(metadata_output)
            logger.info(
                f"{log_prefix} Metadata fetched and parsed successfully for {youtube_url}."
            )
            # Add the original URL for reference
            metadata["input_youtube_url"] = youtube_url
            return metadata

        except json.JSONDecodeError as e:
            logger.error(
                f"{log_prefix} Failed to decode yt-dlp metadata JSON for {youtube_url}: {e}"
            )
            # Log the beginning of the problematic output for debugging
            logger.error(
                f"{log_prefix} Start of problematic metadata output: {metadata_output[:500]}..."
            )
            return None  # Indicate failure
        except Exception as e:
            # Catch other potential errors during parsing
            logger.error(
                f"{log_prefix} Unexpected error processing metadata for {youtube_url}: {e}"
            )
            logger.error(traceback.format_exc())
            return None

    except RuntimeError as e:
        # safe_run already logged the command failure
        logger.error(
            f"{log_prefix} Metadata fetch command failed for {youtube_url}. See previous logs."
        )
        return None
    except Exception as e:
        # Catch unexpected errors during safe_run execution itself
        logger.error(
            f"{log_prefix} Unexpected error during metadata fetch execution for {youtube_url}: {e}"
        )
        return None
