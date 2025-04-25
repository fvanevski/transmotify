# yt/converter.py
"""
Handles converting downloaded media streams to WAV format using ffmpeg
and checking audio duration using ffprobe.
"""

import re
import subprocess  # For duration check fallback if safe_run fails initally
from pathlib import Path
from typing import List, Optional, TextIO, Union

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


# Moved from core/utils.py (via utils/transcripts.py)
# Renamed for clarity within this module
def check_audio_duration(
    audio_path: Path, min_duration_sec: float = 5.0, log_prefix: str = "[FFPROBE]"
) -> bool:
    """
    Checks the duration of an audio file using ffprobe. Logs warnings.

    Args:
        audio_path: Path to the audio file.
        min_duration_sec: Minimum duration in seconds required for processing (e.g., diarization).
        log_prefix: Prefix for log messages.

    Returns:
        True if duration check passes (>= min_duration_sec) or ffprobe fails safely.
        False if audio is shorter than min_duration_sec.
    """
    if not audio_path.is_file():
        log_warning(
            f"{log_prefix} Audio file not found at {audio_path}, cannot check duration."
        )
        return False  # Cannot proceed if file doesn't exist

    log_info(
        f"{log_prefix} Checking audio duration for {audio_path.name} (min: {min_duration_sec}s)..."
    )
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        # Use safe_run for consistency, though direct subprocess call is also possible
        # If safe_run isn't available, this will raise RuntimeError from the fallback above
        duration_str = safe_run(
            command=command,
            log_file_handle=None,  # Don't need separate log for this usually
            log_prefix=log_prefix,
            capture_output=True,
            output_callback=lambda line: None,  # Suppress callback logging for ffprobe
        )

        if duration_str is None:
            # safe_run already logs the error if the command fails
            log_warning(
                f"{log_prefix} ffprobe command failed or returned no output for {audio_path.name}. Proceeding without duration check."
            )
            return True  # Proceed cautiously

        duration_str = duration_str.strip()
        if not duration_str or duration_str == "N/A":
            log_warning(
                f"{log_prefix} ffprobe could not determine duration for {audio_path.name}. Proceeding, quality checks skipped."
            )
            return True  # Proceed if duration is unknown

        duration = float(duration_str)
        log_info(
            f"{log_prefix} Detected audio duration: {duration:.2f} seconds for {audio_path.name}"
        )

        if duration < min_duration_sec:
            log_warning(
                f"{log_prefix} Audio duration ({duration:.1f}s) is less than the minimum "
                f"threshold ({min_duration_sec}s). Downstream processing (e.g., diarization) may be affected or skipped."
            )
            return False  # Indicate check failed (too short)
        else:
            return True  # Indicate check passed (long enough)

    except FileNotFoundError:
        log_warning(
            f"{log_prefix} ffprobe command not found. Cannot perform audio duration check."
        )
        return True  # Proceed if ffprobe is not available
    except ValueError as e:
        log_warning(
            f"{log_prefix} Could not convert ffprobe duration output ('{duration_str}') to float for {audio_path.name}: {e}. Proceeding without check."
        )
        return True
    except Exception as e:
        # Catch any other unexpected errors during the check (e.g., RuntimeError from safe_run)
        log_warning(
            f"{log_prefix} Unexpected error running ffprobe duration check for {audio_path.name}: {e}. Proceeding without check."
        )
        # Consider logging traceback for unexpected errors if needed:
        # import traceback
        # log_warning(traceback.format_exc())
        return True  # Proceed cautiously on other errors


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_channels: int = 1,
    audio_samplerate: int = 16000,
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[FFMPEG]",
) -> Optional[Path]:
    """
    Converts an audio/video file to a standardized WAV format using ffmpeg.

    Args:
        input_path: Path to the input media file.
        output_path: The desired path for the output WAV file.
        audio_channels: Number of audio channels for the output (default: 1 for mono).
        audio_samplerate: Sample rate for the output audio (default: 16000 Hz).
        log_file_handle: Optional file handle for logging subprocess output.
        log_prefix: Prefix for log messages.

    Returns:
        The Path object to the output WAV file if successful, None otherwise.
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    log_info(
        f"{log_prefix} Converting {input_path_obj.name} to WAV at {output_path_obj} (Channels: {audio_channels}, Rate: {audio_samplerate} Hz)"
    )

    if not input_path_obj.is_file():
        log_error(f"{log_prefix} Input file not found: {input_path_obj}")
        return None

    # Ensure output directory exists
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(
            f"{log_prefix} Failed to create parent directory {output_path_obj.parent}: {e}"
        )
        return None

    command: List[str] = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i",
        str(input_path_obj),  # Input file
        "-vn",  # No video output
        "-ac",
        str(audio_channels),  # Set audio channels
        "-ar",
        str(audio_samplerate),  # Set audio sample rate
        "-acodec",
        "pcm_s16le",  # Standard WAV codec (signed 16-bit little-endian PCM)
        "-nostdin",  # Disable interaction
        str(output_path_obj),  # Output file path
    ]

    def ffmpeg_output_callback(line: str):
        """Parses ffmpeg output for logging."""
        # Regex for progress: time=HH:MM:SS.ms bitrate=... speed=X.Yx
        progress_match = re.search(
            r"time=\s*(\d{2}:\d{2}:\d{2}\.\d+).*?speed=\s*([\d.]+)x", line
        )
        size_match = re.search(r"size=\s*(\S+)", line)  # Capture size info if present

        if progress_match:
            time_str = progress_match.group(1)
            speed_str = progress_match.group(2)
            size_str = f" (size: {size_match.group(1)})" if size_match else ""
            log_info(
                f"{log_prefix} Progress: time={time_str}, speed={speed_str}x{size_str}"
            )
        elif "error" in line.lower() or "failed" in line.lower():
            log_error(f"{log_prefix} ffmpeg Error Logged: {line.strip()}")
        # elif "warning" in line.lower(): # Often too verbose
        #    log_warning(f"{log_prefix} ffmpeg Warning: {line.strip()}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=ffmpeg_output_callback,
        )

        if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
            log_error(
                f"{log_prefix} Conversion finished, but output file is missing or empty: {output_path_obj}"
            )
            raise FileNotFoundError(
                f"ffmpeg failed to create a valid output file at {output_path_obj}"
            )

        log_info(
            f"{log_prefix} Conversion to WAV completed successfully: {output_path_obj}"
        )
        return output_path_obj

    except (RuntimeError, FileNotFoundError) as e:
        log_error(f"{log_prefix} Conversion failed for {input_path_obj.name}: {e}")
        output_path_obj.unlink(missing_ok=True)  # Cleanup failed output
        return None
    except Exception as e:
        log_error(
            f"{log_prefix} Unexpected error during conversion for {input_path_obj.name}: {e}"
        )
        output_path_obj.unlink(missing_ok=True)
        return None
