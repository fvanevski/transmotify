# core/utils.py
import os
import subprocess
import tempfile
import traceback
import re
import string
import json  # Added for convert_floats if needed later, good practice
from pathlib import Path
from datetime import datetime  # Added for save_script_transcript timestamps

# Typing imports
from typing import (
    List,
    Optional,
    TextIO,
    Union,
    Dict,
    Callable,
    Any,
    Generator,
    Tuple,
)
from collections import defaultdict, Counter  # Added Counter

# Logging imports
from .logging import log_warning, log_error, log_info

# Import pandas for reading excel and handling NaN
import pandas as pd

# Import constants needed
from .constants import (
    DEFAULT_SNIPPET_MATCH_THRESHOLD,
    SCRIPT_TRANSCRIPT_NAME,  # Added for save_script_transcript
)

# Import fuzz for snippet matching
from rapidfuzz import fuzz

# Define Segment and SegmentsList types
Segment = Dict[str, Any]
SegmentsList = List[Segment]


# --- Existing Utility Functions ---


def create_directory(path: Union[str, Path]) -> None:
    """Creates a directory if it doesn't exist."""
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Log creation (optional, could add later if needed)
        # log_info(f"Ensured directory exists: {dir_path}")
    except OSError as e:
        # Log warning if creation fails
        log_warning(f"Could not create directory {dir_path}: {e}")
    except Exception as e:
        # Catch other potential exceptions during directory creation
        log_error(f"Unexpected error creating directory {dir_path}: {e}")


def get_temp_file(suffix: str = "") -> str:
    """Creates a temporary file and returns its path."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file_path: str = temp_file.name
        temp_file.close()
        return temp_file_path
    except Exception as e:
        log_error(f"Failed to create temporary file: {e}")
        # Re-raise the exception after logging, as the caller expects a path
        raise


def safe_run(
    command: List[str],
    log_file_handle: Optional[TextIO],
    session_id: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Runs an external command safely, logging output and handling errors.

    Args:
        command: The command and arguments as a list of strings.
        log_file_handle: An open file handle for writing process logs.
        session_id: An identifier for the session/process for logging.
        env: Optional dictionary of environment variables for the subprocess.
        output_callback: Optional function to process command output lines in real-time.
    """
    safe_command: List[str] = [str(item) for item in command]
    log_prefix: str = f"[{session_id if session_id else 'PROC'}] "
    process: Optional[subprocess.Popen] = None

    # Prepare the environment for the subprocess
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    try:
        # Start the subprocess
        process = subprocess.Popen(
            safe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_env,
        )

        # Read output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                log_line: str = f"{log_prefix}{line}"
                # Write to the dedicated process log file
                if log_file_handle and not log_file_handle.closed:
                    try:
                        log_file_handle.write(log_line)
                        log_file_handle.flush()
                    except Exception as log_e:
                        # Fallback print if writing to file fails
                        print(
                            f"WARN: Failed to write log line to file handle: {log_e} - Line: {line.strip()}"
                        )

                # Pass the line to the output callback
                if output_callback:
                    try:
                        output_callback(line)
                    except Exception as cb_e:
                        # Log errors in the callback
                        cb_error_msg = f"[{session_id if session_id else 'PROC'}] ERROR in output_callback: {cb_e} - Line: {line.strip()}"
                        log_error(cb_error_msg)
                        # Try to write callback error to the file handle
                        if log_file_handle and not log_file_handle.closed:
                            try:
                                log_file_handle.write(f"{cb_error_msg}\n")
                                log_file_handle.flush()
                            except Exception as log_e2:
                                print(
                                    f"WARN: Failed to write callback error to file handle: {log_e2}"
                                )

            process.stdout.close()

        # Wait for the process and check return code
        return_code: int = process.wait()
        if return_code != 0:
            error_msg: str = (
                f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            )
            # Write error to process log file
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
                except Exception as log_e:
                    print(
                        f"WARN: Failed to write command error to log file handle: {log_e}"
                    )
            # Log error using main logger
            log_error(error_msg)
            raise RuntimeError(error_msg)

    except FileNotFoundError:
        error_msg = (
            f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
            except Exception as log_e:
                print(
                    f"WARN: Failed to write FileNotFoundError to log file handle: {log_e}"
                )
        log_error(error_msg)
        raise FileNotFoundError(error_msg) from None

    except Exception as e:
        error_msg = (
            f"An error occurred while running command {' '.join(safe_command)}: {e}"
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(
                    f"{log_prefix}ERROR: {error_msg}\n{traceback.format_exc()}\n"
                )
            except Exception as log_e:
                print(
                    f"WARN: Failed to write other exception to log file handle: {log_e}"
                )
        log_error(error_msg)
        log_error(traceback.format_exc())

        # Attempt to terminate the process
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"{log_prefix}WARN: Process did not terminate gracefully, killing."
                )
                process.kill()
            except Exception as term_err:
                print(
                    f"{log_prefix}WARN: Error terminating process after failure: {term_err}"
                )
        raise RuntimeError(error_msg) from e


def parse_xlsx_snippets(snippet_string: Any) -> Dict[str, str]:
    """
    Parses a string from an XLSX cell into a Dict[Name, Snippet].
    Handles None, NaN, and non-string types from pandas read_excel.
    """
    mapping = {}
    # Check for pandas NaN or None explicitly
    if (
        pd.isna(snippet_string)
        or not isinstance(snippet_string, str)
        or not snippet_string.strip()
    ):
        return mapping

    lines = snippet_string.strip().split("\n")
    for line in lines:
        # Regex to capture "Speaker Name: Snippet Text"
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
        if match:
            name = match.group(1).strip()
            snippet = match.group(2).strip()
            if name and snippet:
                mapping[name] = snippet
            else:
                log_warning(
                    f"Could not parse snippet line effectively from XLSX: '{line}'"
                )
        else:
            log_warning(f"Ignoring invalid snippet line format from XLSX: '{line}'")
    return mapping


def group_segments_by_speaker(segments: SegmentsList) -> List[Dict[str, Any]]:
    """Group consecutive segments by the same speaker into blocks."""
    blocks = []
    if not segments:  # Handle empty input list
        return blocks

    # Initialize current block with the first segment
    first_seg = segments[0]
    current_speaker = str(first_seg.get("speaker", "unknown"))
    current_text = (first_seg.get("text") or "").strip()
    current_start = first_seg.get("start")
    current_end = first_seg.get("end")
    current_indices = [0]

    # Iterate through the rest of the segments
    for i, seg in enumerate(segments[1:], start=1):
        speaker = str(seg.get("speaker", "unknown"))
        text = (seg.get("text") or "").strip()
        start = seg.get("start")  # Not used for merging, but good to have
        end = seg.get("end")

        # If the speaker is the same, append to the current block
        if speaker == current_speaker:
            if text:
                current_text += (" " if current_text else "") + text
            # Update the end time to the end time of the current segment
            if end is not None:
                current_end = end
            current_indices.append(i)
        else:
            # If the speaker changes, finalize the previous block
            blocks.append(
                {
                    "speaker": current_speaker,
                    "text": current_text,
                    "start": current_start,
                    "end": current_end,
                    "indices": current_indices,
                }
            )
            # Start a new block with the current segment
            current_speaker = speaker
            current_text = text
            current_start = start
            current_end = end
            current_indices = [i]

    # Add the last accumulated block after the loop finishes
    if current_indices:  # Ensure there was at least one segment
        blocks.append(
            {
                "speaker": current_speaker,
                "text": current_text,
                "start": current_start,
                "end": current_end,
                "indices": current_indices,
            }
        )

    log_info(f"Grouped {len(segments)} segments into {len(blocks)} speaker blocks.")
    return blocks


def match_snippets_to_speakers(
    segments: SegmentsList,
    speaker_snippet_map: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Fuzzy-match user snippets to speaker IDs using rapidfuzz.fuzz.partial_ratio
    after aggressive normalization. Returns a mapping from original WhisperX ID
    (e.g., 'SPEAKER_00') to user-provided name (e.g., 'Alice').
    """
    log_info(
        f"Attempting to match {len(speaker_snippet_map)} snippets to speakers using rapidfuzz..."
    )
    if not speaker_snippet_map or not segments:
        log_info("No speaker snippets provided or no segments to match against.")
        return {}

    # Group segments into blocks for more context during matching
    blocks = group_segments_by_speaker(segments)
    if not blocks:
        log_warning(
            "Segment grouping resulted in zero blocks. Cannot perform snippet matching."
        )
        return {}

    speaker_id_to_name_mapping: Dict[str, str] = {}
    best_match_scores: Dict[str, float] = {}  # Track best score per original speaker ID

    # Get threshold from config, convert 0-1 range to 0-100 for rapidfuzz
    thresh = (
        float(config.get("snippet_match_threshold", DEFAULT_SNIPPET_MATCH_THRESHOLD))
        * 100.0
    )
    log_info(f"Using snippet matching threshold: {thresh:.2f} (for rapidfuzz)")

    # Helper function for aggressive normalization (lowercase, remove punctuation, collapse whitespace)
    def aggressively_normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Iterate through each user-provided snippet
    for user_name, snippet in speaker_snippet_map.items():
        if not snippet or len(snippet) < 5:
            log_warning(
                f"Ignoring short/empty snippet for '{user_name}'. Snippet: '{snippet}'"
            )
            continue

        normalized_snippet = aggressively_normalize(snippet)
        if not normalized_snippet:
            log_warning(f"Snippet for '{user_name}' became empty after normalization.")
            continue

        best_match_for_this_snippet = {"score": -1.0, "speaker_id": None}
        match_found_above_threshold = False

        log_info(f"Matching snippet for '{user_name}': '{normalized_snippet}'")

        # Compare snippet against each speaker block
        for blk in blocks:
            original_speaker_id = blk.get("speaker", "unknown")  # This is SPEAKER_XX
            block_text = blk.get("text", "")

            if not block_text:
                continue

            normalized_block_text = aggressively_normalize(block_text)
            if not normalized_block_text:
                continue

            # Calculate similarity score
            ratio = fuzz.partial_ratio(normalized_snippet, normalized_block_text)

            # Update best match found for THIS snippet across all blocks
            if ratio > best_match_for_this_snippet["score"]:
                best_match_for_this_snippet["score"] = ratio
                best_match_for_this_snippet["speaker_id"] = original_speaker_id

            # Check if score meets threshold
            if ratio >= thresh:
                # Check if this is the best score found so far for THIS specific original_speaker_id
                if ratio > best_match_scores.get(original_speaker_id, -1.0):
                    # Assign mapping: original_speaker_id -> user_name
                    speaker_id_to_name_mapping[original_speaker_id] = user_name
                    best_match_scores[original_speaker_id] = ratio
                    log_info(
                        f"Match FOUND for snippet '{user_name}' (score {ratio:.2f}) "
                        f"assigned to speaker '{original_speaker_id}'. Previous best score for this speaker: {best_match_scores.get(original_speaker_id, -1.0):.2f}"
                    )
                    match_found_above_threshold = True
                    # Optimization: If a good match is found, we can potentially break
                    # if we assume one snippet maps to one speaker block uniquely.
                    # However, a speaker might match multiple snippets, or one snippet
                    # might match multiple blocks weakly. Let's continue checking all blocks
                    # for this snippet to ensure the *best* scoring block determines the match,
                    # but only update the mapping if it improves the score for that speaker_id.

        # Log if no match above threshold was found for the current snippet
        if not match_found_above_threshold:
            log_info(
                f"No match >= threshold ({thresh:.2f}) found for snippet '{user_name}'. "
                f"Best match was score {best_match_for_this_snippet['score']:.2f} "
                f"against speaker '{best_match_for_this_snippet['speaker_id']}'."
            )

    log_info(
        f"Final snippet mapping (WhisperX ID -> User Name): {speaker_id_to_name_mapping}"
    )
    return speaker_id_to_name_mapping


# --- NEW Functions Moved from pipeline.py ---


def convert_floats(obj: Any) -> Any:
    """
    Recursively converts numpy float types (like float32) within nested lists
    and dictionaries to standard Python floats for JSON compatibility.
    """
    if isinstance(obj, dict):
        # Recursively process dictionary values
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process list elements
        return [convert_floats(elem) for elem in obj]
    # Check specifically for numpy float types by name if direct type check is problematic
    elif type(obj).__module__ == "numpy" and "float" in type(obj).__name__:
        return float(obj)
    # Handle potential pandas Timestamps or other non-serializable types if needed
    # elif isinstance(obj, pd.Timestamp):
    #     return obj.isoformat()
    else:
        # Return object as is if it's not a dict, list, or numpy float
        return obj


def run_ffprobe_duration_check(audio_path: Path, min_duration: float = 5.0) -> bool:
    """
    Checks the duration of an audio file using ffprobe.

    Args:
        audio_path: Path to the audio file.
        min_duration: Minimum duration in seconds required for the check to pass.

    Returns:
        True if duration check passes (or ffprobe fails safely),
        False if audio is shorter than min_duration.
    """
    try:
        command = [
            "ffprobe",
            "-v",
            "error",  # Only show errors
            "-show_entries",
            "format=duration",  # Get duration from format section
            "-of",
            "default=noprint_wrappers=1:nokey=1",  # Output only the value
            str(audio_path),  # Input file path
        ]
        log_info(f"Running ffprobe check: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raise exception on non-zero exit code
        )
        duration_str = result.stdout.strip()
        if not duration_str or duration_str == "N/A":
            log_warning(
                f"ffprobe could not determine duration for {audio_path}. Proceeding."
            )
            return True  # Proceed if duration is unknown

        duration = float(duration_str)
        log_info(
            f"Detected audio duration: {duration:.2f} seconds for {audio_path.name}"
        )
        if duration < min_duration:
            log_warning(
                f"Audio duration ({duration:.1f}s) is less than the minimum "
                f"threshold ({min_duration}s). Diarization quality may be affected."
            )
            return False  # Indicate check failed (too short)
        else:
            return True  # Indicate check passed (long enough)

    except FileNotFoundError:
        log_warning("ffprobe command not found. Skipping audio duration check.")
        return True  # Skip check if ffprobe isn't installed, don't block processing
    except subprocess.CalledProcessError as e:
        log_error(f"ffprobe failed for {audio_path} with exit code {e.returncode}.")
        log_error(f"ffprobe stderr: {e.stderr}")
        log_warning(
            "Duration check failed due to ffprobe error. Proceeding with caution."
        )
        return True  # Proceed even if ffprobe fails, but log the error
    except ValueError as e:
        log_error(f"Could not parse ffprobe duration output '{duration_str}': {e}")
        log_warning(
            "Duration check failed due to parsing error. Proceeding with caution."
        )
        return True  # Proceed if parsing fails
    except Exception as e:
        log_error(f"An unexpected error occurred during ffprobe duration check: {e}")
        log_warning("Duration check failed unexpectedly. Proceeding with caution.")
        return True  # Proceed on other errors


def save_script_transcript(
    segments: SegmentsList, output_dir: Path, suffix: str
) -> Optional[Path]:
    """
    Saves a plain text transcript formatted like a script (Speaker: Text).

    Args:
        segments: List of segment dictionaries (must contain 'speaker' and 'text').
                  Should ideally have speaker labels already applied/finalized.
        output_dir: The directory where the transcript file will be saved.
        suffix: A string to append to the base filename (e.g., job_id or item_identifier).

    Returns:
        The Path object to the saved transcript file, or None if saving failed.
    """
    # Use the constant for the base filename, add the suffix
    script_filename = f"{Path(SCRIPT_TRANSCRIPT_NAME).stem}_{suffix}.txt"
    script_path = output_dir / script_filename
    log_info(f"Attempting to save script transcript to: {script_path}")

    # Ensure output directory exists
    create_directory(output_dir)

    # Group segments first to combine consecutive utterances by the same speaker
    grouped_blocks = group_segments_by_speaker(segments)
    if not grouped_blocks:
        log_warning(
            "No speaker blocks generated from segments. Cannot save script transcript."
        )
        return None

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            for block in grouped_blocks:
                speaker = block.get("speaker", "UNKNOWN")
                text = block.get("text", "").strip()
                start_time = block.get("start")
                end_time = block.get("end")

                # Format timestamp if available
                time_str = ""
                if start_time is not None and end_time is not None:
                    # Helper to format seconds into HH:MM:SS.fff
                    def format_time(seconds):
                        try:
                            secs = float(seconds)
                            hours = int(secs // 3600)
                            minutes = int((secs % 3600) // 60)
                            seconds_rem = secs % 60
                            return f"{hours:02}:{minutes:02}:{seconds_rem:06.3f}"
                        except (ValueError, TypeError):
                            return "??:??:??.???"  # Fallback for invalid time

                    time_str = f"[{format_time(start_time)} - {format_time(end_time)}] "

                # Write line: [TIMESTAMP] SPEAKER: Text
                f.write(
                    f"{time_str}{speaker}: {text}\n\n"
                )  # Add extra newline for readability

        log_info(f"Script transcript saved successfully to: {script_path}")
        return script_path

    except Exception as e:
        log_error(f"Failed to save script transcript to {script_path}: {e}")
        log_error(traceback.format_exc())  # Log full traceback for debugging
        return None
