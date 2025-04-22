# core/utils.py
import os
import subprocess
import tempfile
import traceback
import re  # Import re for parsing
import string  # Import string for normalization
from pathlib import Path

# Added Callable for type hinting the callback function
# Added Callable for type hinting the callback function
# Added defaultdict, Dict, List, Any, Optional, TextIO, Union, Callable, Generator, Tuple for type hints
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
from collections import defaultdict  # Import defaultdict

# --- ADD LOGGING IMPORTS ---
from .logging import log_warning, log_error, log_info  # Added log_info

# Import pandas to handle potential NaN values from excel reading
import pandas as pd

# Import constants needed for snippet matching
from .constants import DEFAULT_SNIPPET_MATCH_THRESHOLD  # Import the constant

# Import fuzz for snippet matching
from rapidfuzz import fuzz  # Import fuzz

# Define Segment and SegmentsList types
Segment = Dict[str, Any]
SegmentsList = List[Segment]


def create_directory(path: Union[str, Path]) -> None:
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_warning(f"Could not create directory {dir_path}: {e}")


def get_temp_file(suffix: str = "") -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file_path: str = temp_file.name
    temp_file.close()
    return temp_file_path


# safe_run function with both env and output_callback parameters
def safe_run(
    command: List[str],
    log_file_handle: Optional[TextIO],
    session_id: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,  # Parameter for environment variables
    output_callback: Optional[
        Callable[[str], None]
    ] = None,  # Parameter for output processing callback
) -> None:
    safe_command: List[str] = [str(item) for item in command]
    log_prefix: str = f"[{session_id if session_id else 'PROC'}] "
    process: Optional[subprocess.Popen] = None

    # Prepare the environment for the subprocess
    # Start with the current environment and update with provided env dict
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    try:
        # Start the subprocess with the prepared environment
        process = subprocess.Popen(
            safe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=subprocess_env,  # Pass the environment here
        )

        # Read output line by line, write to log file, and pass to callback
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                log_line: str = f"{log_prefix}{line}"
                # Always write to the dedicated process log file if handle is provided
                if log_file_handle:
                    try:
                        log_file_handle.write(log_line)
                        log_file_handle.flush()  # Ensure data is written immediately
                    except Exception as log_e:
                        # Keep fallback print if writing to file handle fails
                        print(
                            f"WARN: Failed to write log line to file handle: {log_e} - Line: {line.strip()}"
                        )  # KEEP PRINT (fallback)

                # Pass the line to the output callback if provided
                if output_callback:
                    try:
                        output_callback(line)
                    except Exception as cb_e:
                        # Log errors occurring within the callback to the main logger
                        cb_error_msg = f"[{session_id if session_id else 'PROC'}] ERROR in output_callback: {cb_e} - Line: {line.strip()}"
                        log_error(cb_error_msg)
                        # Also try to write this callback error to the file handle
                        if log_file_handle:
                            try:
                                log_file_handle.write(f"{cb_error_msg}\n")
                                log_file_handle.flush()
                            except Exception as log_e2:
                                # Fallback print if writing callback error to file handle fails
                                print(
                                    f"WARN: Failed to write callback error to file handle: {log_e2}"
                                )  # KEEP PRINT (fallback)

            process.stdout.close()  # Close the pipe after reading

        # Wait for the process to finish and get the return code
        return_code: int = process.wait()

        # Check for non-zero exit code indicating an error
        if return_code != 0:
            error_msg: str = (
                f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            )
            # Attempt to write the error message to the dedicated process log file
            if log_file_handle:
                try:
                    log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
                except Exception as log_e:
                    # Keep fallback print if writing command error to file handle fails
                    print(
                        f"WARN: Failed to write command error to log file handle: {log_e}"
                    )  # KEEP PRINT (fallback)
            # Log the error using the main application logger
            log_error(error_msg)
            # Raise a Python exception to propagate the failure
            raise RuntimeError(error_msg)

    except FileNotFoundError:
        # Handle case where the command executable is not found
        error_msg = (
            f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        )
        if log_file_handle:
            try:
                log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
            except Exception as log_e:
                print(
                    f"WARN: Failed to write FileNotFoundError to log file handle: {log_e}"
                )  # KEEP PRINT (fallback)
        # Log the error using the main application logger
        log_error(error_msg)
        # Re-raise the error
        raise FileNotFoundError(
            error_msg
        ) from None  # 'from None' prevents linking exceptions

    except Exception as e:
        # Handle any other exceptions that occur during process execution
        error_msg = (
            f"An error occurred while running command {' '.join(safe_command)}: {e}"
        )
        if log_file_handle:
            try:
                log_file_handle.write(
                    f"{log_prefix}ERROR: {error_msg}\n{traceback.format_exc()}\n"
                )
            except Exception as log_e:
                print(
                    f"WARN: Failed to write other exception to log file handle: {log_e}"
                )  # KEEP PRINT (fallback)
        # Log the error and traceback using the main application logger
        log_error(error_msg)
        log_error(traceback.format_exc())

        # Attempt to terminate the subprocess if it's still running
        if process and process.poll() is None:  # poll() returns None if still running
            try:
                process.terminate()  # Request graceful termination
                process.wait(timeout=5)  # Wait a bit for it to terminate
            except subprocess.TimeoutExpired:
                # Keep termination warnings as prints
                print(
                    f"{log_prefix}WARN: Process did not terminate gracefully, killing."
                )  # KEEP PRINT
                process.kill()  # Force kill if terminate timed out
            except Exception as term_err:
                print(
                    f"{log_prefix}WARN: Error terminating process after failure: {term_err}"
                )  # KEEP PRINT
        # Re-raise the original exception, chaining it
        raise RuntimeError(error_msg) from e


# Helper method to parse snippet string from XLSX cell
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
        # Use the same regex logic as the UI parsing
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
        if match:
            name = match.group(1).strip()
            snippet = match.group(2).strip()
            if name and snippet:
                mapping[name] = snippet
            else:
                # Log as warning during development, could be info later
                log_warning(
                    f"Could not parse snippet line effectively from XLSX: '{line}'"
                )
        else:
            log_warning(f"Ignoring invalid snippet line format from XLSX: '{line}'")
    return mapping


# ADDED: Helper function to group consecutive segments by speaker into blocks.
def group_segments_by_speaker(segments: SegmentsList) -> List[Dict[str, Any]]:
    """Group consecutive segments by speaker into blocks."""
    blocks = []
    cur = None
    for i, seg in enumerate(segments):
        spk = seg.get("speaker")
        spk = str(spk) if spk is not None else "unknown"
        text = (seg.get("text") or "").strip()
        start, end = seg.get("start"), seg.get("end")

        if not cur or cur["speaker"] != spk:
            if cur:
                blocks.append(cur)
            cur = {
                "speaker": spk,
                "text": text,
                "start": start,
                "end": end,
                "indices": [i],
            }
        else:
            if text:
                cur["text"] += (" " if cur["text"] else "") + text
            if end is not None:
                cur["end"] = end
            cur["indices"].append(i)

    if cur:
        blocks.append(cur)

    log_info(f"Grouped {len(segments)} segments into {len(blocks)} speaker blocks.")
    return blocks


# ADDED: Fuzzy-match user snippets to speaker IDs using rapidfuzz.fuzz.partial_ratio
def match_snippets_to_speakers(
    segments: SegmentsList,  # Receive segments (should have original SPEAKER_XX IDs)
    speaker_snippet_map: Dict[str, str],
    config: Dict[str, Any],  # Pass config to access threshold
) -> Dict[str, str]:
    """
    Fuzzy-match user snippets to speaker IDs using rapidfuzz.fuzz.partial_ratio
    after aggressive normalization, prioritizing exact substring match.
    Returns a mapping from original WhisperX ID to user-provided name for matched speakers.
    Logs match details or best non-match details.
    Does NOT modify the input segments.
    """
    log_info(
        f"Attempting to match {len(speaker_snippet_map)} snippets to speakers using rapidfuzz.fuzz.partial_ratio after normalization..."
    )
    if not speaker_snippet_map:
        log_info("No speaker snippets provided for matching.")
        return {}

    # Use the helper function to group segments
    blocks = group_segments_by_speaker(segments)

    speaker_id_to_name_mapping: Dict[str, str] = {}
    # This dictionary is used internally to track the best score found for each original speaker ID
    # and prevent a lower-scoring snippet from overriding a higher-scoring one for the same speaker.
    best_match_scores: Dict[str, float] = {}

    # Threshold should be in percentage for rapidfuzz (0-100)
    # Convert the 0-1 threshold from config to 0-100 for rapidfuzz
    thresh = (
        float(
            config.get(  # Use the passed config
                "snippet_match_threshold", DEFAULT_SNIPPET_MATCH_THRESHOLD
            )
        )
        * 100.0
    )
    log_info(f"Using snippet matching threshold: {thresh} (converted for rapidfuzz)")

    # Helper function for aggressive normalization (keep this as it helps comparison)
    # Assumes string and re modules are imported at the top of the file
    def aggressively_normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with a single space and strip leading/trailing whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    for name, snippet in speaker_snippet_map.items():
        if not snippet or len(snippet) < 5:
            log_warning(
                f"Ignoring short or empty snippet for speaker '{name}'. Snippet: '{snippet}')"
            )
            continue

        # Normalize the snippet once
        low = aggressively_normalize(snippet)

        # Store best match info for this snippet across all blocks if no match > thresh is found
        best_match_for_snippet: Dict[str, Any] = {
            "score": -1.0,
            "speaker_id": None,
            "block_text_snippet": "",
            "full_snippet": low,  # Store the normalized snippet itself
        }
        match_found_above_threshold = False

        # Log processed snippet
        log_info(f"Matching snippet for '{name}': '{low}' (length: {len(low)})")
        log_info(
            f"Raw original snippet string repr: {repr(snippet)}"
        )  # Log original snippet repr

        for i, blk in enumerate(blocks):
            sid = blk.get("speaker")
            sid = str(sid) if sid is not None else "unknown"
            txt_raw = blk.get("text", "")

            if not sid or not txt_raw:
                continue

            # Normalize the block text
            txt = aggressively_normalize(txt_raw)

            # Calculate partial_ratio
            ratio = fuzz.partial_ratio(low, txt)

            # Update best match found for THIS snippet (across all blocks)
            if ratio > best_match_for_snippet["score"]:
                best_match_for_snippet["score"] = ratio
                best_match_for_snippet["speaker_id"] = sid
                # Store a snippet of the block text for logging the best match
                best_match_for_snippet["block_text_snippet"] = (
                    txt[:100] + "..." if len(txt) > 100 else txt
                )

            # Compare against the 0-100 threshold
            if ratio >= thresh:
                # Found a match above threshold.
                # Check if this is the best score found *so far* for this specific speaker ID.
                if ratio > best_match_scores.get(sid, -1.0):
                    speaker_id_to_name_mapping[sid] = name
                    best_match_scores[sid] = ratio
                    # Log the successful match
                    log_info(
                        f"Match FOUND for snippet '{name}' (score {ratio:.2f}) against speaker '{sid}'."
                    )
                    # Since we found a match for THIS snippet above the threshold against THIS speaker,
                    # and it's the best score for this speaker so far, we can stop checking blocks for THIS snippet.
                    match_found_above_threshold = True
                    break  # Exit the inner loop (iterating through blocks for the current snippet)

        # After iterating through all blocks for the current snippet:
        # If no match above threshold was found for this snippet, log the best match details found
        if not match_found_above_threshold:
            log_info(
                f"No match >= threshold ({thresh:.2f}) found for snippet '{name}'. Best match found across all blocks was score {best_match_for_snippet['score']:.2f} against speaker '{best_match_for_snippet['speaker_id']}'. Block text snippet: '{best_match_for_snippet['block_text_snippet']}'"
            )

    log_info(
        f"Final snippet mapping (WhisperX ID -> User Name): {speaker_id_to_name_mapping}"
    )
    return speaker_id_to_name_mapping
