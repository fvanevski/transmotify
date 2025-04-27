# utils/transcripts.py
"""
Utility functions for processing, structuring, and manipulating transcript data,
including segment grouping, snippet matching, and saving formats.
"""

import json
import re
import string
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TextIO, Union
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Use rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz

    FUZZ_AVAILABLE = True
except ImportError:
    FUZZ_AVAILABLE = False

# Use pandas for reading excel and NaN handling
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Assuming core.logging is established from Phase 1
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


if not FUZZ_AVAILABLE:
    logger.warning("Rapidfuzz library not found. Snippet matching will be unavailable.")
if not PANDAS_AVAILABLE:
    logger.warning("Pandas library not found. Reading XLSX snippets will be unavailable.")

# Type hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]


# Moved from core/utils.py
def parse_xlsx_snippets(snippet_string: Any) -> Dict[str, str]:
    """
    Parses a string (potentially from an XLSX cell) into a Dict[SpeakerName, SnippetText].
    Handles None, NaN, and non-string types gracefully. Requires pandas.
    """
    if not PANDAS_AVAILABLE:
        logger.error("Pandas not available, cannot parse XLSX snippets.")
        return {}

    mapping = {}
    # Check for pandas NaN or None explicitly, or if not a string
    if (
        pd.isna(snippet_string)
        or not isinstance(snippet_string, str)
        or not snippet_string.strip()
    ):
        return mapping  # Return empty if input is invalid

    lines = snippet_string.strip().split("\n")
    for line in lines:
        # Regex to capture "Speaker Name: Snippet Text" potentially with extra whitespace
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
        if match:
            name = match.group(1).strip()
            snippet = match.group(2).strip()
            if name and snippet:
                mapping[name] = snippet
                logger.info(
                    f"Parsed snippet - Name: '{name}', Snippet: '{snippet[:50]}...'"
                )
            else:
                logger.warning(
                    f"Could not parse snippet line effectively (empty name or snippet): '{line}'"
                )
        else:
            logger.warning(f"Ignoring invalid snippet line format: '{line}'")
    return mapping


# Moved from core/utils.py
def group_segments_by_speaker(segments: SegmentsList) -> List[Dict[str, Any]]:
    """Groups consecutive segments by the same speaker into dialogue blocks."""
    blocks = []
    if not segments:
        logger.info("No segments provided for grouping.")
        return blocks

    logger.info(f"Grouping {len(segments)} segments by speaker...")

    # Use an iterator to handle the first segment initialization cleanly
    segment_iter = iter(segments)
    try:
        first_seg = next(segment_iter)
    except StopIteration:
        logger.info("Segment list was empty after iterator check.")
        return blocks  # Empty list

    # Initialize current block with the first segment
    current_speaker = str(first_seg.get("speaker", "unknown"))
    current_text = (first_seg.get("text") or "").strip()
    current_start = first_seg.get("start")
    current_end = first_seg.get("end")
    current_indices = [0]  # Store original index of segments in the block
    # Add other segment keys if needed in the block (e.g., emotion, words)
    # current_words = first_seg.get("words", [])

    # Iterate through the rest of the segments
    for i, seg in enumerate(segment_iter, start=1):
        speaker = str(seg.get("speaker", "unknown"))
        text = (seg.get("text") or "").strip()
        start = seg.get("start")
        end = seg.get("end")
        # words = seg.get("words", []) # Get words if needed

        # If the speaker is the same, append text and update end time
        if speaker == current_speaker:
            if text:
                current_text += (
                    (" " + text) if current_text else text
                )  # Add space only if needed
            # Update the end time to the end time of the current segment
            if end is not None:
                current_end = end  # Take the latest end time
            current_indices.append(i)
            # current_words.extend(words) # Append words if tracking them per block
        else:
            # Speaker changed: finalize the previous block
            if (
                current_start is not None and current_end is not None
            ):  # Only add blocks with valid times
                blocks.append(
                    {
                        "speaker": current_speaker,
                        "text": current_text,
                        "start": current_start,
                        "end": current_end,
                        "indices": current_indices,
                        # "words": current_words, # Add words if needed
                    }
                )
            else:
                logger.warning(
                    f"Skipping block for speaker '{current_speaker}' due to invalid start/end times."
                )

            # Start a new block with the current segment
            current_speaker = speaker
            current_text = text
            current_start = start
            current_end = end
            current_indices = [i]
            # current_words = words

    # Add the last accumulated block after the loop finishes
    if (
        current_start is not None and current_end is not None
    ):  # Check times for the last block
        blocks.append(
            {
                "speaker": current_speaker,
                "text": current_text,
                "start": current_start,
                "end": current_end,
                "indices": current_indices,
                # "words": current_words,
            }
        )
    else:
        logger.warning(
            f"Skipping final block for speaker '{current_speaker}' due to invalid start/end times."
        )

    logger.info(f"Grouped into {len(blocks)} speaker blocks.")
    return blocks


# Moved from core/utils.py
def match_snippets_to_speakers(
    segments: SegmentsList,
    speaker_snippet_map: Dict[str, str],  # User Name -> Snippet Text
    match_threshold: float = 0.80,  # Default threshold (0.0-1.0)
) -> Dict[str, str]:  # Returns WhisperX ID ('SPEAKER_XX') -> User Name
    """
    Fuzzy-match user snippets to speaker dialogue blocks using rapidfuzz.
    Requires rapidfuzz library.

    Args:
        segments: List of segment dictionaries from transcription.
        speaker_snippet_map: Dictionary mapping user-provided names to text snippets.
        match_threshold: Minimum similarity score (0.0 to 1.0) for a match.

    Returns:
        A dictionary mapping original WhisperX speaker IDs (e.g., 'SPEAKER_00')
        to the user-provided name (e.g., 'Alice') based on the best match above threshold.
    """
    if not FUZZ_AVAILABLE:
        logger.error("Rapidfuzz library not available. Cannot perform snippet matching.")
        return {}

    logger.info(
        f"Attempting to match {len(speaker_snippet_map)} snippets to speakers (threshold: {match_threshold:.2f})..."
    )
    if not speaker_snippet_map or not segments:
        logger.info("No speaker snippets provided or no segments to match against.")
        return {}

    # Group segments into blocks for more context
    blocks = group_segments_by_speaker(segments)
    if not blocks:
        logger.warning(
            "Segment grouping resulted in zero blocks. Cannot perform snippet matching."
        )
        return {}

    speaker_id_to_name_mapping: Dict[str, str] = {}
    # Track best score per original speaker ID to avoid weaker matches overwriting strong ones
    best_match_scores: Dict[str, float] = defaultdict(float)

    # Convert threshold 0-1 range to 0-100 for rapidfuzz partial_ratio
    fuzz_threshold = match_threshold * 100.0

    # Helper for normalization (lowercase, remove punctuation, collapse whitespace)
    def normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Iterate through each user-provided snippet
    for user_name, snippet in speaker_snippet_map.items():
        if not snippet or len(snippet) < 5:  # Basic validation
            logger.warning(
                f"Ignoring short/empty snippet for '{user_name}'. Snippet: '{snippet}'"
            )
            continue

        normalized_snippet = normalize_text(snippet)
        if not normalized_snippet:
            logger.warning(f"Snippet for '{user_name}' became empty after normalization.")
            continue

        best_match_for_this_snippet = {"score": -1.0, "speaker_id": None}
        found_match_above_threshold = False

        logger.info(f"Matching snippet for '{user_name}': '{normalized_snippet[:100]}...'")

        # Compare snippet against each speaker block
        for blk in blocks:
            original_speaker_id = blk.get("speaker")  # This is SPEAKER_XX or unknown
            block_text = blk.get("text", "")
            if not original_speaker_id or not block_text:
                continue

            normalized_block_text = normalize_text(block_text)
            if not normalized_block_text:
                continue

            # Use partial_ratio for finding snippet within larger block
            ratio = fuzz.partial_ratio(normalized_snippet, normalized_block_text)

            # Track the best block match for the current snippet (for logging/debugging)
            if ratio > best_match_for_this_snippet["score"]:
                best_match_for_this_snippet = {
                    "score": ratio,
                    "speaker_id": original_speaker_id,
                }

            # Check if score meets threshold AND is better than any previous match for this speaker_id
            if (
                ratio >= fuzz_threshold
                and ratio > best_match_scores[original_speaker_id]
            ):
                # Assign mapping: original_speaker_id -> user_name
                speaker_id_to_name_mapping[original_speaker_id] = user_name
                best_match_scores[original_speaker_id] = (
                    ratio  # Update best score for this ID
                )
                logger.info(
                    f"  ✅ Match FOUND: Snippet '{user_name}' (score {ratio:.1f}) assigned to speaker '{original_speaker_id}' "
                    f"(overwriting score {best_match_scores.get(original_speaker_id, 0.0):.1f})"
                )
                found_match_above_threshold = True
                # Continue checking other blocks, this snippet might match another speaker even better

        # Log if no match above threshold was found for the current snippet after checking all blocks
        if not found_match_above_threshold:
            best_score = best_match_for_this_snippet["score"]
            best_spk = best_match_for_this_snippet["speaker_id"]
            logger.info(
                f"  ❌ No match >= threshold ({fuzz_threshold:.1f}) found for snippet '{user_name}'. Best was {best_score:.1f} vs '{best_spk}'."
            )

    logger.info(
        f"Final snippet mapping (WhisperX ID -> User Name): {speaker_id_to_name_mapping}"
    )
    return speaker_id_to_name_mapping


# Moved from core/utils.py
def convert_floats(obj: Any) -> Any:
    """
    Recursively converts numpy float types (e.g., float32) and other potentially
    non-JSON-serializable number types within nested structures to standard Python floats.
    """
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(elem) for elem in obj]
    # Check if it quacks like a numpy float (more robust than specific type checks)
    elif hasattr(obj, "item") and callable(obj.item) and isinstance(obj.item(), float):
        return float(obj)
    # Add checks for other types if necessary (e.g., Decimal)
    # elif isinstance(obj, decimal.Decimal):
    #     return float(obj)
    else:
        # Return object as is if it's not a dict, list, or convertible number type
        return obj


# Moved from core/transcription.py
def convert_json_to_structured(json_path: Path) -> SegmentsList:
    """
    Reads a WhisperX JSON output file and converts it into a structured list of segments.

    Args:
        json_path: Path to the WhisperX output JSON file.

    Returns:
        A list of dictionaries, where each represents a segment containing keys like
        'start', 'end', 'text', 'speaker', 'words'. Returns empty list on failure.

    Raises:
        FileNotFoundError: If the json_path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the 'segments' key in the JSON is not a list.
    """
    logger.info(f"Reading and structuring WhisperX JSON output from: {json_path}")
    if not json_path.is_file():
        logger.error(f"WhisperX output JSON file not found at {json_path}")
        raise FileNotFoundError(f"WhisperX output JSON file not found at {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {json_path}: {e}")
        raise  # Re-raise the error
    except Exception as e:
        logger.error(f"Error reading JSON file {json_path}: {e}")
        raise RuntimeError(f"Could not read JSON file {json_path}") from e

    structured: SegmentsList = []
    segments_raw: Any = data.get("segments", [])  # Default to empty list

    if not isinstance(segments_raw, list):
        logger.error(
            f"'segments' key in {json_path} is not a list (type: {type(segments_raw)}). Cannot process."
        )
        raise TypeError(f"'segments' key in {json_path} is not a list.")

    logger.info(f"Structuring {len(segments_raw)} segments from WhisperX output...")

    for i, segment in enumerate(segments_raw):
        if not isinstance(segment, dict):
            logger.warning(f"Item #{i} in 'segments' is not a dictionary. Skipping.")
            continue

        # Extract data with defaults
        text: str = segment.get("text", "").strip()
        start_time: Optional[float] = segment.get("start")
        end_time: Optional[float] = segment.get("end")
        speaker: str = str(segment.get("speaker", "unknown"))  # Ensure string type
        words_raw: Any = segment.get("words", [])
        words: List[Dict[str, Any]] = words_raw if isinstance(words_raw, list) else []

        # Basic validation for timing
        if (
            start_time is None
            or end_time is None
            or not isinstance(start_time, (int, float))
            or not isinstance(end_time, (int, float))
            or start_time > end_time
        ):
            logger.warning(
                f"Segment {i} has invalid/missing time: start={start_time}, end={end_time}. Using raw values."
            )
            # Decide how to handle: skip segment, use None, use 0? Using raw values for now.

        segment_output: Segment = {
            "start": start_time,
            "end": end_time,
            "text": text,
            "speaker": speaker,
            "words": words,  # Ensure words is always a list
        }
        structured.append(segment_output)

    logger.info(f"Finished structuring segments. Returning {len(structured)} segments.")
    return structured


# Moved from core/utils.py
# Assumes group_segments_by_speaker is available in this module
def save_script_transcript(
    segments: SegmentsList, output_path: Path, log_prefix: str = "[Transcript]"
) -> Optional[Path]:
    """
    Saves a simple text transcript grouped by speaker with HH:MM:SS timestamps.

    Args:
        segments: The list of segment dictionaries.
        output_path: The Path object where the transcript file should be saved.
        log_prefix: Prefix for log messages.

    Returns:
        The output Path object if successful, None otherwise.
    """
    logger.info(f"{log_prefix} Attempting to save script transcript to: {output_path}")
    try:
        # Ensure parent directory exists
        if not create_directory(output_path.parent):
            logger.error(
                f"{log_prefix} Failed to create parent directory for {output_path}. Aborting save."
            )
            return None

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {output_path.stem}\n")
            f.write("=" * (len(output_path.stem) + 12) + "\n\n")

            speaker_blocks = group_segments_by_speaker(segments)

            def format_time(seconds: Optional[float]) -> str:
                """Formats seconds into [HH:MM:SS.ss] or [MM:SS.ss]"""
                if (
                    seconds is None
                    or not isinstance(seconds, (int, float))
                    or seconds < 0
                ):
                    return "[??:??.??]"
                try:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = seconds % 60
                    if hours > 0:
                        return (
                            f"[{hours:02d}:{minutes:02d}:{secs:05.2f}]"  # HH:MM:SS.ss
                        )
                    else:
                        return f"[{minutes:02d}:{secs:05.2f}]"  # MM:SS.ss
                except Exception:
                    return "[??:??.??]"  # Fallback

            for block in speaker_blocks:
                speaker = block.get("speaker", "unknown")
                text = block.get("text", "").strip()
                start_time = block.get("start")
                end_time = block.get("end")

                if text:  # Only write blocks with actual text content
                    time_str = f"{format_time(start_time)} -> {format_time(end_time)}"
                    f.write(
                        f"{time_str} {speaker}:\n{text}\n\n"
                    )  # Add newline after speaker for readability

        logger.info(f"{log_prefix} Script transcript saved successfully to {output_path}")
        return output_path

    except Exception as e:
        error_msg = (
            f"{log_prefix} Failed to save script transcript to {output_path}: {e}"
        )
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None


# Moved from core/utils.py
def run_ffprobe_duration_check(audio_path: Path, min_duration: float = 5.0) -> bool:
    """
    Checks the duration of an audio file using ffprobe. Logs warnings.

    Args:
        audio_path: Path to the audio file.
        min_duration: Minimum duration in seconds required for diarization.

    Returns:
        True if duration check passes (or ffprobe fails safely),
        False if audio is shorter than min_duration.
    """
    logger.info(f"Checking audio duration for {audio_path.name} (min: {min_duration}s)...")
    try:
        command = [
            "ffprobe",
            "-v",
            "error",  # Only show errors from ffprobe itself
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",  # Output only the value
            str(audio_path),
        ]
        logger.info(f"Running ffprobe check: {' '.join(command)}")  # Restored log
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raise exception on non-zero exit code from ffprobe
            timeout=30,  # Add a timeout to prevent hanging
        )
        duration_str = result.stdout.strip()

        if not duration_str or duration_str == "N/A":
            logger.warning(
                f"ffprobe could not determine duration for {audio_path.name}. Proceeding, but diarization might be skipped or fail."
            )
            return True  # Proceed cautiously if duration is unknown

        duration = float(duration_str)
        logger.info(
            f"Detected audio duration: {duration:.2f} seconds for {audio_path.name}"
        )  # Restored log

        if duration < min_duration:
            logger.warning(
                f"Audio duration ({duration:.1f}s) is less than the minimum "
                f"threshold ({min_duration}s) required for robust diarization. Quality may be affected."
            )
            return False  # Indicate check failed (too short)
        else:
            return True  # Indicate check passed (long enough)

    except FileNotFoundError:
        logger.warning(
            "ffprobe command not found. Cannot perform audio duration check. Diarization quality check skipped."
        )
        return True  # Proceed if ffprobe is not available, assume long enough
    except subprocess.TimeoutExpired:
        logger.warning(
            f"ffprobe timed out while checking duration for {audio_path.name}. Proceeding without check."
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"ffprobe returned an error (exit code {e.returncode}) while checking duration for {audio_path.name}. Output: {e.stderr or e.stdout}. Proceeding without check."
        )
        return True
    except ValueError as e:
        logger.warning(
            f"Could not convert ffprobe duration output ('{duration_str}') to float for {audio_path.name}: {e}. Proceeding without check."
        )
        return True
    except Exception as e:
        # Catch any other unexpected errors during the check
        logger.warning(
            f"Unexpected error running ffprobe duration check for {audio_path.name}: {e}. Proceeding without check."
        )
        logger.warning(traceback.format_exc())  # Log traceback for unexpected errors
        return True  # Proceed cautiously on other errors
