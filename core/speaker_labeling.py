# core/speaker_labeling.py
"""
Provides functions for identifying speakers eligible for labeling (ordered by speaking time),
and selecting video preview start times.
"""

import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TextIO
from collections import defaultdict
import math

# --- Core Project Imports ---
from .utils import group_segments_by_speaker  # Removed safe_run import
from .logging import log_info, log_warning, log_error

# Type Hinting
Segment = Dict[str, Any]
SegmentsList = List[Segment]
DialogueBlock = Dict[str, Any]


def identify_eligible_speakers(
    segments: SegmentsList, min_total_time: float, min_block_time: float
) -> List[str]:
    """
    Identifies speakers who meet minimum total speaking time and minimum
    continuous block time thresholds.

    Args:
        segments: List of segment dictionaries from transcription.
        min_total_time: Minimum total seconds a speaker must talk across all segments.
        min_block_time: Minimum seconds for at least one continuous dialogue block.

    Returns:
        A list of unique speaker IDs (e.g., 'SPEAKER_00') eligible for labeling,
        SORTED BY TOTAL SPEAKING TIME in descending order.
    """
    log_info(
        f"Identifying speakers eligible for labeling (min_total_time={min_total_time}s, min_block_time={min_block_time}s)..."
    )
    if not segments:
        log_warning("No segments provided to identify_eligible_speakers.")
        return []

    speaker_total_time = defaultdict(float)
    for seg in segments:
        speaker = seg.get("speaker")
        start = seg.get("start")
        end = seg.get("end")
        if speaker and start is not None and end is not None and end > start:
            speaker_total_time[str(speaker)] += end - start

    eligible_speakers_with_time = []
    all_speaker_blocks: List[DialogueBlock] = group_segments_by_speaker(segments)
    speaker_to_blocks_map = defaultdict(list)
    for block in all_speaker_blocks:
        speaker_id = block.get("speaker")
        if speaker_id:
            speaker_to_blocks_map[str(speaker_id)].append(block)

    # Check eligibility for all speakers who have spoken
    unique_speaker_ids = list(speaker_total_time.keys())

    for speaker_id in unique_speaker_ids:
        total_time = speaker_total_time[speaker_id]
        if total_time < min_total_time:
            # log_info(f"Speaker {speaker_id} ineligible: Total time {total_time:.2f}s < {min_total_time}s")
            continue  # Skip if total time is too low

        # Check for at least one long enough block
        has_long_block = False
        speaker_blocks = speaker_to_blocks_map.get(speaker_id, [])
        for block in speaker_blocks:
            block_start = block.get("start")
            block_end = block.get("end")
            if (
                block_start is not None
                and block_end is not None
                and (block_end - block_start) >= min_block_time
            ):
                has_long_block = True
                break

        if not has_long_block:
            # log_info(f"Speaker {speaker_id} ineligible: No single block >= {min_block_time}s")
            continue  # Skip if no suitable block found

        # If both conditions met, add to list with time for sorting
        eligible_speakers_with_time.append({"id": speaker_id, "time": total_time})

    # Sort eligible speakers by total speaking time (descending)
    eligible_speakers_with_time.sort(key=lambda x: x["time"], reverse=True)

    # Extract just the sorted IDs
    sorted_eligible_speaker_ids = [spk["id"] for spk in eligible_speakers_with_time]

    log_info(
        f"Found {len(sorted_eligible_speaker_ids)} eligible speakers (ordered by speaking time): {sorted_eligible_speaker_ids}"
    )
    return sorted_eligible_speaker_ids


def select_preview_time_segments(
    speaker_id: str,
    segments: SegmentsList,
    preview_duration: float,
    min_block_time: float,
) -> List[float]:  # Changed return type to List[float]
    """
    Selects up to 3 start times for video previews for a given speaker.

    Prioritizes the start of the first 3 blocks longer than min_block_time.
    If fewer than 3 exist, uses the start time corresponding to the end of
    the longest block as a fallback. Ensures at least two distinct start times
    if possible.

    Args:
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') to select segments for.
        segments: The full list of segments for the entire audio.
        preview_duration: The desired duration of each preview clip (used for fallback calc).
        min_block_time: Minimum block duration to be considered for primary selection.

    Returns:
        A list of floats, each representing a start time in seconds for a preview clip.
        Returns an empty list if no suitable segments can be found.
    """
    log_info(
        f"Selecting preview start times for {speaker_id} (preview_duration={preview_duration}s, min_block_time={min_block_time}s)..."
    )
    preview_start_times = []
    selected_start_times_set = set()  # To ensure uniqueness

    speaker_segments = [
        seg for seg in segments if str(seg.get("speaker", "")) == speaker_id
    ]
    if not speaker_segments:
        log_warning(f"No segments found for speaker {speaker_id}.")
        return []

    dialogue_blocks = group_segments_by_speaker(speaker_segments)
    if not dialogue_blocks:
        log_warning(f"Could not group segments into blocks for speaker {speaker_id}.")
        return []

    dialogue_blocks.sort(key=lambda b: b.get("start", float("inf")))

    long_blocks = []
    for block in dialogue_blocks:
        start = block.get("start")
        end = block.get("end")
        if start is not None and end is not None and (end - start) >= min_block_time:
            long_blocks.append(block)

    # --- Select primary clips from the start of long blocks ---
    for block in long_blocks:
        if len(preview_start_times) >= 3:
            break
        start = block.get("start")
        # Ensure start time is valid and unique
        if start is not None and start not in selected_start_times_set:
            # Ensure start time is non-negative
            valid_start = max(0.0, start)
            preview_start_times.append(valid_start)
            selected_start_times_set.add(valid_start)

    # --- Fallback: Use time near end of the longest block if needed ---
    if len(preview_start_times) < 3 and dialogue_blocks:
        longest_block = max(
            dialogue_blocks, key=lambda b: b.get("end", 0) - b.get("start", 0)
        )
        l_start = longest_block.get("start")
        l_end = longest_block.get("end")

        # Check if the longest block is long enough to select a preview from its end
        if (
            l_start is not None
            and l_end is not None
            and (l_end - l_start) >= preview_duration
        ):
            # Calculate start time for the *last* N seconds of the block
            fallback_start = max(
                0.0, l_start, l_end - preview_duration
            )  # Ensure non-negative and within block

            # Add only if it's distinct from already selected start times
            if fallback_start not in selected_start_times_set:
                log_info(
                    f"Using fallback start time: End of longest block ({l_start:.2f}-{l_end:.2f}) -> Start at {fallback_start:.2f}s"
                )
                preview_start_times.append(fallback_start)
                selected_start_times_set.add(fallback_start)

    # --- Ensure at least two distinct clips if possible ---
    if len(preview_start_times) == 1 and dialogue_blocks:
        longest_block = max(
            dialogue_blocks, key=lambda b: b.get("end", 0) - b.get("start", 0)
        )
        l_start = longest_block.get("start")
        if l_start is not None:
            valid_l_start = max(0.0, l_start)
            if valid_l_start not in selected_start_times_set:
                log_info(
                    f"Adding second start time from start of longest block ({l_start:.2f}s)"
                )
                preview_start_times.append(valid_l_start)
                selected_start_times_set.add(valid_l_start)

    # Final sort by time
    preview_start_times.sort()

    # Convert to integer seconds for YouTube URL parameter (more compatible)
    preview_start_times_int = [int(math.floor(t)) for t in preview_start_times]
    # Remove potential duplicates after flooring
    unique_preview_start_times_int = sorted(list(set(preview_start_times_int)))

    log_info(
        f"Selected {len(unique_preview_start_times_int)} unique preview start times (seconds) for {speaker_id}: {unique_preview_start_times_int}"
    )
    return unique_preview_start_times_int


# --- download_video_clips function removed as per Option 3 ---
