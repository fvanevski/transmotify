# speaker_id/vid_preview_id.py
"""
Handles identifying speakers eligible for labeling, selecting video preview segments,
and managing the state transitions for the interactive labeling workflow.
"""

import math
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, TextIO

# Assuming utils.transcripts and core.logging are available
try:
    # group_segments_by_speaker is needed for identify_eligible_speakers
    from utils.transcripts import group_segments_by_speaker
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

    # Dummy group_segments_by_speaker if import fails
    def group_segments_by_speaker(segments):
        return []


# Type Hinting
Segment = Dict[str, Any]
SegmentsList = List[Segment]
DialogueBlock = Dict[str, Any]
# Define structure for labeling state items (as used in legacy core/pipeline)
# This state will need to be passed into functions like store_speaker_label
# by the future orchestrator.
LabelingItemState = Dict[str, Any]


# Moved from core/speaker_labeling.py
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
        if (
            speaker
            and start is not None
            and end is not None
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end > start
        ):
            speaker_total_time[str(speaker)] += end - start

    eligible_speakers_with_time = []
    # Use group_segments_by_speaker (now imported from utils.transcripts)
    all_speaker_blocks: List[DialogueBlock] = group_segments_by_speaker(segments)
    speaker_to_blocks_map = defaultdict(list)
    for block in all_speaker_blocks:
        speaker_id = block.get("speaker")
        if speaker_id:
            speaker_to_blocks_map[str(speaker_id)].append(block)

    # Check eligibility for all speakers found
    unique_speaker_ids = list(speaker_total_time.keys())

    for speaker_id in unique_speaker_ids:
        total_time = speaker_total_time[speaker_id]
        if total_time < min_total_time:
            # log_info(f"Speaker {speaker_id} ineligible: Total time {total_time:.2f}s < {min_total_time}s") # Verbose
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
                and isinstance(block_start, (int, float))
                and isinstance(block_end, (int, float))
                and (block_end - block_start) >= min_block_time
            ):
                has_long_block = True
                break

        if not has_long_block:
            # log_info(f"Speaker {speaker_id} ineligible: No single block >= {min_block_time}s") # Verbose
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


# Moved from core/speaker_labeling.py
def select_preview_time_segments(
    speaker_id: str,
    segments: SegmentsList,
    preview_duration: float,
    min_block_time: float,
) -> List[int]:  # Returns list of integer start times in seconds
    """
    Selects up to 3 start times (in seconds) for video previews for a given speaker.
    Prioritizes the start of the first 3 blocks longer than min_block_time.
    Uses fallback logic based on longest block if fewer than 3 suitable blocks exist.

    Args:
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') to select segments for.
        segments: The full list of segments for the entire audio.
        preview_duration: The desired duration of each preview clip (used for fallback calc).
        min_block_time: Minimum block duration to be considered for primary selection.

    Returns:
        A list of unique integers, each representing a start time in seconds for a preview clip.
        Returns an empty list if no suitable segments can be found.
    """
    log_info(
        f"Selecting preview start times for {speaker_id} (preview_duration={preview_duration}s, min_block_time={min_block_time}s)..."
    )
    preview_start_times_float = []
    selected_start_times_set = set()  # To ensure uniqueness

    # Filter segments for the target speaker
    speaker_segments = [
        seg for seg in segments if str(seg.get("speaker", "")) == speaker_id
    ]
    if not speaker_segments:
        log_warning(f"No segments found for speaker {speaker_id}.")
        return []

    # Group the speaker's segments into dialogue blocks
    dialogue_blocks = group_segments_by_speaker(speaker_segments)
    if not dialogue_blocks:
        log_warning(f"Could not group segments into blocks for speaker {speaker_id}.")
        return []

    # Sort blocks by start time
    dialogue_blocks.sort(key=lambda b: b.get("start", float("inf")))

    # Identify blocks long enough for primary selection
    long_blocks = []
    for block in dialogue_blocks:
        start = block.get("start")
        end = block.get("end")
        if (
            start is not None
            and end is not None
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and (end - start) >= min_block_time
        ):
            long_blocks.append(block)

    # --- Select primary clips from the start of long blocks ---
    for block in long_blocks:
        if len(preview_start_times_float) >= 3:
            break
        start = block.get("start")
        if start is not None and isinstance(start, (int, float)):
            # Ensure start time is non-negative and floor it for integer seconds
            valid_start = max(0.0, start)
            start_sec_int = math.floor(valid_start)
            if start_sec_int not in selected_start_times_set:
                preview_start_times_float.append(
                    valid_start
                )  # Keep float for potential sorting
                selected_start_times_set.add(
                    start_sec_int
                )  # Add int to set for uniqueness check

    # --- Fallback: Use time near end of the longest block if needed ---
    if len(preview_start_times_float) < 3 and dialogue_blocks:
        # Find the block with the longest duration
        longest_block = max(
            dialogue_blocks,
            key=lambda b: (b.get("end", 0.0) or 0.0) - (b.get("start", 0.0) or 0.0),
        )
        l_start = longest_block.get("start")
        l_end = longest_block.get("end")

        # Check if the longest block is valid and long enough for a preview from its end
        if (
            l_start is not None
            and l_end is not None
            and isinstance(l_start, (int, float))
            and isinstance(l_end, (int, float))
            and (l_end - l_start) >= preview_duration
        ):
            # Calculate start time for the *last* N seconds of the block
            fallback_start_float = max(
                0.0, l_start, l_end - preview_duration
            )  # Ensure non-negative
            fallback_start_int = math.floor(fallback_start_float)

            # Add only if it's distinct (based on integer seconds)
            if fallback_start_int not in selected_start_times_set:
                log_info(
                    f"Using fallback start time near end of longest block ({l_start:.2f}-{l_end:.2f}s) -> Start at {fallback_start_int}s"
                )
                preview_start_times_float.append(fallback_start_float)
                selected_start_times_set.add(fallback_start_int)

    # --- Ensure at least two distinct clips if possible ---
    # If only one preview time was found, try adding the start of the longest block as a second distinct time
    if len(preview_start_times_float) == 1 and dialogue_blocks:
        longest_block = max(
            dialogue_blocks,
            key=lambda b: (b.get("end", 0.0) or 0.0) - (b.get("start", 0.0) or 0.0),
        )
        l_start = longest_block.get("start")
        if l_start is not None and isinstance(l_start, (int, float)):
            valid_l_start_float = max(0.0, l_start)
            valid_l_start_int = math.floor(valid_l_start_float)
            if valid_l_start_int not in selected_start_times_set:
                log_info(
                    f"Adding second start time from start of longest block ({l_start:.2f}s -> {valid_l_start_int}s)"
                )
                preview_start_times_float.append(valid_l_start_float)
                selected_start_times_set.add(valid_l_start_int)

    # Final conversion to unique integer seconds, sorted
    final_start_times_int = sorted(list(selected_start_times_set))

    log_info(
        f"Selected {len(final_start_times_int)} unique preview start times (seconds) for {speaker_id}: {final_start_times_int}"
    )
    return final_start_times_int


# --- Functions below moved from core/pipeline.py ---
# --- IMPORTANT: These functions rely on external state management ---
# --- (e.g., labeling_state dict) which needs to be provided ---
# --- by the calling context (likely the Phase 9 orchestrator). ---


# Moved from core/pipeline.py
def start_interactive_labeling_for_item(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    labeling_config: Dict[str, Any],  # Expects relevant config values
    log_prefix: str = "[Labeling Start]",
) -> Optional[
    Tuple[str, str, List[int]]
]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
    """
    Prepares and returns data for the first speaker to be labeled for an item.
    Depends on item_state containing 'eligible_speakers', 'youtube_url', 'segments'.

    Args:
        item_state: The dictionary containing the state for this specific item.
        labeling_config: Dictionary with relevant config keys like
                         'speaker_labeling_preview_duration', 'speaker_labeling_min_block_time'.
        log_prefix: Prefix for log messages.

    Returns:
        Tuple (speaker_id, youtube_url, start_times_list) or None if setup fails.
    """
    log_info(f"{log_prefix} Starting interactive labeling...")

    eligible_speakers = item_state.get(
        "eligible_speakers", []
    )  # Already sorted by time
    youtube_url = item_state.get("youtube_url")
    segments = item_state.get("segments", [])

    if not eligible_speakers:
        log_warning(
            f"{log_prefix} No eligible speakers found in state. Cannot start labeling."
        )
        # Orchestrator should handle this - maybe finalize immediately?
        return None
    if not youtube_url:
        log_error(f"{log_prefix} YouTube URL missing from item state.")
        return None
    if not segments:
        log_warning(f"{log_prefix} Segments missing from item state.")
        # Continue, preview selection will just return empty list

    first_speaker_id = eligible_speakers[0]
    log_info(f"{log_prefix} First speaker to label: {first_speaker_id}")

    # Get preview config from passed dict
    preview_duration = float(
        labeling_config.get("speaker_labeling_preview_duration", 5.0)
    )
    min_block_time = float(labeling_config.get("speaker_labeling_min_block_time", 10.0))

    start_times = select_preview_time_segments(
        speaker_id=first_speaker_id,
        segments=segments,
        preview_duration=preview_duration,
        min_block_time=min_block_time,
    )  # Returns List[int]

    if not start_times:
        log_warning(
            f"{log_prefix} Could not select preview start times for {first_speaker_id}."
        )
        # Return speaker ID and URL, but empty times list
        return first_speaker_id, youtube_url, []

    return first_speaker_id, youtube_url, start_times


# Moved from core/pipeline.py
def store_speaker_label(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    speaker_id: str,
    user_label: str,
    log_prefix: str = "[Label Store]",
) -> bool:
    """
    Stores the user-provided label for a speaker within the item's state dict.
    The caller (orchestrator) is responsible for managing and persisting item_state.

    Args:
        item_state: The dictionary containing the state for this specific item.
                    This dictionary will be modified in place.
        speaker_id: The original speaker ID (e.g., SPEAKER_XX).
        user_label: The label provided by the user.
        log_prefix: Prefix for log messages.

    Returns:
        True if label was stored (dictionary updated), False otherwise (e.g., state invalid).
    """
    if not isinstance(item_state, dict):
        log_error(
            f"{log_prefix} Invalid item_state provided (not a dict). Cannot store label."
        )
        return False

    if "collected_labels" not in item_state or not isinstance(
        item_state["collected_labels"], dict
    ):
        item_state["collected_labels"] = {}  # Initialize if missing or wrong type

    cleaned_label = (
        user_label.strip() if user_label else ""
    )  # Store blank if user entered nothing
    item_state["collected_labels"][speaker_id] = cleaned_label
    log_info(
        f"{log_prefix} Stored label for {speaker_id}: '{cleaned_label}' (in provided item_state dict)"
    )
    return True


# Moved from core/pipeline.py
def get_next_speaker_for_labeling(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    current_speaker_index: int,  # Index of the speaker *just* labeled
    labeling_config: Dict[str, Any],  # Expects relevant config values
    log_prefix: str = "[Label Next]",
) -> Optional[
    Tuple[str, str, List[int]]
]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
    """
    Gets the ID, URL, and start times for the next speaker in the item's eligible list.
    Depends on item_state containing 'eligible_speakers', 'youtube_url', 'segments'.

    Args:
        item_state: The dictionary containing the state for this specific item.
        current_speaker_index: The index (in eligible_speakers list) of the speaker
                               that was just processed or labeled.
        labeling_config: Dictionary with relevant config keys like
                         'speaker_labeling_preview_duration', 'speaker_labeling_min_block_time'.
        log_prefix: Prefix for log messages.

    Returns:
        Tuple (speaker_id, youtube_url, start_times_list) for the next speaker,
        or None if all eligible speakers for this item have been processed.
    """

    eligible_speakers = item_state.get("eligible_speakers", [])  # Sorted list
    youtube_url = item_state.get("youtube_url")
    segments = item_state.get("segments", [])

    if not youtube_url:
        log_error(f"{log_prefix} YouTube URL missing from item state for next speaker.")
        return None  # Cannot proceed

    next_speaker_index = current_speaker_index + 1

    if next_speaker_index < len(eligible_speakers):
        next_speaker_id = eligible_speakers[next_speaker_index]
        log_info(
            f"{log_prefix} Getting data for next speaker (Index {next_speaker_index}): {next_speaker_id}"
        )

        preview_duration = float(
            labeling_config.get("speaker_labeling_preview_duration", 5.0)
        )
        min_block_time = float(
            labeling_config.get("speaker_labeling_min_block_time", 10.0)
        )

        start_times = select_preview_time_segments(
            speaker_id=next_speaker_id,
            segments=segments,
            preview_duration=preview_duration,
            min_block_time=min_block_time,
        )

        if not start_times:
            log_warning(
                f"{log_prefix} Could not select preview start times for {next_speaker_id}."
            )
            return next_speaker_id, youtube_url, []

        return next_speaker_id, youtube_url, start_times
    else:
        log_info(f"{log_prefix} All eligible speakers processed for this item.")
        return None  # Signal item completion


# Moved from core/pipeline.py
def skip_labeling_for_item(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    log_prefix: str = "[Label Skip]",
) -> bool:
    """
    Handles the logic when a user skips labeling for remaining speakers in an item.
    Currently, this function primarily serves as a signal; the actual finalization
    (which uses collected labels so far) is handled separately by the orchestrator
    calling the id_mapping functionality.

    Args:
        item_state: The dictionary containing the state for this specific item.
        log_prefix: Prefix for log messages.

    Returns:
        True (as the skip action itself is always considered successful).
    """
    # This function might manipulate item_state in the future if needed,
    # e.g., setting a flag item_state['skipped'] = True
    log_warning(f"{log_prefix} User requested to skip remaining speakers for item.")
    # The orchestrator will see that the next speaker returned is None (implicitly)
    # after this is called and should then trigger finalization using the
    # labels collected so far in item_state['collected_labels'].
    return True
