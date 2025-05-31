# speaker_id/id_mapping.py
"""
Handles applying collected speaker labels to transcript segments.
"""

from typing import List, Dict, Any, Optional

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Type Hinting
Segment = Dict[str, Any]
SegmentsList = List[Segment]
SpeakerLabelMap = Dict[str, str]  # Maps Original ID (e.g., SPEAKER_XX) -> User Label


# Extracted logic from core/pipeline.py::finalize_labeled_item
def apply_speaker_labels(
    segments: SegmentsList,
    speaker_labels: Optional[SpeakerLabelMap],
    log_prefix: str = "[ID Map]",
) -> SegmentsList:
    """
    Applies speaker labels from a mapping to a list of segments.
    Modifies the 'speaker' field in the segment dictionaries in-place OR
    returns a new list (currently modifies in-place for efficiency).

    Args:
        segments: The list of segment dictionaries to modify.
        speaker_labels: A dictionary mapping original speaker IDs (e.g., "SPEAKER_00")
                        to user-provided labels (e.g., "Alice"). If None or empty,
                        no relabeling occurs.
        log_prefix: Prefix for log messages.

    Returns:
        The list of segments with updated speaker labels.
    """
    if not segments:
        log_info(f"{log_prefix} No segments provided for relabeling.")
        return []
    if not speaker_labels:
        log_info(f"{log_prefix} No speaker label map provided. Skipping relabeling.")
        return segments  # Return original segments

    log_info(f"{log_prefix} Applying speaker labels based on mapping: {speaker_labels}")
    segments_relabeled_count = 0
    unique_original_ids = set(
        seg.get("speaker") for seg in segments if seg.get("speaker")
    )

    for seg in segments:
        original_speaker_id = str(seg.get("speaker", "unknown"))

        # Check if this original ID has a mapping provided by the user
        if original_speaker_id in speaker_labels:
            final_label = speaker_labels[original_speaker_id]
            # Apply the label only if it's not blank/empty
            if final_label and isinstance(final_label, str) and final_label.strip():
                new_label = final_label.strip()
                if seg["speaker"] != new_label:  # Check if change occurs
                    seg["speaker"] = new_label
                    segments_relabeled_count += 1
            else:
                # If user provided a blank label for this ID, keep the original ID
                # log_info(f"{log_prefix} Keeping original ID for {original_speaker_id} due to blank user label.")
                pass  # Keep original seg["speaker"]

    # Log summary of changes
    applied_ids = {k for k, v in speaker_labels.items() if v and v.strip()}
    unmapped_original_ids = unique_original_ids - set(speaker_labels.keys())
    blank_mapped_ids = {k for k, v in speaker_labels.items() if not v or not v.strip()}

    log_info(
        f"{log_prefix} Applied {segments_relabeled_count} non-blank speaker label instances."
    )
    if unmapped_original_ids:
        log_info(
            f"{log_prefix} Original IDs without user mapping: {sorted(list(unmapped_original_ids))}"
        )
    if blank_mapped_ids:
        log_info(
            f"{log_prefix} Original IDs mapped to blank labels (kept original ID): {sorted(list(blank_mapped_ids))}"
        )

    return segments
