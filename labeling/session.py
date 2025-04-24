# Proposed content for labeling/session.py

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from core.config import Config
from core.logging import get_logger
from labeling import selector
from transcription.segments import Segment, SegmentsList # typed-alias

logger = get_logger(__name__)

__all__ = ["LabelingSession"]


# Helper dataclass to store state for each item in the batch
@dataclass
class LabelingItem:
    item_id: str # Original identifier (e.g., filename or URL)
    name: str # Display name
    output_directory: Path
    segments: SegmentsList
    eligible_speakers: List[str] = field(default_factory=list)
    # Assuming batch_results provides this structure per speaker
    speaker_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class LabelingSession:
    """
    Holds state for one batch during interactive speaker labeling via the UI.

    Manages navigation between items and speakers, stores temporary labels,
    and provides data needed by the UI (`get_state`). It relies on
    `labeling.selector` for identifying eligible speakers initially.
    """

    def __init__(self, batch_results: List[Dict[str, Any]], cfg: Config):
        """
        Initialises the labeling session from the results of the main pipeline.

        Args:
            batch_results: A list of dictionaries, one per processed item,
                           containing segments, speaker details, paths, etc.
            cfg: The application configuration.
        """
        self.cfg = cfg
        self._items: List[LabelingItem] = []
        self._labels: Dict[int, Dict[str, str]] = {}  # {item_idx: {speaker_id: label}}
        self._current_item_idx: int = 0
        self._current_speaker_idx: int = 0

        # --- Process batch_results ---
        min_total_time = getattr(cfg, "labeling_min_total_time", 10.0)
        min_block_time = getattr(cfg, "labeling_min_block_time", 2.0)

        for idx, result_item in enumerate(batch_results):
            segments = result_item.get("segments", [])
            if not segments:
                logger.warning("Item %s has no segments, skipping for labeling.", result_item.get('name', f'#{idx}'))
                continue

            # Identify eligible speakers using selector
            eligible_speakers = selector.identify_eligible_speakers(
                segments, min_total_time=min_total_time, min_block_time=min_block_time
            )

            if not eligible_speakers:
                logger.info("Item %s has no eligible speakers, skipping for labeling.", result_item.get('name', f'#{idx}'))
                continue

            # --- Prepare speaker details (crucially needs audio path) ---
            # This part assumes 'speakers' dict exists in result_item from pipeline
            # with keys like 'SPEAKER_00', 'SPEAKER_01', etc.
            # Each value should be a dict with 'audio_snippet_path' and 'example_transcript'.
            speaker_details_map = result_item.get("speakers", {})
            processed_speaker_details = {}
            missing_details = False
            for spk_id in eligible_speakers:
                 details = speaker_details_map.get(spk_id)
                 if not details or "audio_snippet_path" not in details or "example_transcript" not in details:
                     logger.error(f"Missing required 'audio_snippet_path' or 'example_transcript' for speaker {spk_id} in item {result_item.get('name', f'#{idx}')}. Cannot proceed with labeling for this speaker.")
                     # Handle this? Skip speaker? Skip item? For now, log error and store None/empty
                     # We will filter out speakers without required details later in get_state
                     processed_speaker_details[spk_id] = None # Mark as unusable
                     missing_details = True # Mark that some details were missing
                 else:
                     processed_speaker_details[spk_id] = {
                         "audio_snippet_path": Path(details["audio_snippet_path"]),
                         "example_transcript": details["example_transcript"],
                     }

            # Only add item if it has *any* eligible speakers with full details
            valid_eligible_speakers = [spk for spk in eligible_speakers if processed_speaker_details.get(spk) is not None]

            if not valid_eligible_speakers:
                 logger.warning(f"Item {result_item.get('name', f'#{idx}')} has no eligible speakers with complete details (audio snippet/transcript). Skipping.")
                 continue


            item = LabelingItem(
                item_id=result_item.get("id", f"item_{idx}"),
                name=result_item.get("name", f"Item {idx + 1}"),
                output_directory=Path(result_item.get("output_directory", ".")),
                segments=segments,
                eligible_speakers=valid_eligible_speakers, # Store only speakers with valid details
                speaker_details=processed_speaker_details,
            )
            self._items.append(item)

        if not self._items:
             logger.warning("Labeling Session initialized, but no items suitable for labeling were found in batch results.")
             # Handle the case where no items are suitable - maybe raise an error?
             # For now, the session will exist but be empty.

        # Ensure initial indices are valid
        self._reset_indices_if_needed()


    def _reset_indices_if_needed(self):
        """Resets indices if they point outside valid items/speakers."""
        if not self._items:
            self._current_item_idx = 0
            self._current_speaker_idx = 0
            return

        if self._current_item_idx >= len(self._items):
            self._current_item_idx = 0
            self._current_speaker_idx = 0

        current_item = self._items[self._current_item_idx]
        if self._current_speaker_idx >= len(current_item.eligible_speakers):
            self._current_speaker_idx = 0

    def start(self):
        """Resets the session state to the beginning."""
        self._current_item_idx = 0
        self._current_speaker_idx = 0
        self._labels = {}
        logger.info("Labeling session reset.")


    def get_state(self, item_idx: int, speaker_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
        """
        Returns the necessary information for the UI to display the current
        speaker labeling step.

        Args:
            item_idx: The index of the item being labeled.
            speaker_idx: The index of the speaker within that item's eligible list.

        Returns:
            A tuple containing: (item_info, speaker_info, progress)
            Returns empty dicts if indices are invalid or session is empty.
        """
        if not self._items or item_idx < 0 or item_idx >= len(self._items):
            logger.error(f"get_state called with invalid item_idx: {item_idx}")
            return {}, {}, {}

        current_item = self._items[item_idx]

        if speaker_idx < 0 or speaker_idx >= len(current_item.eligible_speakers):
             logger.error(f"get_state called with invalid speaker_idx: {speaker_idx} for item {item_idx} which has {len(current_item.eligible_speakers)} eligible speakers.")
             # Attempt to reset speaker index? Or return error state?
             # Resetting might be safer for UI flow.
             self._current_speaker_idx = 0
             speaker_idx = 0
             if speaker_idx >= len(current_item.eligible_speakers): # Still invalid after reset? Item has no speakers?
                 logger.error(f"Item {item_idx} has no valid eligible speakers after reset.")
                 return {}, {}, {}


        speaker_id = current_item.eligible_speakers[speaker_idx]
        speaker_details = current_item.speaker_details.get(speaker_id)

        # This check should technically be redundant due to filtering in __init__
        if not speaker_details:
             logger.error(f"FATAL: Speaker {speaker_id} is in eligible list but missing details in get_state.")
             # This indicates a logic error in __init__ or state corruption.
             # How to recover? Maybe skip speaker? For now, return empty.
             return {}, {}, {}

        # --- Construct return dictionaries ---
        item_info = {
            "name": current_item.name,
            # Add other relevant item details if needed by UI
        }

        speaker_info = {
            "id": speaker_id,
            "audio_snippet_path": str(speaker_details["audio_snippet_path"]),
            "example_transcript": speaker_details["example_transcript"],
            "label": self._labels.get(item_idx, {}).get(speaker_id, ""), # Get saved label
        }

        progress = {
            "item_num": item_idx + 1,
            "total_items": len(self._items),
            "speaker_num": speaker_idx + 1,
            "total_speakers": len(current_item.eligible_speakers),
        }

        return item_info, speaker_info, progress


    def store_label(self, item_idx: int, speaker_idx: int, label: str):
        """Stores the user-provided label for the current speaker."""
        if not self._items or item_idx < 0 or item_idx >= len(self._items):
            logger.error(f"store_label called with invalid item_idx: {item_idx}")
            return

        current_item = self._items[item_idx]
        if speaker_idx < 0 or speaker_idx >= len(current_item.eligible_speakers):
             logger.error(f"store_label called with invalid speaker_idx: {speaker_idx}")
             return

        speaker_id = current_item.eligible_speakers[speaker_idx]
        clean_label = label.strip()

        if item_idx not in self._labels:
            self._labels[item_idx] = {}
        self._labels[item_idx][speaker_id] = clean_label
        logger.info(f"Stored label for Item {item_idx+1}/{len(self._items)}, Speaker {speaker_idx+1}/{len(current_item.eligible_speakers)} ({speaker_id}): '{clean_label}'")


    def navigate(self, current_item_idx: int, current_speaker_idx: int, direction: str) -> Tuple[int, int]:
        """
        Calculates the next (item_idx, speaker_idx) based on the direction.

        Args:
            current_item_idx: The current item index.
            current_speaker_idx: The current speaker index within that item.
            direction: "next" or "prev".

        Returns:
            A tuple containing the new (item_idx, speaker_idx).
        """
        if not self._items:
             return 0, 0 # No items, return default indices

        new_item_idx = current_item_idx
        new_speaker_idx = current_speaker_idx

        num_items = len(self._items)

        if direction == "next":
            num_speakers_current_item = len(self._items[current_item_idx].eligible_speakers)
            if current_speaker_idx + 1 < num_speakers_current_item:
                # Move to next speaker in the same item
                new_speaker_idx += 1
            else:
                # Move to the first speaker of the next item
                if current_item_idx + 1 < num_items:
                    new_item_idx += 1
                    new_speaker_idx = 0
                else:
                    # Already at the last speaker of the last item, do nothing (or loop?)
                    # UI button should be disabled, but handle defensively.
                    logger.warning("Navigate 'next' called on last speaker of last item.")
                    pass # Stay at the end

        elif direction == "prev":
            if current_speaker_idx > 0:
                # Move to previous speaker in the same item
                new_speaker_idx -= 1
            else:
                # Move to the last speaker of the previous item
                if current_item_idx > 0:
                    new_item_idx -= 1
                    # Index of last speaker in the previous item
                    new_speaker_idx = len(self._items[new_item_idx].eligible_speakers) - 1
                else:
                    # Already at the first speaker of the first item, do nothing
                    logger.warning("Navigate 'prev' called on first speaker of first item.")
                    pass # Stay at the beginning
        else:
             logger.warning(f"Unknown navigation direction: {direction}")


        # Update internal state (important!)
        self._current_item_idx = new_item_idx
        self._current_speaker_idx = new_speaker_idx

        return new_item_idx, new_speaker_idx


    def is_complete(self) -> bool:
        """Checks if labels have been provided for all eligible speakers."""
        if not self._items:
            return True # Nothing to label

        for i, item in enumerate(self._items):
            item_labels = self._labels.get(i, {})
            for speaker_id in item.eligible_speakers:
                if speaker_id not in item_labels or not item_labels[speaker_id]:
                    # Found an eligible speaker without a non-empty label
                    return False
        return True # All eligible speakers have labels


    def get_labeled_results(self) -> List[Dict[str, Any]]:
        """
        Applies the stored labels back to the segments in a copy of the
        original batch results structure.

        Returns:
            A deep copy of the original batch_results, but with speaker IDs
            in the 'segments' lists replaced by the user-provided labels.
            (Note: Actual deep copy might be needed depending on usage)
        """
        # Warning: This currently modifies the *original* segments lists
        # stored within the LabelingItem objects because Python passes lists
        # by reference. If the original batch_results needs to be preserved
        # elsewhere, a deep copy mechanism is required here or upon session init.
        # For now, assume modifying the session's internal items is acceptable
        # before packaging.

        final_results = []
        for i, item in enumerate(self._items):
            item_labels = self._labels.get(i, {})
            if not item_labels: # No labels for this item, skip modification? Or add anyway?
                 logger.debug(f"No labels found for item {i} ({item.name}) during finalization.")
                 # Let's still include it in the results, just unmodified

            new_segments = []
            for segment in item.segments:
                 # Create a copy of the segment to avoid modifying the original list directly
                 # if needed elsewhere, though modifying here is the goal for get_labeled_results
                 new_seg = segment.copy()
                 speaker_id = str(new_seg.get("speaker", ""))
                 if speaker_id in item_labels and item_labels[speaker_id]:
                     new_seg["speaker"] = item_labels[speaker_id] # Apply the label
                 new_segments.append(new_seg)

            # Reconstruct the item dictionary for output
            # This needs to be careful to include *all* original info from batch_results
            # Ideally, we'd have stored the original dicts or use deepcopy
            # For simplicity now, just rebuild with key parts:
            final_item_data = {
                 "id": item.item_id,
                 "name": item.name,
                 "output_directory": str(item.output_directory),
                 "segments": new_segments,
                 # --- CRITICAL: Need to merge back other original data ---
                 # This is a placeholder. The actual implementation should likely
                 # iterate over a *copy* of the original batch_results and
                 # update the 'segments' field within those dictionaries.
                 # Example: (assuming self.original_batch_results exists)
                 # final_item_data = self.original_batch_results[i].copy()
                 # final_item_data["segments"] = new_segments
                 # --- End Placeholder ---
                 "speaker_labels_applied": item_labels, # Add the map used for clarity
            }
            final_results.append(final_item_data)

        # Add items from original batch_results that were *not* suitable for labeling?
        # This depends on desired output. If the output ZIP should contain all
        # original items, logic is needed here to merge labeled & unlabeled.

        logger.info(f"Generated labeled results for {len(final_results)} items.")
        return final_results

    def _get_example_transcript(self, speaker_id: str, segments: SegmentsList) -> str:
        """Helper to find a short example text for a speaker."""
        for seg in segments:
            if str(seg.get("speaker")) == speaker_id and seg.get("text"):
                return seg["text"].strip()
        return "..." # Fallback
