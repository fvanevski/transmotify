# core/speaker_labeling.py
"""
Provides functions for identifying speakers eligible for labeling,
selecting video preview time segments, and downloading those clips.
"""

import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TextIO
from collections import defaultdict
import math  # Import math for ceiling function

# --- Core Project Imports ---
from .utils import group_segments_by_speaker, safe_run
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
        A list of unique speaker IDs (e.g., 'SPEAKER_00') eligible for labeling.
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

    eligible_speakers = []
    # Group all segments first to analyze blocks per speaker
    all_speaker_blocks: List[DialogueBlock] = group_segments_by_speaker(segments)

    # Create a map of speaker ID to their blocks
    speaker_to_blocks_map = defaultdict(list)
    for block in all_speaker_blocks:
        speaker_id = block.get("speaker")
        if speaker_id:
            speaker_to_blocks_map[str(speaker_id)].append(block)

    # Now check eligibility
    unique_speaker_ids = sorted(list(speaker_total_time.keys()))

    for speaker_id in unique_speaker_ids:
        total_time = speaker_total_time[speaker_id]
        if total_time < min_total_time:
            log_info(
                f"Speaker {speaker_id} ineligible: Total time {total_time:.2f}s < {min_total_time}s"
            )
            continue

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
                break  # Found one, no need to check further for this speaker

        if not has_long_block:
            log_info(
                f"Speaker {speaker_id} ineligible: No single block >= {min_block_time}s"
            )
            continue

        # If both conditions met
        log_info(
            f"Speaker {speaker_id} IS eligible (Total time: {total_time:.2f}s, Has block >= {min_block_time}s)"
        )
        eligible_speakers.append(speaker_id)

    log_info(f"Found {len(eligible_speakers)} eligible speakers: {eligible_speakers}")
    return eligible_speakers


def select_preview_time_segments(
    speaker_id: str,
    segments: SegmentsList,
    preview_duration: float,
    min_block_time: float,
) -> List[Tuple[float, float]]:
    """
    Selects up to 3 time segments for video previews for a given speaker.

    Prioritizes the start of the first 3 blocks longer than min_block_time.
    If fewer than 3 exist, uses the end of the longest block as a fallback.
    Ensures at least two distinct segments if possible.

    Args:
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') to select segments for.
        segments: The full list of segments for the entire audio.
        preview_duration: The desired duration of each preview clip in seconds.
        min_block_time: Minimum block duration to be considered for primary selection.

    Returns:
        A list of tuples, each containing (start_time, end_time) for a preview clip.
        Returns an empty list if no suitable segments can be found.
    """
    log_info(
        f"Selecting preview segments for {speaker_id} (preview_duration={preview_duration}s, min_block_time={min_block_time}s)..."
    )
    preview_segments = []

    # Filter segments for the target speaker
    speaker_segments = [
        seg for seg in segments if str(seg.get("speaker", "")) == speaker_id
    ]
    if not speaker_segments:
        log_warning(f"No segments found for speaker {speaker_id}.")
        return []

    # Group into dialogue blocks
    dialogue_blocks = group_segments_by_speaker(speaker_segments)
    if not dialogue_blocks:
        log_warning(f"Could not group segments into blocks for speaker {speaker_id}.")
        return []

    # Sort blocks by start time
    dialogue_blocks.sort(key=lambda b: b.get("start", float("inf")))

    # Find blocks longer than the minimum block time
    long_blocks = []
    for block in dialogue_blocks:
        start = block.get("start")
        end = block.get("end")
        if start is not None and end is not None and (end - start) >= min_block_time:
            long_blocks.append(block)

    # --- Select primary clips from the start of long blocks ---
    selected_start_times = set()  # To avoid duplicate start times

    for block in long_blocks:
        if len(preview_segments) >= 3:
            break
        start = block.get("start")
        end = block.get("end")
        if start is not None and end is not None:
            # Ensure the clip doesn't exceed the block boundaries (or audio duration implicitly)
            clip_end = min(start + preview_duration, end)
            # Ensure clip has a positive duration, although start+preview_duration should be >= start
            if clip_end > start and start not in selected_start_times:
                preview_segments.append((start, clip_end))
                selected_start_times.add(start)

    # --- Fallback: Use end of the longest block if needed ---
    if len(preview_segments) < 3 and dialogue_blocks:
        # Find the overall longest block for this speaker
        longest_block = max(
            dialogue_blocks, key=lambda b: b.get("end", 0) - b.get("start", 0)
        )
        l_start = longest_block.get("start")
        l_end = longest_block.get("end")

        if (
            l_start is not None
            and l_end is not None
            and (l_end - l_start) >= preview_duration
        ):
            # Calculate start time for the *last* N seconds of the block
            fallback_start = max(l_start, l_end - preview_duration)
            fallback_end = l_end

            # Add only if it's distinct from already selected start times
            if fallback_start not in selected_start_times:
                log_info(
                    f"Using fallback: End of longest block ({l_start:.2f}-{l_end:.2f}) -> Clip ({fallback_start:.2f}-{fallback_end:.2f})"
                )
                preview_segments.append((fallback_start, fallback_end))
                selected_start_times.add(fallback_start)

    # --- Ensure at least two distinct clips if possible ---
    # This logic might be redundant if the above selection handles distinctness,
    # but serves as a safeguard. If only one clip was found, try adding
    # the start of the longest block if different.
    if len(preview_segments) == 1 and dialogue_blocks:
        longest_block = max(
            dialogue_blocks, key=lambda b: b.get("end", 0) - b.get("start", 0)
        )
        l_start = longest_block.get("start")
        l_end = longest_block.get("end")
        if (
            l_start is not None
            and l_end is not None
            and l_start not in selected_start_times
        ):
            clip_end = min(l_start + preview_duration, l_end)
            if clip_end > l_start:
                log_info(
                    f"Adding second clip from start of longest block ({l_start:.2f}-{l_end:.2f})"
                )
                preview_segments.append((l_start, clip_end))
                selected_start_times.add(l_start)

    # Final sort by start time
    preview_segments.sort(key=lambda x: x[0])

    log_info(
        f"Selected {len(preview_segments)} preview segments for {speaker_id}: {[(f'{s:.2f}', f'{e:.2f}') for s, e in preview_segments]}"
    )
    return preview_segments


def download_video_clips(
    youtube_url: str,
    time_segments: List[Tuple[float, float]],
    output_dir: Path,
    item_identifier: str,
    speaker_id: str,
    log_file_handle: Optional[TextIO] = None,
) -> List[Path]:
    """
    Downloads specific time segments from a YouTube video using yt-dlp.

    Args:
        youtube_url: The URL of the source YouTube video.
        time_segments: List of (start_time, end_time) tuples to download.
        output_dir: Directory to save the downloaded clips.
        item_identifier: Identifier for the batch item (e.g., 'item_001').
        speaker_id: The speaker ID these clips are for (e.g., 'SPEAKER_00').
        log_file_handle: Optional file handle for logging command output.

    Returns:
        A list of Path objects pointing to the successfully downloaded video files.
        Returns an empty list if downloads fail.
    """
    session_id = f"{item_identifier}_{speaker_id}_preview"
    log_info(
        f"[{session_id}] Downloading {len(time_segments)} video clips for {speaker_id} from {youtube_url}..."
    )

    downloaded_files: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    # yt-dlp format string needs care, especially with colons and spaces
    # Example: --download-sections "*1:05-1:15"
    # Example: --download-sections "*5:30-inf" (from time to end)

    for i, (start_time, end_time) in enumerate(time_segments):
        clip_num = i + 1
        # Format times for yt-dlp (HH:MM:SS.ms or just seconds)
        # Using seconds seems safer and less prone to parsing issues if yt-dlp supports it directly.
        # According to docs, "*ss" or "*ss.ms" should work.
        time_range_str = f"*{start_time:.3f}-{end_time:.3f}"

        # Define output filename pattern
        # Sanitize speaker_id just in case it contains odd characters later
        safe_speaker_id = speaker_id.replace(":", "_").replace("/", "_")
        output_filename = f"{item_identifier}_{safe_speaker_id}_clip_{clip_num}.mp4"
        output_path = output_dir / output_filename

        # Construct yt-dlp command
        command = [
            "yt-dlp",
            # Force overwrite to handle potential re-runs during testing
            "--force-overwrite",
            # Download specific section
            "--download-sections",
            time_range_str,
            # Try to get a reasonable quality mp4 directly if possible
            # This format string might need adjustment based on available formats
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            # Output template
            "-o",
            str(output_path),
            # Force IPV4 if IPV6 causes issues (optional)
            # "--force-ipv4",
            # The URL
            "--",  # Use '--' for safety with URLs containing special characters
            youtube_url,
        ]

        log_info(
            f"[{session_id}] Running command for clip {clip_num}: {' '.join(command)}"
        )

        try:
            # Run the command using safe_run
            safe_run(
                command=command,
                log_file_handle=log_file_handle,
                session_id=session_id,
                # Add capture_output=True if you need to check output, but safe_run logs it anyway
            )

            # Check if the file was actually created
            if output_path.exists() and output_path.stat().st_size > 0:
                log_info(
                    f"[{session_id}] Successfully downloaded clip {clip_num} to {output_path}"
                )
                downloaded_files.append(output_path)
            else:
                log_warning(
                    f"[{session_id}] Download command ran for clip {clip_num}, but output file is missing or empty: {output_path}"
                )

        except Exception as e:
            log_error(
                f"[{session_id}] Failed to download clip {clip_num} ({start_time:.2f}-{end_time:.2f}s): {e}"
            )
            log_error(f"[{session_id}] Command attempted: {' '.join(command)}")
            log_error(traceback.format_exc())
            # Continue to try downloading other clips

    log_info(
        f"[{session_id}] Finished download attempts. Successfully downloaded {len(downloaded_files)} clips."
    )
    return downloaded_files
