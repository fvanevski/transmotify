from pathlib import Path
from typing import List, Dict, Any
import subprocess


def select_dialogue_blocks(
    segments: List[Dict[str, Any]], min_block_len: float
) -> List[Dict[str, float]]:
    """
    Merge contiguous segments for a single speaker into continuous dialogue blocks,
    then select up to the first three blocks at least `min_block_len` long. If fewer
    than three, include the final preview window of the longest block to ensure at
    least two previews.
    Returns a list of dicts with 'start' and 'end' times for each block.
    """
    if not segments:
        return []

    # Sort segments by start time
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    # Merge contiguous/overlapping segments
    blocks: List[Dict[str, float]] = []
    cur_start = sorted_segs[0]["start"]
    cur_end = sorted_segs[0]["end"]
    for seg in sorted_segs[1:]:
        s, e = seg["start"], seg["end"]
        if s <= cur_end:
            # Overlaps or touches: extend current block
            cur_end = max(cur_end, e)
        else:
            blocks.append({"start": cur_start, "end": cur_end})
            cur_start, cur_end = s, e
    blocks.append({"start": cur_start, "end": cur_end})

    # Filter blocks by minimum duration
    valid = [b for b in blocks if (b["end"] - b["start"]) >= min_block_len]

    # If enough, return the first three
    if len(valid) >= 3:
        return valid[:3]

    # Otherwise, ensure at least two previews by adding the longest block's final window
    longest = max(blocks, key=lambda b: (b["end"] - b["start"]))
    result = valid.copy()

    # Append longest block if it's not already included
    if longest not in result:
        result.append(longest)

    # Return up to three blocks total
    return result[:3]


def extract_preview_clips(
    source_video: Path,
    blocks: List[Dict[str, float]],
    preview_len: float,
    output_dir: Path,
) -> List[Path]:
    """
    Given a source video path and a list of dialogue blocks (start/end),
    extract clips of length `preview_len` seconds from each block.
    If the block is shorter than `preview_len`, extract the final segment
    of length `preview_len` within the block.
    Clips are saved into `output_dir` as `preview_1.mp4`, etc.
    Returns a list of Paths to the generated clip files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []

    for idx, blk in enumerate(blocks, start=1):
        start, end = blk["start"], blk["end"]
        block_dur = end - start
        # Determine clip start timestamp
        if block_dur >= preview_len:
            clip_start = start
        else:
            clip_start = max(end - preview_len, start)

        out_path = output_dir / f"preview_{idx}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-ss",
            str(clip_start),
            "-t",
            str(preview_len),
            "-c",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        clip_paths.append(out_path)

    return clip_paths
