# core/snippet_matcher.py
import string
import re
from typing import Dict, List, Any

from rapidfuzz import fuzz

from .logging import log_info, log_warning

Segment = Dict[str, Any]
SegmentsList = List[Segment]


# Helper function for aggressive normalization (keep this as it helps comparison)
def aggressively_normalize(text: str) -> str:
    """
    Aggressively normalize text by removing punctuation, converting to lowercase,
    and collapsing whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def match_snippets_to_speakers(
    segments: SegmentsList,  # Receive segments (should have original SPEAKER_XX IDs)
    speaker_snippet_map: Dict[str, str],
    snippet_match_threshold: float = 0.8,  # Default threshold 0-1
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

    blocks = group_segments_by_speaker(segments)

    speaker_id_to_name_mapping: Dict[str, str] = {}
    # This dictionary is used internally to track the best score found for each original speaker ID
    # and prevent a lower-scoring snippet from overriding a higher-scoring one for the same speaker.
    best_match_scores: Dict[str, float] = {}

    # Threshold should be in percentage for rapidfuzz (0-100)
    # Convert the 0-1 threshold from config to 0-100 for rapidfuzz
    thresh = snippet_match_threshold * 100.0
    log_info(f"Using snippet matching threshold: {thresh} (converted for rapidfuzz)")

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
