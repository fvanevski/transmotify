 """speech_analysis.labeling.selector
------------------------------------
Utilities for *automatic* speaker selection and preview‑clip calculation that
support the **interactive labeling** flow exposed in the Gradio UI.

Public API
~~~~~~~~~~
``identify_eligible_speakers``
    Filter and rank speakers by total speaking time & block duration.

``select_preview_time_segments``
    Pick up to three preview‑clip start times (seconds) for the chosen speaker.

``match_snippets_to_speakers``
    Fuzzy‑match user‑supplied snippets to WhisperX speaker IDs using RapidFuzz.
"""

from __future__ import annotations

import math
import string
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Final

from rapidfuzz import fuzz

from speech_analysis.core.logging import get_logger
from speech_analysis.constants import DEFAULT_SNIPPET_MATCH_THRESHOLD
from speech_analysis.transcription.segments import (
    Segment,
    SegmentsList,
    group_segments_by_speaker,
)

logger = get_logger(__name__)

__all__: Final = [
    "identify_eligible_speakers",
    "select_preview_time_segments",
    "match_snippets_to_speakers",
]


# ---------------------------------------------------------------------------
# Speaker eligibility
# ---------------------------------------------------------------------------

def identify_eligible_speakers(
    segments: SegmentsList, *, min_total_time: float, min_block_time: float
) -> List[str]:
    """Return WhisperX speaker IDs that surpass the *total* speaking time **and**
    *at least one* continuous block duration threshold.

    Speakers are returned **sorted by total speaking time (desc)** so the most
    prominent voices are labeled first.
    """
    logger.info(
        "Checking eligible speakers (min_total=%.1fs, min_block=%.1fs)…",
        min_total_time,
        min_block_time,
    )
    if not segments:
        logger.warning("No segments provided – returning empty list.")
        return []

    speaker_total: dict[str, float] = defaultdict(float)
    for seg in segments:
        spk, start, end = seg.get("speaker"), seg.get("start"), seg.get("end")
        if spk is None or start is None or end is None or end <= start:
            continue
        speaker_total[str(spk)] += end - start

    # Map speaker → list of dialogue blocks
    speaker_blocks: dict[str, List[Segment]] = defaultdict(list)
    for block in group_segments_by_speaker(segments):
        speaker_blocks[str(block["speaker"])].append(block)

    eligible: list[tuple[str, float]] = []
    for spk, total in speaker_total.items():
        if total < min_total_time:
            continue
        # has long block?
        if any(
            (blk["end"] - blk["start"] >= min_block_time)
            for blk in speaker_blocks.get(spk, [])
            if blk["start"] is not None and blk["end"] is not None
        ):
            eligible.append((spk, total))

    eligible.sort(key=lambda t: t[1], reverse=True)
    ids = [spk for spk, _ in eligible]
    logger.info("Eligible speakers: %s", ids)
    return ids


# ---------------------------------------------------------------------------
# Preview‑clip selection
# ---------------------------------------------------------------------------

def select_preview_time_segments(
    *,
    speaker_id: str,
    segments: SegmentsList,
    preview_duration: float,
    min_block_time: float,
) -> List[int]:
    """Return up to three **integer** second offsets suitable for preview clips.

    • Start of the first three blocks ⩾ *min_block_time*.
    • Fallback: tail of the longest block.
    • Guarantees at least two distinct times when possible.
    """
    logger.info(
        "Selecting previews for %s (preview=%.1fs, min_block=%.1fs)",
        speaker_id,
        preview_duration,
        min_block_time,
    )

    spk_segments = [s for s in segments if str(s.get("speaker")) == speaker_id]
    if not spk_segments:
        logger.warning("No segments for speaker %s", speaker_id)
        return []

    blocks = group_segments_by_speaker(spk_segments)
    if not blocks:
        logger.warning("No dialogue blocks for speaker %s", speaker_id)
        return []

    blocks.sort(key=lambda b: b.get("start", float("inf")))
    long_blocks = [b for b in blocks if (b["end"] - b["start"] >= min_block_time)]

    starts: list[float] = []
    for blk in long_blocks[:3]:
        starts.append(max(0.0, blk["start"]))

    # Fallback
    if len(starts) < 3:
        longest = max(blocks, key=lambda b: b["end"] - b["start"])
        tail = max(0.0, longest["end"] - preview_duration)
        starts.append(tail)

    # Ensure uniqueness & at least two values if possible
    starts = sorted({math.floor(s) for s in starts})
    if len(starts) == 1 and len(blocks) > 1:
        starts.append(math.floor(blocks[0]["start"]))
    logger.info("Preview starts for %s: %s", speaker_id, starts)
    return starts[:3]


# ---------------------------------------------------------------------------
# Fuzzy snippet‑to‑speaker match
# ---------------------------------------------------------------------------

def match_snippets_to_speakers(
    *,
    segments: SegmentsList,
    speaker_snippet_map: Dict[str, str],
    threshold: float = DEFAULT_SNIPPET_MATCH_THRESHOLD,
) -> Dict[str, str]:
    """Match user‑supplied *snippet* → WhisperX speaker ID using partial‑ratio.

    Returns a mapping ``speaker_id -> user_name``.
    """
    logger.info("Matching %d snippets against %d segments", len(speaker_snippet_map), len(segments))
    if not speaker_snippet_map or not segments:
        return {}

    blocks = group_segments_by_speaker(segments)
    if not blocks:
        logger.warning("No blocks to match snippets against.")
        return {}

    def _norm(txt: str) -> str:
        txt = txt.translate(str.maketrans("", "", string.punctuation)).lower()
        return re.sub(r"\s+", " ", txt).strip()

    speaker_best: dict[str, float] = {}
    mapping: dict[str, str] = {}

    for user_name, snippet in speaker_snippet_map.items():
        base = _norm(snippet)
        best_score, best_id = -1.0, None
        for blk in blocks:
            sid, text = blk["speaker"], blk["text"]
            score = fuzz.partial_ratio(base, _norm(text))
            if score > best_score:
                best_score, best_id = score, sid
        if best_score >= threshold * 100:
            if best_score > speaker_best.get(best_id, -1.0):
                mapping[str(best_id)] = user_name
                speaker_best[str(best_id)] = best_score
                logger.debug("Snippet '%s' matched to %s (%.1f)", user_name, best_id, best_score)
        else:
            logger.info("Snippet '%s' matched below threshold (%.1f)", user_name, best_score)
    logger.info("Final snippet mapping: %s", mapping)
    return mapping
