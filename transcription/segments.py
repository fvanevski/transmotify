 """speech_analysis.transcription.segments
---------------------------------------
Utility functions that operate on the WhisperX JSON transcript structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Final

from speech_analysis.core.logging import get_logger

logger = get_logger(__name__)

__all__: Final = [
    "Segment",
    "load_segments",
    "group_by_speaker",
]


class Segment(TypedDict):
    start: float | None
    end: float | None
    text: str
    speaker: str
    words: List[dict[str, Any]]


def load_segments(json_path: Path) -> List[Segment]:
    """Read WhisperX JSON file and return a validated list of segments."""
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    with open(json_path, "r", encoding="utf-8") as fh:
        try:
            data: dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("Malformed JSON in %s: %s", json_path, exc)
            raise

    raw_segments: Any = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raise TypeError("'segments' key must be a list, got %s" % type(raw_segments))

    cleaned: List[Segment] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            logger.warning("Skipping non-dict segment: %r", seg)
            continue
        cleaned.append(
            Segment(
                start=seg.get("start"),
                end=seg.get("end"),
                text=(seg.get("text") or "").strip(),
                speaker=str(seg.get("speaker", "unknown")),
                words=seg.get("words", []) if isinstance(seg.get("words"), list) else [],
            )
        )
    logger.info("Loaded %d segments from %s", len(cleaned), json_path.name)
    return cleaned


def group_by_speaker(segments: List[Segment]) -> List[dict[str, Any]]:
    """Collapse consecutive segments with same speaker into contiguous blocks."""
    if not segments:
        return []

    blocks: List[dict[str, Any]] = []
    cur = segments[0]
    accum = {
        "speaker": cur["speaker"],
        "text": cur["text"],
        "start": cur["start"],
        "end": cur["end"],
        "indices": [0],
    }
    for idx, seg in enumerate(segments[1:], start=1):
        if seg["speaker"] == accum["speaker"]:
            accum["text"] += (" " if accum["text"] else "") + seg["text"]
            accum["end"] = seg["end"]
            accum["indices"].append(idx)
        else:
            blocks.append(accum)
            accum = {
                "speaker": seg["speaker"],
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "indices": [idx],
            }
    blocks.append(accum)
    logger.info("Grouped %d segments into %d blocks", len(segments), len(blocks))
    return blocks
