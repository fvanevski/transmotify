 """speech_analysis.reporting.summaries
--------------------------------------
Compute per‑speaker and global emotion statistics from the structured segment
list produced by the pipeline.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Final

from speech_analysis.core.logging import get_logger
from speech_analysis.emotion.constants import EMO_VAL

logger = get_logger(__name__)

__all__: Final = [
    "SpeakerStats",
    "build_summary",
    "save_json",
    "save_csv",
]


@dataclass(slots=True)
class SpeakerStats:
    """Aggregated statistics for a single speaker."""

    total_time: float = 0.0  # seconds spoken
    segment_count: int = 0
    emotion_counts: Dict[str, int] | None = None  # lazily filled
    dominant_emotion: str | None = None
    volatility: float | None = None  # std‑dev of valence values

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def _valence_series(emotions: Sequence[str]) -> List[float]:
    """Return numeric valence list for volatility calc (missing → 0)."""
    return [EMO_VAL.get(e, 0.0) for e in emotions]


def build_summary(segments: List[Dict]) -> Dict[str, Dict]:
    """Aggregate statistics per speaker and overall from transcript segments."""

    per_speaker: Dict[str, SpeakerStats] = {}
    global_emotion_counts: Dict[str, int] = {}
    global_valence: List[float] = []

    for seg in segments:
        spk = str(seg.get("speaker", "unknown"))
        start, end = seg.get("start", 0.0), seg.get("end", 0.0)
        emotion = seg.get("emotion", "unknown")
        dur = max(0.0, (end or 0.0) - (start or 0.0))

        # Per‑speaker bucket --------------------------------------------------
        stats = per_speaker.setdefault(spk, SpeakerStats())
        stats.total_time += dur
        stats.segment_count += 1
        stats.emotion_counts = stats.emotion_counts or {}
        stats.emotion_counts[emotion] = stats.emotion_counts.get(emotion, 0) + 1

        # Global accum --------------------------------------------------------
        global_emotion_counts[emotion] = global_emotion_counts.get(emotion, 0) + 1
        global_valence.append(EMO_VAL.get(emotion, 0.0))

    # Post‑processing per speaker -------------------------------------------
    for stats in per_speaker.values():
        if stats.emotion_counts:
            stats.dominant_emotion = max(stats.emotion_counts, key=stats.emotion_counts.get)
        val_series = _valence_series(
            [e for e, c in stats.emotion_counts.items() for _ in range(c)]
        ) if stats.emotion_counts else []
        stats.volatility = statistics.pstdev(val_series) if len(val_series) > 1 else 0.0

    # Global stats -----------------------------------------------------------
    global_dominant = max(global_emotion_counts, key=global_emotion_counts.get) if global_emotion_counts else None
    global_vol = statistics.pstdev(global_valence) if len(global_valence) > 1 else 0.0

    summary = {
        "per_speaker": {spk: st.to_dict() for spk, st in per_speaker.items()},
        "global": {
            "segment_count": len(segments),
            "dominant_emotion": global_dominant,
            "emotion_counts": global_emotion_counts,
            "volatility": global_vol,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------


def save_json(summary: Dict, path: Path) -> Path:
    """Write summary dict to *path* in JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    logger.info("Saved JSON summary to %s", path)
    return path


def save_csv(summary: Dict, path: Path) -> Path:
    """Flatten and persist per‑speaker stats to a CSV file (pandas optional)."""
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        logger.warning("pandas not installed – skipping CSV summary")
        return path

    records = []
    for spk, st in summary["per_speaker"].items():
        rec = {
            "speaker": spk,
            **{k: v for k, v in st.items() if k != "emotion_counts"},
        }
        records.append(rec)

    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV summary to %s", path)
    return path
