# labeling/session.py

"""labeling.session
-----------------------------------
State holder for the **interactive speaker‑labeling** flow used by the Gradio
UI.  It is intentionally lightweight: all heavy computation (preview‑time
selection, snippet matching) lives in :pymod:`labeling.selector`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.logging import get_logger
from labeling import selector
from transcription.segments import Segment  # typed‑alias

logger = get_logger(__name__)

__all__ = ["LabelingSession"]


@dataclass
class _ItemState:
    youtube_url: str
    segments: List[Segment]
    eligible: List[str]
    labels: Dict[str, str] = field(default_factory=dict)

    def next_unlabeled(self) -> Optional[str]:
        for spk in self.eligible:
            if spk not in self.labels:
                return spk
        return None


class LabelingSession:
    """Holds state for one batch that requires user speaker labels.

    The session does **no** file‑system IO; it only keeps an in‑memory mapping
    of item‑IDs to speaker‑label progress and relies on
    :pymod:`labeling.selector` for all heavy math.
    """

    def __init__(
        self,
        *,
        preview_duration: float,
        min_block_time: float,
        min_total_time: float | None = None,
    ):
        self._items: Dict[str, _ItemState] = {}
        self.preview_duration = preview_duration
        self.min_block_time = min_block_time
        self.min_total_time = min_total_time or (min_block_time * 1.5)

    # ------------------------------------------------------------------
    # Public API called by the UI / CLI
    # ------------------------------------------------------------------

    def add_item(self, item_id: str, youtube_url: str, segments: List[Segment]):
        """Register a transcript that still has generic *SPEAKER_XX* IDs."""
        elig = selector.identify_eligible_speakers(
            segments, self.min_total_time, self.min_block_time
        )
        self._items[item_id] = _ItemState(youtube_url, segments, elig)
        logger.info(f"[labeling] Registered {item_id}: {len(elig)} eligible speaker(s)")

    # ---------------- First speaker for an item -------------------
    def start(self, item_id: str):
        """Return *(speaker_id, youtube_url, preview_start_times)* for the first unlabeled speaker."""
        return self._get_data_for_next_speaker(item_id)

    # ---------------- Store user input ---------------------------
    def store_label(self, item_id: str, speaker_id: str, label: str) -> bool:
        st = self._items.get(item_id)
        if not st:
            return False
        st.labels[speaker_id] = label.strip()
        logger.info(f"[labeling] {item_id}: {speaker_id} → '{label}'")
        return True

    # ---------------- Advance within same item -------------------
    def next(self, item_id: str) -> Optional[tuple[str, str, list[int]]]:
        """Return data for the next unlabeled speaker in *item_id* or *None* if finished."""
        return self._get_data_for_next_speaker(item_id)

    # ---------------- Finish & return relabeled segments ---------
    def finalize_item(self, item_id: str):
        """Apply collected labels and pop state.  Returns *(segments, label_map)* or *None*."""
        st = self._items.pop(item_id, None)
        if not st:
            return None
        segs = st.segments
        for seg in segs:
            sid = str(seg.get("speaker", ""))
            if sid in st.labels and st.labels[sid].strip():
                seg["speaker"] = st.labels[sid]
        return segs, st.labels

    # ---------------- Diagnostics ------------------------------
    def pending_items(self) -> List[str]:
        return [k for k, v in self._items.items() if v.next_unlabeled()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data_for_next_speaker(
        self, item_id: str
    ) -> Optional[tuple[str, str, list[int]]]:
        st = self._items.get(item_id)
        if not st:
            return None
        spk = st.next_unlabeled()
        if spk is None:
            return None
        starts = selector.select_preview_time_segments(
            spk, st.segments, self.preview_duration, self.min_block_time
        )
        return spk, st.youtube_url, starts
