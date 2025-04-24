# emotion/analyzer.py

"""emotion.analyzer
----------------------------------
High‑level façade that orchestrates text, audio (SpeechBrain), and visual
emotion models and writes fused labels back into transcript segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torchaudio

from core.config import Config
from core.logging import get_logger
from transcription.segments import Segment

from .text_model import TextEmotionModel
from .audio_model import AudioEmotionModel
from .visual_model import VisualEmotionModel
from .fusion import fuse_weighted_average
from .metrics import significant_text_only

logger = get_logger(__name__)

__all__ = ["MultimodalAnalyzer"]


class MultimodalAnalyzer:
    """Run multimodal affect analysis on WhisperX segments."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Pass specific config values, not the whole cfg object
        self.text_model = TextEmotionModel(
            # Assuming text model name isn't in schema, let it use its default
            # model_name=cfg.text_emotion_model, # Add if you put this in schema
            device=cfg.device
        )
        self.audio_model = AudioEmotionModel(source=cfg.audio_emotion_model,device=cfg.device)

        # visual is optional placeholder
        self.visual_model = VisualEmotionModel(cfg)

        self.text_weight: float = cfg.text_fusion_weight
        self.audio_weight: float = cfg.audio_fusion_weight

        logger.info(
            "MultimodalAnalyzer initialised (text_weight=%.2f, audio_weight=%.2f)",
            self.text_weight,
            self.audio_weight,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(
        self,
        segments: Sequence[Segment],
        audio_path: Path,
        video_path: Path | None = None,
    ) -> List[Segment]:
        """Update *segments* in‑place with emotion predictions and return it."""

        # Make a list to ensure mutability in caller
        segs: List[Segment] = list(segments)

        # ---- TEXT --------------------------------------------------------
        logger.info("Running text‑based emotion classification …")
        for seg in segs:
            seg["text_emotion"] = self.text_model.predict(seg.get("text", ""))

        # ---- AUDIO -------------------------------------------------------
        logger.info("Running audio‑based emotion classification …")
        wave, sr = torchaudio.load(audio_path)
        wave = wave.to(self.audio_model.device)
        for seg in segs:
            start, end = seg.get("start", 0), seg.get("end", 0)
            if end <= start:
                seg["audio_emotion"] = {}
                continue
            frame_offset = int(start * sr)
            num_frames = int((end - start) * sr)
            snippet = wave[:, frame_offset : frame_offset + num_frames]
            seg["audio_emotion"] = self.audio_model.predict(snippet, sr)

        # ---- VISUAL (placeholder) ---------------------------------------
        for seg in segs:
            seg["visual_emotion"] = None

        # ---- FUSION ------------------------------------------------------
        logger.info("Fusing modalities …")
        for seg in segs:
            text_probs: Dict[str, float] = seg.get("text_emotion", {})
            audio_probs: Dict[str, float] = seg.get("audio_emotion", {})

            label, conf = fuse_weighted_average(
                text_probs,
                audio_probs,
                self.text_weight,
                self.audio_weight,
            )
            seg["fused_emotion"] = label
            seg["fused_emotion_confidence"] = conf

            # significant text‑only
            text_only = {
                lab: prob
                for lab, prob in text_probs.items()
                if lab not in self.audio_model.labels
            }
            seg["significant_text_emotions"] = significant_text_only(
                text_only,
                max(audio_probs.values()) if audio_probs else 0.0,
            )

            seg["emotion"] = label or "unknown"

        logger.info("Multimodal analysis complete (%d segments).", len(segs))
        return segs


#
