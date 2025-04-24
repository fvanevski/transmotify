# emotion/visual_model.py

"""emotion.visual_model
--------------------------------------
Placeholder for frame‑level facial‑affect inference (currently unimplemented).

The interface mirrors the audio/text models so that the orchestrator doesn’t
need to special‑case missing modalities.
"""

from __future__ import annotations

from typing import Dict

from core.logging import get_logger

logger = get_logger(__name__)

__all__ = ["VisualEmotionModel"]


class VisualEmotionModel:
    """Stub that always returns ``{"analysis_skipped": 1.0}``."""

    def __init__(self, config, *, device: str = "cpu") -> None:  # noqa: D401
        self.config = config
        self.device = device
        logger.info(
            "VisualEmotionModel initialised as placeholder – no ops will be performed."
        )

    def predict(self, frame) -> Dict[str, float]:  # noqa: D401
        """Return a dummy distribution signalling the analysis was skipped."""
        return {"analysis_skipped": 1.0}
