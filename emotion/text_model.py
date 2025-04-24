# emotion/text_model.py

"""emotion.text_model
-------------------------------------
Wrapper around a HuggingFace text‑classification pipeline that predicts
**emotion probabilities** for a single utterance.
"""

from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Final

from transformers import pipeline as hf_pipeline

from core.logging import get_logger

logger = get_logger(__name__)

__all__: Final = [
    "TextEmotionModel",
]

_LABEL_FALLBACK = {"label": "no_text", "score": 1.0}


class TextEmotionModel:
    """Lazy‑loads a HuggingFace emotion classifier and returns *probability dicts*."""

    def __init__(
        self, model_name: str | None = None, *, device: str | int | None = None
    ):
        self.model_name = model_name or "j-hartmann/emotion-english-distilroberta-base"
        self.device = device  # -1 = CPU, 0+ = CUDA device id, None = auto

    # ------------------------------------------------------------------
    # Lazy resources
    # ------------------------------------------------------------------

    @cached_property
    def _pipe(self):
        logger.info(
            f"Loading text emotion model '{self.model_name}' on device={self.device} …"
        )
        try:
            pipe = hf_pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=self.device if self.device is not None else -1,
            )
            return pipe
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load HuggingFace emotion model: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> List[Dict[str, float]]:  # noqa: D401
        """Return a list like ``[{"label": "joy", "score": 0.87}, …]``.

        If the pipeline isn’t loaded or *text* is blank, returns a single
        fallback entry indicating no analysis.
        """
        if not text or not text.strip():
            return [_LABEL_FALLBACK]

        if self._pipe is None:
            return [{"label": "analysis_failed", "score": 1.0}]

        try:
            result = self._pipe(text)
            # HF pipeline wraps results in an extra list dim
            return result[0] if result else [_LABEL_FALLBACK]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Text emotion inference error: {exc}")
            return [{"label": "analysis_failed", "score": 1.0}]
