# emotion/metrics.py

"""emotion.metrics
---------------------------------
Utility functions for post‑processing emotion predictions (e.g. volatility,
text‑only significance).
"""

from __future__ import annotations

from typing import Dict

from core.logging import get_logger

logger = get_logger(__name__)

__all__ = ["significant_text_only"]


def significant_text_only(
    text_only_probs: Dict[str, float],
    max_audio_prob: float,
) -> Dict[str, float]:
    """Return text‑only emotions whose probability beats every audio label."""
    sig = {
        lab: prob
        for lab, prob in text_only_probs.items()
        if prob > max_audio_prob and prob > 0
    }
    if sig:
        logger.debug("Significant text‑only emotions: %s", sig)
    return sig
