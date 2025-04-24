# emotion/fusion.py

"""emotion.fusion
--------------------------------
Stateless helpers for fusing emotion probability dictionaries coming from
text & audio classifiers.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from core.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = ["fuse_weighted_average"]


def fuse_weighted_average(
    text_probs: Dict[str, float],
    audio_probs: Dict[str, float],
    *,
    text_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> Tuple[str, float, Dict[str, float]]:
    """Blend two probability maps with given weights.

    Returns a tuple ``(label, confidence, fused_probs)`` where *label* is the
    max‑probability emotion among the union of keys, *confidence* its fused
    probability, and *fused_probs* the full combined distribution.
    """

    if text_weight + audio_weight == 0:
        logger.warning("Fusion weights sum to zero – defaulting to equal weights.")
        text_weight = audio_weight = 0.5

    # normalise weights
    total = text_weight + audio_weight
    text_w = text_weight / total
    audio_w = audio_weight / total

    logger.debug(f"text_probs type: {type(text_probs)}, content: {text_probs}")
    logger.debug(f"audio_probs type: {type(audio_probs)}, content: {audio_probs}")

    if not isinstance(text_probs, dict) or not isinstance(audio_probs, dict):
        logger.error(
            f"Expected dict for emotion probabilities, but got text_probs type: {type(text_probs)}, audio_probs type: {type(audio_probs)}"
        )
        # Return default values or raise a more specific error
        return (
            "unknown",
            0.0,
            {},
        )  # Assuming the function should return label, confidence, and fused probabilities

    labels = set(text_probs) | set(audio_probs)
    fused: Dict[str, float] = {}
    for lab in labels:
        fused[lab] = text_w * text_probs.get(lab, 0.0) + audio_w * audio_probs.get(
            lab, 0.0
        )

    if not fused or all(v == 0 for v in fused.values()):
        return "unknown", 0.0, fused

    best_label = max(fused, key=fused.get)
    return best_label, fused[best_label], fused
