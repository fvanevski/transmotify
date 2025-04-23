 """speech_analysis.emotion.constants
-----------------------------------
Canonical mapping from emotion labels → scalar valence values.
The map is imported by both :pymod:`speech_analysis.emotion.metrics` and
:pymod:`speech_analysis.reporting.plotting` to keep sentiment math
consistent across the codebase.
"""

from __future__ import annotations

from typing import Dict, Final

EMO_VAL: Final[Dict[str, float]] = {
    "joy": 1.0,
    "love": 0.8,
    "surprise": 0.5,
    "neutral": 0.0,
    "fear": -1.5,
    "sadness": -1.0,
    "disgust": -1.8,
    "anger": -2.0,
    # Pipeline bookkeeping labels
    "unknown": 0.0,
    "analysis_skipped": 0.0,
    "analysis_failed": 0.0,
    "no_text": 0.0,
}

__all__ = ["EMO_VAL"]