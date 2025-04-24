 """speech_analysis.reporting.plotting
-------------------------------------
Matplotlib visualisations for the emotion analysis pipeline.  All functions
create exactly *one* figure, save it to disk, and then close the figure to
avoid global‑state leaks.

Per dev‑rules: no explicit colours are set; matplotlib will choose defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

from speech_analysis.core.logging import get_logger
from speech_analysis.emotion.constants import EMO_VAL

logger = get_logger(__name__)

__all__ = [
    "plot_trajectory",
    "plot_distribution",
    "plot_volatility",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_out_dir(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_trajectory(segments: List[Dict], out: Path) -> Path:
    """Line plot of emotion valence over time."""
    _ensure_out_dir(out)

    xs = [seg.get("start", 0.0) for seg in segments]
    ys = [EMO_VAL.get(seg.get("emotion", "unknown"), 0.0) for seg in segments]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Valence (mapped)")
    ax.set_title("Emotion trajectory")

    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved trajectory plot to %s", out)
    return out


def plot_distribution(summary: Dict, out: Path) -> Path:
    """Bar chart of emotion distribution across all segments."""
    _ensure_out_dir(out)

    emotion_counts = summary.get("global", {}).get("emotion_counts", {})
    labels = list(emotion_counts.keys())
    values = [emotion_counts[lbl] for lbl in labels]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Count")
    ax.set_title("Emotion distribution (global)")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved distribution plot to %s", out)
    return out


def plot_volatility(summary: Dict, out: Path) -> Path:
    """Bar chart of per‑speaker emotion volatility (σ of valence)."""
    _ensure_out_dir(out)

    per_spk = summary.get("per_speaker", {})
    speakers = list(per_spk.keys())
    volt = [per_spk[s]["volatility"] for s in speakers]

    fig, ax = plt.subplots()
    ax.bar(speakers, volt)
    ax.set_ylabel("σ(Valence)")
    ax.set_title("Emotion volatility by speaker")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved volatility plot to %s", out)
    return out
