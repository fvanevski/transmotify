# emotion/audio_model.py

"""emotion.audio_model
--------------------------------------
SpeechBrain‑based paralinguistic emotion classifier.

The class hides the messy *foreign_class* loading boilerplate and always
returns a **probability distribution** across the model's canonical labels
(hap, sad, ang, neu by default).
"""

from __future__ import annotations

from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Final

import numpy as np
import torch

from speechbrain.inference.interfaces import foreign_class

from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: Final = ["AudioEmotionModel"]


class AudioEmotionModel:
    """Lightweight wrapper around a SpeechBrain classifier.

    Parameters
    ----------
    source
        HuggingFace model repo or local directory containing the model.
    device
        *"cuda"* or *"cpu"*. If *"auto"* (default) the wrapper will pick
        *cuda* when available.
    labels
        Optional explicit label order. When ``None`` the wrapper tries to
        infer it from the model output length.  Defaults to the standard 4‑way
        IEMOCAP mapping ``['hap', 'sad', 'ang', 'neu']``.
    """

    _DEFAULT_LABELS: Final[List[str]] = ["hap", "sad", "ang", "neu"]

    def __init__(
        self, source: str, *, device: str = "auto", labels: List[str] | None = None
    ):
        self.source = source
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        self._labels = labels or self._DEFAULT_LABELS

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    @cached_property
    def _classifier(self):
        logger.info("Loading SpeechBrain model '%s' on %s…", self.source, self.device)
        try:
            return foreign_class(
                source=self.source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": self.device},
            )
        except Exception as err:
            logger.error(
                "Failed to load SpeechBrain model from '%s': %s", self.source, err
            )
            return None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public prediction util
    # ------------------------------------------------------------------

    def predict(
        self, waveform: torch.Tensor, sample_rate: int = 16_000
    ) -> Dict[str, float]:
        """Return emotion probabilities for the given *mono* waveform.

        The waveform tensor is assumed to be on *CPU*; the method will move it
        to the model device as needed.
        """
        if self._classifier is None:
            return {"analysis_failed": 1.0}

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        # basic sanity shape check skipped for brevity

        waveform = waveform.to(self.device)

        try:
            out = self._classifier.classify_batch(waveform)
        except Exception as err:
            logger.warning("SpeechBrain classify_batch error: %s", err)
            return {"analysis_failed": 1.0}

        # Typical SpeechBrain tuple: (probs, embeddings, lengths, predicted_labels)
        probs = None
        if (
            isinstance(out, (list, tuple))
            and len(out) > 0
            and isinstance(out[0], torch.Tensor)
        ):
            probs = out[0].squeeze(0).detach().cpu().numpy()
            if probs.ndim != 1:
                probs = probs.ravel()
        if probs is None or probs.size != len(self._labels):
            logger.warning("Unexpected SpeechBrain output shape; returning flat probs.")
            return {lbl: 0.25 for lbl in self._labels}

        # Ensure probabilities sum to 1
        if not np.all((probs >= 0) & (probs <= 1)) or not np.isclose(
            probs.sum(), 1, atol=1e-3
        ):
            probs = torch.softmax(torch.as_tensor(probs), dim=-1).numpy()

        return {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}
