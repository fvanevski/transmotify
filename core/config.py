 """speech_analysis.core.config
--------------------------------
Runtime configuration facade that wraps :pymod:`speech_analysis.config.schema`
so that callers can interact with *one* object regardless of whether settings
originated from environment variables, a user‑supplied JSON file, or defaults.

Key features
~~~~~~~~~~~~
* **Dot‑access & mapping‑like API** – ``cfg.output_dir`` works alongside
  ``cfg["output_dir"]`` for backwards compatibility.
* **Cross‑field validation** – ensures fusion weights sum ~1, directories exist.
* **Save** – persists *only* values that differ from defaults to a JSON file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, MutableMapping

import pydantic

from speech_analysis.config.schema import Settings
from speech_analysis.core.logging import get_logger

logger = get_logger(__name__)


class Config(MutableMapping[str, Any]):
    """A lightweight wrapper around :class:`~speech_analysis.config.schema.Settings`."""

    _settings: Settings
    _defaults: MappingProxyType

    def __init__(self, json_path: str | Path | None = None):
        # 1. Load env / defaults via BaseSettings
        settings = Settings()

        # 2. Optionally overlay JSON – only non‑unset keys override
        if json_path is not None:
            json_path = Path(json_path)
            if json_path.exists():
                logger.info("Loading config overrides from %s", json_path)
                overrides = Settings.parse_file(json_path)
                settings = settings.model_copy(update=overrides.model_dump(exclude_unset=True))
            else:
                logger.warning("Config JSON %s not found – using env/defaults", json_path)

        self._settings = settings
        self._defaults = MappingProxyType(Settings().model_dump())  # frozen baseline

        # 3. Cross‑field sanity checks & side‑effects
        self._post_init()

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------
    def _post_init(self) -> None:
        """Create directories and warn for unusual combos."""
        # ensure dirs
        for d in (self.output_dir, self.temp_dir):
            d.mkdir(parents=True, exist_ok=True)

        # fusion weight sanity
        s = self.text_fusion_weight + self.audio_fusion_weight
        if not 0.99 <= s <= 1.01:
            logger.warning(
                "Fusion weights do not sum to 1. Normalising (text %.2f, audio %.2f).",
                self.text_fusion_weight,
                self.audio_fusion_weight,
            )
            total = max(s, 1e-6)
            self._settings = self._settings.model_copy(
                update={
                    "text_fusion_weight": self.text_fusion_weight / total,
                    "audio_fusion_weight": self.audio_fusion_weight / total,
                }
            )

        if self.hf_token is None:
            logger.warning(
                "HF_TOKEN missing – certain models may fail to download (pyannote)."
            )

    # ------------------------------------------------------------------
    # Mapping / attribute proxy
    # ------------------------------------------------------------------
    def __getattr__(self, item):  # dot access
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._settings, item)

    def __setattr__(self, key, value):  # to keep immutability illusion
        if key in {"_settings", "_defaults"}:
            super().__setattr__(key, value)
        else:
            self._settings = self._settings.model_copy(update={key: value})

    # Mapping interface
    def __getitem__(self, key: str) -> Any:
        return getattr(self._settings, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __delitem__(self, key: str) -> None:  # not supported but keep Mapping contract
        raise TypeError("Config keys cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(self._settings.model_dump().keys())

    def __len__(self) -> int:
        return len(self._settings.model_dump())

    # convenience
    def get(self, key: str, default: Any | None = None):
        return getattr(self._settings, key, default)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path = "config.json") -> None:
        """Save only the diff vs defaults so the file stays minimal."""
        path = Path(path)
        diff = {
            k: v
            for k, v in self._settings.model_dump().items()
            if v != self._defaults[k]
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(diff, fh, indent=2, ensure_ascii=False)
            logger.info("Configuration saved to %s", path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save config to %s – %s", path, exc)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – cosmetic
        return f"<Config {self._settings.model_dump()!r}>"
