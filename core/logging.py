# core/logging.py

"""core.logging
--------------------------------
Centralised, coloured, rotating logging for the whole package.
Every executable entry‑point (CLI, Gradio UI, notebooks) **must** call
`core.logging.init(config)` exactly once before importing heavy modules.

This module intentionally has *no* side‑effects on import so that unit tests
can configure logging however they like.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Final

try:
    from rich.logging import RichHandler  # type: ignore

    _RICH_AVAILABLE: Final[bool] = True
except ModuleNotFoundError:  # pragma: no cover – rich is an optional dep
    _RICH_AVAILABLE = False

__all__ = [
    "init",
    "get_logger",
    "LoggingInitError",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LoggingInitError(RuntimeError):
    """Raised when the logger cannot be initialised."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logger(name: str | None = None) -> logging.Logger:  # noqa: D401
    """Return a ``logging.Logger`` (thin re‑export).

    Usage::

        from core.logging import get_logger
        logger = get_logger(__name__)
    """

    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_FMT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT: Final[str] = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Bootstrap API
# ---------------------------------------------------------------------------


def init(config: Any, *, console: bool = True) -> None:  # noqa: D401
    """Initialise the root logger.

    Parameters
    ----------
    config
        A validated settings object (e.g., from ``core.config``)
        exposing at minimum the following *attributes* (not dict keys)::

            log_level: str = "INFO"
            log_dir: str | pathlib.Path = "output"
            log_filename: str = "app.log"
            log_rotate_bytes: int = 10 * 1024 * 1024  # 10 MB
            log_backup_count: int = 3

        Unknown attributes are ignored so tests can pass simple ``types.SimpleNamespace``.
    console
        Attach a coloured console handler (Rich <https://rich.readthedocs.io/>)
        if available, otherwise fall back to an uncoloured ``StreamHandler``.
    """

    root_logger = logging.getLogger()

    # Do not re‑initialise if already configured (allows test‑specific setups).
    if root_logger.handlers:
        return

    try:
        level_name: str = str(getattr(config, "log_level", "INFO"))
        level: int = getattr(logging, level_name.upper(), logging.INFO)

        log_dir = Path(
            getattr(config, "log_dir", getattr(config, "output_dir", "output"))
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / str(getattr(config, "log_filename", "app.log"))

        max_bytes = int(getattr(config, "log_rotate_bytes", 10 * 1024 * 1024))
        backup_cnt = int(getattr(config, "log_backup_count", 3))

        # ------------------------------------------------------------------
        # File handler (rotates)
        # ------------------------------------------------------------------
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_cnt, encoding="utf‑8"
        )
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FMT, _DATE_FMT))
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

        # ------------------------------------------------------------------
        # Optional console handler
        # ------------------------------------------------------------------
        if console:
            if _RICH_AVAILABLE:
                rich_handler = RichHandler(
                    level=level,
                    show_time=False,  # already in our file
                    show_level=False,
                    show_path=False,
                    rich_tracebacks=True,
                )
                rich_handler.setFormatter(logging.Formatter(_DEFAULT_FMT, "%H:%M:%S"))
                root_logger.addHandler(rich_handler)
            else:  # pragma: no cover – rich not installed
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(logging.Formatter(_DEFAULT_FMT, "%H:%M:%S"))
                stream_handler.setLevel(level)
                root_logger.addHandler(stream_handler)

        root_logger.setLevel(level)

        # Quieten noisy third‑party libraries unless user asked for DEBUG.
        for noisy in ("urllib3", "botocore", "boto3", "transformers"):
            logging.getLogger(noisy).setLevel(max(level + 10, logging.WARNING))

        root_logger.debug("Logging initialised → %s", log_path)

    except Exception as exc:  # pragma: no cover – rare path
        # Remove partial handlers to avoid duplicate logs if caller retries.
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
            h.close()
        raise LoggingInitError(str(exc)) from exc
