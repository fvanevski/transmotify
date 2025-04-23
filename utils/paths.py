 """speech_analysis.utils.paths
--------------------------------
Filesystem convenience helpers that are safe for concurrent use and free of
pipeline‑specific logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Final

from speech_analysis.core.logging import get_logger

logger = get_logger(__name__)

__all__: Final = [
    "ensure_dir",
    "get_temp_file",
]


def ensure_dir(path: str | Path) -> Path:  # noqa: D401 – Imperative helper
    """Create *path* (recursively) if it does not exist and return it as :class:`Path`.

    Parameters
    ----------
    path:
        Directory to create.  May be a :class:`str` or :class:`~pathlib.Path`.

    Returns
    -------
    pathlib.Path
        The absolute, expanded path.

    Notes
    -----
    * Multiple processes can safely race on this function.
    * Exceptions bubble up – callers decide whether failure is fatal.
    """

    p = Path(path).expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", p)
    except Exception:  # pragma: no cover – propagated
        logger.exception("Unable to create directory: %s", p)
        raise
    return p



def get_temp_file(suffix: str = "") -> Path:  # noqa: D401 – Imperative helper
    """Return a *pathlib.Path* to a **named** temporary file.

    The file is created with *delete=False* so that downstream subprocesses can
    reopen it by name.
    """

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    finally:
        # We close immediately; caller re‑opens as needed.
        tmp.close()
    logger.debug("Created temporary file: %s", tmp.name)
    return Path(tmp.name)
