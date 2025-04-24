# utils/paths.py

"""utils.paths
--------------------------------
Filesystem convenience helpers that are safe for concurrent use and free of
pipeline‑specific logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Final

from core.logging import get_logger

logger = get_logger(__name__)

__all__: Final = [
    "ensure_dir",
    "get_temp_file",
    "find_unique_filename",
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


def find_unique_filename(directory: str | Path, filename: str) -> Path:
    """Find a unique filename in the given directory.

    If the filename already exists, appends a counter in parentheses
    before the extension (e.g., 'file (1).txt').
    """
    directory_path = Path(directory).expanduser().resolve()
    base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
    counter = 0
    unique_filename = filename
    while (directory_path / unique_filename).exists():
        counter += 1
        if ext:
            unique_filename = f"{base} ({counter}).{ext}"
        else:
            unique_filename = f"{base} ({counter})"
    return directory_path / unique_filename
