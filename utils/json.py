 """speech_analysis.utils.json
--------------------------------
Helpers for making arbitrary Python data structures JSON‑serialisable.
Currently focuses on numpy scalar promotion but is easily extensible.
"""

from __future__ import annotations

from typing import Any

import numpy as _np  # soft dependency; tiny import overhead

__all__ = ["sanitize", "convert_floats"]


def _is_numpy_scalar(obj: Any) -> bool:
    """True if *obj* is a NumPy scalar (e.g., np.float32, np.int64)."""
    return isinstance(obj, _np.generic)


def sanitize(obj: Any) -> Any:  # noqa: C901 – recursion switch‑yard
    """Recursively convert non‑JSON‑serialisable scalars into built‑ins.

    * NumPy floating / integer scalars → builtin ``float`` / ``int``.
    * ``Path`` objects → ``str`` path.
    * Any nested combo of ``dict`` / ``list`` / ``tuple`` / ``set`` is walked.
    """
    # Primitive fast‑path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalars → built‑ins
    if _is_numpy_scalar(obj):  # floats & ints
        return obj.item()

    from pathlib import Path  # local import to avoid heavy runtime deps

    # pathlib.Path → str
    if isinstance(obj, Path):
        return str(obj)

    # dict → dict
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}

    # list / tuple / set → list
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]

    # fallback – let ``json.dump`` raise, but at least fail clearly
    return obj


# backward‑compat alias
convert_floats = sanitize
