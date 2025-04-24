# test/unit/test_utils.py

"""Unit tests for generic utilities (paths, subprocess, json)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from utils.paths import ensure_dir, get_temp_file
from utils.subprocess import run, SubprocessError
from utils.json import sanitize


# ---------------------------------------------------------------------------
# paths helpers
# ---------------------------------------------------------------------------


def test_ensure_dir(tmp_path: Path) -> None:
    """ensure_dir should create nested directories idempotently."""
    target = tmp_path / "nested" / "dir"
    assert not target.exists()
    returned = ensure_dir(target)
    assert returned == target
    assert target.is_dir()


def test_get_temp_file() -> None:
    """get_temp_file must return a writable path that actually exists."""
    p = get_temp_file(".txt")
    assert p.exists()
    p.write_text("hi")
    assert p.read_text() == "hi"
    p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# subprocess wrapper
# ---------------------------------------------------------------------------


def test_run_success_capture_output() -> None:
    """run() should capture stdout when requested and return it."""
    out = run([sys.executable, "-c", "print('hello')"], capture_output=True)
    assert out.strip() == "hello"


def test_run_failure_raises() -> None:
    """A non‑zero exit code must raise SubprocessError."""
    with pytest.raises(SubprocessError):
        run([sys.executable, "-c", "import sys; sys.exit(1)"])


# ---------------------------------------------------------------------------
# json sanitize
# ---------------------------------------------------------------------------


def test_sanitize_numpy_scalars() -> None:
    """sanitize should replace numpy scalars with native Python types."""
    original = {"a": np.float32(1.5), "b": [np.int64(2)]}
    sanitized = sanitize(original)
    assert isinstance(sanitized["a"], float)
    assert isinstance(sanitized["b"][0], int)
    # Values should remain equal
    assert sanitized == {"a": 1.5, "b": [2]}
