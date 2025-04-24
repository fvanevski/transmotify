# test/unit/test_logging.py

"""Unit tests for core.logging"""

from types import SimpleNamespace
import tempfile
import os
import importlib
import logging
import pytest

import core.logging as clog


def _make_cfg(tmp_path):
    """Return a minimal config-like object expected by clog.init."""
    return SimpleNamespace(
        log_dir=str(tmp_path), log_filename="test.log", log_level="INFO"
    )


def test_init_creates_logfile(tmp_path):
    """clog.init should create the log directory & file and write a line."""
    cfg = _make_cfg(tmp_path)
    clog.init(cfg)  # should not raise

    logger = clog.get_logger(__name__)
    logger.info("hello world")

    log_file = tmp_path / "test.log"
    # Ensure handler buffers are flushed
    for h in logging.getLogger().handlers:
        if hasattr(h, "flush"):
            h.flush()

    assert log_file.exists(), "Log file was not created"
    assert "hello world" in log_file.read_text(encoding="utf-8")


def test_reinit_is_noop(tmp_path):
    """Calling init twice should not duplicate handlers or raise."""
    cfg = _make_cfg(tmp_path)
    clog.init(cfg)
    n_handlers_first = len(logging.getLogger().handlers)

    # Second call: should leave handler count unchanged
    clog.init(cfg)
    n_handlers_second = len(logging.getLogger().handlers)

    assert n_handlers_first == n_handlers_second


def test_invalid_dir_raises(tmp_path):
    """An uncreatable directory should raise LoggingInitError."""
    # Create a file where the directory should be to simulate inability to mkdir
    bad_path = tmp_path / "occupied"
    bad_path.write_text("occupied")

    cfg = SimpleNamespace(log_dir=str(bad_path), log_filename="x.log", log_level="INFO")

    with pytest.raises(clog.LoggingInitError):
        clog.init(cfg)
