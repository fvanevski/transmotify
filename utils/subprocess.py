# utils/subprocess.py

"""utils.subprocess
-----------------------------------
Safe wrapper around :pymod:`subprocess` that logs *every* stdout / stderr line,
optionally streams it to a callback, and raises a dedicated exception on
non‑zero exit codes.

Usage example
-------------

```python
from utils.subprocess import run
from pathlib import Path

out = run(["ffmpeg", "-i", "in.mp4", "out.wav"], capture_output=True)
print(out)
```

The function is intentionally blocking; async execution belongs in a higher‑
level job scheduler.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from core.logging import get_logger

logger = get_logger(__name__)

__all__ = ["run", "SubprocessError"]


class SubprocessError(RuntimeError):
    """Raised when an external process exits with a non‑zero status."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    command: Iterable[str | os.PathLike[str]],
    *,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
    stream_callback: Callable[[str], None] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    check: bool = True,
    text_mode: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str | None:
    """Execute *command* synchronously.

    Parameters
    ----------
    command
        Command and its arguments.  Accepts any iterable; each part is coerced
        to :class:`str`.
    env
        Optional environment overrides.  Merged with the parent `os.environ`.
    capture_output
        If *True*, accumulates the child process' combined STDOUT/STDERR and
        returns it as a single string.  Mutually compatible with
        *stream_callback*; the same line can be both returned and streamed.
    stream_callback
        Function called with each line *after* it has been logged, allowing
        callers (e.g. Gradio) to update UI in real‑time.
    cwd
        Working directory for the child process.
    check
        If *True* (default) raise :class:`SubprocessError` when the exit code
        is non‑zero.
    text_mode, encoding, errors
        Passed directly to :pyclass:`subprocess.Popen`.

    Returns
    -------
    ``str | None``
        Captured output if *capture_output* is enabled, else ``None``.
    """

    safe_cmd: List[str] = [str(p) for p in command]
    full_cmd = " ".join(safe_cmd)

    # Build env
    proc_env: MutableMapping[str, str] = os.environ.copy()
    if env:
        proc_env.update(env)

    logger.debug("Running external command: %s", full_cmd)

    captured_lines: List[str] = [] if capture_output else None  # type: ignore[assignment]

    try:
        with subprocess.Popen(
            safe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=text_mode,
            encoding=encoding,
            errors=errors,
            env=proc_env,
        ) as proc:
            assert proc.stdout is not None  # for mypy
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                logger.info("%s", line)

                if capture_output:
                    captured_lines.append(line)
                if stream_callback is not None:
                    try:
                        stream_callback(line)
                    except Exception as cb_exc:  # pragma: no cover – user code
                        logger.warning(
                            "stream_callback raised %s: %s",
                            cb_exc.__class__.__name__,
                            cb_exc,
                        )

            return_code = proc.wait()

    except FileNotFoundError as fnf:
        logger.error("Command not found: %s", safe_cmd[0])
        raise SubprocessError(str(fnf)) from None
    except Exception as exc:  # pragma: no cover – defensive
        logger.error(
            "Unexpected error running command '%s': %s\n%s",
            full_cmd,
            exc,
            traceback.format_exc(),
        )
        raise

    if check and return_code != 0:
        msg = f"Command exited with status {return_code}: {full_cmd}"
        logger.error(msg)
        raise SubprocessError(msg)

    if capture_output:
        return "\n".join(captured_lines)  # type: ignore[return-value]
    return None
