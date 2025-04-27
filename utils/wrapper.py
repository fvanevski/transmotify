# utils/wrapper.py
"""
Provides utility functions for wrapping and safely executing external processes.
"""

import os
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

# Moved from core/utils.py
def safe_run(
    command: List[str],
    log_file_handle: Optional[TextIO],
    log_prefix: str = "[PROC]",
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
) -> Optional[str]:
    """
    Runs an external command safely, logging output and handling errors.

    Args:
        command: The command and arguments as a list of strings.
        log_file_handle: An open file handle for writing process logs (optional).
        log_prefix: A prefix string for log messages (e.g., session ID).
        env: Optional dictionary of environment variables for the subprocess.
        output_callback: Optional function to process command output lines in real-time.
        capture_output: If True, captures and returns the standard output as a string.
                        Note: If True, output_callback might receive lines twice if
                        it also prints them. Best used when callback doesn't print.

    Returns:
        The captured standard output as a single string if capture_output is True,
        otherwise None.

    Raises:
        FileNotFoundError: If the command executable is not found.
        RuntimeError: If the command returns a non-zero exit code or another
                      error occurs during execution.
    """
    safe_command: List[str] = [str(item) for item in command]
    process: Optional[subprocess.Popen] = None
    captured_output_lines: List[str] = [] if capture_output else None

    # Prepare the environment for the subprocess
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    logger.info(
        f"{log_prefix} Executing command: {' '.join(safe_command)}"
    )  # Log command execution

    try:
        # Start the subprocess
        process = subprocess.Popen(
            safe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 chars
            env=subprocess_env,
            bufsize=1,  # Line buffered
        )

        # Read output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                stripped_line = line.strip()  # Strip leading/trailing whitespace
                log_line: str = (
                    f"{log_prefix} {stripped_line}\n"  # Add prefix for log file
                )

                # Write raw line (with prefix) to the dedicated process log file
                if log_file_handle and not log_file_handle.closed:
                    try:
                        log_file_handle.write(log_line)
                        log_file_handle.flush()
                    except Exception as log_e:
                        # Fallback print if writing to file fails
                        print(
                            f"WARNING: Failed to write log line to file handle: {log_e} - Line: {stripped_line}"
                        )

                # Capture output line if requested
                if capture_output:
                    captured_output_lines.append(
                        line
                    )  # Append original line with newline

                # Pass the stripped line to the output callback
                if output_callback:
                    try:
                        output_callback(stripped_line)
                    except Exception as cb_e:
                        cb_error_msg = f"{log_prefix} ERROR in output_callback: {cb_e} - Line: {stripped_line}"
                        logger.error(cb_error_msg)
                        if log_file_handle and not log_file_handle.closed:
                            try:
                                log_file_handle.write(f"{cb_error_msg}\n")
                                log_file_handle.flush()
                            except Exception as log_e2:
                                print(
                                    f"WARNING: Failed to write callback error to file handle: {log_e2}"
                                )

            process.stdout.close()

        # Wait for the process and check return code
        return_code: int = process.wait()

        if return_code != 0:
            error_msg = (
                f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            )
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_file_handle.write(f"{log_prefix} ERROR: {error_msg}\n")
                except Exception as log_e:
                    print(
                        f"WARNING: Failed to write command error to log file handle: {log_e}"
                    )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.info(f"{log_prefix} Command finished successfully.")

        return "".join(captured_output_lines) if capture_output else None

    except FileNotFoundError:
        error_msg = (
            f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(f"{log_prefix} ERROR: {error_msg}\n")
            except Exception as log_e:
                print(
                    f"WARN: Failed to write FileNotFoundError to log file handle: {log_e}"
                )
        logger.error(error_msg)
        raise

    except Exception as e:
        error_msg = (
            f"An error occurred while running command {' '.join(safe_command)}: {e}"
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(
                    f"{log_prefix} ERROR: {error_msg}\n{traceback.format_exc()}\n"
                )
            except Exception as log_e:
                print(
                    f"WARN: Failed to write other exception to log file handle: {log_e}"
                )
        logger.error(error_msg)
        logger.exception(traceback.format_exc())
        if process and process.poll() is None:
            logger.warning(
                f"{log_prefix} Attempting to terminate process {process.pid}..."
            )
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.warning(f"{log_prefix} Process terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"{log_prefix} Process did not terminate gracefully, killing."
                )
                process.kill()
            except Exception as term_err:
                logger.warning(
                    f"{log_prefix} Error terminating process after failure: {term_err}"
                )
        raise RuntimeError(error_msg) from e
