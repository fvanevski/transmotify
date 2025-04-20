# core/utils.py
import os
import subprocess
import tempfile
import traceback
import re # Import re for parsing
from pathlib import Path
# Added Callable for type hinting the callback function
from typing import List, Optional, TextIO, Union, Dict, Callable, Any

# --- ADD LOGGING IMPORTS ---
from .logging import log_warning, log_error
# Import pandas to handle potential NaN values from excel reading
import pandas as pd


def create_directory(path: Union[str, Path]) -> None:
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_warning(f"Could not create directory {dir_path}: {e}")

def get_temp_file(suffix: str = "") -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file_path: str = temp_file.name
    temp_file.close()
    return temp_file_path

# safe_run function with both env and output_callback parameters
def safe_run(
    command: List[str],
    log_file_handle: Optional[TextIO],
    session_id: Optional[str] = None,
    env: Optional[Dict[str, str]] = None, # Parameter for environment variables
    output_callback: Optional[Callable[[str], None]] = None # Parameter for output processing callback
) -> None:
    safe_command: List[str] = [str(item) for item in command]
    log_prefix: str = f"[{session_id if session_id else 'PROC'}] "
    process: Optional[subprocess.Popen] = None

    # Prepare the environment for the subprocess
    # Start with the current environment and update with provided env dict
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    try:
        # Start the subprocess with the prepared environment
        process = subprocess.Popen(
            safe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=subprocess_env # Pass the environment here
        )

        # Read output line by line, write to log file, and pass to callback
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                log_line: str = f"{log_prefix}{line}"
                # Always write to the dedicated process log file if handle is provided
                if log_file_handle:
                    try:
                        log_file_handle.write(log_line)
                        log_file_handle.flush() # Ensure data is written immediately
                    except Exception as log_e:
                        # Keep fallback print if writing to file handle fails
                        print(f"WARN: Failed to write log line to file handle: {log_e} - Line: {line.strip()}") # KEEP PRINT (fallback)

                # Pass the line to the output callback if provided
                if output_callback:
                    try:
                        output_callback(line)
                    except Exception as cb_e:
                        # Log errors occurring within the callback to the main logger
                        cb_error_msg = f"[{session_id if session_id else 'PROC'}] ERROR in output_callback: {cb_e} - Line: {line.strip()}"
                        log_error(cb_error_msg)
                        # Also try to write this callback error to the file handle
                        if log_file_handle:
                            try:
                                log_file_handle.write(f"{cb_error_msg}\n")
                                log_file_handle.flush()
                            except Exception as log_e2:
                                # Fallback print if writing callback error to file handle fails
                                print(f"WARN: Failed to write callback error to file handle: {log_e2}") # KEEP PRINT (fallback)


            process.stdout.close() # Close the pipe after reading

        # Wait for the process to finish and get the return code
        return_code: int = process.wait()

        # Check for non-zero exit code indicating an error
        if return_code != 0:
            error_msg: str = f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            # Attempt to write the error message to the dedicated process log file
            if log_file_handle:
                try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
                except Exception as log_e:
                     # Keep fallback print if writing command error to file handle fails
                     print(f"WARN: Failed to write command error to log file handle: {log_e}") # KEEP PRINT (fallback)
            # Log the error using the main application logger
            log_error(error_msg)
            # Raise a Python exception to propagate the failure
            raise RuntimeError(error_msg)

    except FileNotFoundError:
        # Handle case where the command executable is not found
        error_msg = f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        if log_file_handle:
            try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n")
            except Exception as log_e:
                 print(f"WARN: Failed to write FileNotFoundError to log file handle: {log_e}") # KEEP PRINT (fallback)
        # Log the error using the main application logger
        log_error(error_msg)
        # Re-raise the error
        raise FileNotFoundError(error_msg) from None # 'from None' prevents linking exceptions

    except Exception as e:
        # Handle any other exceptions that occur during process execution
        error_msg = f"An error occurred while running command {' '.join(safe_command)}: {e}"
        if log_file_handle:
            try: log_file_handle.write(f"{log_prefix}ERROR: {error_msg}\n{traceback.format_exc()}\n")
            except Exception as log_e:
                 print(f"WARN: Failed to write other exception to log file handle: {log_e}") # KEEP PRINT (fallback)
        # Log the error and traceback using the main application logger
        log_error(error_msg)
        log_error(traceback.format_exc())

        # Attempt to terminate the subprocess if it's still running
        if process and process.poll() is None: # poll() returns None if still running
            try:
                process.terminate() # Request graceful termination
                process.wait(timeout=5) # Wait a bit for it to terminate
            except subprocess.TimeoutExpired:
                 # Keep termination warnings as prints
                 print(f"{log_prefix}WARN: Process did not terminate gracefully, killing.") # KEEP PRINT
                 process.kill() # Force kill if terminate timed out
            except Exception as term_err:
                print(f"{log_prefix}WARN: Error terminating process after failure: {term_err}") # KEEP PRINT
        # Re-raise the original exception, chaining it
        raise RuntimeError(error_msg) from e

# ADDED: Helper method to parse snippet string from XLSX cell
def parse_xlsx_snippets(snippet_string: Any) -> Dict[str, str]:
    """
    Parses a string from an XLSX cell into a Dict[Name, Snippet].
    Handles None, NaN, and non-string types from pandas read_excel.
    """
    mapping = {}
    # Check for pandas NaN or None explicitly
    if pd.isna(snippet_string) or not isinstance(snippet_string, str) or not snippet_string.strip():
        return mapping

    lines = snippet_string.strip().split('\n')
    for line in lines:
        # Use the same regex logic as the UI parsing
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
        if match:
            name = match.group(1).strip()
            snippet = match.group(2).strip()
            if name and snippet:
                mapping[name] = snippet
            else:
                # Log as warning during development, could be info later
                log_warning(f"Could not parse snippet line effectively from XLSX: '{line}'")
        else:
            log_warning(f"Ignoring invalid snippet line format from XLSX: '{line}'")
    return mapping