# asr/asr.py
"""
Handles Automatic Speech Recognition (ASR) using the WhisperX tool.
Provides functions to run the transcription and diarization pipeline.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any

logger = logging.getLogger(__name__)

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str, **kwargs):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str, **kwargs):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str, **kwargs):
        print(f"INFO (logging unavailable): {message}")

    # Dummy safe_run if wrapper is missing
    def safe_run(*args, **kwargs):
        raise RuntimeError("utils.wrapper.safe_run not available")


def run_whisperx(
    # Input/Output
    audio_path: Path,
    output_dir: Path,
    # Core Model Config
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "float16",
    # Language & Batching
    language: Optional[str] = None,  # e.g., "en", "es", None for auto-detect
    batch_size: int = 16,
    # Diarization Config
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    # Output Finding Config (Passed from main config)
    output_filename_exclusions: List[str] = [],
    # Logging
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[ASR WhisperX]",
) -> Optional[Path]:
    """
    Runs the WhisperX transcription and diarization pipeline via command line.

    Args:
        audio_path: Path to the input audio file (WAV format recommended).
        output_dir: Directory where WhisperX should save its output files.
        model_size: Whisper model size (e.g., "tiny", "base", "small", "medium", "large-v3").
        device: Device to run on ('cuda' or 'cpu').
        compute_type: Data type for computation (e.g., "float16", "int8").
        language: Language code for transcription (or None for auto-detect).
        batch_size: Batch size for transcription inference.
        hf_token: Hugging Face token (required for Pyannote diarization models).
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        output_filename_exclusions: List of filenames to ignore when searching for the primary JSON output.
        log_file_handle: Optional file handle for logging subprocess output.
        log_prefix: Prefix for log messages.

    Returns:
        Path object to the primary WhisperX JSON output file if successful, None otherwise.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If the WhisperX command fails execution.
    """
    logger.info(f"{log_prefix} Starting WhisperX process for: {audio_path.name}")

    if not audio_path.is_file():
        logger.error(f"{log_prefix} Input audio file not found: {audio_path}")
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"{log_prefix} Failed to create output directory {output_dir}: {e}")
        # Decide if this should raise or return None. Returning None seems safer.
        return None

    # --- Build WhisperX Command ---
    command: List[str] = [
        "whisperx",
        str(audio_path),
        "--model",
        model_size,
        "--diarize",  # Always enable diarization as per original logic
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",  # Ensure JSON for structured output parsing later
        "--device",
        device,
    ]

    # Add compute type if specified
    if compute_type:
        command.extend(["--compute_type", compute_type])

    # Add batch size
    command.extend(["--batch_size", str(batch_size)])

    # Add language if specified (and not 'auto', which is the default)
    if language and language.lower() != "auto":
        command.extend(["--language", language])

    # Add Hugging Face token if available (required for pyannote)
    if hf_token:
        command.extend(["--hf_token", hf_token])
    else:
        # Log warning if diarization might fail due to missing token
        logger.warning(
            f"{log_prefix} Hugging Face token not provided. Diarization may fail if using models like Pyannote."
        )

    # Add diarization speaker count hints if provided
    if min_speakers is not None:
        command.extend(["--min_speakers", str(min_speakers)])
    if max_speakers is not None:
        command.extend(["--max_speakers", str(max_speakers)])

    # --- Define Output Callback for safe_run ---
    # (Integrates logic from legacy _run_whisperx helper)
    def whisperx_output_callback(line: str):
        """Parses whisperx output for logging key stages."""
        if "Loading model" in line or "Loading faster-whisper model" in line:
            logger.info(f"{log_prefix} Loading model...")
        elif "Detected language:" in line:
            logger.info(f"{log_prefix} {line.strip()}")
        elif re.search(r"Transcribing \d+ segments \|", line):  # Transcription progress
            logger.info(f"{log_prefix} Progress: {line.strip()}")
        elif re.search(r"Aligning \d+ segments \|", line):  # Alignment progress
            logger.info(f"{log_prefix} Progress: {line.strip()}")
        elif "Performing diarization" in line:
            logger.info(f"{log_prefix} Starting diarization...")
        elif "Diarization complete" in line:
            logger.info(f"{log_prefix} Diarization finished.")
        elif "Saving transcriptions to" in line:
            logger.info(f"{log_prefix} {line.strip()}")
        # Catch potential Torch/WhisperX warnings/errors
        elif re.search(r"(warning|error|traceback)", line.lower()) and (
            "torch" in line.lower() or "whisper" in line.lower()
        ):
            logger.warning(f"{log_prefix} WhisperX/Torch Log: {line.strip()}")

    # --- Execute WhisperX Command ---
    masked_command_log = []
    skip_next = False
    for arg in command:
        if skip_next:
            skip_next = False
            continue
        if arg == "--hf_token":
            masked_command_log.append(arg)
            masked_command_log.append("*****")  # Mask token in log
            skip_next = True
        else:
            masked_command_log.append(arg)
    logger.info(f"{log_prefix} Executing command: {' '.join(masked_command_log)}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=whisperx_output_callback,
        )
        logger.info(f"{log_prefix} WhisperX command finished execution attempt.")

    except RuntimeError as e:
        # safe_run already logged the error details
        logger.error(f"{log_prefix} WhisperX command failed execution. See logs above.")
        # Re-raise the error to indicate failure to the caller
        raise
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during WhisperX execution: {e}")
        raise RuntimeError(f"Unexpected error running WhisperX: {e}") from e

    # --- Find WhisperX Output JSON ---
    # (Integrates logic from legacy run_whisperx)
    logger.info(f"{log_prefix} Locating WhisperX output JSON in: {output_dir}")
    # Expected name based on WhisperX default naming convention
    expected_json_path = output_dir / f"{audio_path.stem}.json"

    if expected_json_path.is_file():
        logger.info(
            f"{log_prefix} Found WhisperX output at expected path: {expected_json_path}"
        )
        return expected_json_path
    else:
        logger.warning(
            f"{log_prefix} Expected WhisperX output '{expected_json_path.name}' not found."
        )
        logger.info(
            f"{log_prefix} Searching directory {output_dir} for other potential '.json' files..."
        )

        found_json_files = []
        try:
            for f in output_dir.iterdir():
                # Check if it's a file, has .json suffix, and is NOT in the exclusion list
                if (
                    f.is_file()
                    and f.suffix == ".json"
                    and f.name not in output_filename_exclusions
                ):
                    found_json_files.append(f)
        except Exception as e:
            logger.error(
                f"{log_prefix} Error iterating through output directory {output_dir}: {e}"
            )
            # Fall through to the error below

        if found_json_files:
            # Sort by modification time (most recent first) as a heuristic? Or just take first?
            # Taking the first found seems simplest based on original logic.
            selected_file = found_json_files[0]
            if len(found_json_files) > 1:
                logger.warning(
                    f"{log_prefix} Multiple potential WhisperX JSON files found in {output_dir} (excluding standard names). "
                    f"Using the first one found: {selected_file.name}"
                )
            else:
                logger.info(
                    f"{log_prefix} Found WhisperX output by searching: {selected_file}"
                )
            return selected_file
        else:
            # If no file is found after checking expected path and searching
            error_msg = f"{log_prefix} Could not locate WhisperX JSON output in {output_dir} after command execution."
            logger.error(error_msg)
            # Use FileNotFoundError to be consistent with original intent if expected file is missing
            raise FileNotFoundError(error_msg)
