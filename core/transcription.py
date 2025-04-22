# core/transcription.py
import json
import os
import traceback
import re
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Any, Tuple

# Import yt-dlp to ensure it's checked/available, even if only used via command line
try:
    import yt_dlp  # noqa: F401
except ImportError:
    # Log or print a warning if yt-dlp is not installed
    print("WARN: yt-dlp package not found. YouTube downloads will fail.")


# Logging imports
from .logging import log_info, log_warning, log_error

# Utility imports
from .utils import safe_run  # safe_run handles command execution

# Constants import for finding WhisperX output
from .constants import (
    INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
    FINAL_STRUCTURED_TRANSCRIPT_NAME,
    EMOTION_SUMMARY_CSV_NAME,
    EMOTION_SUMMARY_JSON_NAME,
    SCRIPT_TRANSCRIPT_NAME,
)


# Type hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]


class Transcription:
    """
    Handles audio acquisition (YouTube) and runs external tools
    (yt-dlp, ffmpeg, whisperx) for transcription and diarization.
    """

    config: Dict[str, Any]
    device: str
    hf_token: Optional[str]

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Transcription module with configuration.

        Args:
            config: The application configuration dictionary.
        """
        self.config = config
        self.device = config.get("device", "cpu")  # e.g., 'cpu', 'cuda'
        self.hf_token = config.get("hf_token")  # Hugging Face token for Pyannote models

        log_info("Transcription module initialized.")
        log_info(f"Using device: {self.device}")
        if not self.hf_token:
            log_warning(
                "Hugging Face token (hf_token) not found in config. Diarization might fail."
            )

    # --- Wrapper function for yt-dlp ---
    def _run_yt_dlp(self, command: List[str], log_file_handle: TextIO, session_id: str):
        """Runs yt-dlp command using safe_run with specific output parsing."""
        log_info(f"[{session_id}] Running yt-dlp...")

        def yt_dlp_output_callback(line: str):
            # Regex for download progress: percentage, size, speed, ETA
            progress_match = re.search(
                r"\[download\]\s+([\d.]+%) of\s+~?([\d.]+\s*\w+)\s+at\s+([\d.]+\s*\w+/s)\s+ETA\s+([\d:]+)",
                line,
            )
            if progress_match:
                log_info(
                    f"[{session_id}] yt-dlp: {progress_match.group(1)} at {progress_match.group(3)} ETA {progress_match.group(4)}"
                )
            elif "[download] Destination:" in line:
                log_info(
                    f"[{session_id}] yt-dlp destination: {line.split(':', 1)[1].strip()}"
                )
            elif (
                "[info]" in line and "Downloading" not in line
            ):  # Avoid duplicate download messages
                log_info(
                    f"[{session_id}] yt-dlp info: {line.split(':', 1)[-1].strip()}"
                )
            elif "[ExtractAudio]" in line:
                log_info(f"[{session_id}] yt-dlp: Extracting audio...")
            elif "ERROR:" in line:  # Catch generic errors
                log_error(f"[{session_id}] yt-dlp error: {line.strip()}")

        try:
            safe_run(
                command,
                log_file_handle,
                session_id,
                output_callback=yt_dlp_output_callback,
            )
            log_info(f"[{session_id}] yt-dlp finished successfully.")
        except RuntimeError as e:
            log_error(f"[{session_id}] yt-dlp command failed: {e}")
            raise  # Re-raise the error to be caught by the caller

    # --- Wrapper function for ffmpeg ---
    def _run_ffmpeg(self, command: List[str], log_file_handle: TextIO, session_id: str):
        """Runs ffmpeg command using safe_run with specific output parsing."""
        log_info(f"[{session_id}] Running ffmpeg...")

        def ffmpeg_output_callback(line: str):
            # Regex for progress: time=HH:MM:SS.ms speed=X.Yx
            progress_match = re.search(
                r"time=\s*(\d{2}:\d{2}:\d{2}\.\d+).*?speed=\s*([\d.]+)x", line
            )
            if progress_match:
                log_info(
                    f"[{session_id}] ffmpeg progress: time={progress_match.group(1)}, speed={progress_match.group(2)}x"
                )
            elif "error" in line.lower() or "failed" in line.lower():
                log_error(f"[{session_id}] ffmpeg error: {line.strip()}")
            elif "warning" in line.lower():
                log_warning(f"[{session_id}] ffmpeg warning: {line.strip()}")

        try:
            safe_run(
                command,
                log_file_handle,
                session_id,
                output_callback=ffmpeg_output_callback,
            )
            log_info(f"[{session_id}] ffmpeg finished successfully.")
        except RuntimeError as e:
            log_error(f"[{session_id}] ffmpeg command failed: {e}")
            raise  # Re-raise the error

    # --- Wrapper function for whisperx ---
    def _run_whisperx(
        self, command: List[str], log_file_handle: TextIO, session_id: str
    ):
        """Runs whisperx command using safe_run with specific output parsing."""
        # Build command log, masking the token
        command_log: List[str] = []
        skip_next = False
        for i, arg in enumerate(command):
            if skip_next:
                skip_next = False
                continue
            if arg == "--hf_token":
                command_log.append(arg)
                command_log.append("*****")  # Mask token
                skip_next = True
            else:
                command_log.append(arg)
        log_info(f"[{session_id}] Running WhisperX command: {' '.join(command_log)}")

        def whisperx_output_callback(line: str):
            # Log key stages and progress
            if "Loading model" in line:
                log_info(f"[{session_id}] WhisperX: Loading model...")
            elif "Detected language:" in line:
                log_info(f"[{session_id}] WhisperX: {line.strip()}")
            elif re.search(
                r"Transcribing \d+ segments \|", line
            ):  # VAD/Transcription progress
                log_info(f"[{session_id}] WhisperX progress: {line.strip()}")
            elif re.search(r"Aligning \d+ segments \|", line):  # Alignment progress
                log_info(f"[{session_id}] WhisperX progress: {line.strip()}")
            elif "Performing diarization" in line:
                log_info(f"[{session_id}] WhisperX: Starting diarization...")
            elif "Diarization complete" in line:
                log_info(f"[{session_id}] WhisperX: Diarization finished.")
            # Catch potential warnings/errors specifically from whisperx/torch
            elif (
                re.search(r"(warning|error|traceback)", line.lower())
                and "torch" in line.lower()
            ):
                log_warning(
                    f"[{session_id}] WhisperX/Torch warning/error: {line.strip()}"
                )
            elif "Saving transcriptions to" in line:
                log_info(f"[{session_id}] WhisperX: {line.strip()}")

        try:
            safe_run(
                command,
                log_file_handle,
                session_id,
                output_callback=whisperx_output_callback,
            )
            log_info(f"[{session_id}] WhisperX command finished successfully.")
        except RuntimeError as e:
            log_error(f"[{session_id}] WhisperX command failed: {e}")
            raise  # Re-raise the error

    # --- download_audio_from_youtube ---
    def download_audio_from_youtube(
        self, youtube_url: str, temp_dir: str, log_file_handle: TextIO, session_id: str
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Downloads audio from a YouTube URL, converts it to WAV, and returns the path.

        Args:
            youtube_url: The URL of the YouTube video.
            temp_dir: The directory to store intermediate and final audio files.
            log_file_handle: File handle for logging subprocess output.
            session_id: Identifier for logging.

        Returns:
            Path object to the final WAV audio file.

        Raises:
            RuntimeError: If download or conversion fails.
            FileNotFoundError: If the final WAV file is not found.
        """
        temp_dir_path = Path(temp_dir)
        # Use a unique base name including session_id to avoid collisions
        basename: str = f"audio_input_{session_id}"
        # Define paths for intermediate webm and final wav
        webm_path: Path = temp_dir_path / f"{basename}.webm"
        wav_path: Path = temp_dir_path / f"{basename}.wav"

        # Get config values or use defaults
        yt_dlp_format: str = self.config.get(
            "youtube_dl_format", "bestaudio/best"
        )  # More robust default
        ffmpeg_ac: str = str(
            self.config.get("ffmpeg_audio_channels", 1)
        )  # Audio channels (1 for mono)
        ffmpeg_ar: str = str(
            self.config.get("ffmpeg_audio_samplerate", 16000)
        )  # Sample rate (16kHz standard)

        try:
            # Run yt-dlp to download audio
            self._run_yt_dlp(
                [
                    "yt-dlp",
                    "-f",
                    yt_dlp_format,
                    "-o",
                    str(webm_path),
                    "--",
                    youtube_url,
                ],  # Use '--' for safety with URLs
                log_file_handle,
                session_id,
            )

            if not webm_path.exists():
                raise FileNotFoundError(
                    f"yt-dlp failed to download audio to {webm_path}"
                )

            # Run ffmpeg to convert to WAV
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-i",
                    str(webm_path),  # Input file
                    "-ac",
                    ffmpeg_ac,  # Set audio channels
                    "-ar",
                    ffmpeg_ar,  # Set audio sample rate
                    "-vn",  # No video output
                    "-nostdin",  # Disable interaction
                    str(wav_path),  # Output file path
                ],
                log_file_handle,
                session_id,
            )

            if not wav_path.exists():
                raise FileNotFoundError(f"ffmpeg failed to convert audio to {wav_path}")

            log_info(
                f"[{session_id}] Audio downloaded and converted successfully to: {wav_path}"
            )

            # --- Fetch Metadata ---
            log_info(f"[{session_id}] Fetching metadata for {youtube_url}...")
            metadata_command = [
                "yt-dlp",
                "-j",  # Output JSON
                "--",
                youtube_url,
            ]
            metadata_output = safe_run(
                metadata_command,
                log_file_handle,
                session_id,
                capture_output=True,  # Capture stdout
            )

            metadata: Dict[str, Any] = {}
            if (
                metadata_output
            ):  # safe_run now returns the output string if capture_output is True
                try:
                    video_info = json.loads(
                        metadata_output
                    )  # Load from the returned string
                    # Extract required fields, providing defaults for safety
                    metadata = {
                        "youtube_url": youtube_url,  # Include the URL itself
                        "video_title": video_info.get("fulltitle"),
                        "video_description": video_info.get("description"),
                        "video_uploader": video_info.get("uploader"),
                        "video_creators": video_info.get(
                            "creators"
                        ),  # This might be a list or None
                        "upload_date": video_info.get("upload_date"),  # Format YYYYMMDD
                        "release_date": video_info.get(
                            "release_date"
                        ),  # Format YYYY-MM-DD
                    }
                    log_info(f"[{session_id}] Metadata fetched successfully.")
                except json.JSONDecodeError as e:
                    log_error(
                        f"[{session_id}] Failed to decode yt-dlp metadata JSON: {e}"
                    )
                    metadata = {
                        "youtube_url": youtube_url,
                        "error": "Failed to decode metadata JSON",
                    }
                except Exception as e:
                    log_error(
                        f"[{session_id}] Unexpected error processing metadata: {e}"
                    )
                    metadata = {
                        "youtube_url": youtube_url,
                        "error": f"Unexpected metadata error: {e}",
                    }
            else:
                log_warning(f"[{session_id}] No metadata output from yt-dlp.")
                metadata = {
                    "youtube_url": youtube_url,
                    "warning": "No metadata output from yt-dlp",
                }

            return wav_path, metadata  # Return both audio path and metadata

        except (RuntimeError, FileNotFoundError) as e:
            log_error(
                f"[{session_id}] YouTube download/conversion failed for {youtube_url}: {e}"
            )
            raise  # Re-raise the specific error
        except Exception as e:
            log_error(
                f"[{session_id}] Unexpected error during download/conversion for {youtube_url}: {e}"
            )
            log_error(traceback.format_exc())
            raise RuntimeError(f"Unexpected download/conversion error: {e}") from e
        finally:
            # Clean up intermediate webm file
            if webm_path.exists():
                try:
                    webm_path.unlink()
                    log_info(f"[{session_id}] Removed intermediate file: {webm_path}")
                except OSError as e:
                    # Log cleanup failure but don't raise an error for this
                    warn_msg = (
                        f"WARN: Failed to remove intermediate file {webm_path}: {e}"
                    )
                    log_warning(warn_msg)
                    if log_file_handle and not log_file_handle.closed:
                        try:
                            log_file_handle.write(f"[{session_id}] {warn_msg}\n")
                        except Exception:
                            pass  # Ignore errors writing warning to log

    # --- run_whisperx --- MODIFIED TO RETURN PATH
    def run_whisperx(
        self,
        audio_path: Path,
        output_dir: Path,
        log_file_handle: TextIO,
        session_id: str,
    ) -> Path:
        """
        Runs the WhisperX transcription and diarization pipeline.

        Args:
            audio_path: Path to the input audio file (WAV format recommended).
            output_dir: Directory where WhisperX should save its output files.
            log_file_handle: File handle for logging subprocess output.
            session_id: Identifier for logging.

        Returns:
            Path object to the primary WhisperX JSON output file.

        Raises:
            RuntimeError: If the WhisperX command fails.
            FileNotFoundError: If the WhisperX output JSON cannot be located.
        """
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

        # --- Build WhisperX Command ---
        command: List[str] = [
            "whisperx",
            str(audio_path),  # Input audio file
            "--model",
            self.config.get("whisper_model_size", "large-v3"),  # Model size
            "--diarize",  # Enable diarization
            "--output_dir",
            str(output_dir),  # Output directory
            "--output_format",
            "json",  # Ensure JSON output for parsing
            "--device",
            self.device,  # 'cuda' or 'cpu'
        ]

        # Add Hugging Face token if available (required for pyannote)
        if self.hf_token:
            command.extend(["--hf_token", self.hf_token])
        else:
            # This was checked in __init__, but double-check here.
            # The command might still run without diarization if token is missing,
            # but diarization is explicitly requested, so we should expect issues.
            log_warning(
                f"[{session_id}] Running WhisperX diarization without HF token. This is likely to fail."
            )

        # Optional arguments from config
        lang: Optional[str] = self.config.get("whisper_language")
        if lang and lang.lower() != "auto":
            command.extend(["--language", lang])

        batch_size_val: Optional[Any] = self.config.get("whisper_batch_size")
        if batch_size_val is not None:
            command.extend(["--batch_size", str(batch_size_val)])

        compute_type: Optional[str] = self.config.get("whisper_compute_type")
        if compute_type:
            command.extend(["--compute_type", compute_type])

        # Add diarization specific args if needed (min/max speakers)
        min_speakers = self.config.get("diarization_min_speakers")
        if min_speakers is not None:
            command.extend(["--min_speakers", str(min_speakers)])
        max_speakers = self.config.get("diarization_max_speakers")
        if max_speakers is not None:
            command.extend(["--max_speakers", str(max_speakers)])

        # --- Run WhisperX Command ---
        try:
            self._run_whisperx(command, log_file_handle, session_id)
        except RuntimeError as e:
            # Error is already logged by _run_whisperx and safe_run
            # Re-raise to indicate failure to the caller
            raise

        # --- Find WhisperX Output JSON (Integrated Logic) ---
        log_info(f"[{session_id}] Locating WhisperX output JSON in: {output_dir}")
        expected_json_path = output_dir / f"{audio_path.stem}.json"

        if expected_json_path.exists():
            log_info(
                f"[{session_id}] Found WhisperX output at expected path: {expected_json_path}"
            )
            return expected_json_path
        else:
            log_warning(
                f"[{session_id}] Expected WhisperX output '{expected_json_path.name}' not found. "
                f"Searching directory {output_dir} for other '.json' files..."
            )
            # Define names of files *not* to consider as the primary whisperx output
            standard_output_names = [
                INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
                FINAL_STRUCTURED_TRANSCRIPT_NAME,
                EMOTION_SUMMARY_JSON_NAME,
                EMOTION_SUMMARY_CSV_NAME,  # Although not json, good to list known outputs
                SCRIPT_TRANSCRIPT_NAME,  # Although not json, good to list known outputs
                # Add others if necessary
            ]
            found_json_files = [
                f
                for f in output_dir.iterdir()
                if f.is_file()
                and f.suffix == ".json"
                and f.name not in standard_output_names
            ]

            if found_json_files:
                if len(found_json_files) > 1:
                    log_warning(
                        f"[{session_id}] Multiple potential WhisperX JSON files found in {output_dir}. "
                        f"Using the first one found: {found_json_files[0].name}"
                    )
                log_info(
                    f"[{session_id}] Found WhisperX output by searching: {found_json_files[0]}"
                )
                return found_json_files[0]  # Return the first match found
            else:
                # If no file is found after checking expected and searching
                error_msg = f"[{session_id}] Could not locate WhisperX JSON output in {output_dir} after command execution."
                log_error(error_msg)
                raise FileNotFoundError(error_msg)

    # --- convert_json_to_structured ---
    def convert_json_to_structured(self, json_path: Path) -> SegmentsList:
        """
        Reads the WhisperX JSON output and converts it into a structured list of segments.

        Args:
            json_path: Path to the WhisperX output JSON file.

        Returns:
            A list of dictionaries, where each dictionary represents a segment
            containing 'start', 'end', 'text', 'speaker', and 'words'.

        Raises:
            FileNotFoundError: If the json_path does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            TypeError: If the 'segments' key in the JSON is not a list.
        """
        log_info(f"Reading and structuring WhisperX JSON output from: {json_path}")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"WhisperX output JSON file not found at {json_path}"
            )

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)
        except json.JSONDecodeError as e:
            log_error(f"Failed to decode JSON from {json_path}: {e}")
            raise  # Re-raise the error
        except Exception as e:
            log_error(f"Error reading JSON file {json_path}: {e}")
            raise RuntimeError(f"Could not read JSON file {json_path}") from e

        structured: SegmentsList = []
        # Get segments, default to empty list if key is missing
        segments_raw: Any = data.get("segments", [])

        # Validate that segments is actually a list
        if not isinstance(segments_raw, list):
            log_error(
                f"'segments' key in {json_path} is not a list (type: {type(segments_raw)}). Cannot process."
            )
            # Raise TypeError or return empty list depending on desired strictness
            raise TypeError(f"'segments' key in {json_path} is not a list.")
            # Or: return []

        log_info(f"Structuring {len(segments_raw)} segments from WhisperX output...")

        # Process each segment dictionary
        for i, segment in enumerate(segments_raw):
            # Ensure segment is a dictionary before accessing keys
            if not isinstance(segment, dict):
                log_warning(f"Item #{i} in 'segments' is not a dictionary. Skipping.")
                continue

            text: str = segment.get("text", "").strip()
            start_time: Optional[float] = segment.get("start")
            end_time: Optional[float] = segment.get("end")
            # Ensure speaker ID is treated as a string, default to 'unknown'
            speaker: str = str(segment.get("speaker", "unknown"))
            # Get words, default to empty list if missing or not a list
            words_raw: Any = segment.get("words", [])
            words: List[Dict[str, Any]] = (
                words_raw if isinstance(words_raw, list) else []
            )

            # Basic validation for timing (optional but good practice)
            if start_time is None or end_time is None or start_time > end_time:
                log_warning(
                    f"Segment {i} has invalid/missing time: start={start_time}, end={end_time}. Using as is."
                )

            segment_output: Segment = {
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,  # Already ensured it's a string
                "words": words,  # Already ensured it's a list
            }
            structured.append(segment_output)

        log_info(
            f"Finished structuring segments. Returning {len(structured)} segments."
        )
        return structured
