# core/transcription.py
import json
import os
import traceback
import re
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Any

import yt_dlp  # noqa: F401

from .logging import log_info, log_warning, log_error
# Ensure safe_run accepts output_callback
from .utils import safe_run


Segment = Dict[str, Any]
SegmentsList = List[Segment]

class Transcription:
    config: Dict[str, Any]
    device: str
    hf_token: Optional[str]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
        self.hf_token = config.get("hf_token")

        log_info("Transcription module initialized.")

    # --- Wrapper function for yt-dlp ---
    def _run_yt_dlp(self, command: List[str], log_file_handle: TextIO, session_id: str):
        log_info(f"[{session_id}] Running yt-dlp...")

        def yt_dlp_output_callback(line: str):
            # Refined regex for yt-dlp download progress: looks for percentage, size, speed, ETA
            progress_match = re.search(
                r"\[download\]\s+([\d.]+%) of\s+~?([\d.]+\s*\w+)\s+at\s+([\d.]+\s*\w+/s)\s+ETA\s+([\d:]+)",
                line
            )
            if progress_match:
                # Log progress at INFO level
                log_info(f"[{session_id}] yt-dlp download progress: {progress_match.group(1).strip()} at {progress_match.group(3).strip()} (ETA: {progress_match.group(4).strip()})")
            elif "[download] Destination:" in line:
                 log_info(f"[{session_id}] yt-dlp destination: {line.split(':', 1)[1].strip()}")
            elif "[download] 100%" in line:
                 log_info(f"[{session_id}] yt-dlp download complete.")
            elif "[info]" in line:
                 # Log general info messages from yt-dlp
                 log_info(f"[{session_id}] yt-dlp info: {line.split(':', 1)[1].strip()}")
            elif "[ExtractAudio]" in line:
                 log_info(f"[{session_id}] yt-dlp: Extracting audio...")
            elif "[error]" in line.lower():
                 # Log explicit errors from yt-dlp as errors
                 log_error(f"[{session_id}] yt-dlp error: {line.strip()}")
            # Note: Many yt-dlp informational messages are useful; adjust logging level or patterns as needed.


        safe_run(
            command,
            log_file_handle,
            session_id,
            output_callback=yt_dlp_output_callback # Pass the callback here
        )
        log_info(f"[{session_id}] yt-dlp finished.")


    # --- Wrapper function for ffmpeg ---
    def _run_ffmpeg(self, command: List[str], log_file_handle: TextIO, session_id: str):
        log_info(f"[{session_id}] Running ffmpeg...")

        def ffmpeg_output_callback(line: str):
             # Refined regex for ffmpeg progress: looks for time and speed
             # This pattern is common on stderr during encoding/conversion
             progress_match = re.search(
                 r"frame=\s*\d+\s+.*?time=\s*(\d{2}:\d{2}:\d{2}\.\d+).*?speed=\s*([\d.]+)x",
                 line
             )
             if progress_match:
                 # Log progress at INFO level
                 log_info(f"[{session_id}] ffmpeg progress: time={progress_match.group(1)}, speed={progress_match.group(2)}x")
             elif line.strip().startswith("Output #0"):
                  log_info(f"[{session_id}] ffmpeg output configuration detected.")
             elif line.strip().startswith("Stream mapping:"):
                  log_info(f"[{session_id}] ffmpeg stream mapping: {line.split(':', 1)[1].strip()}")
             elif "deprecated" in line.lower() or re.search(r":?\s*warning", line.lower()): # More robust warning check
                  # Log warnings from ffmpeg
                  log_warning(f"[{session_id}] ffmpeg warning: {line.strip()}")
             elif "error" in line.lower() or "failed" in line.lower() or "fehler" in line.lower():
                  # Log errors from ffmpeg
                  log_error(f"[{session_id}] ffmpeg error: {line.strip()}")
             # else: logging.debug(f"[{session_id}] ffmpeg raw: {line.strip()}") # Option to log unhandled lines


        safe_run(
            command,
            log_file_handle,
            session_id,
            output_callback=ffmpeg_output_callback # Pass the callback here
        )
        log_info(f"[{session_id}] ffmpeg finished.")


    # --- Wrapper function for whisperx ---
    def _run_whisperx(self, command: List[str], log_file_handle: TextIO, session_id: str):
        # Build command log, masking the token (kept here as it's specific to whisperx command)
        command_log: List[str] = []
        skip_next = False
        for i, arg in enumerate(command):
            if skip_next:
                skip_next = False
                continue
            if arg == "--hf_token":
                command_log.append(arg)
                command_log.append("*****")
                skip_next = True
            else:
                command_log.append(arg)
        log_info(f"[{session_id}] Running WhisperX command: {' '.join(command_log)}")

        def whisperx_output_callback(line: str):
             # Refined regex for whisperx output parsing
             if "Loading model" in line:
                 log_info(f"[{session_id}] WhisperX: Loading model...")
             elif "Detected language:" in line:
                 log_info(f"[{session_id}] WhisperX: {line.strip()}")
             elif ">>Performing transcription..." in line:
                 log_info(f"[{session_id}] WhisperX: Starting transcription...")
             elif ">>Performing alignment..." in line:
                 log_info(f"[{session_id}] WhisperX: Starting alignment...")
             elif ">>Performing diarization..." in line:
                 log_info(f"[{session_id}] WhisperX: Starting diarization...")
             # Check for progress percentages like [ 50%] or 50%
             elif re.search(r"\[?\s*\d+%\]?", line):
                 log_info(f"[{session_id}] WhisperX progress: {line.strip()}")
             elif "finished ASR inference" in line:
                 log_info(f"[{session_id}] WhisperX: ASR inference finished.")
             elif "finished alignment" in line:
                 log_info(f"[{session_id}] WhisperX: Alignment finished.")
             elif "finished diarization" in line:
                 log_info(f"[{session_id}] WhisperX: Diarization finished.")
             elif "CPU threads:" in line or "CUDA extensions are installed" in line or "device is" in line:
                 log_info(f"[{session_id}] WhisperX setup: {line.strip()}")
             # Catch general warnings/errors that don't contain 'Transcript:'
             # Avoid logging the Transcript lines themselves as warnings/errors
             elif ("error" in line.lower() or "fail" in line.lower() or re.search(r":?\s*warning", line.lower())) and "Transcript:" not in line:
                 log_warning(f"[{session_id}] WhisperX warning/error: {line.strip()}")
             # else: logging.debug(f"[{session_id}] WhisperX raw: {line.strip()}") # Option to log unhandled lines


        safe_run(
            command,
            log_file_handle,
            session_id,
            output_callback=whisperx_output_callback # Pass the callback here
        )
        log_info(f"[{session_id}] WhisperX finished.")


    # --- download_audio_from_youtube now uses wrappers ---
    def download_audio_from_youtube(
        self, youtube_url: str, temp_dir: str, log_file_handle: TextIO, session_id: str
    ) -> str:
        temp_dir_path = Path(temp_dir)
        basename: str = "audio_input"
        webm_path: Path = temp_dir_path / f"{basename}_{session_id}.webm"
        wav_path: Path = temp_dir_path / f"{basename}_{session_id}.wav"
        webm_path_str: str = str(webm_path)
        wav_path_str: str = str(wav_path)

        yt_dlp_format: str = self.config.get("youtube_dl_format", "251")
        ffmpeg_ac: str = str(self.config.get("ffmpeg_audio_channels", 1))
        ffmpeg_ar: str = str(self.config.get("ffmpeg_audio_samplerate", 16000))

        try:
            self._run_yt_dlp(
                ["yt-dlp", "-f", yt_dlp_format, "-o", webm_path_str, youtube_url],
                log_file_handle,
                session_id,
            )

            self._run_ffmpeg(
                [
                    "ffmpeg", "-y", "-i", webm_path_str,
                    "-ac", ffmpeg_ac, "-ar", ffmpeg_ar,
                    "-vn", "-nostdin",
                    wav_path_str,
                ],
                log_file_handle,
                session_id,
            )
            return wav_path_str
        except Exception as e:
            log_error(f"YouTube download/conversion failed for {youtube_url}: {e}")
            raise
        finally:
            if webm_path.exists():
                try:
                    webm_path.unlink()
                    log_info(f"Removed intermediate file: {webm_path}")
                except OSError as e:
                    warn_msg = f"WARN: Failed to remove intermediate file {webm_path}: {e}"
                    try:
                        if log_file_handle and not log_file_handle.closed:
                            log_file_handle.write(f"[{session_id}] {warn_msg}\n")
                        else:
                            print(warn_msg)
                    except:
                        print(warn_msg)


    # --- run_whisperx now uses its wrapper ---
    def run_whisperx(
        self, audio_path: str, output_dir: str, log_file_handle: TextIO, session_id: str
    ) -> None:
        command: List[str] = [
            "whisperx", audio_path,
            "--model", self.config.get("whisper_model_size", "large-v2"),
            "--diarize",
            "--hf_token", self.hf_token or "",
            "--output_dir", output_dir,
            "--output_format", self.config.get("whisper_output_format", "json"),
            "--device", self.device,
        ]
        lang: Optional[str] = self.config.get("whisper_language")
        if lang: command.extend(["--language", lang])
        batch_size_val: Optional[Any] = self.config.get("whisper_batch_size")
        if batch_size_val is not None: command.extend(["--batch_size", str(batch_size_val)])
        compute_type: Optional[str] = self.config.get("whisper_compute_type")
        if compute_type: command.extend(["--compute_type", compute_type])

        self._run_whisperx(command, log_file_handle, session_id)


    def convert_json_to_structured(self, json_path: str) -> SegmentsList:
        log_info(f"Reading WhisperX JSON output from: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)
        except FileNotFoundError as e:
            log_error(f"WhisperX output JSON file not found at {json_path}")
            raise e
        except json.JSONDecodeError as e:
            log_error(f"Failed to decode JSON from {json_path}")
            raise e

        structured: SegmentsList = []
        segments: List[Dict] = data.get("segments", [])
        if not isinstance(segments, list):
            log_warning(f"'segments' key in {json_path} is not a list. Processing as empty.")
            segments = []

        log_info(f"Structuring {len(segments)} segments from WhisperX output...")

        for i, segment in enumerate(segments):
            text: str = segment.get("text", "").strip()
            start_time: Optional[float] = segment.get("start")
            end_time: Optional[float] = segment.get("end")
            speaker: str = segment.get("speaker", "unknown")
            words: List[Dict[str, Any]] = segment.get("words", [])

            segment_output: Segment = {
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker,
                "words": words,
            }
            structured.append(segment_output)

        log_info(f"Finished structuring segments. Returning {len(structured)} segments.")
        return structured
