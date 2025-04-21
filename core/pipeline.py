# core/pipeline.py

import csv
import json
import shutil
import subprocess
import traceback
import zipfile
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, TextIO, Tuple


from .constants import (
    EMO_VAL,
    LOG_FILE_NAME,
    INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
    FINAL_STRUCTURED_TRANSCRIPT_NAME,
    EMOTION_SUMMARY_CSV_NAME,
    EMOTION_SUMMARY_JSON_NAME,
    SCRIPT_TRANSCRIPT_NAME,
    FINAL_ZIP_SUFFIX,
    DEFAULT_SNIPPET_MATCH_THRESHOLD,
)
from .logging import log_error, log_info, log_warning
from .plotting import generate_all_plots
from .transcription import Transcription  # Import Transcription
from .multimodal_analysis import MultimodalAnalysis

# Ensure parse_xlsx_snippets is imported from utils
from .utils import create_directory, get_temp_file, safe_run, parse_xlsx_snippets
from .snippet_matcher import group_segments_by_speaker  # Added


Segment = Dict[str, Any]
SegmentsList = List[Segment]
SpeakerPreview = Dict[str, str]
SpeakerPreviewsList = List[SpeakerPreview]
EmotionSummary = Dict[str, Dict[str, Any]]


# Helper function to convert numpy.float32 objects to standard Python floats
def convert_floats(obj):
    """
    Recursively convert numpy.float32 objects to standard Python floats
    in a dictionary or list.
    """
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(elem) for elem in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


class Pipeline:
    """
    Orchestrates the end-to-end speech processing workflow:
      1) Prepare audio (download or copy)
      2) WhisperX transcription + diarization
      3) Multimodal emotion & deception analysis
      4) Multimodal emotion & deception analysis
      5) Generate speaker previews (for UI labeling - Single Process only)
      6) Save intermediate structured JSON (with original IDs - Single Process only)
      7) Relabel & finalize (apply user labels, save final JSON, summaries, plots, script, ZIP - Single Process)
      8) Process batch XLSX (handle multiple URLs, include metadata, finalize each item - Batch Process)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transcription = Transcription(config)  # Initialize Transcription
        log_info("Initializing MultimodalAnalysis in Pipeline...")
        self.mm = MultimodalAnalysis(config)  # Pass config to MultimodalAnalysis
        # Optional fallback diarizer - Keep existing Pyannote logic if needed
        try:
            from pyannote.audio import Pipeline as PyannotePipeline

            pyannote_model_source = config.get(
                "pyannote_diarization_model", "pyannote/speaker-diarization"
            )
            hf_token = config.get("hf_token")
            if not hf_token:
                log_warning(
                    "Hugging Face token is not set. Pyannote diarization may fail."
                )

            # Diarizer loading in Pipeline init is only for potential fallback,
            # WhisperX handles diarization primarily. Keep this but note its role.
            self.diarizer = PyannotePipeline.from_pretrained(
                pyannote_model_source, use_auth_token=hf_token
            )
            log_info(
                f"Pyannote diarization model '{pyannote_model_source}' loaded successfully."
            )
        except Exception as e:
            log_warning(f"Failed to load Pyannote diarization model: {e}")
            self.diarizer = None

    def _prepare_audio_input(
        self,
        input_source: str,
        temp_dir: Path,
        log_file: TextIO,
        session_id: str,  # This will be the item_identifier in batch mode
    ) -> Path:
        """Download from URL or copy local file; return path to input audio."""
        log_info(f"[{session_id}] Preparing audio input from: {input_source}")
        if input_source.startswith(("http://", "https://")):
            # Pass session_id to download_audio_from_youtube for filename uniqueness
            dl = self.transcription.download_audio_from_youtube(
                input_source, str(temp_dir), log_file, session_id
            )
            if not dl:
                raise RuntimeError(f"Audio download failed for URL: {input_source}")
            audio_path = Path(dl)
        else:
            src = Path(input_source)
            if not src.exists():
                raise ValueError(f"Invalid input source: {input_source}")
            # Use session_id/item_identifier in the filename for uniqueness within the batch temp dir
            unique_filename = f"{session_id}_{src.name}"
            dest = temp_dir / unique_filename
            shutil.copy(src, dest)
            audio_path = dest

        if not audio_path.exists():
            raise FileNotFoundError(f"Could not obtain audio from: {input_source}")
        log_info(f"[{session_id}] Audio prepared at: {audio_path}")
        return audio_path

    def _run_ffprobe_duration_check(self, audio_path: Path) -> bool:
        """Check audio length via ffprobe; skip if not available."""
        try:
            res = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            duration = float(res.stdout.strip())
            min_dur = float(self.config.get("min_diarization_duration", 5.0))
            if duration < min_dur:
                log_warning(
                    f"Audio too short ({duration:.1f}s < {min_dur}s) for optimal diarization."
                )
                return False
            return True  # Return True if check passes (audio is long enough)
        except FileNotFoundError:
            log_warning("ffprobe not installed; skipping duration check.")
            return False
        except Exception as e:
            log_error(f"ffprobe error: {e}")
            log_warning(
                "Duration check failed, proceeding but diarization quality may be affected."
            )
            return False

    # Remove _find_whisperx_output as batch processing reads from a combined output dir

    def _save_script_transcript(
        self, segments: SegmentsList, output_dir: Path, suffix: str
    ) -> Optional[Path]:
        """Saves a plain text transcript formatted like a script."""
        script_path = output_dir / f"{Path(SCRIPT_TRANSCRIPT_NAME).stem}_{suffix}.txt"
        log_info(f"Saving script transcript to: {script_path}")

        grouped_blocks = group_segments_by_speaker(segments)

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                for block in grouped_blocks:
                    speaker = block.get("speaker", "UNKNOWN")
                    text = block.get("text", "").strip()
                    start_time = block.get("start")
                    end_time = block.get("end")

                    time_str = ""
                    if start_time is not None and end_time is not None:
                        # Format time to HH:MM:SS.ms
                        start_h = int(start_time // 3600)
                        start_m = int((start_time % 3600) // 60)
                        start_s = start_time % 60
                        end_h = int(end_time // 3600)
                        end_m = int((end_time % 3600) // 60)
                        end_s = end_time % 60
                        time_str = f"[{start_h:02}:{start_m:02}:{start_s:06.3f} - {end_h:02}:{end_m:02}:{end_s:06.3f}] "

                    f.write(f"{time_str}{speaker}: {text}\n\n")

            log_info(f"Script transcript saved successfully to: {script_path}")
            return script_path

        except Exception as e:
            log_error(f"Failed to save script transcript to {script_path}: {e}")
            return None

    def _calculate_emotion_summary(
        self, segments: SegmentsList, include_timeline: bool = False
    ) -> EmotionSummary:
        speaker_stats = defaultdict(list)
        timeline = defaultdict(list)

        # No longer filtering to summary_emotions here, include all analyzed
        # summary_emotions = ['anger', 'joy', 'sadness', 'neutral']

        for seg in segments:
            # Use the 'speaker' ID assigned by WhisperX (potentially batched)
            spk = str(seg.get("speaker", "unknown"))
            # Use the final 'emotion' key which comes from fused results or fallbacks
            emo = seg.get("emotion", "unknown")

            speaker_stats[spk].append(emo)

            if include_timeline:
                timeline_entry: Dict[str, Any] = {
                    "time": seg.get("start", 0.0),
                    "emotion": emo,
                }
                if "fused_emotion_confidence" in seg:
                    timeline_entry["fused_emotion_confidence"] = seg[
                        "fused_emotion_confidence"
                    ]
                if "significant_text_emotions" in seg:
                    timeline_entry["significant_text_emotions"] = seg[
                        "significant_text_emotions"
                    ]
                # Include source_id in timeline entry for batch processing
                if "source_id" in seg:
                    timeline_entry["source_id"] = seg["source_id"]

                timeline[spk].append(timeline_entry)

        summary = {}
        for spk, emos in speaker_stats.items():
            # Calculate transitions including all observed emotion labels
            transitions = sum(1 for i in range(1, len(emos)) if emos[i] != emos[i - 1])

            emotion_counts = Counter(emos)
            # Filterable emotions for dominant calculation: exclude unknown, skipped, failed, no_text
            filterable_emotions = [
                e
                for e in emos
                if e
                not in ["unknown", "analysis_skipped", "analysis_failed", "no_text"]
            ]

            # Determine dominant emotion from filterable ones, fallback to "neutral" or "unknown"
            dominant = (
                Counter(filterable_emotions).most_common(1)[0][0]
                if filterable_emotions
                else "unknown"
            )  # Fallback to unknown

            # Calculate volatility and mean score using numeric EMO_VAL
            # Ensure there are values before calculating statistics
            vals = [EMO_VAL.get(e, 0.0) for e in emos]
            vol = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mean_score = statistics.mean(vals) if vals else 0.0

            entry = {
                "total_segments": len(emos),
                "emotion_transitions": transitions,
                "dominant_emotion": dominant,
                "emotion_volatility": round(vol, 3),
                "emotion_score_mean": round(mean_score, 3),
                "emotion_counts": dict(emotion_counts),
            }
            if include_timeline:
                # Timeline entries are already added with source_id if available
                entry["emotion_timeline"] = sorted(
                    timeline[spk], key=lambda x: x.get("time", 0.0)
                )

            summary[spk] = entry

        return summary

    def _save_emotion_summary(
        self,
        stats: EmotionSummary,
        out_dir: Path,
        suffix: str,  # Use suffix to include item identifier
    ) -> Tuple[Optional[Path], Optional[Path]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"{Path(EMOTION_SUMMARY_CSV_NAME).stem}_{suffix}.csv"
        json_filename = f"{Path(EMOTION_SUMMARY_JSON_NAME).stem}_{suffix}.json"
        csv_path = out_dir / csv_filename
        json_path = out_dir / json_filename

        stats_serializable = convert_floats(stats)
        try:
            json_path.write_text(
                json.dumps(stats_serializable, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log_info(f"Emotion summary JSON saved to: {json_path}")
        except Exception as e:
            log_error(f"Failed to save emotion summary JSON to {json_path}: {e}")
            json_path = None  # Indicate failure

        all_emotion_types_in_counts = sorted(
            list(
                set(
                    e
                    for count_dict in [
                        data.get("emotion_counts", {}) for data in stats.values()
                    ]
                    for e in count_dict.keys()
                )
            )
        )

        standard_headers = [
            "speaker",
            "total_segments",
            "emotion_transitions",
            "dominant_emotion",
            "emotion_volatility",
            "emotion_score_mean",
        ]
        final_headers = standard_headers + [
            h for h in all_emotion_types_in_counts if h not in standard_headers
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=final_headers)
                writer.writeheader()
                for spk, data in stats.items():
                    row = {"speaker": spk}
                    row.update(
                        {k: data.get(k, "") for k in standard_headers if k != "speaker"}
                    )

                    emotion_counts_data = data.get("emotion_counts", {})
                    for emo_header in all_emotion_types_in_counts:
                        row[emo_header] = emotion_counts_data.get(emo_header, 0)

                    writer.writerow(row)
            log_info(f"Emotion summary CSV saved to: {csv_path}")
        except Exception as e:
            log_error(f"Failed to save emotion summary CSV to {csv_path}: {e}")
            csv_path = None  # Indicate failure

        return csv_path, json_path

    def _create_final_zip(
        self,
        zip_path: Path,
        files_to_add: Dict[Path, str],
        plot_files: List[Path],
        log_file: Path,  # This should be the main batch log file
    ) -> Optional[Path]:
        log_info(f"Creating ZIP: {zip_path}")
        final_output_dir = zip_path.parent
        final_output_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"Ensured final output directory exists: {final_output_dir}")

        temp_zip_path = zip_path.with_suffix(".temp.zip")

        try:
            with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                # Add the main batch log file at the root of the zip
                if log_file.exists():
                    zf.write(log_file, arcname=log_file.name)
                    log_info(f"Added batch log file to master zip: {log_file.name}.")

                for fp, arc_name in files_to_add.items():
                    if fp and fp.exists():  # Check if path is not None and exists
                        zf.write(fp, arcname=arc_name)
                        log_info(f"Added {fp.name} to zip as {arc_name}.")
                    else:
                        log_warning(f"Missing file to add to zip: {fp}. Skipping.")

                for p in plot_files:
                    if p and p.exists():  # Check if path is not None and exists
                        zf.write(p, arcname=f"plots/{p.name}")
                        log_info(f"Added plot {p.name} to zip as plots/{p.name}.")
                    else:
                        log_warning(f"Missing plot to add to zip: {p}. Skipping.")

            shutil.move(str(temp_zip_path), str(zip_path))
            log_info(f"Successfully created final ZIP: {zip_path}")
            return zip_path  # Return the created zip path

        except Exception as e:
            log_error(
                f"Failed to create ZIP file {zip_path}: {e}\n{traceback.format_exc()}"
            )
            if temp_zip_path.exists():
                try:
                    temp_zip_path.unlink()
                    log_info(f"Cleaned up temporary zip file: {temp_zip_path}")
                except Exception as unlink_e:
                    log_warning(
                        f"Failed to clean up temporary zip file {temp_zip_path}: {unlink_e}"
                    )
            return None  # Indicate failure

    # MODIFIED: _finalize_batch_item to save files locally within item_work_path, no zipping or cleanup
    def _finalize_batch_item(
        self,
        segments: SegmentsList,
        metadata: Optional[Any],
        # audio_path_in_work_dir: Optional[Path], # Not strictly needed here anymore as we have segments with source_id
        item_work_path: Path,  # The work path for THIS batch item
        log_file_handle: TextIO,  # Pass the main batch log handle for logging
        item_identifier: str,  # e.g., "youtube_video_id" or "row_number"
        original_source: str,  # Pass the original source for this item
    ) -> str:  # Return only a status message string for this item
        """
        Finalizes a single batch item: save final JSON, generate summaries,
        plots, script transcript. Files are saved within the item_work_path/output
        directory. Does NOT create a ZIP or clean up. Includes user metadata in JSON.
        Returns a status message for this item.
        """

        # Use the provided log file handle for logging - Keeping the local item_log function for structured messages
        def item_log(level, message):
            full_message = f"[{item_identifier}] {message}"
            if log_file_handle:
                try:
                    log_file_handle.write(
                        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                    )
                    log_file_handle.flush()
                except Exception as log_e:
                    log_error(
                        f"[{item_identifier}] Failed to write to batch log file: {log_e} - Message: {message}"
                    )
                    print(
                        f"WARN: Failed to write to batch log file: {log_e} - Message: {message}"
                    )  # Fallback print
            # Also log to main application logger
            if level == "info":
                log_info(full_message)
            elif level == "warning":
                log_warning(full_message)
            elif level == "error":
                log_error(full_message)

        item_log(
            "info", f"Starting finalization for {item_identifier} in: {item_work_path}"
        )
        status_msg = f"[{item_identifier}] Starting finalization..."

        try:
            if not segments:
                status_msg = (
                    f"[{item_identifier}] ERROR: No segments found for finalization."
                )
                item_log("error", status_msg)
                return status_msg

            # Save the FINAL structured JSON to the output directory within item_work_path
            item_log("info", "Saving final structured transcript JSON...")
            output_temp_dir = (
                item_work_path / "output"
            )  # Save within the item's output subdir
            output_temp_dir.mkdir(exist_ok=True)  # Ensure output subdir exists
            # Use a consistent filename for the item's final JSON within its folder
            final_json_name = (
                f"{item_identifier}_structured_transcript.json"  # Consistent name
            )
            final_json_path = output_temp_dir / final_json_name

            segments_final_serializable = {"segments": convert_floats(segments)}
            if metadata is not None:
                segments_final_serializable["user_metadata"] = str(
                    metadata
                )  # Ensure metadata is serializable

            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(segments_final_serializable, f, indent=2, ensure_ascii=False)
            item_log(
                "info",
                f"Final structured transcript saved to: {final_json_path}",
            )

            # Save the script-formatted plain text transcript
            item_log("info", "Saving script-formatted transcript...")
            # Use the item_identifier in the script filename suffix
            script_transcript_path = self._save_script_transcript(
                segments, output_temp_dir, item_identifier
            )
            if script_transcript_path:
                item_log(
                    "info", f"Script transcript saved to: {script_transcript_path}"
                )

            # Calculate and save emotion summary (CSV and JSON)
            item_log("info", "Calculating and saving emotion summary...")
            stats = self._calculate_emotion_summary(segments, include_timeline=True)
            # Use the item_identifier in the summary filenames suffix
            csv_path, json_summary_path = self._save_emotion_summary(
                stats, output_temp_dir, item_identifier
            )
            if csv_path and json_summary_path:
                item_log(
                    "info", f"Emotion summary saved to: {csv_path}, {json_summary_path}"
                )
            else:
                item_log(
                    "warning", "Failed to save all emotion summary files for this item."
                )

            # Generate plots
            item_log("info", "Generating plots...")
            plots: List[str] = []
            try:
                # Use the item_identifier in the plot filenames suffix, save to item's output subdir
                plots = generate_all_plots(stats, str(output_temp_dir), item_identifier)
                item_log("info", f"Generated {len(plots)} plot file(s) for this item.")
            except Exception as p:
                item_log(
                    "warning", f"Plot generation failed for {item_identifier}: {p}"
                )
                plots = []

            # Note: audio_path_in_work_dir (the WAV file) is already in the item_work_path.
            # All other outputs are now saved in item_work_path/output.
            # No zip creation or cleanup is done here.
            status_msg = f"[{item_identifier}] ✅ Processing and finalization complete."
            item_log("info", status_msg)
            return status_msg  # Return success status

        except Exception as e:
            err_msg = f"[{item_identifier}] ERROR during finalization: {e}"
            item_log("error", err_msg + "\n" + traceback.format_exc())
            status_msg = err_msg  # Return error status
            return status_msg

    # MODIFIED: process_batch_xlsx to handle batch audio processing with WhisperX
    def process_batch_xlsx(self, xlsx_filepath: str) -> Tuple[str, str]:
        """
        Processes an XLSX file containing YouTube URLs and optional metadata in batch.
        Downloads/copies all audio, runs WhisperX once on the batch, and generates
        per-item results including metadata. Creates a single master ZIP.
        Returns (status_message, results_summary_string).
        """
        batch_job_id = datetime.utcnow().strftime("batch-%Y%m%dT%H%M%S")
        log_info(f"[{batch_job_id}] Starting batch processing for {xlsx_filepath}")
        batch_status_message = f"[{batch_job_id}] Starting batch processing..."
        batch_results_summary = ""
        base_temp_dir = Path(self.config.get("temp_dir", "./temp"))
        batch_work_path = (
            base_temp_dir / batch_job_id
        )  # Main working directory for the batch
        batch_log_path = (
            batch_work_path / LOG_FILE_NAME
        )  # Main log file for the batch process
        log_file_handle = None
        processed_item_identifiers: List[
            str
        ] = []  # To track which items were processed for cleanup
        audio_paths_for_batch: List[
            Path
        ] = []  # List to hold paths of downloaded/copied audio files for the batch run
        item_metadata_map: Dict[str, Any] = {}  # Map item_identifier to metadata
        item_source_map: Dict[
            str, str
        ] = {}  # Map item_identifier to original source (URL/path)
        source_audio_map: Dict[
            str, Path
        ] = {}  # Map source_id (original URL/path) to local audio file Path
        source_video_map: Dict[
            str, Path
        ] = {}  # Map source_id (original URL/path) to local video file Path (Placeholder - video download not implemented)

        # ADDED: Map from local temporary audio Path to original source identifier (string)
        local_audio_path_to_original_source_map: Dict[Path, str] = {}

        # Define a local batch logging function to use the specific batch log file handle
        def batch_log(level, message):
            full_message = f"[{batch_job_id}] {message}"
            # Use the main batch log file handle if available, otherwise fallback
            if log_file_handle:
                try:
                    log_file_handle.write(
                        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                    )
                    log_file_handle.flush()
                except Exception as log_e:
                    # Fallback print if writing to file handle fails
                    print(
                        f"WARN: [{batch_job_id}] Failed to write to batch log file: {log_e} - Message: {message}"
                    )
            # Also log to main application logger regardless
            if level == "info":
                log_info(full_message)
            elif level == "warning":
                log_warning(full_message)
            elif level == "error":
                log_error(full_message)

        try:
            batch_work_path.mkdir(parents=True, exist_ok=True)
            # Open the main batch log file immediately
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            batch_log("info", f"Batch log file created at: {batch_log_path}")

            batch_log("info", batch_status_message)

            # Read the XLSX file
            batch_log("info", f"Reading batch file: {xlsx_filepath}")
            try:
                df = pd.read_excel(xlsx_filepath)
                batch_log(
                    "info", f"Successfully read {len(df)} rows from {xlsx_filepath}"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Batch input file not found at {xlsx_filepath}"
                )
            except pd.errors.EmptyDataError:
                raise pd.errors.EmptyDataError(
                    f"The uploaded Excel file {xlsx_filepath} is empty."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read Excel file {xlsx_filepath}: {e}")

            results: List[str] = []
            url_col = self.config.get("batch_url_column", "YouTube URL")
            video_col = self.config.get(
                "batch_video_column", "Video URL"
            )  # Added video column config

            if url_col not in df.columns:
                raise ValueError(
                    f"Required column '{url_col}' not found in the XLSX file."
                )

            # Identify metadata column (the second column if it exists and is not the URL or Video column)
            metadata_col = None
            other_cols = [col for col in df.columns if col not in [url_col, video_col]]
            if other_cols:
                metadata_col = other_cols[0]  # Take the first other column as metadata
                batch_log("info", f"Identified metadata column: '{metadata_col}'")
            else:
                batch_log("info", "No additional columns found for metadata.")

            batch_log(
                "info",
                f"Using URL column: '{url_col}'",
            )
            if video_col in df.columns:
                batch_log("info", f"Using Video URL column: '{video_col}'")
            else:
                batch_log(
                    "info",
                    f"Video URL column '{video_col}' not found. Visual analysis for batch items will be skipped.",
                )

            # --- Prepare all audio and video inputs for the batch ---
            batch_log("info", "Preparing audio inputs for the entire batch...")
            audio_prep_successful = True
            # i = 0 # Initialize a counter for item identifiers - using index from iterrows instead
            valid_items_for_processing: List[
                str
            ] = []  # Track items with successfully prepared audio

            for index, row in df.iterrows():
                # Use a consistent item identifier based on row number (1-based index)
                item_identifier = f"row_{index + 1}"

                youtube_url_raw = row.get(
                    url_col
                )  # Use .get to handle potential missing column gracefully if needed, though validation above checks presence
                if not isinstance(youtube_url_raw, str) or not youtube_url_raw.strip():
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Skipping row {index + 1} due to missing or invalid YouTube URL.",
                    )
                    results.append(
                        f"[{item_identifier}] Skipped: Missing/Invalid YouTube URL."
                    )
                    # audio_prep_successful remains True unless a preparation step *fails* for a valid URL
                    continue

                # Create a temporary working directory for THIS batch item within the main batch dir
                item_work_path = batch_work_path / item_identifier
                item_work_path.mkdir(exist_ok=True)  # Ensure directory exists

                youtube_url = youtube_url_raw.strip()
                original_source_id = (
                    youtube_url  # Use the URL as the source_id for mapping
                )

                item_source_map[item_identifier] = (
                    original_source_id  # Map item identifier to original source ID
                )

                # Get metadata for this row if metadata_col was identified
                metadata_value = row.get(metadata_col) if metadata_col else None
                item_metadata_map[item_identifier] = metadata_value  # Store metadata

                # Get video URL if available
                video_url_raw = row.get(video_col) if video_col in df.columns else None
                video_url = (
                    video_url_raw.strip() if isinstance(video_url_raw, str) else None
                )

                try:
                    # Prepare audio for this item within its specific work path, using item_identifier as session_id for logging/filenames
                    # Use original_source_id as session_id for download_audio_from_youtube filenames
                    audio_path = self._prepare_audio_input(
                        youtube_url, item_work_path, log_file_handle, item_identifier
                    )
                    audio_paths_for_batch.append(
                        audio_path
                    )  # Collect audio path for batch processing
                    valid_items_for_processing.append(
                        item_identifier
                    )  # Add to list ONLY if audio prep succeeded
                    source_audio_map[original_source_id] = (
                        audio_path  # Map original source ID (string) to the prepared audio file path (Path)
                    )

                    # ADDED: Populate the map from local temporary audio Path to original source identifier
                    local_audio_path_to_original_source_map[audio_path] = (
                        original_source_id
                    )

                    batch_log(
                        "info", f"[{item_identifier}] Audio prepared at: {audio_path}"
                    )

                    # --- Handle Video Download/Preparation (Placeholder) ---
                    # You would add similar logic here to download/copy video if needed for visual analysis
                    # For now, we'll just map the source_id to a dummy video path if a URL was provided
                    if video_url:
                        # **TODO: Implement actual video download/copy logic here**
                        # For now, create a dummy path and log a warning
                        dummy_video_path = (
                            item_work_path / f"{item_identifier}_video_placeholder.mp4"
                        )
                        source_video_map[original_source_id] = (
                            dummy_video_path  # Map source_id to video path (even if dummy)
                        )
                        batch_log(
                            "warning",
                            f"[{item_identifier}] Video download/preparation not fully implemented. Using placeholder path: {dummy_video_path}",
                        )
                    else:
                        batch_log(
                            "info",
                            f"[{item_identifier}] No video URL provided. Visual analysis will be skipped for this item.",
                        )

                except Exception as e:
                    batch_log(
                        "error",
                        f"[{item_identifier}] Failed to prepare audio input: {e}",
                    )
                    results.append(
                        f"[{item_identifier}] Failed during audio preparation."
                    )
                    audio_prep_successful = (
                        False  # Mark batch prep as failed if any item fails
                    )

            if not audio_prep_successful or not audio_paths_for_batch:
                batch_status_message = f"[{batch_job_id}] ERROR: Batch audio preparation failed or no valid audio sources found."
                batch_log("error", batch_status_message)
                batch_results_summary = batch_status_message + "\n" + "\n".join(results)
                return batch_status_message, batch_results_summary

            # --- Run WhisperX once on all prepared audio files ---
            batch_log(
                "info",
                "Running WhisperX for transcription and diarization on the entire batch...",
            )
            # The output directory for the batch WhisperX run will be the main batch_work_path
            whisperx_output_dir = batch_work_path
            try:
                # Pass the list of string audio paths to run_whisperx
                self.transcription.run_whisperx(
                    [str(p) for p in audio_paths_for_batch],  # Pass list of strings
                    str(whisperx_output_dir),  # Pass the output directory
                    log_file_handle,
                    batch_job_id,  # Use batch_job_id for the WhisperX session ID
                )
            except Exception as e:
                err_msg = (
                    f"[{batch_job_id}] ERROR during batch WhisperX processing: {e}"
                )
                batch_log("error", err_msg + "\n" + traceback.format_exc())
                batch_status_message = err_msg
                batch_results_summary = batch_status_message + "\n" + "\n".join(results)
                return batch_status_message, batch_results_summary

            # --- Structure the combined WhisperX output and perform analysis ---
            batch_log(
                "info",
                "Structuring combined WhisperX output and performing analysis...",
            )
            all_segments: SegmentsList = []
            try:
                # Pass the output directory and the mapping from local path to original source ID
                all_segments = self.transcription.convert_json_to_structured(
                    str(whisperx_output_dir),  # Pass the output directory
                    local_audio_path_to_original_source_map,  # Pass the new map
                )

                if not all_segments:
                    raise RuntimeError(
                        "Failed to structure segments from WhisperX batch output."
                    )

                # Perform Multimodal Emotion Analysis on all segments (which now have correct source_id)
                batch_log(
                    "info", "Running multimodal emotion analysis on all segments..."
                )
                # Pass the source_audio_map and source_video_map to the analyze method
                # source_audio_map is keyed by original source ID (string) -> local Path
                # source_video_map is keyed by original source ID (string) -> local Path
                # Segments now have source_id as original source ID (string)
                self.mm.analyze(
                    all_segments,
                    source_audio_map,
                    source_video_map,  # These maps are correct
                )  # analyze updates segments in-place

            except Exception as e:
                err_msg = f"[{batch_job_id}] ERROR during structuring or analysis: {e}"
                batch_log("error", err_msg + "\n" + traceback.format_exc())
                batch_status_message = err_msg
                batch_results_summary = batch_status_message + "\n" + "\n".join(results)
                return batch_status_message, batch_results_summary

            # --- Finalize each batch item individually ---
            batch_log("info", "Finalizing each batch item...")
            # Iterate through items that had successfully prepared audio
            for item_identifier in valid_items_for_processing:
                item_work_path = batch_work_path / item_identifier
                item_metadata = item_metadata_map.get(item_identifier)
                original_source_id = item_source_map.get(
                    item_identifier, "unknown_source"
                )  # Get original source ID

                # Filter segments for this specific item using the source identifier
                # Segments from convert_json_to_structured should have 'source_id' matching original audio path string or original URL
                # Assuming source_id in segments is the original URL/path string
                item_segments = [
                    seg
                    for seg in all_segments
                    if seg.get("source_id") == original_source_id
                ]

                # Ensure item_segments is not empty before finalizing
                if not item_segments:
                    batch_log(
                        "warning",
                        f"[{item_identifier}] No segments found for this item after batch processing. Skipping finalization.",
                    )
                    results.append(
                        f"[{item_identifier}] Skipped finalization: No segments found."
                    )
                    continue

                item_status_msg = self._finalize_batch_item(
                    item_segments,
                    item_metadata,
                    item_work_path,  # Pass the specific item's work path
                    log_file_handle,  # Pass the main batch log file handle
                    item_identifier,
                    original_source_id,  # Pass the original source ID
                )
                results.append(f"[{item_identifier}] {item_status_msg}")

            # --- Create the single master ZIP bundle ---
            batch_log("info", "Creating single master ZIP bundle for the batch...")
            permanent_output_dir = Path(self.config.get("output_dir", "./output"))
            permanent_output_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure permanent output directory exists

            # Use the batch job ID for the master zip filename
            master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
            master_zip_location = permanent_output_dir / master_zip_name

            batch_log("info", f"Master ZIP will be saved to: {master_zip_location}")

            temp_master_zip_path = master_zip_location.with_suffix(".temp.zip")

            files_to_add_to_master_zip: Dict[Path, str] = {}
            plot_files_for_master_zip: List[
                Path
            ] = []  # Plots are added separately in _create_final_zip if we use it

            # Collect files from each item's output subdirectory
            for item_id in (
                valid_items_for_processing
            ):  # Only collect from items that were successfully processed
                item_output_dir = batch_work_path / item_id / "output"
                if item_output_dir.exists():
                    for item_file_path in item_output_dir.rglob(
                        "*"
                    ):  # Use rglob to get files in subdirectories like 'plots'
                        if item_file_path.is_file():
                            # Create archive name like item_id/output/filename.json or item_id/output/plots/plot.png
                            archive_name = f"{item_id}/output/{item_file_path.relative_to(item_output_dir)}"
                            files_to_add_to_master_zip[item_file_path] = archive_name

            # Also add the original input XLSX to the zip
            original_xlsx_path = Path(xlsx_filepath)
            if original_xlsx_path.exists():
                files_to_add_to_master_zip[original_xlsx_path] = original_xlsx_path.name

            try:
                # Use the _create_final_zip helper, which can handle adding log file and plots
                # Note: Plots are already included via rglob in files_to_add_to_master_zip,
                # so passing plot_files_for_master_zip separately might lead to duplicates or is unnecessary.
                # Let's modify _create_final_zip to just take files_to_add and the log file.
                # OR, adjust the collection logic above to separate plots if _create_final_zip expects them that way.
                # Let's adjust collection above and keep _create_final_zip as is for now.
                # Update collection logic:
                files_to_add_to_master_zip = {}
                plot_files_for_master_zip = []
                for item_id in valid_items_for_processing:
                    item_output_dir = batch_work_path / item_id / "output"
                    if item_output_dir.exists():
                        for item_file_path in item_output_dir.rglob("*"):
                            if item_file_path.is_file():
                                if item_file_path.parent.name == "plots":
                                    # Plot files go to plot_files_for_master_zip
                                    plot_files_for_master_zip.append(item_file_path)
                                else:
                                    # Other files go to files_to_add_to_master_zip
                                    archive_name = f"{item_id}/output/{item_file_path.relative_to(item_output_dir)}"
                                    files_to_add_to_master_zip[item_file_path] = (
                                        archive_name
                                    )
                # Add the original input XLSX
                if original_xlsx_path.exists():
                    files_to_add_to_master_zip[original_xlsx_path] = (
                        original_xlsx_path.name
                    )

                final_zip_path = self._create_final_zip(
                    master_zip_location,  # Destination path
                    files_to_add_to_master_zip,  # Other files
                    plot_files_for_master_zip,  # Plot files
                    batch_log_path,  # The main batch log file
                )

                if final_zip_path:
                    master_zip_path_str = str(final_zip_path)
                    batch_status_message = f"[{batch_job_id}] ✅ Batch processing complete. Download ready: {master_zip_path_str}"
                    batch_results_summary = (
                        "Batch Results Summary:\n"
                        + "\n".join(results)
                        + f"\n\nMaster ZIP: {master_zip_path_str}"
                    )
                    batch_log("info", batch_status_message)
                    batch_log("info", batch_results_summary)
                else:
                    # Handle case where _create_final_zip failed
                    err_msg = (
                        f"[{batch_job_id}] ERROR: Failed to create master ZIP file."
                    )
                    batch_log("error", err_msg)
                    batch_status_message = err_msg
                    batch_results_summary = (
                        "Batch Results Summary:\n"
                        + "\n".join(results)
                        + f"\n\n{err_msg}"
                    )

            except Exception as zip_e:
                err_msg = f"[{batch_job_id}] ERROR creating master ZIP file: {zip_e}"
                batch_log("error", err_msg + "\n" + traceback.format_exc())
                batch_status_message = err_msg
                batch_results_summary = (
                    "Batch Results Summary:\n"
                    + "\n".join(results)
                    + f"\n\nERROR creating master ZIP: {zip_e}"
                )

        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            ValueError,
            RuntimeError,
        ) as e:
            # Catch specific anticipated errors from setup/reading/prep
            batch_status_message = f"[{batch_job_id}] ERROR: {e}"
            batch_log("error", batch_status_message)
            batch_results_summary = batch_status_message
        except Exception as e:
            # Catch any other unexpected errors
            err_msg = f"[{batch_job_id}] An unexpected error occurred during batch processing: {e}"
            batch_log("error", err_msg + "\n" + traceback.format_exc())
            batch_status_message = err_msg
            batch_results_summary = batch_status_message

        finally:
            # Ensure the main batch log file is closed
            if log_file_handle and not log_file_handle.closed:
                log_file_handle.close()

            # Clean up the main batch temporary directory after processing and zipping
            # MODIFIED: Add config check to keep temporary files
            keep_temp = self.config.get("keep_temp_files", False)
            if batch_work_path.exists() and not keep_temp:
                batch_log(
                    "info", f"Cleaning up batch temporary directory: {batch_work_path}"
                )
                try:
                    shutil.rmtree(batch_work_path)
                    batch_log(
                        "info", f"Batch temporary directory removed: {batch_work_path}"
                    )
                except OSError as e:
                    batch_log(
                        "warning",
                        f"Failed to remove batch temporary directory {batch_work_path}: {e}",
                    )
            elif batch_work_path.exists() and keep_temp:
                batch_log(
                    "info",
                    f"Keeping batch temporary directory for debugging: {batch_work_path}",
                )

        # This return statement should be aligned with the main try/except/finally block
        return batch_status_message, batch_results_summary
