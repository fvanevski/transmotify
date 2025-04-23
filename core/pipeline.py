# core/pipeline.py
import shutil
import traceback
import zipfile
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union  # Added Union

# --- Core Module Imports ---
from .constants import (
    LOG_FILE_NAME,
    FINAL_STRUCTURED_TRANSCRIPT_NAME,
    FINAL_ZIP_SUFFIX,
)
from .logging import log_error, log_info, log_warning
from .transcription import Transcription
from .multimodal_analysis import MultimodalAnalysis
import core.utils as utils
import core.reporting as reporting

# --- NEW: Import speaker labeling module ---
import core.speaker_labeling as speaker_labeling

# Type Hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]
EmotionSummary = Dict[str, Dict[str, Any]]
SpeakerLabels = Optional[Dict[str, str]]  # Renamed from SpeakerMapping for clarity

# --- NEW: Define structure for labeling state ---
LabelingItemState = Dict[
    str, Any
]  # Holds 'youtube_url', 'segments', 'eligible_speakers', 'collected_labels', 'audio_path', 'metadata'
LabelingBatchState = Dict[str, LabelingItemState]  # Maps item_identifier to its state
LabelingState = Dict[str, LabelingBatchState]  # Maps batch_job_id to its state


class Pipeline:
    """
    Orchestrates the end-to-end speech processing workflow, including
    optional interactive speaker labeling.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline components and state storage."""
        self.config = config
        self.transcription = Transcription(config)
        log_info("Initializing MultimodalAnalysis in Pipeline...")
        self.mm = MultimodalAnalysis(config)

        # --- NEW: State management for interactive labeling ---
        # Stores intermediate data for items awaiting user labeling
        # Structure: {batch_id: {item_id: {data...}}}
        self.labeling_state: LabelingState = {}

        # --- NEW: State management for collecting final output files ---
        # Structure: {batch_id: {archive_path: local_path}}
        self.batch_output_files: Dict[str, Dict[str, Path]] = {}

    # --- Helper: Safely get item state ---
    def _get_item_state(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[LabelingItemState]:
        """Retrieves the stored state for a specific item needing labeling."""
        batch_state = self.labeling_state.get(batch_job_id)
        if not batch_state:
            log_error(f"Labeling state not found for batch '{batch_job_id}'.")
            return None
        item_state = batch_state.get(item_identifier)
        if not item_state:
            log_error(
                f"Labeling state not found for item '{item_identifier}' in batch '{batch_job_id}'."
            )
            return None
        return item_state

    # --- Helper: Safely remove item state ---
    def _remove_item_state(self, batch_job_id: str, item_identifier: str):
        """Removes the state for an item after it's finalized."""
        if batch_job_id in self.labeling_state:
            if item_identifier in self.labeling_state[batch_job_id]:
                del self.labeling_state[batch_job_id][item_identifier]
                log_info(
                    f"Removed labeling state for item '{item_identifier}' in batch '{batch_job_id}'."
                )
            if not self.labeling_state[batch_job_id]:  # Remove batch entry if empty
                del self.labeling_state[batch_job_id]
                log_info(f"Removed empty labeling state for batch '{batch_job_id}'.")

    def _prepare_audio_input(
        self,
        input_source: str,
        item_work_dir: Path,
        log_file_handle: TextIO,
        session_id: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Downloads or copies audio and gets metadata."""
        # (No changes needed in this method's logic)
        log_info(f"[{session_id}] Preparing audio input from: {input_source}")
        audio_path: Optional[Path] = None
        metadata: Dict[str, Any] = {}

        if input_source.startswith(("http://", "https://")):
            try:
                audio_path, metadata = self.transcription.download_audio_from_youtube(
                    input_source, str(item_work_dir), log_file_handle, session_id
                )
            except (RuntimeError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"Audio download/conversion failed for URL: {input_source}"
                ) from e
        else:
            # Handling local files remains the same
            src_path = Path(input_source)
            if not src_path.is_file():
                raise ValueError(f"Invalid local input file path: {input_source}")
            unique_filename = f"{src_path.stem}_{session_id}{src_path.suffix}"
            dest_path = item_work_dir / unique_filename
            try:
                log_info(f"[{session_id}] Copying local file {src_path} to {dest_path}")
                shutil.copy(str(src_path), str(dest_path))
                audio_path = dest_path
                metadata = {
                    "source_path": str(src_path),
                    "filename": src_path.name,
                    "prepared_path": str(audio_path),
                    "source_type": "local_file",
                }
                if audio_path.suffix.lower() != ".wav":
                    log_warning(
                        f"[{session_id}] Input file {audio_path.name} is not WAV. WhisperX compatibility depends on ffmpeg."
                    )
            except Exception as e:
                log_error(
                    f"[{session_id}] Failed to copy local file {src_path} to {dest_path}: {e}"
                )
                raise RuntimeError(
                    f"Failed to prepare local audio file {input_source}"
                ) from e

        if audio_path is None or not audio_path.is_file():
            raise FileNotFoundError(
                f"Could not obtain valid audio file from source: {input_source}"
            )

        log_info(f"[{session_id}] Audio prepared successfully at: {audio_path}")
        return audio_path, metadata

    def _run_initial_item_processing(
        self,
        input_source: str,  # Can be URL or local path
        item_work_path: Path,
        log_file_handle: TextIO,
        item_identifier: str,
    ) -> Tuple[Optional[SegmentsList], Optional[Path], Optional[Dict[str, Any]]]:
        """
        Runs the initial processing steps: audio prep, transcription, analysis.
        Returns segments, audio path, and metadata, or Nones on failure.
        """
        audio_path_in_work_dir: Optional[Path] = None
        segments: Optional[SegmentsList] = None
        metadata: Optional[Dict[str, Any]] = None

        try:
            output_temp_dir = item_work_path / "output"
            output_temp_dir.mkdir(parents=True, exist_ok=True)

            # 1. Prepare Audio & Get Metadata
            audio_path_in_work_dir, metadata = self._prepare_audio_input(
                input_source, item_work_path, log_file_handle, item_identifier
            )
            if audio_path_in_work_dir is None:
                raise RuntimeError(
                    "Audio preparation failed."
                )  # Error logged in _prepare_audio_input

            # 2. Duration Check (Optional but recommended)
            min_duration = float(self.config.get("min_diarization_duration", 5.0))
            duration_ok = utils.run_ffprobe_duration_check(
                audio_path_in_work_dir, min_duration
            )
            if not duration_ok:
                log_warning(
                    f"[{item_identifier}] Audio duration potentially too short for reliable diarization."
                )

            # 3. Run WhisperX
            log_info(
                f"[{item_identifier}] Running WhisperX transcription and diarization..."
            )
            whisperx_json_path = self.transcription.run_whisperx(
                audio_path_in_work_dir,
                output_temp_dir,
                log_file_handle,
                item_identifier,
            )
            log_info(
                f"[{item_identifier}] WhisperX completed. Output JSON: {whisperx_json_path}"
            )

            # 4. Structure WhisperX Output
            log_info(f"[{item_identifier}] Structuring WhisperX output...")
            segments = self.transcription.convert_json_to_structured(whisperx_json_path)
            log_info(f"[{item_identifier}] Structured {len(segments)} segments.")

            # 5. Run Multimodal Analysis
            log_info(f"[{item_identifier}] Running multimodal emotion analysis...")
            # Use the input_source (original URL or path) for potential video analysis
            segments = self.mm.analyze(
                segments, str(audio_path_in_work_dir), input_source
            )
            log_info(f"[{item_identifier}] Multimodal analysis complete.")

            return segments, audio_path_in_work_dir, metadata

        except Exception as e:
            err_msg = f"[{item_identifier}] ERROR during initial item processing: {e}"
            log_error(err_msg)
            log_error(traceback.format_exc())
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_entry = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {err_msg}\n{traceback.format_exc()}\n"
                    log_file_handle.write(log_entry)
                    log_file_handle.flush()
                except Exception as log_e:
                    print(
                        f"WARN: Failed to write item processing error to log file: {log_e}"
                    )
            # Return None for all outputs on failure
            return (
                None,
                audio_path_in_work_dir,
                metadata,
            )  # Return audio path and metadata even if analysis failed

    def _finalize_batch_item(
        self,
        segments: SegmentsList,
        speaker_labels: SpeakerLabels,  # Changed from speaker_mapping
        item_identifier: str,
        item_work_path: Path,
        log_file_handle: TextIO,
        include_json_summary: bool,
        include_csv_summary: bool,
        include_script: bool,
        include_plots: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Union[Path, List[Path]]]]:  # Updated return type hint
        """Finalizes a single item: relabels, saves reports, returns file paths."""
        item_output_dir = item_work_path / "output"
        item_output_dir.mkdir(parents=True, exist_ok=True)
        generated_files_paths: Dict[
            str, Union[Path, List[Path]]
        ] = {}  # Store paths here

        def item_log(level, message):
            # (Logging helper remains the same)
            full_message = f"[{item_identifier}] {message}"
            log_func = log_info
            if level == "warning":
                log_func = log_warning
            elif level == "error":
                log_func = log_error
            log_func(full_message)
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_entry = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                    log_file_handle.write(log_entry)
                    log_file_handle.flush()
                except Exception as log_e:
                    print(
                        f"WARN: Failed to write to item log file: {log_e} - Message: {message}"
                    )

        item_log("info", f"Starting finalization for item in: {item_output_dir}")
        try:
            if not segments:
                item_log("error", "No segments provided for finalization.")
                return None

            # --- Relabeling using provided speaker_labels map ---
            if speaker_labels:  # speaker_labels is the Dict[SPEAKER_XX, UserLabel]
                item_log(
                    "info",
                    f"Applying speaker labels based on mapping: {speaker_labels}",
                )
                segments_relabeled_count = 0
                for seg in segments:
                    original_speaker_id = str(seg.get("speaker", "unknown"))
                    if original_speaker_id in speaker_labels:
                        # Use the label ONLY if it's not empty/None
                        final_label = speaker_labels[original_speaker_id]
                        if final_label and str(final_label).strip():
                            seg["speaker"] = str(final_label).strip()
                            segments_relabeled_count += 1
                        else:
                            # Keep original SPEAKER_XX if user submitted blank label
                            item_log(
                                "info",
                                f"Keeping original ID for {original_speaker_id} due to blank user label.",
                            )
                item_log(
                    "info",
                    f"Applied non-blank speaker labels to {segments_relabeled_count} segments.",
                )
            else:
                item_log(
                    "info",
                    "No speaker labels provided. Keeping original SPEAKER_XX IDs.",
                )

            # --- Save Final Structured Transcript (including metadata) ---
            final_json_name = (
                f"{item_identifier}_{Path(FINAL_STRUCTURED_TRANSCRIPT_NAME).name}"
            )
            final_json_path = item_output_dir / final_json_name
            item_log(
                "info", f"Saving final structured transcript to: {final_json_path}"
            )
            try:
                final_output_data: Dict[str, Any] = {
                    "segments": utils.convert_floats(segments),
                    "metadata": metadata if metadata is not None else {},
                }
                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(final_output_data, f, indent=2, ensure_ascii=False)
                item_log("info", "Final structured transcript saved successfully.")
                generated_files_paths["final_structured_json"] = final_json_path
            except Exception as e:
                item_log(
                    "error",
                    f"Failed to save final structured transcript: {e}\n{traceback.format_exc()}",
                )
                return None  # Critical failure

            # --- Generate Optional Report Outputs ---
            item_log("info", "Generating optional report outputs...")
            try:
                report_outputs = reporting.generate_item_report_outputs(
                    segments=segments,  # Use potentially relabeled segments
                    item_identifier=item_identifier,
                    item_output_dir=item_output_dir,
                    config=self.config,
                    log_file_handle=log_file_handle,
                    include_json_summary=include_json_summary,
                    include_csv_summary=include_csv_summary,
                    include_script=include_script,
                    include_plots=include_plots,
                )
                # report_outputs is Dict[str, Optional[Path] or List[Path]]
                for key, path_or_list in report_outputs.items():
                    if path_or_list:  # Only add if path/list is not None/empty
                        generated_files_paths[key] = path_or_list

                item_log(
                    "info", f"Generated report outputs: {list(report_outputs.keys())}"
                )
            except Exception as e:
                item_log(
                    "error",
                    f"Error during optional report generation step: {e}\n{traceback.format_exc()}",
                )
                # Continue finalization even if reports fail

            item_log("info", f"Finalization complete for item.")
            return generated_files_paths  # Return dict of created file paths

        except Exception as e:
            item_log(
                "error",
                f"Unexpected error during finalization: {e}\n{traceback.format_exc()}",
            )
            return None

    def create_final_zip(
        self,
        zip_path: Path,
        files_to_add: Dict[str, Path],  # Maps archive path to local path
        log_file_handle: Optional[TextIO] = None,
        batch_job_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Creates the final ZIP archive from a dictionary of files."""
        log_prefix = f"[{batch_job_id}] " if batch_job_id else ""
        log_info(f"{log_prefix}Attempting to create final ZIP archive: {zip_path}")

        if not files_to_add:
            log_warning(
                f"{log_prefix}No files provided to add to the zip archive. Skipping zip creation."
            )
            return None

        final_output_dir = zip_path.parent
        try:
            final_output_dir.mkdir(parents=True, exist_ok=True)
            log_info(
                f"{log_prefix}Ensured final output directory exists: {final_output_dir}"
            )
        except Exception as e:
            log_error(
                f"{log_prefix}Failed to create final output directory {final_output_dir}: {e}"
            )
            return None

        temp_zip_path = zip_path.with_suffix(f".{os.getpid()}.temp.zip")
        files_added_count = 0
        files_skipped_count = 0
        try:
            with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                # Add files from the dictionary
                for arc_name, local_path in files_to_add.items():
                    if local_path and local_path.is_file():
                        try:
                            zf.write(local_path, arcname=arc_name)
                            files_added_count += 1
                        except Exception as e:
                            log_warning(
                                f"{log_prefix}Failed to add file {local_path} to zip as {arc_name}: {e}"
                            )
                            files_skipped_count += 1
                    elif local_path:
                        log_warning(
                            f"{log_prefix}File not found or is not a file, skipping: {local_path}"
                        )
                        files_skipped_count += 1
                    else:
                        log_warning(
                            f"{log_prefix}Invalid path for archive name '{arc_name}', skipping."
                        )
                        files_skipped_count += 1

            if files_added_count == 0:
                log_error(
                    f"{log_prefix}No files were successfully added to the zip. Aborting zip creation."
                )
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                return None

            # Move temporary zip to final path
            shutil.move(str(temp_zip_path), str(zip_path))
            log_info(f"{log_prefix}Successfully created final ZIP: {zip_path}")
            log_info(
                f"{log_prefix}Files added: {files_added_count}, Files skipped: {files_skipped_count}"
            )
            return zip_path

        except Exception as e:
            log_error(
                f"{log_prefix}Failed to create ZIP file {zip_path}: {e}\n{traceback.format_exc()}"
            )
            if temp_zip_path.exists():
                try:
                    temp_zip_path.unlink()
                    log_info(f"{log_prefix}Cleaned up temporary zip file.")
                except Exception as unlink_e:
                    log_warning(
                        f"{log_prefix}Failed to clean up temporary zip file {temp_zip_path}: {unlink_e}"
                    )
            return None
        finally:
            # Clean up batch output file tracking for this batch ID
            if batch_job_id and batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
                log_info(f"{log_prefix}Cleared output file tracking for batch.")

    # --- BATCH PROCESSING ENTRY POINT ---
    def process_batch_xlsx(
        self,
        xlsx_filepath: str,
        include_source_audio: bool,
        include_json_summary: bool,
        include_csv_summary: bool,
        include_script_transcript: bool,
        include_plots: bool,
    ) -> Tuple[str, str, Optional[str]]:  # Added Optional[str] for batch_job_id
        """
        Processes a batch defined in an Excel file. Handles initial processing
        and sets up state for interactive labeling if enabled and needed.

        Returns:
            Tuple[status_message, results_summary, batch_job_id (if labeling needed else None)]
        """
        batch_job_id = f"batch-{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]}"
        log_info(f"[{batch_job_id}] Starting batch processing for: {xlsx_filepath}")
        batch_status_message = f"[{batch_job_id}] Reading batch file..."
        batch_results_list: List[str] = []
        base_temp_dir = Path(self.config.get("temp_dir", "./temp"))
        batch_work_path = base_temp_dir / batch_job_id
        log_file_handle: Optional[TextIO] = None
        total_items = 0
        processed_immediately_count = 0
        pending_labeling_count = 0
        failed_count = 0
        labeling_is_required_overall = False  # Flag if any item needs labeling

        # --- Initialize output file tracking for this batch ---
        self.batch_output_files[batch_job_id] = {}

        try:
            batch_work_path.mkdir(parents=True, exist_ok=True)
            batch_log_path = batch_work_path / LOG_FILE_NAME
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            log_info(f"[{batch_job_id}] Batch log file created at: {batch_log_path}")
            # Add log file to the batch outputs immediately
            self.batch_output_files[batch_job_id][batch_log_path.name] = batch_log_path

            def batch_log(level, message):
                # (Logging helper remains the same)
                full_message = f"[{batch_job_id}] {message}"
                log_func = log_info
                if level == "warning":
                    log_func = log_warning
                elif level == "error":
                    log_func = log_error
                log_func(full_message)
                if log_file_handle and not log_file_handle.closed:
                    try:
                        log_entry = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                        log_file_handle.write(log_entry)
                        log_file_handle.flush()
                    except Exception as log_e:
                        print(
                            f"WARN: Failed to write to batch log file: {log_e} - Message: {message}"
                        )

            batch_log("info", f"Batch temporary directory: {batch_work_path}")
            batch_log("info", f"Reading batch file: {xlsx_filepath}")
            try:
                df = pd.read_excel(xlsx_filepath, sheet_name=0)
                total_items = len(df)
                batch_log("info", f"Read {total_items} rows from {xlsx_filepath}")
                if df.empty:
                    raise ValueError("Excel file contains no data rows.")
            except FileNotFoundError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Failed to read or parse Excel file {xlsx_filepath}: {e}"
                ) from e

            url_col = self.config.get(
                "batch_url_column", "YouTube URL"
            )  # Assuming config still has this key
            if url_col not in df.columns:
                raise ValueError(
                    f"Required column '{url_col}' not found in the Excel file."
                )

            # --- Get labeling config settings ---
            enable_labeling = self.config.get("enable_interactive_labeling", False)
            labeling_min_total_time = float(
                self.config.get("speaker_labeling_min_total_time", 15.0)
            )
            labeling_min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            batch_log("info", f"Interactive Labeling Enabled: {enable_labeling}")

            # --- Initialize labeling state for this batch ---
            self.labeling_state[batch_job_id] = {}

            for sequential_index, (index, row) in enumerate(df.iterrows()):
                item_index = sequential_index + 1
                item_identifier = f"item_{item_index:03d}"
                batch_log(
                    "info",
                    f"--- Processing item {item_index}/{total_items} ({item_identifier}) ---",
                )

                source_url_or_path = row.get(url_col)  # Get URL or Path from Excel
                if (
                    not isinstance(source_url_or_path, str)
                    or not source_url_or_path.strip()
                ):
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Skipping row {item_index}: Invalid or missing source URL/Path ('{source_url_or_path}').",
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Skipped: Invalid Source."
                    )
                    failed_count += 1
                    continue

                source_url_or_path = source_url_or_path.strip()
                # Check if it looks like a URL, needed for video preview download check
                is_youtube_url = source_url_or_path.startswith(("http:", "https:"))

                item_work_path = batch_work_path / item_identifier
                item_work_path.mkdir(exist_ok=True)

                # --- Run Initial Processing ---
                segments, audio_path, metadata = self._run_initial_item_processing(
                    source_url_or_path, item_work_path, log_file_handle, item_identifier
                )

                if segments is None or audio_path is None:
                    batch_log(
                        "error", f"[{item_identifier}] Core initial processing failed."
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Core processing error."
                    )
                    failed_count += 1
                    continue  # Skip to next item

                # --- Check if Interactive Labeling is Needed ---
                needs_labeling = False
                eligible_speakers = []
                if (
                    enable_labeling and is_youtube_url
                ):  # Only enable for YouTube URLs for now
                    eligible_speakers = speaker_labeling.identify_eligible_speakers(
                        segments, labeling_min_total_time, labeling_min_block_time
                    )
                    if eligible_speakers:
                        needs_labeling = True
                        labeling_is_required_overall = (
                            True  # Mark that the batch needs UI interaction
                        )
                        pending_labeling_count += 1
                        batch_results_list.append(
                            f"[{item_identifier}] Success (Pending Labeling)."
                        )
                        batch_log(
                            "info",
                            f"[{item_identifier}] Item requires interactive labeling for speakers: {eligible_speakers}.",
                        )

                        # --- Store state for later ---
                        self.labeling_state[batch_job_id][item_identifier] = {
                            "youtube_url": source_url_or_path,  # Store the URL
                            "segments": segments,
                            "eligible_speakers": eligible_speakers,
                            "collected_labels": {},  # Initialize empty labels
                            "audio_path": audio_path,  # Store audio path needed later
                            "metadata": metadata,  # Store metadata needed later
                            "item_work_path": item_work_path,  # Store work path
                        }
                        # Add source audio to zip collection immediately if requested, even if labeling pending
                        if include_source_audio and audio_path and audio_path.is_file():
                            self.batch_output_files[batch_job_id][
                                f"{item_identifier}/{audio_path.name}"
                            ] = audio_path

                    else:
                        batch_log(
                            "info",
                            f"[{item_identifier}] Interactive labeling enabled, but no eligible speakers found.",
                        )
                elif enable_labeling and not is_youtube_url:
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Interactive labeling enabled, but input is not a YouTube URL. Skipping labeling step.",
                    )

                # --- Finalize Immediately OR Defer ---
                if not needs_labeling:
                    batch_log(
                        "info",
                        f"[{item_identifier}] Finalizing item immediately (no labeling required).",
                    )
                    # Pass None or empty dict for speaker_labels
                    generated_files = self._finalize_batch_item(
                        segments=segments,
                        speaker_labels={},  # No labels if not interactive
                        item_identifier=item_identifier,
                        item_work_path=item_work_path,
                        log_file_handle=log_file_handle,
                        include_json_summary=include_json_summary,
                        include_csv_summary=include_csv_summary,
                        include_script=include_script_transcript,
                        include_plots=include_plots,
                        metadata=metadata,
                    )

                    if generated_files is None:
                        batch_log(
                            "error",
                            f"[{item_identifier}] Immediate finalization failed critically.",
                        )
                        batch_results_list.append(
                            f"[{item_identifier}] Failed: Finalization error."
                        )
                        failed_count += 1
                    else:
                        processed_immediately_count += 1
                        batch_results_list.append(
                            f"[{item_identifier}] Success (Finalized)."
                        )
                        # --- Add generated files to batch output collection ---
                        if batch_job_id in self.batch_output_files:
                            arc_folder_base = item_identifier
                            for key, path_or_list in generated_files.items():
                                arc_folder = (
                                    f"{arc_folder_base}/plots"
                                    if key == "plot_paths"
                                    else arc_folder_base
                                )
                                if key == "plot_paths" and isinstance(
                                    path_or_list, list
                                ):
                                    for p_path in path_or_list:
                                        if p_path and p_path.is_file():
                                            self.batch_output_files[batch_job_id][
                                                f"{arc_folder}/{p_path.name}"
                                            ] = p_path
                                elif (
                                    isinstance(path_or_list, Path)
                                    and path_or_list.is_file()
                                ):
                                    self.batch_output_files[batch_job_id][
                                        f"{arc_folder}/{path_or_list.name}"
                                    ] = path_or_list

                            # Add source audio if requested
                            if (
                                include_source_audio
                                and audio_path
                                and audio_path.is_file()
                            ):
                                self.batch_output_files[batch_job_id][
                                    f"{arc_folder_base}/{audio_path.name}"
                                ] = audio_path
                        else:
                            log_warning(
                                f"[{item_identifier}] Cannot add output files to zip collection - batch ID '{batch_job_id}' not found."
                            )

                batch_log(
                    "info",
                    f"--- Finished item {item_index}/{total_items} ({item_identifier}) ---",
                )

            # --- End of loop ---
            total_processed_or_pending = (
                processed_immediately_count + pending_labeling_count
            )
            batch_log("info", f"Batch processing loop complete.")
            batch_log(
                "info", f"  Items Finalized Immediately: {processed_immediately_count}"
            )
            batch_log(
                "info", f"  Items Pending Labeling:      {pending_labeling_count}"
            )
            batch_log("info", f"  Items Failed/Skipped:        {failed_count}")

            if total_processed_or_pending == 0 and failed_count > 0:
                raise RuntimeError(
                    "No items were processed successfully or queued for labeling in the batch."
                )

            # --- Determine final status message ---
            if labeling_is_required_overall:
                batch_status_message = f"[{batch_job_id}] Initial processing complete. {pending_labeling_count} item(s) require speaker labeling via the UI."
                # Return batch_job_id so UI knows which session to continue
                return_batch_id = batch_job_id
            else:
                # If no labeling needed, create the zip now
                batch_log(
                    "info",
                    f"No interactive labeling required for this batch. Creating final ZIP.",
                )
                permanent_output_dir = Path(self.config.get("output_dir", "./output"))
                master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
                master_zip_path = permanent_output_dir / master_zip_name
                created_zip_path = self.create_final_zip(
                    master_zip_path,
                    self.batch_output_files.get(
                        batch_job_id, {}
                    ),  # Get files for this batch
                    log_file_handle,
                    batch_job_id,
                )
                if created_zip_path:
                    batch_status_message = f"[{batch_job_id}] ✅ Batch processing complete. Download ready: {created_zip_path}"
                else:
                    batch_status_message = f"[{batch_job_id}] ❗️ Batch processing finished, but failed to create final ZIP bundle."
                return_batch_id = None  # No further interaction needed

            batch_log("info", f"Batch Status: {batch_status_message}")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            err_msg = f"[{batch_job_id}] ERROR: {e}"
            batch_log("error", err_msg)
            batch_status_message = err_msg
            return_batch_id = None
            # Clean up potentially created batch output file entry on early error
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]

        except Exception as e:
            err_msg = f"[{batch_job_id}] An unexpected error occurred during batch processing: {e}"
            batch_log("error", err_msg + "\n" + traceback.format_exc())
            batch_status_message = err_msg
            return_batch_id = None
            # Clean up potentially created batch output file entry on early error
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
        finally:
            # *** Indentation Correction Start ***
            # Close log file ONLY if labeling is NOT required (otherwise UI needs it open)
            # We might need a separate mechanism to close the log later if labeling occurs.
            # For now, let's keep it simple: close if no labeling needed overall.
            if (
                log_file_handle
                and not log_file_handle.closed
                and not labeling_is_required_overall
            ):
                log_info(
                    f"[{batch_job_id}] Closing batch log file (no labeling required)."
                )
                log_file_handle.close()
            elif (
                log_file_handle
                and not log_file_handle.closed
                and labeling_is_required_overall
            ):
                log_info(
                    f"[{batch_job_id}] Keeping batch log file open for interactive labeling."
                )

            # Cleanup logic needs adjustment - only cleanup if successful *and* no labeling pending
            cleanup_temp = self.config.get("cleanup_temp_on_success", True)
            # Check if the final status indicates success (ends with .zip path potentially)
            batch_succeeded_without_labeling = (
                "✅" in batch_status_message
            ) and not labeling_is_required_overall
            should_cleanup = cleanup_temp and batch_succeeded_without_labeling

            if batch_work_path.exists():
                if should_cleanup:
                    batch_log(
                        "info",
                        f"Cleaning up batch temporary directory: {batch_work_path}",
                    )
                    try:
                        shutil.rmtree(batch_work_path)
                        batch_log("info", f"Successfully removed batch temp directory.")
                    except OSError as e:
                        batch_log(
                            "warning",
                            f"Failed to remove batch temporary directory {batch_work_path}: {e}",
                        )
                elif labeling_is_required_overall:
                    batch_log(
                        "info",
                        f"Keeping temporary directory for interactive labeling: {batch_work_path}",
                    )
                else:  # Keep if errors occurred or labeling required but didn't succeed fully before error
                    batch_log(
                        "warning",
                        f"Skipping cleanup of temporary directory due to errors, pending labeling, or config: {batch_work_path}",
                    )
            # *** Indentation Correction End ***

        # --- Construct results summary ---
        results_summary_string = (
            f"Batch Processing Summary ({batch_job_id}):\n"
            f"- Total Items Read: {total_items}\n"
            f"- Finalized Immediately: {processed_immediately_count}\n"
            f"- Pending Labeling: {pending_labeling_count}\n"
            f"- Failed/Skipped: {failed_count}\n"
            f"--------------------\n"
            + "\n".join(batch_results_list)
            + f"\n--------------------\n"
            f"Overall Status: {batch_status_message}"
        )
        return batch_status_message, results_summary_string, return_batch_id

    # --- NEW INTERFACE FUNCTIONS for UI ---

    def start_interactive_labeling_for_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Prepares and returns data for the first speaker to be labeled for an item.
        Downloads initial video clips.
        """
        log_info(f"[{batch_job_id}-{item_identifier}] Starting interactive labeling...")
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return None  # Error logged in helper

        eligible_speakers = item_state.get("eligible_speakers", [])
        if not eligible_speakers:
            log_warning(
                f"[{batch_job_id}-{item_identifier}] No eligible speakers found in stored state."
            )
            # Finalize immediately if no speakers ended up needing labeling
            self.finalize_labeled_item(batch_job_id, item_identifier)  # Try to finalize
            self._remove_item_state(
                batch_job_id, item_identifier
            )  # Ensure state is cleaned up
            return None  # Signal UI no labeling needed

        first_speaker_id = eligible_speakers[0]
        log_info(
            f"[{batch_job_id}-{item_identifier}] First speaker to label: {first_speaker_id}"
        )

        # Get preview segments
        preview_duration = float(
            self.config.get("speaker_labeling_preview_duration", 5.0)
        )
        min_block_time = float(
            self.config.get("speaker_labeling_min_block_time", 10.0)
        )  # Needed again here
        time_segments = speaker_labeling.select_preview_time_segments(
            speaker_id=first_speaker_id,
            segments=item_state.get("segments", []),
            preview_duration=preview_duration,
            min_block_time=min_block_time,
        )

        if not time_segments:
            log_warning(
                f"[{batch_job_id}-{item_identifier}] Could not select preview segments for {first_speaker_id}."
            )
            # Consider how UI should handle this - maybe skip speaker? For now, return empty list.
            return first_speaker_id, []

        # Download clips
        youtube_url = item_state.get("youtube_url")
        item_work_path = item_state.get("item_work_path")  # Get work path from state
        if not youtube_url or not item_work_path:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Missing youtube_url or item_work_path in state for downloading clips."
            )
            return first_speaker_id, []  # Cannot download

        # Need log file handle - retrieve or reopen? Let's try reopening in append mode.
        batch_work_dir = item_work_path.parent  # Parent is the batch directory
        log_file_path = batch_work_dir / LOG_FILE_NAME  # Get batch log path
        log_handle = None
        clip_paths: List[Path] = []
        try:
            # Ensure batch dir exists before trying to open log
            batch_work_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file_path, "a", encoding="utf-8")
            clip_paths = speaker_labeling.download_video_clips(
                youtube_url=youtube_url,
                time_segments=time_segments,
                output_dir=item_work_path / "previews",  # Save clips in a subfolder
                item_identifier=item_identifier,
                speaker_id=first_speaker_id,
                log_file_handle=log_handle,
            )
        except Exception as e:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Error opening log or downloading clips for {first_speaker_id}: {e}"
            )
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()

        # Return speaker ID and paths as strings
        return first_speaker_id, [str(p) for p in clip_paths]

    def store_speaker_label(
        self, batch_job_id: str, item_identifier: str, speaker_id: str, user_label: str
    ) -> bool:
        """Stores the user-provided label for a speaker."""
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return False  # Error logged in helper

        if "collected_labels" not in item_state:
            item_state["collected_labels"] = {}  # Initialize if missing

        # Store the label (even if blank, _finalize_batch_item handles logic)
        item_state["collected_labels"][speaker_id] = user_label
        log_info(
            f"[{batch_job_id}-{item_identifier}] Stored label for {speaker_id}: '{user_label}'"
        )
        return True

    def get_next_speaker_for_labeling(
        self, batch_job_id: str, item_identifier: str, current_speaker_index: int
    ) -> Optional[Tuple[str, List[str]]]:
        """Gets the ID and video clip paths for the next speaker, or None if done."""
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return None  # Error logged in helper

        eligible_speakers = item_state.get("eligible_speakers", [])
        next_speaker_index = current_speaker_index + 1

        if next_speaker_index < len(eligible_speakers):
            next_speaker_id = eligible_speakers[next_speaker_index]
            log_info(
                f"[{batch_job_id}-{item_identifier}] Getting data for next speaker (Index {next_speaker_index}): {next_speaker_id}"
            )

            # --- Logic duplicated from start_interactive_labeling_for_item ---
            # Refactor Opportunity: Create a helper function for segment selection & download
            preview_duration = float(
                self.config.get("speaker_labeling_preview_duration", 5.0)
            )
            min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            time_segments = speaker_labeling.select_preview_time_segments(
                speaker_id=next_speaker_id,
                segments=item_state.get("segments", []),
                preview_duration=preview_duration,
                min_block_time=min_block_time,
            )

            if not time_segments:
                log_warning(
                    f"[{batch_job_id}-{item_identifier}] Could not select preview segments for {next_speaker_id}."
                )
                return next_speaker_id, []

            youtube_url = item_state.get("youtube_url")
            item_work_path = item_state.get("item_work_path")
            if not youtube_url or not item_work_path:
                log_error(
                    f"[{batch_job_id}-{item_identifier}] Missing state data for downloading clips for {next_speaker_id}."
                )
                return next_speaker_id, []

            batch_work_dir = item_work_path.parent
            log_file_path = batch_work_dir / LOG_FILE_NAME
            log_handle = None
            clip_paths: List[Path] = []
            try:
                batch_work_dir.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_file_path, "a", encoding="utf-8")
                clip_paths = speaker_labeling.download_video_clips(
                    youtube_url=youtube_url,
                    time_segments=time_segments,
                    output_dir=item_work_path / "previews",
                    item_identifier=item_identifier,
                    speaker_id=next_speaker_id,
                    log_file_handle=log_handle,
                )
            except Exception as e:
                log_error(
                    f"[{batch_job_id}-{item_identifier}] Error opening log or downloading clips for {next_speaker_id}: {e}"
                )
            finally:
                if log_handle and not log_handle.closed:
                    log_handle.close()

            return next_speaker_id, [str(p) for p in clip_paths]
            # --- End of duplicated logic ---

        else:
            log_info(
                f"[{batch_job_id}-{item_identifier}] All eligible speakers have been processed for this item."
            )
            return None  # Signal that labeling for this item is complete

    def finalize_labeled_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """Finalizes an item after interactive labeling is complete."""
        log_info(
            f"[{batch_job_id}-{item_identifier}] Finalizing item after labeling..."
        )
        item_state = self._get_item_state(batch_job_id, item_identifier)
        # IMPORTANT: Keep state until after finalization logic runs, then remove

        if not item_state:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Cannot finalize - item state not found."
            )
            return None

        # Retrieve necessary data from state
        segments = item_state.get("segments")
        collected_labels = item_state.get("collected_labels", {})
        item_work_path = item_state.get("item_work_path")
        metadata = item_state.get("metadata")
        audio_path = item_state.get("audio_path")  # Get audio path from state

        if not segments or not item_work_path:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Cannot finalize - missing segments or work path in state."
            )
            self._remove_item_state(
                batch_job_id, item_identifier
            )  # Clean up broken state
            return None

        # Need log file handle again for finalization
        batch_work_dir = item_work_path.parent
        log_file_path = batch_work_dir / LOG_FILE_NAME
        log_handle = None
        generated_files: Optional[Dict[str, Any]] = None
        try:
            batch_work_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file_path, "a", encoding="utf-8")

            # --- Call the existing finalization logic ---
            generated_files = self._finalize_batch_item(
                segments=segments,
                speaker_labels=collected_labels,  # Pass the collected labels
                item_identifier=item_identifier,
                item_work_path=item_work_path,
                log_file_handle=log_handle,
                # Retrieve output flags from config
                include_json_summary=self.config.get("include_json_summary", True),
                include_csv_summary=self.config.get("include_csv_summary", False),
                include_script=self.config.get("include_script_transcript", False),
                include_plots=self.config.get("include_plots", False),
                metadata=metadata,
            )

            # --- Add generated files to the main batch output collection ---
            if generated_files and batch_job_id in self.batch_output_files:
                arc_folder_base = item_identifier
                for key, path_or_list in generated_files.items():
                    arc_folder = (
                        f"{arc_folder_base}/plots"
                        if key == "plot_paths"
                        else arc_folder_base
                    )
                    if key == "plot_paths" and isinstance(path_or_list, list):
                        for p_path in path_or_list:
                            if p_path and p_path.is_file():
                                self.batch_output_files[batch_job_id][
                                    f"{arc_folder}/{p_path.name}"
                                ] = p_path
                    elif isinstance(path_or_list, Path) and path_or_list.is_file():
                        self.batch_output_files[batch_job_id][
                            f"{arc_folder}/{path_or_list.name}"
                        ] = path_or_list

                # Add source audio if requested (might already be added, but safer to check)
                include_source_audio = self.config.get("include_source_audio", True)
                if include_source_audio and audio_path and audio_path.is_file():
                    if (
                        f"{arc_folder_base}/{audio_path.name}"
                        not in self.batch_output_files[batch_job_id]
                    ):
                        self.batch_output_files[batch_job_id][
                            f"{arc_folder_base}/{audio_path.name}"
                        ] = audio_path

            elif not generated_files:
                log_error(
                    f"[{batch_job_id}-{item_identifier}] Finalization process failed."
                )
            else:  # generated_files is ok, but batch_job_id missing from collection
                log_warning(
                    f"[{batch_job_id}-{item_identifier}] Cannot add output files to zip collection - batch ID not found in tracking."
                )

        except Exception as e:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Error during item finalization process: {e}\n{traceback.format_exc()}"
            )
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()
            # Clean up state for this item now that it's finalized (or failed)
            self._remove_item_state(batch_job_id, item_identifier)
            # Clean up preview clips?
            preview_dir = item_work_path / "previews"
            if preview_dir.exists():
                try:
                    shutil.rmtree(preview_dir)
                    log_info(
                        f"[{batch_job_id}-{item_identifier}] Removed preview clip directory: {preview_dir}"
                    )
                except OSError as e:
                    log_warning(
                        f"[{batch_job_id}-{item_identifier}] Failed to remove preview clip directory {preview_dir}: {e}"
                    )

        return generated_files

    def check_batch_completion_and_zip(self, batch_job_id: str) -> Optional[Path]:
        """
        Checks if all items needing labeling in a batch are done. If so, creates the final zip.
        Should be called by the UI after the last item's labeling is submitted/finalized.
        """
        # Check if the batch ID still exists in the labeling state
        if batch_job_id in self.labeling_state and self.labeling_state[batch_job_id]:
            # If there are still items left in the state for this batch, it's not complete
            log_info(
                f"[{batch_job_id}] Batch labeling not yet complete. Items remaining: {list(self.labeling_state[batch_job_id].keys())}"
            )
            return None  # Indicate not complete
        else:
            # Batch state is gone or empty, meaning all items are finalized
            log_info(
                f"[{batch_job_id}] All items finalized or batch complete. Proceeding with ZIP creation."
            )

            # Need log file handle one last time for zipping
            batch_work_path = Path(self.config.get("temp_dir", "./temp")) / batch_job_id
            log_file_path = batch_work_path / LOG_FILE_NAME
            log_handle = None
            zip_path: Optional[Path] = None
            try:
                # Open log in append mode, create batch dir if it was somehow removed
                batch_work_path.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_file_path, "a", encoding="utf-8")  # Append to log

                permanent_output_dir = Path(self.config.get("output_dir", "./output"))
                master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
                master_zip_path = permanent_output_dir / master_zip_name

                # Get the collected file paths for this batch
                files_for_zip = self.batch_output_files.get(batch_job_id, {})

                zip_path = self.create_final_zip(
                    master_zip_path,
                    files_for_zip,
                    log_handle,
                    batch_job_id,
                )
                # Cleanup temp dir after successful zip (if configured)
                cleanup_temp = self.config.get("cleanup_temp_on_success", True)
                if zip_path and cleanup_temp and batch_work_path.exists():
                    log_info(
                        f"[{batch_job_id}] Cleaning up batch temporary directory after successful zip: {batch_work_path}"
                    )
                    try:
                        shutil.rmtree(batch_work_path)
                        log_info(
                            f"[{batch_job_id}] Successfully removed batch temp directory."
                        )
                    except OSError as e:
                        log_warning(
                            f"[{batch_job_id}] Failed to remove batch temporary directory {batch_work_path}: {e}"
                        )
                elif zip_path and not cleanup_temp:
                    log_info(
                        f"[{batch_job_id}] Skipping cleanup of temporary directory as per config: {batch_work_path}"
                    )
                elif not zip_path and batch_work_path.exists():  # Keep if zip failed
                    log_warning(
                        f"[{batch_job_id}] Keeping temporary directory due to ZIP creation failure: {batch_work_path}"
                    )

            except Exception as e:
                log_error(
                    f"[{batch_job_id}] Error during final zip creation or cleanup check: {e}"
                )
            finally:
                if log_handle and not log_handle.closed:
                    log_handle.close()
                # Ensure state is clean regardless of zip success/failure now
                if batch_job_id in self.labeling_state:
                    del self.labeling_state[batch_job_id]
                # batch_output_files cleaned up inside create_final_zip finally block

            return zip_path  # Return path if successful, None otherwise
