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
import core.speaker_labeling as speaker_labeling  # Keep this import

# Type Hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]
EmotionSummary = Dict[str, Dict[str, Any]]
SpeakerLabels = Optional[Dict[str, str]]

# Define structure for labeling state
# --- ADDED 'output_flags' to item state ---
LabelingItemState = Dict[
    str, Any
]  # Holds 'youtube_url', 'segments', 'eligible_speakers', 'collected_labels', 'audio_path', 'metadata', 'item_work_path'
# --- ADDED 'output_flags' to batch state ---
LabelingBatchState = Dict[
    str, Union[LabelingItemState, List[str], Dict[str, bool]]
]  # Allow item state, item order list, and flags dict
LabelingState = Dict[str, LabelingBatchState]


class Pipeline:
    """
    Orchestrates the end-to-end speech processing workflow, including
    optional interactive speaker labeling using YouTube embeds.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline components and state storage."""
        self.config = config
        self.transcription = Transcription(config)
        log_info("Initializing MultimodalAnalysis in Pipeline...")
        self.mm = MultimodalAnalysis(config)
        self.labeling_state: LabelingState = {}
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
        # --- Ensure we get the item dict, not flags or order list ---
        item_state = batch_state.get(item_identifier)
        if (
            not item_state
            or not isinstance(item_state, dict)
            or "segments" not in item_state
        ):  # Basic check for item state structure
            log_error(
                f"Labeling state not found or invalid for item '{item_identifier}' in batch '{batch_job_id}'."
            )
            return None
        return item_state  # Type hint checker might complain, but logic is sound

    # --- Helper: Safely remove item state ---
    def _remove_item_state(self, batch_job_id: str, item_identifier: str):
        """Removes the state for an item after it's finalized or skipped."""
        if batch_job_id in self.labeling_state:
            if item_identifier in self.labeling_state[batch_job_id]:
                # Ensure we're deleting an item state dict
                item_state_to_remove = self.labeling_state[batch_job_id].get(
                    item_identifier
                )
                if (
                    isinstance(item_state_to_remove, dict)
                    and "item_work_path" in item_state_to_remove
                ):
                    # Clean up preview dir if it exists before removing state
                    item_work_path = item_state_to_remove.get("item_work_path")
                    if item_work_path:
                        preview_dir = Path(item_work_path) / "previews"
                        if preview_dir.exists():
                            try:
                                shutil.rmtree(preview_dir)
                                log_info(
                                    f"[{batch_job_id}-{item_identifier}] Removed preview clip directory during state removal: {preview_dir}"
                                )
                            except OSError as e:
                                log_warning(
                                    f"[{batch_job_id}-{item_identifier}] Failed to remove preview clip directory {preview_dir} during state removal: {e}"
                                )

                    del self.labeling_state[batch_job_id][item_identifier]
                    log_info(
                        f"Removed labeling state for item '{item_identifier}' in batch '{batch_job_id}'."
                    )
                else:
                    log_warning(
                        f"Attempted to remove non-item state for '{item_identifier}' in batch '{batch_job_id}'."
                    )

            # --- Check if only helper keys remain before deleting batch state ---
            remaining_keys = [
                k
                for k in self.labeling_state[batch_job_id].keys()
                if k.startswith("item_")
            ]
            if not remaining_keys:
                del self.labeling_state[batch_job_id]
                log_info(
                    f"Removed empty (or helper-only) labeling state for batch '{batch_job_id}'."
                )

    # --- prepare_audio_input remains the same ---
    def _prepare_audio_input(
        self,
        input_source: str,
        item_work_dir: Path,
        log_file_handle: TextIO,
        session_id: str,
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:  # Return Optional Path/Dict
        """Downloads or copies audio and gets metadata."""
        log_info(f"[{session_id}] Preparing audio input from: {input_source}")
        audio_path: Optional[Path] = None
        metadata: Optional[Dict[str, Any]] = None  # Initialize as Optional

        try:
            if input_source.startswith(("http://", "https://")):
                # --- Make sure transcription module is initialized ---
                if not hasattr(self, "transcription") or self.transcription is None:
                    self.transcription = Transcription(
                        self.config
                    )  # Re-initialize if needed

                audio_path, metadata = self.transcription.download_audio_from_youtube(
                    input_source, str(item_work_dir), log_file_handle, session_id
                )
            else:
                # Handling local files
                src_path = Path(input_source)
                if not src_path.is_file():
                    raise ValueError(f"Invalid local input file path: {input_source}")
                unique_filename = f"{src_path.stem}_{session_id}{src_path.suffix}"
                dest_path = item_work_dir / unique_filename
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

            if audio_path is None or not audio_path.is_file():
                raise FileNotFoundError(
                    f"Could not obtain valid audio file from source: {input_source}"
                )

            log_info(f"[{session_id}] Audio prepared successfully at: {audio_path}")
            return audio_path, metadata

        except Exception as e:
            log_error(
                f"[{session_id}] Failed to prepare audio input from {input_source}: {e}"
            )
            log_error(traceback.format_exc())
            # Ensure we return None, None on failure
            return None, None

    # --- _run_initial_item_processing remains the same ---
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
            # --- Crucial Check: Stop if audio prep failed ---
            if audio_path_in_work_dir is None:
                raise RuntimeError(
                    f"Audio preparation failed for {input_source}, cannot proceed."
                )

            # 2. Duration Check
            min_duration = float(self.config.get("min_diarization_duration", 5.0))
            utils.run_ffprobe_duration_check(
                audio_path_in_work_dir, min_duration
            )  # Ignore return, just log warning

            # 3. Run WhisperX
            # --- Make sure transcription module is initialized ---
            if not hasattr(self, "transcription") or self.transcription is None:
                self.transcription = Transcription(
                    self.config
                )  # Re-initialize if needed
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
            if not segments:  # Handle case where structuring fails or yields empty list
                log_warning(
                    f"[{item_identifier}] No segments found after structuring WhisperX output."
                )
                # Still return audio_path and metadata, but segments will be None/empty

            # 5. Run Multimodal Analysis (Only if segments exist)
            if segments:
                # --- Ensure multimodal analysis module is initialized ---
                if not hasattr(self, "mm") or self.mm is None:
                    log_info(
                        "Re-initializing MultimodalAnalysis in _run_initial_item_processing..."
                    )
                    self.mm = MultimodalAnalysis(self.config)
                log_info(f"[{item_identifier}] Running multimodal emotion analysis...")
                # Determine video path - use input_source if URL, else None (or handle local video paths if added later)
                video_path_for_analysis = (
                    input_source if input_source.startswith(("http:", "https:")) else ""
                )  # Pass empty string if local audio
                segments = self.mm.analyze(
                    segments, str(audio_path_in_work_dir), video_path_for_analysis
                )
                log_info(f"[{item_identifier}] Multimodal analysis complete.")
            else:
                log_warning(
                    f"[{item_identifier}] Skipping multimodal analysis due to missing segments."
                )

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
            # Return None for segments, but keep audio_path/metadata if available
            return None, audio_path_in_work_dir, metadata

    # --- MODIFIED: Added batch_job_id argument to retrieve flags ---
    def _finalize_batch_item(
        self,
        segments: SegmentsList,
        speaker_labels: SpeakerLabels,
        item_identifier: str,
        item_work_path: Path,
        log_file_handle: TextIO,
        # --- Removed individual flags, pass batch_job_id instead ---
        batch_job_id: str,  # NEW: Pass batch ID
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Union[Path, List[Path]]]]:
        """Finalizes a single item: relabels, saves reports, returns file paths."""
        item_output_dir = item_work_path / "output"
        item_output_dir.mkdir(parents=True, exist_ok=True)
        generated_files_paths: Dict[str, Union[Path, List[Path]]] = {}

        # --- Retrieve output flags from batch state ---
        output_flags = self.labeling_state.get(batch_job_id, {}).get("output_flags", {})
        include_json_summary = output_flags.get(
            "include_json", self.config.get("include_json_summary", True)
        )
        include_csv_summary = output_flags.get(
            "include_csv", self.config.get("include_csv_summary", False)
        )
        include_script = output_flags.get(
            "include_script", self.config.get("include_script_transcript", False)
        )
        include_plots = output_flags.get(
            "include_plots", self.config.get("include_plots", False)
        )
        log_info(
            f"[{item_identifier}] Finalizing with flags: JSON={include_json_summary}, CSV={include_csv_summary}, Script={include_script}, Plots={include_plots}"
        )

        def item_log(level, message):
            # ... (item_log helper remains the same) ...
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
                item_log(
                    "warning",
                    "No segments provided for finalization, but proceeding to allow metadata/empty report saving.",
                )
                segments = []

            # --- Relabeling using provided speaker_labels map ---
            # (Logic remains the same)
            if speaker_labels:
                item_log(
                    "info",
                    f"Applying speaker labels based on mapping: {speaker_labels}",
                )
                segments_relabeled_count = 0
                for seg in segments:  # Safe even if segments is empty
                    original_speaker_id = str(seg.get("speaker", "unknown"))
                    if original_speaker_id in speaker_labels:
                        final_label = speaker_labels[original_speaker_id]
                        if final_label and str(final_label).strip():
                            seg["speaker"] = str(final_label).strip()
                            segments_relabeled_count += 1
                        else:
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

            # --- Save Final Structured Transcript ---
            # (Logic remains the same)
            final_json_name = (
                f"{item_identifier}_{Path(FINAL_STRUCTURED_TRANSCRIPT_NAME).name}"
            )
            final_json_path = item_output_dir / final_json_name
            item_log(
                "info", f"Saving final structured transcript to: {final_json_path}"
            )
            try:
                final_output_data: Dict[str, Any] = {
                    "segments": utils.convert_floats(segments if segments else []),
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

            # --- Generate Optional Report Outputs ---
            # (Logic remains the same, uses flags retrieved above)
            item_log("info", "Generating optional report outputs...")
            try:
                report_outputs = reporting.generate_item_report_outputs(
                    segments=segments,
                    item_identifier=item_identifier,
                    item_output_dir=item_output_dir,
                    config=self.config,
                    log_file_handle=log_file_handle,
                    include_json_summary=include_json_summary,  # Use retrieved flag
                    include_csv_summary=include_csv_summary,  # Use retrieved flag
                    include_script=include_script,  # Use retrieved flag
                    include_plots=include_plots,  # Use retrieved flag
                )
                generated_report_keys = []
                for key, path_or_list in report_outputs.items():
                    if path_or_list:
                        generated_files_paths[key] = path_or_list
                        generated_report_keys.append(
                            key
                        )  # Track which files were actually generated
                item_log("info", f"Generated report outputs: {generated_report_keys}")
            except Exception as e:
                item_log(
                    "error",
                    f"Error during optional report generation step: {e}\n{traceback.format_exc()}",
                )

            item_log("info", f"Finalization complete for item.")
            return generated_files_paths

        except Exception as e:
            item_log(
                "error",
                f"Unexpected error during finalization: {e}\n{traceback.format_exc()}",
            )
            return None

    # --- create_final_zip remains the same ---
    def create_final_zip(
        self,
        zip_path: Path,
        files_to_add: Dict[str, Path],
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
                for arc_name, local_path in files_to_add.items():
                    # Ensure local_path is a Path object
                    if not isinstance(local_path, Path):
                        try:
                            local_path = Path(local_path)
                        except TypeError:
                            log_warning(
                                f"{log_prefix}Invalid path type for archive name '{arc_name}', skipping: {local_path}"
                            )
                            files_skipped_count += 1
                            continue

                    if local_path.is_file():
                        try:
                            zf.write(local_path, arcname=arc_name)
                            files_added_count += 1
                        except Exception as e:
                            log_warning(
                                f"{log_prefix}Failed to add file {local_path} to zip as {arc_name}: {e}"
                            )
                            files_skipped_count += 1
                    else:
                        log_warning(
                            f"{log_prefix}File not found or is not a file, skipping: {local_path}"
                        )
                        files_skipped_count += 1

            if files_added_count == 0:
                log_error(
                    f"{log_prefix}No files were successfully added to the zip. Aborting zip creation."
                )
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                return None

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
            # Clear tracking AFTER zip attempt (success or fail)
            if batch_job_id and batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
                log_info(f"{log_prefix}Cleared output file tracking for batch.")

    # --- BATCH PROCESSING ENTRY POINT ---
    # --- MODIFIED: Store flags in batch state ---
    def process_batch_xlsx(
        self,
        xlsx_filepath: str,
        include_source_audio: bool,
        include_json_summary: bool,
        include_csv_summary: bool,
        include_script_transcript: bool,
        include_plots: bool,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Processes a batch defined in an Excel file. Handles initial processing
        and sets up state for interactive labeling if enabled and needed.
        Returns: Tuple[status_message, results_summary, batch_job_id | None]
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
        labeling_is_required_overall = False

        # Initialize tracking for this batch's output files
        self.batch_output_files[batch_job_id] = {}

        try:
            batch_work_path.mkdir(parents=True, exist_ok=True)
            batch_log_path = batch_work_path / LOG_FILE_NAME
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            log_info(f"[{batch_job_id}] Batch log file created at: {batch_log_path}")
            # Add the log file itself to the list of files to zip
            self.batch_output_files[batch_job_id][batch_log_path.name] = batch_log_path

            # --- batch_log helper function (remains the same) ---
            def batch_log(level, message):
                full_message = f"[{batch_job_id}] {message}"
                log_func = (
                    log_info
                    if level == "info"
                    else log_warning
                    if level == "warning"
                    else log_error
                )
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

            url_col = self.config.get("batch_url_column", "YouTube URL")
            if url_col not in df.columns:
                raise ValueError(
                    f"Required column '{url_col}' not found in the Excel file."
                )

            enable_labeling = self.config.get("enable_interactive_labeling", False)
            labeling_min_total_time = float(
                self.config.get("speaker_labeling_min_total_time", 15.0)
            )
            labeling_min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            batch_log("info", f"Interactive Labeling Enabled: {enable_labeling}")

            # --- Initialize state for this batch, including output flags ---
            self.labeling_state[batch_job_id] = {
                "output_flags": {
                    "include_audio": include_source_audio,
                    "include_json": include_json_summary,
                    "include_csv": include_csv_summary,
                    "include_script": include_script_transcript,
                    "include_plots": include_plots,
                },
                "items_requiring_labeling_order": [],  # Initialize empty list for item order
            }
            batch_log(
                "info",
                f"Stored output flags for batch: {self.labeling_state[batch_job_id]['output_flags']}",
            )

            # --- Item Processing Loop ---
            items_requiring_labeling_list = []  # Keep track of items needing labeling
            for sequential_index, (index, row) in enumerate(df.iterrows()):
                item_index = sequential_index + 1
                item_identifier = f"item_{item_index:03d}"
                batch_log(
                    "info",
                    f"--- Processing item {item_index}/{total_items} ({item_identifier}) ---",
                )

                source_url_or_path = row.get(url_col)
                if (
                    not isinstance(source_url_or_path, str)
                    or not source_url_or_path.strip()
                ):
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Skipping row {item_index}: Invalid or missing source URL/Path.",
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Skipped: Invalid Source."
                    )
                    failed_count += 1
                    continue
                source_url_or_path = source_url_or_path.strip()
                is_youtube_url = source_url_or_path.startswith(("http:", "https:"))

                item_work_path = batch_work_path / item_identifier
                item_work_path.mkdir(exist_ok=True)

                # --- Run Initial Processing ---
                segments, audio_path, metadata = self._run_initial_item_processing(
                    source_url_or_path, item_work_path, log_file_handle, item_identifier
                )

                if audio_path is None and segments is None:
                    batch_log(
                        "error",
                        f"[{item_identifier}] Critical failure during initial processing (likely audio prep).",
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Critical processing error."
                    )
                    failed_count += 1
                    continue

                # --- Check if Labeling Needed ---
                needs_labeling = False
                eligible_speakers = []
                if enable_labeling and is_youtube_url and segments:
                    eligible_speakers = speaker_labeling.identify_eligible_speakers(
                        segments, labeling_min_total_time, labeling_min_block_time
                    )
                    if eligible_speakers:
                        needs_labeling = True
                        labeling_is_required_overall = True
                        pending_labeling_count += 1
                        batch_results_list.append(
                            f"[{item_identifier}] Success (Pending Labeling)."
                        )
                        items_requiring_labeling_list.append(item_identifier)
                        batch_log(
                            "info",
                            f"[{item_identifier}] Item requires interactive labeling for speakers: {eligible_speakers}.",
                        )

                        # Store item state under the batch ID
                        self.labeling_state[batch_job_id][item_identifier] = {
                            "youtube_url": source_url_or_path,
                            "segments": segments,
                            "eligible_speakers": eligible_speakers,
                            "collected_labels": {},
                            "audio_path": audio_path,
                            "metadata": metadata,
                            "item_work_path": item_work_path,
                        }
                        # Add source audio to zip list *if* pending labeling and flag is true
                        if include_source_audio and audio_path and audio_path.is_file():
                            self.batch_output_files[batch_job_id][
                                f"{item_identifier}/{audio_path.name}"
                            ] = audio_path
                    else:
                        batch_log(
                            "info",
                            f"[{item_identifier}] Labeling enabled, but no eligible speakers found.",
                        )
                elif enable_labeling and not is_youtube_url:
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Labeling enabled, but input is not YouTube URL. Skipping labeling.",
                    )
                elif enable_labeling and not segments:
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Labeling enabled, but no segments found. Skipping labeling.",
                    )

                # --- Finalize Immediately OR Defer ---
                if not needs_labeling:
                    batch_log(
                        "info", f"[{item_identifier}] Finalizing item immediately."
                    )
                    # --- Pass batch_job_id to retrieve flags ---
                    generated_files = self._finalize_batch_item(
                        segments=segments if segments else [],
                        speaker_labels={},
                        item_identifier=item_identifier,
                        item_work_path=item_work_path,
                        log_file_handle=log_file_handle,
                        batch_job_id=batch_job_id,  # Pass batch ID here
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
                        # Add generated files to zip list
                        if batch_job_id in self.batch_output_files:
                            arc_folder_base = item_identifier
                            for key, path_or_list in generated_files.items():
                                # Determine the correct subfolder in the zip
                                arc_folder = (
                                    f"{arc_folder_base}/plots"
                                    if key == "plot_paths"
                                    else arc_folder_base
                                )
                                if key == "plot_paths" and isinstance(
                                    path_or_list, list
                                ):
                                    for p_path in path_or_list:
                                        if (
                                            isinstance(p_path, Path)
                                            and p_path.is_file()
                                        ):  # Check type and existence
                                            self.batch_output_files[batch_job_id][
                                                f"{arc_folder}/{p_path.name}"
                                            ] = p_path
                                elif (
                                    isinstance(path_or_list, Path)
                                    and path_or_list.is_file()
                                ):  # Check type and existence
                                    self.batch_output_files[batch_job_id][
                                        f"{arc_folder}/{path_or_list.name}"
                                    ] = path_or_list
                            # Add source audio if needed
                            if (
                                include_source_audio
                                and audio_path
                                and audio_path.is_file()
                            ):
                                self.batch_output_files[batch_job_id][
                                    f"{arc_folder_base}/{audio_path.name}"
                                ] = audio_path
                        else:
                            batch_log(
                                "warning",
                                f"[{item_identifier}] Cannot add output files to zip collection - batch ID '{batch_job_id}' not found.",
                            )

                batch_log(
                    "info",
                    f"--- Finished item {item_index}/{total_items} ({item_identifier}) ---",
                )
            # --- End of Item Loop ---

            # Store the order of items needing labeling in the batch state
            if labeling_is_required_overall and batch_job_id in self.labeling_state:
                self.labeling_state[batch_job_id]["items_requiring_labeling_order"] = (
                    items_requiring_labeling_list
                )

            total_processed_or_pending = (
                processed_immediately_count + pending_labeling_count
            )
            batch_log("info", f"Batch processing loop complete.")
            batch_log(
                "info", f"  Items Finalized Immediately: {processed_immediately_count}"
            )
            batch_log(
                "info",
                f"  Items Pending Labeling:      {pending_labeling_count} ({items_requiring_labeling_list})",
            )
            batch_log("info", f"  Items Failed/Skipped:        {failed_count}")

            if total_processed_or_pending == 0 and failed_count > 0:
                raise RuntimeError(
                    "No items were processed successfully or queued for labeling."
                )

            # --- Determine Final Status Message ---
            if labeling_is_required_overall:
                batch_status_message = f"[{batch_job_id}] Initial processing complete. {pending_labeling_count} item(s) require speaker labeling via the UI."
                return_batch_id = batch_job_id
            else:
                batch_log(
                    "info", f"No interactive labeling required. Creating final ZIP."
                )
                permanent_output_dir = Path(self.config.get("output_dir", "./output"))
                master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
                master_zip_path = permanent_output_dir / master_zip_name
                created_zip_path = self.create_final_zip(
                    master_zip_path,
                    self.batch_output_files.get(batch_job_id, {}),
                    log_file_handle,
                    batch_job_id,
                )
                if created_zip_path:
                    batch_status_message = f"[{batch_job_id}] ✅ Batch processing complete. Download ready: {created_zip_path}"
                else:
                    batch_status_message = f"[{batch_job_id}] ❗️ Batch processing finished, but failed to create final ZIP bundle."
                return_batch_id = (
                    None  # No labeling, so no batch ID needed for UI state
                )

            batch_log("info", f"Batch Status: {batch_status_message}")

        # --- Exception Handling and Finally Block (remain largely the same) ---
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            err_msg = f"[{batch_job_id}] ERROR: {e}"
            batch_log("error", err_msg)
            batch_status_message = err_msg
            return_batch_id = None
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
        except Exception as e:
            err_msg = f"[{batch_job_id}] An unexpected error occurred: {e}"
            batch_log("error", err_msg + "\n" + traceback.format_exc())
            batch_status_message = err_msg
            return_batch_id = None
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
        finally:
            # --- Finally Block ---
            if (
                log_file_handle
                and not log_file_handle.closed
                and not labeling_is_required_overall  # Close log only if NOT pending labeling
            ):
                batch_log("info", f"Closing batch log file (no labeling required).")
                log_file_handle.close()
            elif (
                log_file_handle
                and not log_file_handle.closed
                and labeling_is_required_overall
            ):
                log_info(
                    f"[{batch_job_id}] Keeping batch log file open for interactive labeling."
                )

            cleanup_temp = self.config.get("cleanup_temp_on_success", True)
            batch_succeeded_without_labeling = (
                (
                    "✅" in batch_status_message or "❗️" in batch_status_message
                )  # Check for success OR zip failure
                and not labeling_is_required_overall
            )
            # Cleanup if configured AND (batch succeeded without labeling OR batch failed entirely)
            should_cleanup = cleanup_temp and (
                batch_succeeded_without_labeling or return_batch_id is None
            )

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
                        # Use batch_log helper if available, else print
                        log_func = (
                            batch_log
                            if log_file_handle and not log_file_handle.closed
                            else print
                        )
                        log_func(
                            "warning",
                            f"Failed to remove batch temporary directory {batch_work_path}: {e}",
                        )
                elif labeling_is_required_overall:
                    batch_log(
                        "info",
                        f"Keeping temporary directory for interactive labeling: {batch_work_path}",
                    )
                else:  # Keep on other errors if not cleaning up on success
                    batch_log(
                        "warning",
                        f"Skipping cleanup of temporary directory due to errors or config: {batch_work_path}",
                    )
            # --- End Finally Block ---

        results_summary_string = (
            f"Batch Summary ({batch_job_id}):\n"
            f"- Total Items: {total_items}\n"
            f"- Finalized Immediately: {processed_immediately_count}\n"
            f"- Pending Labeling: {pending_labeling_count}\n"
            f"- Failed/Skipped: {failed_count}\n"
            f"--------------------\n"
            + "\n".join(batch_results_list)
            + f"\n--------------------\nOverall Status: {batch_status_message}"
        )
        return batch_status_message, results_summary_string, return_batch_id

    # --- start_interactive_labeling_for_item remains the same ---
    def start_interactive_labeling_for_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[
        Tuple[str, str, List[float]]
    ]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
        """
        Prepares and returns data for the first speaker to be labeled for an item.
        Gets preview start times.
        """
        log_info(f"[{batch_job_id}-{item_identifier}] Starting interactive labeling...")
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return None

        eligible_speakers = item_state.get("eligible_speakers", [])  # Already sorted
        youtube_url = item_state.get("youtube_url")

        if not eligible_speakers:
            log_warning(
                f"[{batch_job_id}-{item_identifier}] No eligible speakers found in state. Attempting to finalize."
            )
            self.finalize_labeled_item(batch_job_id, item_identifier)  # Try finalize
            # No need to remove state here, finalize_labeled_item does it
            return None
        if not youtube_url:
            log_error(
                f"[{batch_job_id}-{item_identifier}] YouTube URL missing from state."
            )
            self._remove_item_state(
                batch_job_id, item_identifier
            )  # Remove broken state
            return None

        first_speaker_id = eligible_speakers[0]
        log_info(
            f"[{batch_job_id}-{item_identifier}] First speaker to label: {first_speaker_id}"
        )

        preview_duration = float(
            self.config.get("speaker_labeling_preview_duration", 5.0)
        )
        min_block_time = float(self.config.get("speaker_labeling_min_block_time", 10.0))
        start_times = speaker_labeling.select_preview_time_segments(
            speaker_id=first_speaker_id,
            segments=item_state.get("segments", []),
            preview_duration=preview_duration,
            min_block_time=min_block_time,
        )  # Returns List[int]

        if not start_times:
            log_warning(
                f"[{batch_job_id}-{item_identifier}] Could not select preview start times for {first_speaker_id}."
            )
            # Return speaker ID and URL, but empty times list
            return first_speaker_id, youtube_url, []

        # No download needed, just return speaker, URL, and start times
        return first_speaker_id, youtube_url, start_times

    # --- store_speaker_label remains the same ---
    def store_speaker_label(
        self, batch_job_id: str, item_identifier: str, speaker_id: str, user_label: str
    ) -> bool:
        """Stores the user-provided label for a speaker."""
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return False

        if "collected_labels" not in item_state:
            item_state["collected_labels"] = {}
        item_state["collected_labels"][speaker_id] = user_label
        log_info(
            f"[{batch_job_id}-{item_identifier}] Stored label for {speaker_id}: '{user_label}'"
        )
        return True

    # --- get_next_speaker_for_labeling remains the same ---
    def get_next_speaker_for_labeling(
        self, batch_job_id: str, item_identifier: str, current_speaker_index: int
    ) -> Optional[
        Tuple[str, str, List[float]]
    ]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
        """Gets the ID, URL, and start times for the next speaker, or None if done for item."""
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            return None

        eligible_speakers = item_state.get("eligible_speakers", [])  # Sorted list
        youtube_url = item_state.get("youtube_url")
        if not youtube_url:
            log_error(
                f"[{batch_job_id}-{item_identifier}] YouTube URL missing from state for next speaker."
            )
            return None  # Cannot proceed

        next_speaker_index = current_speaker_index + 1

        if next_speaker_index < len(eligible_speakers):
            next_speaker_id = eligible_speakers[next_speaker_index]
            log_info(
                f"[{batch_job_id}-{item_identifier}] Getting data for next speaker (Index {next_speaker_index}): {next_speaker_id}"
            )

            preview_duration = float(
                self.config.get("speaker_labeling_preview_duration", 5.0)
            )
            min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            start_times = speaker_labeling.select_preview_time_segments(
                speaker_id=next_speaker_id,
                segments=item_state.get("segments", []),
                preview_duration=preview_duration,
                min_block_time=min_block_time,
            )

            if not start_times:
                log_warning(
                    f"[{batch_job_id}-{item_identifier}] Could not select preview start times for {next_speaker_id}."
                )
                return next_speaker_id, youtube_url, []

            return next_speaker_id, youtube_url, start_times
        else:
            log_info(
                f"[{batch_job_id}-{item_identifier}] All eligible speakers processed for this item."
            )
            return None  # Signal item completion

    # --- MODIFIED: Pass batch_job_id to _finalize_batch_item ---
    def finalize_labeled_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """Finalizes an item after interactive labeling is complete OR skipped."""
        log_info(f"[{batch_job_id}-{item_identifier}] Finalizing item...")
        item_state = self._get_item_state(batch_job_id, item_identifier)
        if not item_state:
            log_error(
                f"[{batch_job_id}-{item_identifier}] Cannot finalize - item state not found."
            )
            # Since state is gone, we can't really finalize. Return None.
            # Attempting to remove state again is harmless but redundant.
            return None

        segments = item_state.get("segments")
        collected_labels = item_state.get("collected_labels", {})
        item_work_path_obj = item_state.get("item_work_path")  # Use obj suffix
        metadata = item_state.get("metadata")
        audio_path_obj = item_state.get("audio_path")  # Use obj suffix

        if not item_work_path_obj or not isinstance(item_work_path_obj, Path):
            log_error(
                f"[{batch_job_id}-{item_identifier}] Cannot finalize - missing or invalid work path in state."
            )
            self._remove_item_state(batch_job_id, item_identifier)
            return None

        # Need log file handle (ensure batch dir exists for appending)
        batch_work_dir = item_work_path_obj.parent
        log_file_path = batch_work_dir / LOG_FILE_NAME
        log_handle = None
        generated_files: Optional[Dict[str, Any]] = None
        try:
            batch_work_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file_path, "a", encoding="utf-8")

            # --- Pass batch_job_id here ---
            generated_files = self._finalize_batch_item(
                segments=segments if segments else [],
                speaker_labels=collected_labels,
                item_identifier=item_identifier,
                item_work_path=item_work_path_obj,  # Pass Path object
                log_file_handle=log_handle,
                batch_job_id=batch_job_id,  # Pass batch ID here
                metadata=metadata,
            )

            # Add generated files to the main batch output collection
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
                            if (
                                isinstance(p_path, Path) and p_path.is_file()
                            ):  # Check type and existence
                                self.batch_output_files[batch_job_id][
                                    f"{arc_folder}/{p_path.name}"
                                ] = p_path
                    elif (
                        isinstance(path_or_list, Path) and path_or_list.is_file()
                    ):  # Check type and existence
                        self.batch_output_files[batch_job_id][
                            f"{arc_folder}/{path_or_list.name}"
                        ] = path_or_list

                # Retrieve the include_audio flag specific to this batch run
                batch_flags = self.labeling_state.get(batch_job_id, {}).get(
                    "output_flags", {}
                )
                include_source_audio = batch_flags.get(
                    "include_audio", self.config.get("include_source_audio", True)
                )

                if (
                    include_source_audio
                    and audio_path_obj
                    and isinstance(audio_path_obj, Path)
                    and audio_path_obj.is_file()
                ):
                    # Check if it's already added (might happen during initial processing if pending)
                    audio_arc_name = f"{arc_folder_base}/{audio_path_obj.name}"
                    if audio_arc_name not in self.batch_output_files[batch_job_id]:
                        self.batch_output_files[batch_job_id][audio_arc_name] = (
                            audio_path_obj
                        )
            elif not generated_files:
                log_error(
                    f"[{batch_job_id}-{item_identifier}] Finalization process failed."
                )
            else:  # generated_files is not None, but batch_job_id might be missing (shouldn't happen)
                log_warning(
                    f"[{batch_job_id}-{item_identifier}] Cannot add output files - batch ID not found in tracking."
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

        return generated_files

    # --- skip_labeling_for_item remains the same ---
    def skip_labeling_for_item(self, batch_job_id: str, item_identifier: str) -> bool:
        """
        Skips labeling for remaining speakers in an item and finalizes it
        with labels collected so far.
        """
        log_warning(
            f"[{batch_job_id}-{item_identifier}] User requested to skip remaining speakers."
        )
        # The finalization process automatically uses current labels and removes state.
        result = self.finalize_labeled_item(batch_job_id, item_identifier)
        return result is not None

    # --- MODIFIED: Refined completion check ---
    def check_batch_completion_and_zip(self, batch_job_id: str) -> Optional[Path]:
        """
        Checks if all items requiring labeling are done. If so, creates final zip.
        """
        batch_state = self.labeling_state.get(batch_job_id)

        # Check if batch state exists and if any actual item keys are left
        items_remaining = False
        if batch_state:
            # Check for keys that represent items (e.g., start with "item_")
            item_keys_present = [k for k in batch_state.keys() if k.startswith("item_")]
            if item_keys_present:
                items_remaining = True
                log_info(
                    f"[{batch_job_id}] Batch labeling not yet complete. Items remaining: {item_keys_present}"
                )

        if items_remaining:
            return None  # Not complete

        # Otherwise, batch state is gone or has no item keys left -> Finalize
        log_info(
            f"[{batch_job_id}] All items finalized or batch complete. Proceeding with ZIP creation."
        )
        batch_work_path = Path(self.config.get("temp_dir", "./temp")) / batch_job_id
        log_file_path = batch_work_path / LOG_FILE_NAME
        log_handle = None
        zip_path: Optional[Path] = None
        try:
            # Ensure batch work dir exists before trying to open log for appending
            batch_work_path.mkdir(parents=True, exist_ok=True)
            # Open log in append mode ('a') to add zip messages
            log_handle = open(log_file_path, "a", encoding="utf-8")

            permanent_output_dir = Path(self.config.get("output_dir", "./output"))
            master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
            master_zip_path = permanent_output_dir / master_zip_name
            files_for_zip = self.batch_output_files.get(batch_job_id, {})

            zip_path = self.create_final_zip(
                master_zip_path, files_for_zip, log_handle, batch_job_id
            )

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
                    # Use log_handle if available for logging cleanup error
                    log_func = (
                        log_handle.write
                        if log_handle and not log_handle.closed
                        else print
                    )
                    log_func(
                        f"WARNING: Failed to remove batch temporary directory {batch_work_path}: {e}\n"
                    )
            elif zip_path and not cleanup_temp:
                log_info(
                    f"[{batch_job_id}] Skipping cleanup of temporary directory as per config: {batch_work_path}"
                )
            elif not zip_path and batch_work_path.exists():
                log_warning(
                    f"[{batch_job_id}] Keeping temporary directory due to ZIP creation failure: {batch_work_path}"
                )

        except Exception as e:
            log_error(
                f"[{batch_job_id}] Error during final zip creation or cleanup check: {e}\n{traceback.format_exc()}"
            )
            if log_handle and not log_handle.closed:
                log_handle.write(f"ERROR: Error during final zip creation: {e}\n")
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()
            # Clean up batch state if it still exists (e.g., only helper keys left)
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
            # Note: batch_output_files is cleaned up within create_final_zip

        return zip_path
