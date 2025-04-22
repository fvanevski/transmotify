# core/pipeline.py
import shutil
import traceback
import zipfile
import json
import os  # <--- ADDED IMPORT
from collections import defaultdict
from datetime import (
    datetime,
)  # <--- ENSURED DATETIME IMPORT (needed for logging helpers)
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, TextIO, Tuple

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

# Optional fallback diarizer (Pyannote)
try:
    from pyannote.audio import Pipeline as PyannotePipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    log_warning("Pyannote.audio not found. Fallback diarization will not be available.")
    PyannotePipeline = None
    PYANNOTE_AVAILABLE = False
except Exception as e:
    log_warning(
        f"Failed to import Pyannote.audio: {e}. Fallback diarization unavailable."
    )
    PyannotePipeline = None
    PYANNOTE_AVAILABLE = False

# Type Hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]
EmotionSummary = Dict[str, Dict[str, Any]]
SpeakerMapping = Optional[Dict[str, str]]


class Pipeline:
    """
    Orchestrates the end-to-end speech processing workflow for batch processing.
    Focuses on sequence: audio prep -> transcription -> analysis -> labeling -> reporting -> packaging.
    Speaker labeling via snippets is DEPRECATED. Placeholder for video-based labeling exists.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the pipeline components."""
        self.config = config
        self.transcription = Transcription(config)
        log_info("Initializing MultimodalAnalysis in Pipeline...")
        self.mm = MultimodalAnalysis(config)
        self.diarizer = None
        if PYANNOTE_AVAILABLE and PyannotePipeline:
            try:
                pyannote_model_source = config.get(
                    "pyannote_diarization_model", "pyannote/speaker-diarization-3.1"
                )
                hf_token = config.get("hf_token")
                if hf_token:
                    self.diarizer = PyannotePipeline.from_pretrained(
                        pyannote_model_source, use_auth_token=hf_token
                    )
                    log_info(
                        f"Pyannote diarization model '{pyannote_model_source}' loaded successfully."
                    )
                else:
                    log_warning(
                        "Pyannote diarization model requires HF token, but none found. Skipping load."
                    )
            except Exception as e:
                log_warning(
                    f"Failed to load Pyannote diarization model '{pyannote_model_source}': {e}"
                )
                self.diarizer = None

    def _prepare_audio_input(
        self,
        input_source: str,
        item_work_dir: Path,
        log_file_handle: TextIO,
        session_id: str,
    ) -> Path:
        # (No changes to this method's logic)
        log_info(f"[{session_id}] Preparing audio input from: {input_source}")
        audio_path: Optional[Path] = None
        if input_source.startswith(("http://", "https://")):
            try:
                audio_path = self.transcription.download_audio_from_youtube(
                    input_source, str(item_work_dir), log_file_handle, session_id
                )
            except (RuntimeError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"Audio download/conversion failed for URL: {input_source}"
                ) from e
        else:
            src_path = Path(input_source)
            if not src_path.is_file():
                raise ValueError(f"Invalid local input file path: {input_source}")
            unique_filename = f"{src_path.stem}_{session_id}{src_path.suffix}"
            dest_path = item_work_dir / unique_filename
            try:
                log_info(f"[{session_id}] Copying local file {src_path} to {dest_path}")
                shutil.copy(str(src_path), str(dest_path))
                audio_path = dest_path
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
        return audio_path

    def _process_batch_item(
        self,
        input_source: str,
        item_work_path: Path,
        log_file_handle: TextIO,
        item_identifier: str,
    ) -> Tuple[Optional[SegmentsList], Optional[Path], SpeakerMapping]:
        # (No changes to this method's logic)
        audio_path_in_work_dir: Optional[Path] = None
        segments: Optional[SegmentsList] = None
        speaker_mapping: SpeakerMapping = None
        whisperx_json_path: Optional[Path] = None
        try:
            output_temp_dir = item_work_path / "output"
            output_temp_dir.mkdir(parents=True, exist_ok=True)
            audio_path_in_work_dir = self._prepare_audio_input(
                input_source, item_work_path, log_file_handle, item_identifier
            )
            min_duration = float(self.config.get("min_diarization_duration", 5.0))
            duration_ok = utils.run_ffprobe_duration_check(
                audio_path_in_work_dir, min_duration
            )
            if not duration_ok:
                log_warning(
                    f"[{item_identifier}] Audio duration potentially too short."
                )
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
            log_info(f"[{item_identifier}] Structuring WhisperX output...")
            segments = self.transcription.convert_json_to_structured(whisperx_json_path)
            log_info(f"[{item_identifier}] Structured {len(segments)} segments.")
            log_info(f"[{item_identifier}] Running multimodal emotion analysis...")
            segments = self.mm.analyze(
                segments, str(audio_path_in_work_dir), input_source
            )
            log_info(f"[{item_identifier}] Multimodal analysis complete.")
            log_info(
                f"[{item_identifier}] Placeholder for video preview speaker labeling."
            )
            speaker_mapping = {}
            log_info(
                f"[{item_identifier}] Speaker mapping (currently empty): {speaker_mapping}"
            )
            return segments, audio_path_in_work_dir, speaker_mapping
        except Exception as e:
            err_msg = f"[{item_identifier}] ERROR during batch item processing: {e}"
            log_error(err_msg)
            log_error(traceback.format_exc())
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_file_handle.write(
                        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {err_msg}\n{traceback.format_exc()}\n"
                    )
                    log_file_handle.flush()
                except Exception as log_e:
                    print(
                        f"WARN: Failed to write item processing error to log file: {log_e}"
                    )
            return None, None, None

    def _finalize_batch_item(
        self,
        segments: SegmentsList,
        speaker_mapping: SpeakerMapping,
        item_identifier: str,
        item_work_path: Path,
        log_file_handle: TextIO,
        include_json_summary: bool,
        include_csv_summary: bool,
        include_script: bool,
        include_plots: bool,
    ) -> Optional[Dict[str, Any]]:
        # (No changes to this method's logic, relies on imported datetime)
        item_output_dir = item_work_path / "output"
        item_output_dir.mkdir(parents=True, exist_ok=True)

        def item_log(level, message):
            full_message = f"[{item_identifier}] {message}"
            log_func = log_info
            if level == "warning":
                log_func = log_warning
            elif level == "error":
                log_func = log_error
            log_func(full_message)
            if log_file_handle and not log_file_handle.closed:
                try:
                    # Uses datetime here
                    log_file_handle.write(
                        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                    )
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
            if speaker_mapping:
                item_log(
                    "info",
                    f"Applying speaker labels based on mapping: {speaker_mapping}",
                )
                segments_relabeled_count = 0
                for seg in segments:
                    original_speaker_id = seg.get("speaker", "unknown")
                    if original_speaker_id in speaker_mapping:
                        final_label = speaker_mapping[original_speaker_id]
                        if final_label:
                            seg["speaker"] = final_label
                            segments_relabeled_count += 1
                item_log(
                    "info",
                    f"Applied speaker labels to {segments_relabeled_count} segments.",
                )
            else:
                item_log(
                    "info",
                    "No speaker mapping provided (video labeling pending). Keeping original SPEAKER_XX IDs.",
                )
            final_json_name = (
                f"{item_identifier}_{Path(FINAL_STRUCTURED_TRANSCRIPT_NAME).name}"
            )
            final_json_path = item_output_dir / final_json_name
            item_log(
                "info", f"Saving final structured transcript to: {final_json_path}"
            )
            generated_files_paths = {}
            try:
                segments_final_serializable = utils.convert_floats(segments)
                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        segments_final_serializable, f, indent=2, ensure_ascii=False
                    )
                item_log("info", "Final structured transcript saved successfully.")
                generated_files_paths["final_structured_json"] = final_json_path
            except Exception as e:
                item_log(
                    "error",
                    f"Failed to save final structured transcript: {e}\n{traceback.format_exc()}",
                )
                return None
            item_log("info", "Generating optional report outputs...")
            try:
                report_outputs = reporting.generate_item_report_outputs(
                    segments=segments,
                    item_identifier=item_identifier,
                    item_output_dir=item_output_dir,
                    config=self.config,
                    log_file_handle=log_file_handle,
                    include_json_summary=include_json_summary,
                    include_csv_summary=include_csv_summary,
                    include_script=include_script,
                    include_plots=include_plots,
                )
                generated_files_paths.update(report_outputs)
                item_log(
                    "info", f"Generated report outputs: {list(report_outputs.keys())}"
                )
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

    def create_final_zip(
        self,
        zip_path: Path,
        files_to_add: Dict[str, Path],
        log_file_handle: Optional[TextIO] = None,
        batch_job_id: Optional[str] = None,
    ) -> Optional[Path]:
        # (No changes to this method's logic, relies on imported os)
        log_prefix = f"[{batch_job_id}] " if batch_job_id else ""
        log_info(f"{log_prefix}Attempting to create final ZIP archive: {zip_path}")
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
        # Uses os.getpid() here
        temp_zip_path = zip_path.with_suffix(f".{os.getpid()}.temp.zip")
        files_added_count = 0
        files_skipped_count = 0
        try:
            with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
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
                    f"{log_prefix}No files were successfully added to the zip. Aborting."
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

    def process_batch_xlsx(self, xlsx_filepath: str) -> Tuple[str, str]:
        # (No changes to this method's logic, relies on imported datetime)
        batch_job_id = f"batch-{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]}"
        log_info(f"[{batch_job_id}] Starting batch processing for: {xlsx_filepath}")
        batch_status_message = f"[{batch_job_id}] Reading batch file..."
        batch_results_list: List[str] = []
        base_temp_dir = Path(self.config.get("temp_dir", "./temp"))
        batch_work_path = base_temp_dir / batch_job_id
        log_file_handle: Optional[TextIO] = None
        all_output_files_for_zip: Dict[str, Path] = {}
        total_items = 0
        processed_count = 0
        failed_count = 0
        try:
            batch_work_path.mkdir(parents=True, exist_ok=True)
            batch_log_path = batch_work_path / LOG_FILE_NAME
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            log_info(f"[{batch_job_id}] Batch log file created at: {batch_log_path}")
            all_output_files_for_zip[batch_log_path.name] = batch_log_path

            def batch_log(level, message):
                full_message = f"[{batch_job_id}] {message}"
                log_func = log_info
                if level == "warning":
                    log_func = log_warning
                elif level == "error":
                    log_func = log_error
                log_func(full_message)
                if log_file_handle and not log_file_handle.closed:
                    try:
                        # Uses datetime here
                        log_file_handle.write(
                            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                        )
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
            batch_log("info", f"Using URL column: '{url_col}'")
            snippet_col_name = self.config.get(
                "batch_snippet_column", "Speaker Snippets"
            )
            if snippet_col_name in df.columns:
                log_info(
                    f"Note: Snippet column '{snippet_col_name}' found but will be ignored."
                )
            # Read flags from config
            include_json_summary = self.config.get("include_json_summary", True)
            include_csv_summary = self.config.get("include_csv_summary", False)
            include_script = self.config.get("include_script_transcript", False)
            include_plots = self.config.get("include_plots", False)
            include_audio_in_zip = self.config.get("include_source_audio", True)
            batch_log(
                "info",
                f"Optional outputs - JSON Summary: {include_json_summary}, CSV Summary: {include_csv_summary}, Script: {include_script}, Plots: {include_plots}, Source Audio: {include_audio_in_zip}",
            )
            for index, row in df.iterrows():
                item_index = index + 1
                item_identifier = f"item_{item_index:03d}"
                batch_log(
                    "info",
                    f"--- Processing item {item_index}/{total_items} ({item_identifier}) ---",
                )
                youtube_url_raw = row.get(url_col)
                if not isinstance(
                    youtube_url_raw, str
                ) or not youtube_url_raw.strip().startswith(("http:", "https:")):
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Skipping row {item_index}: Invalid or missing URL ('{youtube_url_raw}').",
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Skipped: Invalid URL."
                    )
                    failed_count += 1
                    continue
                youtube_url = youtube_url_raw.strip()
                item_work_path = batch_work_path / item_identifier
                item_work_path.mkdir(exist_ok=True)
                segments, audio_path, speaker_mapping = self._process_batch_item(
                    youtube_url, item_work_path, log_file_handle, item_identifier
                )
                if segments is None or audio_path is None:
                    batch_log("error", f"[{item_identifier}] Core processing failed.")
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Core processing error."
                    )
                    failed_count += 1
                    continue
                generated_files = self._finalize_batch_item(
                    segments=segments,
                    speaker_mapping=speaker_mapping,
                    item_identifier=item_identifier,
                    item_work_path=item_work_path,
                    log_file_handle=log_file_handle,
                    include_json_summary=include_json_summary,
                    include_csv_summary=include_csv_summary,
                    include_script=include_script,
                    include_plots=include_plots,
                )
                if generated_files is None:
                    batch_log(
                        "error", f"[{item_identifier}] Finalization failed critically."
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Finalization error."
                    )
                    failed_count += 1
                    continue
                processed_count += 1
                batch_results_list.append(f"[{item_identifier}] Success.")
                for key, path_or_list in generated_files.items():
                    arc_folder = (
                        f"{item_identifier}/plots"
                        if key == "plot_paths"
                        else item_identifier
                    )
                    if key == "plot_paths" and isinstance(path_or_list, list):
                        for p_path in path_or_list:
                            if p_path and p_path.is_file():
                                all_output_files_for_zip[
                                    f"{arc_folder}/{p_path.name}"
                                ] = p_path
                    elif isinstance(path_or_list, Path) and path_or_list.is_file():
                        f_path = path_or_list
                        all_output_files_for_zip[f"{arc_folder}/{f_path.name}"] = f_path
                if include_audio_in_zip and audio_path.is_file():
                    all_output_files_for_zip[f"{item_identifier}/{audio_path.name}"] = (
                        audio_path
                    )
                batch_log(
                    "info",
                    f"--- Finished item {item_index}/{total_items} ({item_identifier}) ---",
                )
            batch_log(
                "info",
                f"Batch processing loop complete. Processed: {processed_count}, Failed/Skipped: {failed_count}",
            )
            if processed_count == 0:
                raise RuntimeError("No items were processed successfully in the batch.")
            permanent_output_dir = Path(self.config.get("output_dir", "./output"))
            master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
            master_zip_path = permanent_output_dir / master_zip_name
            batch_log("info", f"Creating master ZIP bundle: {master_zip_path}")
            created_zip_path = self.create_final_zip(
                master_zip_path, all_output_files_for_zip, log_file_handle, batch_job_id
            )
            if created_zip_path:
                batch_status_message = f"[{batch_job_id}] ✅ Batch processing complete. Download ready: {created_zip_path}"
            else:
                batch_status_message = f"[{batch_job_id}] ❗️ Batch processing finished, but failed to create final ZIP bundle."
            batch_log("info", batch_status_message)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            err_msg = f"[{batch_job_id}] ERROR: {e}"
            batch_log("error", err_msg)
            batch_status_message = err_msg
        except Exception as e:
            err_msg = f"[{batch_job_id}] An unexpected error occurred during batch processing: {e}"
            batch_log("error", err_msg + "\n" + traceback.format_exc())
            batch_status_message = err_msg
        finally:
            if log_file_handle and not log_file_handle.closed:
                log_info(f"[{batch_job_id}] Closing batch log file.")
                log_file_handle.close()
            cleanup_temp = self.config.get("cleanup_temp_on_success", True)
            should_cleanup = cleanup_temp and "✅" in batch_status_message
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
                else:
                    batch_log(
                        "warning",
                        f"Skipping cleanup of temporary directory due to errors or config: {batch_work_path}",
                    )
        results_summary_string = (
            f"Batch Processing Summary ({batch_job_id}):\n"
            f"- Total Items Read: {total_items}\n"
            f"- Successfully Processed: {processed_count}\n"
            f"- Failed/Skipped: {failed_count}\n"
            f"--------------------\n"
            + "\n".join(batch_results_list)
            + f"\n--------------------\n"
            f"Overall Status: {batch_status_message}"
        )
        return batch_status_message, results_summary_string
