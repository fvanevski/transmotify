# core/orchestrator.py
# REVISED: Skips visual analysis call for YouTube URLs based on user clarification.
# REVISED: Integrates Riva ASR call.
"""
Orchestrates the end-to-end speech analysis pipeline, coordinating calls to
specialized modules for configuration, I/O, ASR, emotion analysis, speaker ID, etc.
Manages state for batch processing and interactive labeling.
"""

import shutil
import traceback
import json
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union, cast  # Added cast
import logging
from core.errors import (
    TransmotifyError,
    PipelineStageError,
    ConfigurationError,
    InputDataError,
    ProcessingError,
    ResourceAccessError,
)

logger = logging.getLogger(__name__)

# --- Core Components ---
from core.logging import log_error, log_info, log_warning  # Moved outside try

try:
    from config.config import Config
    from utils.file_manager import (
        create_directory,
        get_temp_file_path,
        save_text_file,
        cleanup_directory,
        create_zip_archive,
        copy_local_file,
    )
    from utils.transcripts import (
        parse_xlsx_snippets,
        group_segments_by_speaker,
        match_snippets_to_speakers,
        convert_floats,
        convert_json_to_structured,
        save_script_transcript,
    )
    from yt.downloader import download_youtube_stream
    from yt.converter import convert_to_wav, check_audio_duration
    from yt.metadata import fetch_youtube_metadata
    from asr.asr import run_riva_asr # Updated import for Riva ASR
    from emotion.text_model import TextEmotionModel
    from emotion.audio_model import AudioEmotionModel
    from emotion.visual_model import VisualEmotionModel
    from analysis.emotion_fusion import fuse_emotions
    from emotion.metrics import calculate_emotion_summary  # Corrected import source
    from analysis.visualization import generate_all_plots  # Removed incorrect import
    from speaker_id.id_mapping import SpeakerLabelMap, apply_speaker_labels
    from speaker_id.vid_preview_id import (
        identify_eligible_speakers,
        start_interactive_labeling_for_item,
        store_speaker_label,
        get_next_speaker_for_labeling,
        skip_labeling_for_item,
    )

    # Add missing transcript types
    from utils.transcripts import SegmentsList, Segment
except ImportError as e:
    logger.error(f"Orchestrator failed to import core components: {e}", exc_info=True)
    raise ResourceAccessError(f"Orchestrator failed to import core components: {e}") from e

# Type Aliases (Consider moving to a types file if complex)
BatchOutputFiles = Dict[
    str, Dict[str, Union[str, Path]]
]  # batch_job_id -> {file_type: path} # Match create_zip_archive
EmotionSummary = Dict[str, Dict[str, Any]]
SpeakerLabels = Optional[Dict[str, str]]
LabelingItemState = Dict[str, Any]
LabelingBatchState = Dict[str, Union[LabelingItemState, List[str], Dict[str, bool]]]
LabelingState = Dict[str, LabelingBatchState]


class Orchestrator:
    """Manages the speech analysis workflow, state, and interactions."""

    def __init__(self, config: Config):
        self.config = config
        self.labeling_state: LabelingState = {}
        self.batch_output_files: Dict[str, Dict[str, Path]] = {}
        logger.info("Orchestrator: Initializing emotion models...")
        self._init_emotion_models()
        logger.info("Orchestrator initialized.")

    def _init_emotion_models(self):
        """Helper to initialize emotion model instances."""
        self.text_emotion_model = TextEmotionModel(
            model_name=self.config.get(
                "transformers_text_emotion_model",
                "j-hartmann/emotion-english-distilroberta-base",
            ),
            device=self.config.get("device"),
        )
        self.audio_emotion_model = AudioEmotionModel(
            model_source=self.config.get(
                "audio_emotion_model",
                "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            ),
            device=self.config.get("device"),
        )
        self.visual_emotion_model = VisualEmotionModel(
            detector_backend=self.config.get("deepface_detector_backend", "opencv"),
            analysis_frame_rate=self.config.get("visual_frame_rate", 1),
            device=self.config.get("device"),
        )

    def _get_log_prefix(
        self, batch_job_id: Optional[str] = None, item_identifier: Optional[str] = None
    ) -> str:
        parts = []
        if batch_job_id:
            parts.append(batch_job_id)
        if item_identifier:
            parts.append(item_identifier)
        return f"[{'-'.join(parts)}]" if parts else "[Orchestrator]"

    def _save_json_summary(
        self, summary_data: Dict, output_path: Path, log_prefix: str
    ) -> Optional[Path]:
        logger.info(
            f"{log_prefix} Attempting to save detailed emotion summary JSON to: {output_path}"
        )
        try:
            summary_serializable = convert_floats(summary_data)
            if save_text_file(
                json.dumps(summary_serializable, indent=2, ensure_ascii=False),
                output_path,
            ):
                logger.info(
                    f"{log_prefix} Detailed emotion summary JSON saved successfully."
                )
                return output_path
            else:
                logger.error(
                    f"{log_prefix} Failed to save emotion summary JSON using file manager."
                )
                return None
        except Exception as e:
            logger.exception(
                f"{log_prefix} Failed to serialize or save emotion summary JSON to {output_path}: {e}"
            )
            return None

    def _save_csv_summary(
        self, summary_data: EmotionSummary, output_path: Path, log_prefix: str
    ) -> Optional[Path]:
        logger.info(
            f"{log_prefix} Attempting to save high-level emotion summary CSV to: {output_path}"
        )
        try:
            standard_headers = [
                "speaker",
                "total_segments",
                "dominant_emotion",
                "emotion_volatility",
                "emotion_score_mean",
                "emotion_transitions",
            ]
            all_emotion_count_keys = sorted(
                list(
                    set(
                        emotion
                        for speaker_data in summary_data.values()
                        for emotion in speaker_data.get("emotion_counts", {}).keys()
                    )
                )
            )
            final_headers = standard_headers + [
                f"count_{emo}" for emo in all_emotion_count_keys
            ]
            import io, csv

            output = io.StringIO()
            writer = csv.DictWriter(
                output, fieldnames=final_headers, extrasaction="ignore"
            )
            writer.writeheader()
            for speaker_id, data in summary_data.items():
                row_data = {"speaker": speaker_id}
                row_data.update(
                    {
                        h: str(data.get(h, ""))
                        for h in standard_headers
                        if h != "speaker"
                    }
                )
                emotion_counts = data.get("emotion_counts", {})
                row_data.update(
                    {
                        f"count_{emo}": str(emotion_counts.get(emo, 0))
                        for emo in all_emotion_count_keys
                    }
                )
                for key in ["emotion_volatility", "emotion_score_mean"]:
                    if key in row_data and row_data[key]:
                        try:
                            row_data[key] = f"{float(row_data[key]):.4f}"
                        except:
                            pass
                writer.writerow(row_data)
            if save_text_file(output.getvalue(), output_path):
                logger.info(
                    f"{log_prefix} High-level emotion summary CSV saved successfully."
                )
                return output_path
            else:
                logger.error(
                    f"{log_prefix} Failed to save emotion summary CSV using file manager."
                )
                return None
        except Exception as e:
            logger.exception(
                f"{log_prefix} Failed to generate or save emotion summary CSV to {output_path}: {e}"
            )
            return None

    # --- Main Batch Processing Method ---
    def process_batch(
        self,
        input_source: Union[str, Path],
        include_source_audio: bool,
        include_json_summary: bool,
        include_csv_summary: bool,
        include_script_transcript: bool,
        include_plots: bool,
    ) -> Tuple[str, str, Optional[str]]:
        batch_job_id = (
            f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]}"
        )
        log_prefix = self._get_log_prefix(batch_job_id)
        logger.info(f"{log_prefix} Starting batch processing for: {input_source}")

        batch_status_message = f"{log_prefix} Reading batch definition..."
        batch_results_list: List[str] = []
        base_temp_dir = Path(self.config.get("temp_dir", "./temp"))
        batch_work_path = base_temp_dir / batch_job_id
        log_file_handle: Optional[TextIO] = None
        total_items = 0
        processed_immediately_count = 0
        pending_labeling_count = 0
        failed_count = 0
        labeling_is_required_overall = False
        return_batch_id: Optional[str] = None
        total_processed_or_pending: int = 0
        self.batch_output_files[batch_job_id] = {}
        self.labeling_state[batch_job_id] = {
            "output_flags": {
                "include_audio": include_source_audio,
                "include_json": include_json_summary,
                "include_csv": include_csv_summary,
                "include_script": include_script_transcript,
                "include_plots": include_plots,
            },
            "items_requiring_labeling_order": [],
        }

        try:
            if not create_directory(batch_work_path):
                raise ResourceAccessError(
                    f"Failed to create batch work directory: {batch_work_path}"
                )
            batch_log_path = batch_work_path / self.config.get(
                "log_file_name", "process_log.txt"
            )
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            logger.info(f"{log_prefix} Batch log file created at: {batch_log_path}")
            self.batch_output_files[batch_job_id][batch_log_path.name] = batch_log_path

            if not Path(input_source).is_file() or not str(input_source).endswith(
                ".xlsx"
            ):
                raise InputDataError(
                    f"Invalid input source. Expecting an XLSX file path: {input_source}"
                )
            logger.info(f"{log_prefix} Reading batch file: {input_source}")
            try:
                df = pd.read_excel(input_source, sheet_name=0)
                total_items = len(df)
                url_col_name = df.columns[0]
                if df.empty:
                    raise ValueError("Excel file contains no data rows.")
                logger.info(
                    f"{log_prefix} Read {total_items} rows. Using column '{url_col_name}'."
                )
            except Exception as e:
                logger.exception(f"{log_prefix} Failed to read Excel file: {e}")
                raise InputDataError(f"Failed to read Excel file: {input_source}") from e

            enable_labeling = self.config.get("enable_interactive_labeling", True)
            labeling_min_total_time = float(
                self.config.get("speaker_labeling_min_total_time", 15.0)
            )
            labeling_min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            logger.info(f"{log_prefix} Interactive Labeling Enabled: {enable_labeling}")
            items_requiring_labeling_list = []

            for item_index, (sequential_index, row) in enumerate(
                df.iterrows(), start=1
            ):
                item_identifier = (
                    f"item_{item_index:03d}"
                )
                item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
                logger.info(
                    f"{item_log_prefix} --- Processing item {item_index}/{total_items} ---"
                )

                source_url_or_path = row.get(url_col_name)
                if (
                    not isinstance(source_url_or_path, str)
                    or not source_url_or_path.strip()
                ):
                    logger.warning(
                        f"{item_log_prefix} Skipping row {item_index}: Invalid or missing source."
                    )
                    batch_results_list.append(
                        f"[{item_identifier}] Skipped: Invalid Source."
                    )
                    failed_count += 1
                    continue
                source_url_or_path = source_url_or_path.strip()
                is_youtube_url = source_url_or_path.startswith(("http:", "https:"))
                is_local_file = not is_youtube_url and Path(source_url_or_path).exists()

                item_work_path = batch_work_path / item_identifier
                if not create_directory(item_work_path):
                    logger.error(
                        f"{item_log_prefix} Failed to create item work directory. Skipping."
                    )
                    failed_count += 1
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Cannot create work directory."
                    )
                    continue

                audio_input_path: Optional[Path] = None
                metadata: Optional[Dict[str, Any]] = None
                segments: Optional[SegmentsList] = None
                processed_segments: Optional[SegmentsList] = None

                try:
                    temp_audio_dir = item_work_path / "audio_temp"
                    create_directory(temp_audio_dir)
                    raw_dl_path = temp_audio_dir / f"raw_download_{item_identifier}"
                    final_wav_path = item_work_path / f"audio_{item_identifier}.wav"
                    if is_youtube_url:
                        logger.info(f"{item_log_prefix} Downloading YouTube stream...")
                        dl_stream_path = download_youtube_stream(
                            youtube_url=source_url_or_path,
                            output_path=raw_dl_path,
                            youtube_dl_format=self.config.get(
                                "youtube_dl_format", "bestaudio/best"
                            ),
                            log_file_handle=log_file_handle,
                            log_prefix=item_log_prefix,
                        )
                        if not dl_stream_path:
                            raise RuntimeError("YouTube download failed.")
                        logger.info(f"{item_log_prefix} Converting to WAV...")
                        audio_input_path = convert_to_wav(
                            input_path=dl_stream_path,
                            output_path=final_wav_path,
                            audio_channels=self.config.get("ffmpeg_audio_channels", 1),
                            audio_samplerate=self.config.get(
                                "ffmpeg_audio_samplerate", 16000
                            ),
                            log_file_handle=log_file_handle,
                            log_prefix=item_log_prefix,
                        )
                        if not audio_input_path:
                            raise RuntimeError("Audio conversion failed.")
                        dl_stream_path.unlink(missing_ok=True)
                        logger.info(f"{item_log_prefix} Fetching metadata...")
                        metadata = fetch_youtube_metadata(
                            youtube_url=source_url_or_path, log_prefix=item_log_prefix
                        )
                        if metadata:
                            metadata["prepared_audio_path"] = str(audio_input_path)
                        else:
                            logger.warning(f"{item_log_prefix} Failed to fetch metadata.")
                    elif is_local_file:
                        logger.info(f"{item_log_prefix} Processing local file...")
                        audio_input_path = convert_to_wav(
                            input_path=source_url_or_path,
                            output_path=final_wav_path,
                            audio_channels=self.config.get("ffmpeg_audio_channels", 1),
                            audio_samplerate=self.config.get(
                                "ffmpeg_audio_samplerate", 16000
                            ),
                            log_file_handle=log_file_handle,
                            log_prefix=item_log_prefix,
                        )
                        if not audio_input_path:
                            raise RuntimeError("Local audio conversion/copying failed.")
                        metadata = {
                            "source_path": str(source_url_or_path),
                            "filename": Path(source_url_or_path).name,
                            "prepared_audio_path": str(audio_input_path),
                            "source_type": "local_file",
                        }
                    else:
                        raise ValueError(
                            f"Input source is not a valid URL or existing local file: {source_url_or_path}"
                        )

                    min_duration = float(
                        self.config.get("min_diarization_duration", 5.0)
                    )
                    if not check_audio_duration(
                        audio_input_path, min_duration, item_log_prefix
                    ):
                        logger.warning(f"{item_log_prefix} Audio duration too short.")

                    asr_output_dir = item_work_path / "asr_output"
                    create_directory(asr_output_dir)
                    excluded_asr_outputs = [
                        self.config.get(k)
                        for k in [
                            "intermediate_structured_transcript_name",
                            "final_structured_transcript_name",
                            "emotion_summary_json_name",
                            "emotion_summary_csv_name",
                            "script_transcript_name",
                        ]
                    ]
                    # Updated to call run_riva_asr with new parameters from config
                    asr_json_path = run_riva_asr(
                        audio_path=audio_input_path,
                        output_dir=asr_output_dir,
                        riva_server_uri=self.config.get("riva_server_uri", "localhost:50051"),
                        language_code=self.config.get("riva_asr_language_code", "en-US"),
                        max_speakers_diarization=self.config.get("riva_max_speakers_diarization", 2),
                        enable_automatic_punctuation=self.config.get("riva_enable_automatic_punctuation", True),
                        # device parameter removed as it's not directly used by Riva client call
                        output_filename_exclusions=[
                            name for name in excluded_asr_outputs if name
                        ],
                        log_file_handle=log_file_handle,
                        log_prefix=item_log_prefix,
                    )
                    if not asr_json_path: # Use the new path variable
                        raise RuntimeError("ASR (Riva) execution failed.")

                    segments = convert_json_to_structured(asr_json_path) # Use the new path variable
                    if not segments:
                        logger.warning(
                            f"{item_log_prefix} No segments found after structuring ASR output."
                        )

                    if segments:
                        logger.info(
                            f"{item_log_prefix} Running emotion analysis for {len(segments)} segments..."
                        )
                        processed_segments = []
                        visual_emotion_map: Optional[Dict[int, Optional[str]]] = None
                        video_path_for_visual: Optional[Path] = None
                        if is_local_file:
                            vid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
                            local_path_obj = Path(source_url_or_path)
                            if local_path_obj.suffix.lower() in vid_extensions:
                                video_path_for_visual = local_path_obj

                        if video_path_for_visual:
                            logger.info(
                                f"{item_log_prefix} Running visual emotion analysis on local file: {video_path_for_visual.name}"
                            )
                            visual_emotion_map = (
                                self.visual_emotion_model.predict_video_segments(
                                    video_path_for_visual, segments
                                )
                            )
                        elif is_youtube_url:
                            logger.info(
                                f"{item_log_prefix} Skipping visual emotion analysis for YouTube URL (no local video file)."
                            )
                            visual_emotion_map = {}

                        for i, seg in enumerate(segments):
                            seg_log_prefix = f"{item_log_prefix} [Seg {i}]"
                            text_scores = self.text_emotion_model.predict(
                                seg.get("text", "")
                            )
                            seg["text_emotion"] = text_scores
                            audio_scores = self.audio_emotion_model.predict_segment(
                                audio_path=audio_input_path,
                                start_time=seg.get("start", 0.0),
                                end_time=seg.get("end", 0.0),
                            )
                            seg["audio_emotion"] = audio_scores
                            seg["visual_emotion"] = (
                                visual_emotion_map.get(i)
                                if visual_emotion_map
                                else None
                            )
                            fused_label, fused_conf, sig_text = fuse_emotions(
                                text_emotion_scores=text_scores,
                                audio_emotion_scores=audio_scores,
                                text_fusion_weight=self.config.get(
                                    "text_fusion_weight", 0.6
                                ),
                                audio_fusion_weight=self.config.get(
                                    "audio_fusion_weight", 0.4
                                ),
                                log_prefix=seg_log_prefix,
                            )
                            seg["emotion"] = fused_label
                            seg["fused_emotion_confidence"] = fused_conf
                            seg["significant_text_emotions"] = sig_text
                            processed_segments.append(seg)
                        logger.info(
                            f"{item_log_prefix} Emotion analysis and fusion complete."
                        )
                    else:
                        logger.warning(
                            f"{item_log_prefix} Skipping emotion analysis due to missing segments."
                        )
                        processed_segments = segments
                    segments = processed_segments

                except Exception as e:
                    err_msg = (
                        f"{item_log_prefix} ERROR during item processing pipeline: {e}"
                    )
                    logger.error(err_msg)
                    logger.exception(traceback.format_exc())
                    if log_file_handle:
                        log_file_handle.write(f"{err_msg}\n{traceback.format_exc()}\n")
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Pipeline error."
                    )
                    failed_count += 1
                    if item_work_path.exists():
                        try:
                            shutil.rmtree(item_work_path)
                        except Exception:
                            pass
                    continue

                needs_labeling = False
                eligible_speakers = []
                if enable_labeling and is_youtube_url and segments:
                    eligible_speakers = identify_eligible_speakers(
                        segments, labeling_min_total_time, labeling_min_block_time
                    )
                    if eligible_speakers:
                        needs_labeling = True
                        labeling_is_required_overall = True
                        pending_labeling_count += 1
                        items_requiring_labeling_list.append(item_identifier)
                        batch_results_list.append(
                            f"[{item_identifier}] Success (Pending Labeling)"
                        )
                        logger.info(
                            f"{item_log_prefix} Item requires labeling: {eligible_speakers}"
                        )
                        self.labeling_state[batch_job_id][item_identifier] = {
                            "youtube_url": source_url_or_path,
                            "segments": segments,
                            "eligible_speakers": eligible_speakers,
                            "collected_labels": {},
                            "audio_path": audio_input_path,
                            "metadata": metadata if metadata else {},
                            "item_work_path": item_work_path,
                            "status": "pending_labeling",
                        }
                        if (
                            include_source_audio
                            and audio_input_path
                            and audio_input_path.is_file()
                        ):
                            arc_name = f"{item_identifier}/{audio_input_path.name}"
                            self.batch_output_files[batch_job_id][
                                arc_name
                            ] = audio_input_path
                    else:
                        logger.info(
                            f"{item_log_prefix} Labeling enabled, but no eligible speakers."
                        )
                elif enable_labeling:
                    logger.info(
                        f"{item_log_prefix} Labeling enabled, but not YT URL or no segments."
                    )

                if not needs_labeling:
                    logger.info(f"{item_log_prefix} Finalizing item immediately.")
                    try:
                        finalized_files = self._finalize_batch_item(
                            batch_job_id=batch_job_id,
                            item_identifier=item_identifier,
                            segments=segments if segments else [],
                            speaker_labels={},
                            metadata=metadata if metadata else {},
                            audio_path=audio_input_path,
                            item_work_path=item_work_path,
                            log_file_handle=log_file_handle,
                        )
                        if finalized_files is None:
                            raise RuntimeError("Finalization helper returned None.")
                        processed_immediately_count += 1
                        batch_results_list.append(
                            f"[{item_identifier}] Success (Finalized)"
                        )
                    except Exception as e:
                        logger.exception(f"{item_log_prefix} Error during immediate finalization: {e}")
                        batch_results_list.append(
                            f"[{item_identifier}] Failed: Finalization error."
                        )
                        failed_count += 1
                        if item_work_path.exists():
                            try:
                                shutil.rmtree(item_work_path)
                            except Exception:
                                pass
                logger.info(
                    f"{item_log_prefix} --- Finished item {item_index}/{total_items} ---"
                )

            if labeling_is_required_overall:
                self.labeling_state[batch_job_id][
                    "items_requiring_labeling_order"
                ] = items_requiring_labeling_list
            total_processed_or_pending = (
                processed_immediately_count + pending_labeling_count
            )
            logger.info(
                f"{log_prefix} Batch loop complete. Finalized: {processed_immediately_count}, Pending: {pending_labeling_count}, Failed: {failed_count}"
            )
            if total_processed_or_pending == 0 and failed_count > 0:
                raise RuntimeError("No items processed successfully or queued.")

            if labeling_is_required_overall:
                batch_status_message = f"{log_prefix} Initial processing complete. {pending_labeling_count} item(s) require labeling."
                return_batch_id = batch_job_id
            else:
                logger.info(f"{log_prefix} No labeling needed. Creating final ZIP.")
                permanent_output_dir = Path(self.config.get("output_dir", "./output"))
                zip_suffix = self.config.get("final_zip_suffix", "_final_bundle.zip")
                master_zip_name = f"{batch_job_id}_batch_results{zip_suffix}"
                master_zip_path = permanent_output_dir / master_zip_name
                files_to_add_cast = cast(
                    Dict[str, Union[str, Path]],
                    self.batch_output_files.get(batch_job_id, {}),
                )
                created_zip_path = create_zip_archive(
                    zip_path=master_zip_path,
                    files_to_add=files_to_add_cast,
                    log_prefix=log_prefix,
                )
                batch_status_message = (
                    f"{log_prefix} ✅ Batch complete. Download: {created_zip_path}"
                    if created_zip_path
                    else f"{log_prefix} ❗️ Batch finished, but ZIP creation failed."
                )
                return_batch_id = None
            logger.info(f"{log_prefix} Batch Status: {batch_status_message}")

        except (FileNotFoundError, ValueError, ResourceAccessError, InputDataError) as e:
            err_msg = f"{log_prefix} BATCH ERROR: {e}"
            logger.error(err_msg)
            if log_file_handle:
                log_file_handle.write(f"{err_msg}\n")
            batch_status_message = err_msg
            return_batch_id = None
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
        except Exception as e:
            err_msg = f"{log_prefix} UNEXPECTED BATCH ERROR: {e}"
            logger.exception(err_msg)
            if log_file_handle:
                log_file_handle.write(f"{err_msg}\n{traceback.format_exc()}\n")
            batch_status_message = err_msg
            return_batch_id = None
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
        finally:
            should_close_log = (
                log_file_handle
                and not log_file_handle.closed
                and not labeling_is_required_overall
            )
            if should_close_log:
                logger.info(f"{log_prefix} Closing batch log file.")
                if log_file_handle:
                    log_file_handle.close()
            elif (
                log_file_handle
                and not log_file_handle.closed
                and labeling_is_required_overall
            ):
                logger.info(f"{log_prefix} Keeping batch log open.")
            cleanup_temp = self.config.get("cleanup_temp_on_success", False)
            batch_finished_without_labeling = (return_batch_id is None) and (
                total_processed_or_pending > 0 or failed_count == total_items
            )
            batch_failed_entirely = (return_batch_id is None) and (
                total_processed_or_pending == 0 and failed_count > 0
            )
            should_cleanup = cleanup_temp and (
                batch_finished_without_labeling or batch_failed_entirely
            )
            if batch_work_path.exists():
                if should_cleanup and not labeling_is_required_overall:
                    logger.info(
                        f"{log_prefix} Cleaning up batch temp dir: {batch_work_path}"
                    )
                    if not cleanup_directory(batch_work_path, recreate=False):
                        logger.warning(f"{log_prefix} Failed remove batch temp dir.")
                elif labeling_is_required_overall:
                    logger.info(f"{log_prefix} Keeping batch temp dir: {batch_work_path}")
                else:
                    logger.warning(
                        f"{log_prefix} Skipping cleanup of batch temp dir: {batch_work_path}"
                    )

        results_summary_string = (
            f"Batch Summary ({batch_job_id}):\n"
            + f"- Total Items: {total_items}\n"
            + f"- Finalized Immediately: {processed_immediately_count}\n"
            + f"- Pending Labeling: {pending_labeling_count}\n"
            + f"- Failed/Skipped: {failed_count}\n"
            + f"--------------------\n"
            + "\n".join(batch_results_list)
            + f"\n--------------------\nOverall Status: {batch_status_message}"
        )

        return batch_status_message, "\n".join(batch_results_list), return_batch_id

    def _finalize_batch_item(
        self,
        batch_job_id: str,
        item_identifier: str,
        segments: SegmentsList,
        speaker_labels: SpeakerLabelMap,
        metadata: Dict[str, Any],
        audio_path: Path,
        item_work_path: Path,
        log_file_handle: Optional[TextIO] = None,
    ) -> Optional[Dict[str, Union[Path, List[Path]]]]:
        item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(f"{item_log_prefix} Starting finalization...")
        item_output_dir = item_work_path / "output"
        create_directory(item_output_dir)
        generated_files: Dict[str, Union[Path, List[Path]]] = {}
        try:
            relabeled_segments = apply_speaker_labels(
                segments=segments,
                speaker_labels=speaker_labels,
                log_prefix=item_log_prefix,
            )
            final_json_name = f"{item_identifier}_{self.config.get('final_structured_transcript_name', 'structured_transcript_final.json')}"
            final_json_path = item_output_dir / final_json_name
            final_output_data = {
                "segments": convert_floats(relabeled_segments),
                "metadata": metadata if metadata else {},
            }
            if save_text_file(
                json.dumps(final_output_data, indent=2, ensure_ascii=False),
                final_json_path,
            ):
                generated_files["final_structured_json"] = final_json_path
            else:
                logger.error(
                    f"{item_log_prefix} Failed to save final structured transcript."
                )
            output_flags_dict = self.labeling_state.get(batch_job_id, {}).get(
                "output_flags", {}
            )
            batch_flags = cast(Dict[str, bool], output_flags_dict)
            include_json = batch_flags.get("include_json", False)
            include_csv = batch_flags.get("include_csv", False)
            include_script = batch_flags.get("include_script", False)
            include_plots = batch_flags.get("include_plots", False)
            include_audio = batch_flags.get("include_audio", False)
            summary_data: Optional[EmotionSummary] = None
            if include_json or include_csv or include_plots:
                logger.info(f"{item_log_prefix} Calculating emotion summary...")
                summary_data = calculate_emotion_summary(
                    segments=relabeled_segments,
                    emotion_value_map=self.config.get("emotion_value_map", {}),
                    include_timeline=True,
                    log_prefix=item_log_prefix,
                )
                if not summary_data:
                    logger.warning(
                        f"{item_log_prefix} Emotion summary calculation yielded no data."
                    )
            if include_json and summary_data:
                json_summary_name = f"{item_identifier}_{self.config.get('emotion_summary_json_name', 'emotion_summary.json')}"
                json_summary_path = item_output_dir / json_summary_name
                saved_path = self._save_json_summary(
                    summary_data, json_summary_path, item_log_prefix
                )
                if saved_path:
                    generated_files["json_summary_path"] = saved_path
            if include_csv and summary_data:
                csv_summary_name = f"{item_identifier}_{self.config.get('emotion_summary_csv_name', 'emotion_summary.csv')}"
                csv_summary_path = item_output_dir / csv_summary_name
                saved_path = self._save_csv_summary(
                    summary_data, csv_summary_path, item_log_prefix
                )
                if saved_path:
                    generated_files["csv_summary_path"] = saved_path
            if include_script:
                script_name = f"{item_identifier}_{self.config.get('script_transcript_name', 'script_transcript.txt')}"
                script_path = item_output_dir / script_name
                saved_path = save_script_transcript(
                    relabeled_segments, script_path, item_log_prefix
                )
                if saved_path:
                    generated_files["script_path"] = saved_path
            if include_plots and summary_data:
                logger.info(f"{item_log_prefix} Generating plots...")
                plot_output_dir = item_output_dir / "plots"
                create_directory(plot_output_dir)
                plot_paths = generate_all_plots(
                    summary_data=summary_data,
                    output_dir=plot_output_dir,
                    file_prefix=f"{item_identifier}",
                    emotion_value_map=self.config.get("emotion_value_map", {}),
                    emotion_colors=self.config.get("emotion_colors", {}),
                    log_prefix=item_log_prefix,
                )
                if plot_paths:
                    generated_files["plot_paths"] = plot_paths
            if batch_job_id not in self.batch_output_files:
                self.batch_output_files[batch_job_id] = {}
            for key, path_or_list in generated_files.items():
                arc_folder_base = item_identifier
                arc_folder = (
                    f"{arc_folder_base}/plots"
                    if key == "plot_paths"
                    else arc_folder_base
                )
                if isinstance(path_or_list, list):
                    for p_path in path_or_list:
                        if isinstance(p_path, Path) and p_path.is_file():
                            self.batch_output_files[batch_job_id][
                                f"{arc_folder}/{p_path.name}"
                            ] = p_path
                elif isinstance(path_or_list, Path) and path_or_list.is_file():
                    self.batch_output_files[batch_job_id][
                        f"{arc_folder}/{path_or_list.name}"
                    ] = path_or_list
            if include_audio and audio_path and audio_path.is_file():
                self.batch_output_files[batch_job_id][
                    f"{item_identifier}/{audio_path.name}"
                ] = audio_path
            logger.info(f"{item_log_prefix} Finalization complete.")
            return generated_files
        except Exception as e:
            logger.exception(
                f"{item_log_prefix} Unexpected error during finalization: {e}"
            )
            return None

    def start_labeling_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[Tuple[str, str, List[int]]]:
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(f"{log_prefix} UI Request: Start labeling item.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            logger.error(f"{log_prefix} Item state invalid.")
            return None
        labeling_config = {
            "speaker_labeling_preview_duration": self.config.get(
                "speaker_labeling_preview_duration", 5.0
            ),
            "speaker_labeling_min_block_time": self.config.get(
                "speaker_labeling_min_block_time", 10.0
            ),
        }
        return start_interactive_labeling_for_item(
            item_state=item_state,
            labeling_config=labeling_config,
            log_prefix=log_prefix,
        )

    def store_label(
        self, batch_job_id: str, item_identifier: str, speaker_id: str, user_label: str
    ) -> bool:
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(f"{log_prefix} UI Request: Store label {speaker_id} = '{user_label}'.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            logger.error(f"{log_prefix} Item state invalid.")
            return False
        return store_speaker_label(
            item_state=item_state,
            speaker_id=speaker_id,
            user_label=user_label,
            log_prefix=log_prefix,
        )

    def get_next_labeling_speaker(
        self, batch_job_id: str, item_identifier: str, current_speaker_index: int
    ) -> Optional[Tuple[str, str, List[int]]]:
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(
            f"{log_prefix} UI Request: Get next speaker after index {current_speaker_index}."
        )
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            logger.error(f"{log_prefix} Item state invalid.")
            return None
        labeling_config = {
            "speaker_labeling_preview_duration": self.config.get(
                "speaker_labeling_preview_duration", 5.0
            ),
            "speaker_labeling_min_block_time": self.config.get(
                "speaker_labeling_min_block_time", 10.0
            ),
        }
        return get_next_speaker_for_labeling(
            item_state=item_state,
            current_speaker_index=current_speaker_index,
            labeling_config=labeling_config,
            log_prefix=log_prefix,
        )

    def skip_item_labeling(self, batch_job_id: str, item_identifier: str) -> bool:
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(f"{log_prefix} UI Request: Skip remaining speakers.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            logger.error(f"{log_prefix} Item state invalid.")
            return False
        success = skip_labeling_for_item(item_state=item_state, log_prefix=log_prefix)
        if success:
            logger.info(f"{log_prefix} Triggering finalization after skip.")
            finalized_ok = self.finalize_item(batch_job_id, item_identifier)
            return finalized_ok
        else:
            logger.error(f"{log_prefix} skip_labeling_for_item returned False.")
            return False

    def finalize_item(self, batch_job_id: str, item_identifier: str) -> bool:
        item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        logger.info(f"{item_log_prefix} Orchestrator finalizing item...")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            logger.error(f"{item_log_prefix} Cannot finalize - item state invalid.")
            return False
        segments = item_state.get("segments")
        collected_labels = cast(SpeakerLabelMap, item_state.get("collected_labels", {}))
        metadata = cast(Dict[str, Any], item_state.get("metadata", {}))
        audio_path = item_state.get("audio_path")
        item_work_path = item_state.get("item_work_path")
        if not segments or not audio_path or not item_work_path:
            logger.error(f"{item_log_prefix} Cannot finalize - missing data in state.")
            self._remove_item_state(batch_job_id, item_identifier)
            return False
        segments = cast(SegmentsList, segments)
        audio_path = cast(Path, audio_path)
        item_work_path = cast(Path, item_work_path)
        batch_work_path = item_work_path.parent
        log_file_path = batch_work_path / self.config.get(
            "log_file_name", "process_log.txt"
        )
        log_handle: Optional[TextIO] = None
        finalization_success = False
        try:
            if log_file_path.exists():
                log_handle = open(log_file_path, "a", encoding="utf-8")
            generated_files = self._finalize_batch_item(
                batch_job_id=batch_job_id,
                item_identifier=item_identifier,
                segments=segments,
                speaker_labels=collected_labels,
                metadata=metadata,
                audio_path=audio_path,
                item_work_path=item_work_path,
                log_file_handle=log_handle,
            )
            finalization_success = generated_files is not None
        except Exception as e:
            logger.exception(
                f"{item_log_prefix} Unexpected error during item finalization: {e}\n{traceback.format_exc()}"
            )
            finalization_success = False
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()
            self._remove_item_state(batch_job_id, item_identifier)
        return finalization_success

    def check_completion_and_zip(self, batch_job_id: str) -> Optional[Path]:
        log_prefix = self._get_log_prefix(batch_job_id)
        logger.info(f"{log_prefix} UI Request: Check batch completion and ZIP.")
        batch_state = self.labeling_state.get(batch_job_id)
        items_remaining = False
        if batch_state:
            item_keys_present = [k for k in batch_state.keys() if k.startswith("item_")]
            items_remaining = bool(item_keys_present)
        if items_remaining:
            logger.info(f"{log_prefix} Batch labeling not complete.")
            return None
        logger.info(f"{log_prefix} All items finalized. Proceeding with ZIP creation.")
        permanent_output_dir = Path(self.config.get("output_dir", "./output"))
        zip_suffix = self.config.get("final_zip_suffix", "_final_bundle.zip")
        master_zip_name = f"{batch_job_id}_batch_results{zip_suffix}"
        master_zip_path = permanent_output_dir / master_zip_name
        files_for_zip = self.batch_output_files.get(batch_job_id, {})
        batch_work_path = Path(self.config.get("temp_dir", "./temp")) / batch_job_id
        log_file_path = batch_work_path / self.config.get(
            "log_file_name", "process_log.txt"
        )
        log_handle = None
        zip_path: Optional[Path] = None
        try:
            files_for_zip_cast = cast(Dict[str, Union[str, Path]], files_for_zip)
            if log_file_path.exists():
                log_handle = open(log_file_path, "a", encoding="utf-8")
                files_for_zip_cast[log_file_path.name] = log_file_path
            zip_path = create_zip_archive(
                zip_path=master_zip_path,
                files_to_add=files_for_zip_cast,
                log_prefix=log_prefix,
            )
            cleanup_temp = self.config.get("cleanup_temp_on_success", False)
            if zip_path and cleanup_temp and batch_work_path.exists():
                logger.info(
                    f"{log_prefix} Cleaning up batch temp dir after ZIP: {batch_work_path}"
                )
                cleanup_directory(batch_work_path, recreate=False)
            elif zip_path and not cleanup_temp:
                logger.info(
                    f"{log_prefix} Skipping cleanup of temp dir: {batch_work_path}"
                )
            elif not zip_path and batch_work_path.exists():
                logger.warning(
                    f"{log_prefix} Keeping temp dir due to ZIP failure: {batch_work_path}"
                )
        except Exception as e:
            logger.exception(
                f"{log_prefix} Error during final zip/cleanup: {e}"
            )
            zip_path = None
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
        return zip_path

    def _remove_item_state(self, batch_job_id: str, item_identifier: str):
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        if batch_job_id in self.labeling_state:
            batch_state = self.labeling_state[batch_job_id]
            if item_identifier in batch_state:
                item_state_to_remove = batch_state.get(item_identifier)
                if (
                    isinstance(item_state_to_remove, dict)
                    and "item_work_path" in item_state_to_remove
                ):
                    del batch_state[item_identifier]
                    logger.info(f"{log_prefix} Removed labeling state for item.")
                    if "items_requiring_labeling_order" in batch_state and isinstance(
                        batch_state["items_requiring_labeling_order"], list
                    ):
                        try:
                            batch_state["items_requiring_labeling_order"].remove(
                                item_identifier
                            )
                        except ValueError:
                            pass
                else:
                    logger.warning(
                        f"{log_prefix} Attempted to remove non-item state for '{item_identifier}'."
                    )
                remaining_item_keys = [
                    k for k in batch_state.keys() if k.startswith("item_")
                ]
                if not remaining_item_keys:
                    logger.info(
                        f"{self._get_log_prefix(batch_job_id)} No items left. Removing batch state."
                    )
                    del self.labeling_state[batch_job_id]
