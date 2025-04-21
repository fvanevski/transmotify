# core/pipeline.py
import csv
import json
import shutil
import subprocess
import traceback
import zipfile
import string  # Add this line to import the string module
import re  # Add this line to import the regular expression module
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, TextIO, Tuple

from rapidfuzz import fuzz  # Import the fuzz module from fuzzywuzzy

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
from .transcription import Transcription
from .multimodal_analysis import MultimodalAnalysis
from .utils import (
    parse_xlsx_snippets,
)  # Assuming parse_xlsx_snippets is in core/utils.py


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
      4) Snippet-based speaker matching (for UI suggestion or batch relabeling)
      5) Generate speaker previews (for UI labeling - Single Process only)
      6) Save intermediate structured JSON (with original IDs - Single Process only)
      7) Relabel & finalize (apply user labels, save final JSON, summaries, plots, script, ZIP - Single Process)
      8) Process batch XLSX (handle multiple URLs, apply snippet labels directly, finalize each item - Batch Process)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transcription = Transcription(config)
        log_info("Initializing MultimodalAnalysis in Pipeline...")
        self.mm = MultimodalAnalysis(config)
        # Optional fallback diarizer
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
        session_id: str,
    ) -> Path:
        """Download from URL or copy local file; return path to input audio."""
        log_info(f"[{session_id}] Preparing audio input from: {input_source}")
        if input_source.startswith(("http://", "https://")):
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
            unique_filename = f"{src.stem}_{session_id}{src.suffix}"
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
        except FileNotFoundError:
            log_warning("ffprobe not installed; skipping duration check.")
            return False
        except Exception as e:
            log_error(f"ffprobe error: {e}")
            log_warning(
                "Duration check failed, proceeding but diarization quality may be affected."
            )
            return False
        return True

    def _find_whisperx_output(self, out_dir: Path, audio_path: Path) -> Path:
        """Locate the primary WhisperX JSON in output directory."""
        expected = out_dir / f"{audio_path.stem}.json"
        if expected.exists():
            log_info(f"Found WhisperX output at expected path: {expected}")
            return expected
        log_warning(
            f"Expected WhisperX output {expected.name} not found; searching directory {out_dir}..."
        )
        standard_output_names = [
            INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME,
            FINAL_STRUCTURED_TRANSCRIPT_NAME,
            EMOTION_SUMMARY_JSON_NAME,
            EMOTION_SUMMARY_CSV_NAME,
            SCRIPT_TRANSCRIPT_NAME,
        ]
        found_json_files = [
            f
            for f in out_dir.iterdir()
            if f.suffix == ".json" and f.name not in standard_output_names
        ]
        if found_json_files:
            if len(found_json_files) > 1:
                log_warning(
                    f"Multiple potential WhisperX JSON files found in {out_dir}. Using the first one: {found_json_files[0].name}"
                )
            log_info(f"Found WhisperX output by searching: {found_json_files[0]}")
            return found_json_files[0]

        raise FileNotFoundError(f"No suitable WhisperX JSON output found in {out_dir}")

    def _group_segments_by_speaker(
        self, segments: SegmentsList
    ) -> List[Dict[str, Any]]:
        """Group consecutive segments by speaker into blocks."""
        blocks = []
        cur = None
        for i, seg in enumerate(segments):
            spk = seg.get("speaker")
            spk = str(spk) if spk is not None else "unknown"
            text = (seg.get("text") or "").strip()
            start, end = seg.get("start"), seg.get("end")

            if not cur or cur["speaker"] != spk:
                if cur:
                    blocks.append(cur)
                cur = {
                    "speaker": spk,
                    "text": text,
                    "start": start,
                    "end": end,
                    "indices": [i],
                }
            else:
                if text:
                    cur["text"] += (" " if cur["text"] else "") + text
                if end is not None:
                    cur["end"] = end
                cur["indices"].append(i)

        if cur:
            blocks.append(cur)

        log_info(f"Grouped {len(segments)} segments into {len(blocks)} speaker blocks.")
        return blocks

    # MODIFIED: _match_snippets_to_speakers using rapidfuzz, aggressive normalization, and refined logging
    def _match_snippets_to_speakers(
        self,
        segments: SegmentsList,  # Receive segments (should have original SPEAKER_XX IDs)
        speaker_snippet_map: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Fuzzy-match user snippets to speaker IDs using rapidfuzz.fuzz.partial_ratio
        after aggressive normalization, prioritizing exact substring match.
        Returns a mapping from original WhisperX ID to user-provided name for matched speakers.
        Logs match details or best non-match details.
        Does NOT modify the input segments.
        """
        log_info(
            f"Attempting to match {len(speaker_snippet_map)} snippets to speakers using rapidfuzz.fuzz.partial_ratio after normalization..."
        )
        if not speaker_snippet_map:
            log_info("No speaker snippets provided for matching.")
            return {}

        blocks = self._group_segments_by_speaker(segments)

        speaker_id_to_name_mapping: Dict[str, str] = {}
        # This dictionary is used internally to track the best score found for each original speaker ID
        # and prevent a lower-scoring snippet from overriding a higher-scoring one for the same speaker.
        best_match_scores: Dict[str, float] = {}

        # Threshold should be in percentage for rapidfuzz (0-100)
        # Convert the 0-1 threshold from config to 0-100 for rapidfuzz
        thresh = (
            float(
                self.config.get(
                    "snippet_match_threshold", DEFAULT_SNIPPET_MATCH_THRESHOLD
                )
            )
            * 100.0
        )
        log_info(
            f"Using snippet matching threshold: {thresh} (converted for rapidfuzz)"
        )

        # Helper function for aggressive normalization (keep this as it helps comparison)
        # Assumes string and re modules are imported at the top of the file
        def aggressively_normalize(text: str) -> str:
            if not isinstance(text, str):
                return ""
            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Convert to lowercase
            text = text.lower()
            # Replace multiple spaces with a single space and strip leading/trailing whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        for name, snippet in speaker_snippet_map.items():
            if not snippet or len(snippet) < 5:
                log_warning(
                    f"Ignoring short or empty snippet for speaker '{name}'. Snippet: '{snippet}')"
                )
                continue

            # Normalize the snippet once
            low = aggressively_normalize(snippet)

            # Store best match info for this snippet across all blocks if no match > thresh is found
            best_match_for_snippet: Dict[str, Any] = {
                "score": -1.0,
                "speaker_id": None,
                "block_text_snippet": "",
                "full_snippet": low,  # Store the normalized snippet itself
            }
            match_found_above_threshold = False

            # Log processed snippet
            log_info(f"Matching snippet for '{name}': '{low}' (length: {len(low)})")
            log_info(
                f"Raw original snippet string repr: {repr(snippet)}"
            )  # Log original snippet repr

            for i, blk in enumerate(blocks):
                sid = blk.get("speaker")
                sid = str(sid) if sid is not None else "unknown"
                txt_raw = blk.get("text", "")

                if not sid or not txt_raw:
                    continue

                # Normalize the block text
                txt = aggressively_normalize(txt_raw)

                # Calculate partial_ratio
                ratio = fuzz.partial_ratio(low, txt)

                # Update best match found for THIS snippet (across all blocks)
                if ratio > best_match_for_snippet["score"]:
                    best_match_for_snippet["score"] = ratio
                    best_match_for_snippet["speaker_id"] = sid
                    # Store a snippet of the block text for logging the best match
                    best_match_for_snippet["block_text_snippet"] = (
                        txt[:100] + "..." if len(txt) > 100 else txt
                    )

                # Compare against the 0-100 threshold
                if ratio >= thresh:
                    # Found a match above threshold.
                    # Check if this is the best score found *so far* for this specific speaker ID.
                    if ratio > best_match_scores.get(sid, -1.0):
                        speaker_id_to_name_mapping[sid] = name
                        best_match_scores[sid] = ratio
                        # Log the successful match
                        log_info(
                            f"Match FOUND for snippet '{name}' (score {ratio:.2f}) against speaker '{sid}'."
                        )
                        # Since we found a match for THIS snippet above the threshold against THIS speaker,
                        # and it's the best score for this speaker so far, we can stop checking blocks for THIS snippet.
                        match_found_above_threshold = True
                        break  # Exit the inner loop (iterating through blocks for the current snippet)

            # After iterating through all blocks for the current snippet:
            # If no match above threshold was found for this snippet, log the best match details found
            if not match_found_above_threshold:
                log_info(
                    f"No match >= threshold ({thresh:.2f}) found for snippet '{name}'. Best match found across all blocks was score {best_match_for_snippet['score']:.2f} against speaker '{best_match_for_snippet['speaker_id']}'. Block text snippet: '{best_match_for_snippet['block_text_snippet']}'"
                )

        log_info(
            f"Final snippet mapping (WhisperX ID -> User Name): {speaker_id_to_name_mapping}"
        )
        return speaker_id_to_name_mapping

    def _save_script_transcript(
        self, segments: SegmentsList, output_dir: Path, suffix: str
    ) -> Optional[Path]:
        """Saves a plain text transcript formatted like a script."""
        script_path = output_dir / f"{Path(SCRIPT_TRANSCRIPT_NAME).stem}_{suffix}.txt"
        log_info(f"Saving script transcript to: {script_path}")

        grouped_blocks = self._group_segments_by_speaker(segments)

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                for block in grouped_blocks:
                    speaker = block.get("speaker", "UNKNOWN")
                    text = block.get("text", "").strip()
                    start_time = block.get("start")
                    end_time = block.get("end")

                    time_str = ""
                    if start_time is not None and end_time is not None:
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
                entry["emotion_timeline"] = sorted(
                    timeline[spk], key=lambda x: x.get("time", 0.0)
                )

            summary[spk] = entry

        return summary

    def _save_emotion_summary(
        self,
        stats: EmotionSummary,
        out_dir: Path,
        suffix: str,
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
        log_file: Path,
    ) -> Optional[Path]:
        log_info(f"Creating ZIP: {zip_path}")
        final_output_dir = zip_path.parent
        final_output_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"Ensured final output directory exists: {final_output_dir}")

        temp_zip_path = zip_path.with_suffix(".temp.zip")

        try:
            with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
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

    def _process_batch_item(
        self,
        input_source: str,
        speaker_snippet_map: Optional[Dict[str, str]],
        work_path: Path,
        log_file_handle: TextIO,
        job_id: str,
    ) -> Tuple[Optional[SegmentsList], Optional[Path], Optional[Dict[str, str]]]:
        """
        Processes a single audio source (URL or file) for batch mode:
        download/copy -> ffprobe -> whisperx -> structuring -> multimodal analysis -> snippet matching.
        Returns (segments_with_original_ids, audio_path_in_work_dir, snippet_mapping).
        Does NOT perform finalization or save intermediate JSON/previews.
        """
        audio_path = None
        segments: SegmentsList = []
        snippet_mapping: Dict[str, str] = {}
        audio_path_in_work_dir = (
            None  # To store the path to the copied/downloaded WAV in work_path
        )

        try:
            output_temp_dir = work_path / "output"
            output_temp_dir.mkdir(exist_ok=True)

            # --- Prepare Audio Input ---
            status_message = f"[{job_id}] Preparing audio input from: {input_source}"
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            audio_path = self._prepare_audio_input(
                input_source, work_path, log_file_handle, job_id
            )
            audio_path_in_work_dir = (
                audio_path  # This is the path within the job's temp directory
            )

            # --- Check Audio Duration ---
            status_message = f"[{job_id}] Checking audio duration..."
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            duration_check_ok = self._run_ffprobe_duration_check(audio_path)
            if not duration_check_ok:
                log_warning(
                    f"[{job_id}] Audio duration check issues or ffprobe not available."
                )

            # --- Run WhisperX ---
            status_message = (
                f"[{job_id}] Running WhisperX for transcription and diarization..."
            )
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            self.transcription.run_whisperx(
                str(audio_path), str(output_temp_dir), log_file_handle, job_id
            )

            status_message = f"[{job_id}] Structuring WhisperX output..."
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            whisperx_json_path = self._find_whisperx_output(output_temp_dir, audio_path)
            segments = self.transcription.convert_json_to_structured(
                str(whisperx_json_path)
            )  # Segments have original SPEAKER_XX IDs

            # --- Perform Multimodal Emotion Analysis ---
            status_message = f"[{job_id}] Running multimodal emotion analysis..."
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            # Analyze segments with original SPEAKER_XX IDs
            segments = self.mm.analyze(
                segments, str(audio_path), input_source
            )  # Assuming analyze can handle audio_path

            # --- Perform Snippet-Based Speaker Matching ---
            status_message = f"[{job_id}] Performing snippet-based speaker matching..."
            log_info(status_message)
            log_file_handle.write(f"{status_message}\n")
            log_file_handle.flush()
            # Match snippets using segments with original SPEAKER_XX IDs. Segments are NOT modified here.
            snippet_mapping = self._match_snippets_to_speakers(
                segments, speaker_snippet_map or {}
            )

            return segments, audio_path_in_work_dir, snippet_mapping

        except Exception as e:
            err_msg = f"[{job_id}] ERROR during batch item processing: {e}"
            log_error(err_msg + "\n" + traceback.format_exc())
            if log_file_handle and not log_file_handle.closed:
                log_file_handle.write(err_msg + "\n" + traceback.format_exc() + "\n")
                log_file_handle.flush()
            return None, None, None  # Indicate failure

    # MODIFIED: _finalize_batch_item to save files locally within item_work_path, no zipping or cleanup
    def _finalize_batch_item(
        self,
        segments: SegmentsList,
        snippet_mapping: Optional[Dict[str, str]],
        audio_path_in_work_dir: Optional[
            Path
        ],  # Path to the WAV file in the item work directory
        item_work_path: Path,  # The work path for THIS batch item
        log_file_handle: TextIO,  # Pass the main batch log handle for logging
        item_identifier: str,  # e.g., "youtube_video_id" or "row_number"
    ) -> str:  # Return only a status message string for this item
        """
        Finalizes a single batch item: apply snippet labels, save final JSON,
        generate summaries, plots, script transcript. Files are saved within
        the item_work_path/output directory. Does NOT create a ZIP or clean up.
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

            # Apply snippet-matched labels (ONLY for batch processing)
            item_log("info", "Applying snippet-matched speaker labels...")
            segments_relabeled_count = 0
            for seg in segments:
                original_speaker_id_in_segment = seg.get(
                    "speaker"
                )  # This is the original SPEAKER_XX ID
                original_speaker_id_str = (
                    str(original_speaker_id_in_segment)
                    if original_speaker_id_in_segment is not None
                    else "unknown"
                )

                # Apply the snippet-matched label if a mapping exists for this original SPEAKER_XX ID
                if (
                    snippet_mapping is not None
                    and original_speaker_id_str in snippet_mapping
                ):
                    user_label = snippet_mapping[original_speaker_id_str]
                    if (
                        user_label
                    ):  # Only relabel if the snippet mapping provided a non-empty label
                        seg["speaker"] = user_label
                        segments_relabeled_count += 1
                    else:
                        # If snippet mapping was empty string, keep original SPEAKER_XX ID
                        seg["speaker"] = (
                            original_speaker_id_str  # Explicitly set to original ID if empty label provided
                        )
                # If original_speaker_id_str is NOT in snippet_mapping, it remains its original SPEAKER_XX ID

            item_log(
                "info",
                f"Applied snippet labels to {segments_relabeled_count} segments.",
            )

            # Save the FINAL relabeled structured JSON to the output directory within item_work_path
            item_log("info", "Saving final relabeled structured transcript JSON...")
            output_temp_dir = (
                item_work_path / "output"
            )  # Save within the item's output subdir
            output_temp_dir.mkdir(exist_ok=True)  # Ensure output subdir exists
            final_json_name = (
                f"{item_identifier}_{Path(FINAL_STRUCTURED_TRANSCRIPT_NAME).name}"
            )
            final_json_path = output_temp_dir / final_json_name

            segments_final_serializable = convert_floats(segments)

            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(segments_final_serializable, f, indent=2, ensure_ascii=False)
            item_log(
                "info",
                f"Final relabeled structured transcript saved to: {final_json_path}",
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

    # MODIFIED: process_batch_xlsx to create a single master zip and handle cleanup
    def process_batch_xlsx(self, xlsx_filepath: str) -> Tuple[str, str]:
        """
        Processes an XLSX file containing YouTube URLs and optional speaker snippets in batch.
        Reads the file, processes each item, and generates results for each in subdirectories.
        Creates a single master ZIP containing all item results.
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

        try:
            batch_work_path.mkdir(parents=True, exist_ok=True)
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            log_info(f"[{batch_job_id}] Batch log file created at: {batch_log_path}")

            def batch_log(level, message):
                full_message = f"[{batch_job_id}] {message}"
                if log_file_handle:
                    try:
                        log_file_handle.write(
                            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                        )
                        log_file_handle.flush()
                    except Exception as log_e:
                        log_error(
                            f"[{batch_job_id}] Failed to write to batch log file: {log_e} - Message: {message}"
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

            batch_log("info", batch_status_message)

            # Read the XLSX file
            batch_log("info", f"Reading batch file: {xlsx_filepath}")
            df = pd.read_excel(xlsx_filepath)
            batch_log("info", f"Successfully read {len(df)} rows from {xlsx_filepath}")

            results: List[str] = []
            url_col = self.config.get("batch_url_column", "YouTube URL")
            snippet_col = self.config.get("batch_snippet_column", "Speaker Snippets")

            if url_col not in df.columns:
                raise ValueError(
                    f"Required column '{url_col}' not found in the XLSX file."
                )

            batch_log(
                "info",
                f"Using URL column: '{url_col}' and Snippet column: '{snippet_col}'",
            )

            i = 0  # Initialize a counter
            for (
                _,
                row,
            ) in (
                df.iterrows()
            ):  # Iterate using iterrows to get row as Series, ignore index
                i += 1  # Increment counter for 1-based item_identifier
                item_identifier = f"item_{i}"  # Use the counter for identifier

                # Ensure youtube_url is valid before adding to processed_item_identifiers
                youtube_url_raw = row[url_col]
                if not isinstance(youtube_url_raw, str) or not youtube_url_raw.strip():
                    batch_log(
                        "warning",
                        f"[{item_identifier}] Skipping row {i} due to missing or invalid YouTube URL.",  # Use counter here
                    )
                    results.append(
                        f"[{item_identifier}] Skipped: Missing/Invalid YouTube URL."
                    )
                    continue

                processed_item_identifiers.append(
                    item_identifier
                )  # Add to list ONLY if URL is valid

                youtube_url = youtube_url_raw.strip()
                snippets_string = row.get(
                    snippet_col, ""
                )  # Use .get() for optional column with default

                batch_log("info", f"[{item_identifier}] Processing URL: {youtube_url}")

                # Parse snippets for this item (assuming parse_xlsx_snippets is in utils or imported)
                speaker_snippet_map = parse_xlsx_snippets(
                    snippets_string
                )  # Call function from utils
                batch_log(
                    "info",
                    f"[{item_identifier}] Parsed snippets: {speaker_snippet_map}",
                )

                # Create a temporary working directory for THIS batch item within the main batch dir
                item_work_path = batch_work_path / item_identifier
                item_work_path.mkdir(exist_ok=True)  # Create item-specific subdir

                # Process the single item using the internal method
                segments, audio_path_in_work_dir, snippet_mapping = (
                    self._process_batch_item(
                        youtube_url,
                        speaker_snippet_map,
                        item_work_path,  # Pass the item's work path
                        log_file_handle,  # Pass the main batch log handle
                        item_identifier,  # Use item identifier for logging inside the item process
                    )
                )

                if segments is not None:
                    # Finalize the single item (save files locally)
                    item_status_msg = self._finalize_batch_item(
                        segments,
                        snippet_mapping,
                        audio_path_in_work_dir,  # Pass audio path in work dir
                        item_work_path,  # Pass the item's work path for saving outputs
                        log_file_handle,  # Pass the main batch log handle
                        item_identifier,  # Use item identifier for finalization logging prefix
                    )
                    results.append(f"[{item_identifier}] {item_status_msg}")
                else:
                    # _process_batch_item failed, status is already logged inside that method
                    results.append(
                        f"[{item_identifier}] Failed during processing step."
                    )

            # --- Batch Processing Loop Finished ---

            # --- Create the single master ZIP bundle ---
            batch_log("info", "Creating single master ZIP bundle for the batch...")
            permanent_output_dir = Path(self.config.get("output_dir", "./output"))
            # Use the batch job ID for the master zip filename
            master_zip_name = f"{batch_job_id}_batch_results{FINAL_ZIP_SUFFIX}"
            master_zip_location = permanent_output_dir / master_zip_name

            batch_log("info", f"Master ZIP will be saved to: {master_zip_location}")

            temp_master_zip_path = master_zip_location.with_suffix(".temp.zip")

            try:
                with zipfile.ZipFile(
                    str(temp_master_zip_path), "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    # Add the main batch log file at the root of the zip
                    if batch_log_path.exists():
                        zf.write(batch_log_path, arcname=batch_log_path.name)
                        batch_log(
                            "info",
                            f"Added batch log file to master zip: {batch_log_path.name}",
                        )

                    # Iterate through each item's working directory and add its contents to the zip
                    for item_id in (
                        processed_item_identifiers
                    ):  # Only iterate through successfully identified items
                        item_work_path = batch_work_path / item_id
                        if item_work_path.exists():
                            # Walk through the item's directory
                            for item_file_path in item_work_path.rglob("*"):
                                if item_file_path.is_file():
                                    # Create an archive name that includes the item's folder
                                    archive_name = f"{item_id}/{item_file_path.relative_to(item_work_path)}"
                                    zf.write(item_file_path, arcname=archive_name)
                                    # batch_log('info', f"Added {item_file_path.name} to master zip as {archive_name}") # Can be very verbose

                shutil.move(str(temp_master_zip_path), str(master_zip_location))
                batch_log(
                    "info", f"Successfully created master ZIP: {master_zip_location}"
                )
                master_zip_path_str = str(master_zip_location)

                batch_status_message = f"[{batch_job_id}] ✅ Batch processing complete. Download ready: {master_zip_path_str}"
                batch_results_summary = (
                    "Batch Results Summary:\n"
                    + "\n".join(results)
                    + f"\n\nMaster ZIP: {master_zip_path_str}"
                )
                batch_log("info", batch_status_message)
                batch_log("info", batch_results_summary)

            except Exception as zip_e:
                err_msg = f"[{batch_job_id}] ERROR creating master ZIP file: {zip_e}"
                batch_log("error", err_msg + "\n" + traceback.format_exc())
                batch_status_message = err_msg
                batch_results_summary = (
                    "Batch Results Summary:\n"
                    + "\n".join(results)
                    + f"\n\nERROR creating master ZIP: {zip_e}"
                )
                master_zip_path_str = None  # Indicate failure

        except FileNotFoundError:
            batch_status_message = (
                f"[{batch_job_id}] ERROR: Batch input file not found at {xlsx_filepath}"
            )
            batch_log("error", batch_status_message)
            batch_results_summary = batch_status_message
        except pd.errors.EmptyDataError:
            batch_status_message = f"[{batch_job_id}] ERROR: The uploaded Excel file {xlsx_filepath} is empty."
            batch_log("error", batch_status_message)
            batch_results_summary = batch_status_message
        except Exception as e:
            err_msg = f"[{batch_job_id}] An unexpected error occurred during batch processing setup or iteration: {e}"
            batch_log("error", err_msg + "\n" + traceback.format_exc())
            batch_status_message = err_msg
            batch_results_summary = batch_status_message

        finally:
            # Ensure the main batch log file is closed
            if log_file_handle and not log_file_handle.closed:
                log_file_handle.close()

            # Clean up the main batch temporary directory after processing and zipping
            # This block should be aligned with the try/except block
            if batch_work_path.exists():
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

        # This return statement should be aligned with the try/except/finally block
        return batch_status_message, batch_results_summary

    # Assuming parse_xlsx_snippets is in core/utils.py and imported
