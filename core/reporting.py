# core/reporting.py
"""
Handles calculation of summaries, saving report files (JSON, CSV),
and triggering plot generation based on processed segment data.
"""

import json
import csv
import traceback
from pathlib import Path
from collections import Counter, defaultdict
import statistics
from typing import Dict, List, Any, Optional, Tuple, TextIO

# Logging imports
from .logging import log_info, log_warning, log_error

# Utility imports
from .utils import (
    convert_floats,  # For saving JSON compatible numbers
    save_script_transcript,  # To save the text script if requested
)

# Plotting import
from .plotting import generate_all_plots  # To generate plots if requested

# Constants import
from .constants import (
    EMO_VAL,  # Emotion value mapping
    EMOTION_SUMMARY_CSV_NAME,
    EMOTION_SUMMARY_JSON_NAME,
)

# Type hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]
EmotionSummary = Dict[str, Dict[str, Any]]  # Speaker -> Stats Dict


# --- Functions Moved/Adapted from pipeline.py ---


def calculate_emotion_summary(
    segments: SegmentsList, include_timeline: bool = False
) -> EmotionSummary:
    """
    Calculates emotion statistics per speaker from a list of segments.

    Args:
        segments: List of segment dictionaries, expected to have 'speaker' and 'emotion' keys.
        include_timeline: If True, adds a detailed timestamped emotion timeline to the output.

    Returns:
        A dictionary where keys are speaker IDs and values are dictionaries
        containing calculated statistics (counts, dominant, volatility, etc.).
    """
    # Use defaultdict to easily append emotions per speaker
    speaker_emotions = defaultdict(list)
    # Use defaultdict for timeline if requested
    speaker_timeline = defaultdict(list) if include_timeline else None

    log_info(f"Calculating emotion summary for {len(segments)} segments...")

    for seg in segments:
        # Ensure speaker is treated as a string
        spk = str(seg.get("speaker", "unknown"))
        # Use the final 'emotion' key from multimodal analysis
        emo = seg.get("emotion", "unknown")  # e.g., 'joy', 'anger', 'unknown'

        speaker_emotions[spk].append(emo)

        # If timeline requested, capture emotion and time
        if include_timeline and speaker_timeline is not None:
            timeline_entry: Dict[str, Any] = {
                "time": seg.get("start", 0.0),  # Use segment start time
                "emotion": emo,
                # Optionally add confidence scores if available in segment
                "fused_confidence": seg.get("fused_emotion_confidence"),
                "significant_text": seg.get("significant_text_emotions"),
            }
            # Filter out None values from the timeline entry for cleaner output
            speaker_timeline[spk].append(
                {k: v for k, v in timeline_entry.items() if v is not None}
            )

    summary: EmotionSummary = {}
    # Calculate stats for each speaker found
    for spk, emotions_list in speaker_emotions.items():
        if not emotions_list:  # Skip if a speaker somehow has no emotions recorded
            continue

        # Count occurrences of each emotion
        emotion_counts = Counter(emotions_list)

        # Calculate transitions (changes in emotion from one segment to the next)
        transitions = sum(
            1
            for i in range(len(emotions_list) - 1)
            if emotions_list[i + 1] != emotions_list[i]
        )

        # --- Determine Dominant Emotion ---
        # Filter out non-indicative emotions before finding the most common
        filterable_emotions = [
            e
            for e in emotions_list
            if e not in ["unknown", "analysis_skipped", "analysis_failed", "no_text"]
        ]
        if filterable_emotions:
            dominant_emotion = Counter(filterable_emotions).most_common(1)[0][0]
        else:
            # If only non-indicative emotions are present, check if 'neutral' is there
            dominant_emotion = "neutral" if "neutral" in emotion_counts else "unknown"

        # --- Calculate Volatility and Mean Score ---
        # Use EMO_VAL mapping from constants
        emotion_values = [EMO_VAL.get(e, 0.0) for e in emotions_list]
        volatility = (
            statistics.stdev(emotion_values) if len(emotion_values) > 1 else 0.0
        )
        mean_score = statistics.mean(emotion_values) if emotion_values else 0.0

        # Build the summary entry for this speaker
        speaker_summary_entry: Dict[str, Any] = {
            "total_segments": len(emotions_list),
            "dominant_emotion": dominant_emotion,
            "emotion_volatility": round(volatility, 4),  # More precision?
            "emotion_score_mean": round(mean_score, 4),
            "emotion_transitions": transitions,
            "emotion_counts": dict(emotion_counts),  # Convert Counter to dict for JSON
        }

        # Add timeline if requested and available
        if (
            include_timeline
            and speaker_timeline is not None
            and spk in speaker_timeline
        ):
            # Sort timeline by time
            speaker_summary_entry["emotion_timeline"] = sorted(
                speaker_timeline[spk], key=lambda x: x.get("time", 0.0)
            )

        summary[spk] = speaker_summary_entry

    log_info(f"Emotion summary calculated for {len(summary)} speakers.")
    return summary


def save_emotion_summary(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_suffix: str,
    save_json: bool = True,  # Default to save JSON
    save_csv: bool = False,  # Default to NOT save CSV
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Saves the calculated emotion summary data to JSON and/or CSV files.

    Args:
        summary_data: The dictionary returned by calculate_emotion_summary.
        output_dir: The directory where the summary files will be saved.
        file_suffix: A string to append to the base filenames (e.g., item identifier).
        save_json: If True, save the detailed summary (including timeline) as JSON.
        save_csv: If True, save the high-level summary (without timeline) as CSV.

    Returns:
        A tuple containing the Path objects for the saved (json_path, csv_path).
        Paths will be None if the corresponding file type was not saved or failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None

    # --- Save JSON ---
    if save_json:
        json_filename = f"{Path(EMOTION_SUMMARY_JSON_NAME).stem}_{file_suffix}.json"
        json_path = output_dir / json_filename
        log_info(f"Attempting to save detailed emotion summary JSON to: {json_path}")
        try:
            # Convert numpy floats etc., before saving
            summary_serializable = convert_floats(summary_data)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
            log_info(f"Detailed emotion summary JSON saved successfully.")
        except Exception as e:
            log_error(f"Failed to save emotion summary JSON to {json_path}: {e}")
            log_error(traceback.format_exc())
            json_path = None  # Indicate failure

    # --- Save CSV ---
    if save_csv:
        csv_filename = f"{Path(EMOTION_SUMMARY_CSV_NAME).stem}_{file_suffix}.csv"
        csv_path = output_dir / csv_filename
        log_info(f"Attempting to save high-level emotion summary CSV to: {csv_path}")

        # Define standard headers for the CSV (excluding the detailed timeline)
        standard_headers = [
            "speaker",
            "total_segments",
            "dominant_emotion",
            "emotion_volatility",
            "emotion_score_mean",
            "emotion_transitions",
        ]
        # Dynamically find all unique emotion types mentioned in the counts across all speakers
        all_emotion_count_keys = sorted(
            list(
                set(
                    emotion
                    for speaker_data in summary_data.values()
                    for emotion in speaker_data.get("emotion_counts", {}).keys()
                )
            )
        )
        # Combine standard headers with emotion count headers
        final_headers = standard_headers + [
            f"count_{emo}" for emo in all_emotion_count_keys
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(
                    cf, fieldnames=final_headers, extrasaction="ignore"
                )
                writer.writeheader()
                # Write data for each speaker
                for speaker_id, data in summary_data.items():
                    row_data = {"speaker": speaker_id}
                    # Add standard fields
                    row_data.update(
                        {h: data.get(h) for h in standard_headers if h != "speaker"}
                    )
                    # Add emotion counts, prefixing keys with 'count_'
                    emotion_counts = data.get("emotion_counts", {})
                    row_data.update(
                        {
                            f"count_{emo}": emotion_counts.get(emo, 0)
                            for emo in all_emotion_count_keys
                        }
                    )

                    # Ensure float values are rounded for CSV readability (optional)
                    for key in ["emotion_volatility", "emotion_score_mean"]:
                        if key in row_data and row_data[key] is not None:
                            try:
                                row_data[key] = round(float(row_data[key]), 4)
                            except:
                                pass  # Ignore errors if value isn't float

                    writer.writerow(row_data)
            log_info(f"High-level emotion summary CSV saved successfully.")
        except Exception as e:
            log_error(f"Failed to save emotion summary CSV to {csv_path}: {e}")
            log_error(traceback.format_exc())
            csv_path = None  # Indicate failure

    return json_path, csv_path


# --- Main Orchestrator Function ---


def generate_item_report_outputs(
    segments: SegmentsList,
    item_identifier: str,
    item_output_dir: Path,
    config: Dict[str, Any],
    log_file_handle: TextIO,  # Use main log handle for messages here
    # --- Flags for optional outputs ---
    include_json_summary: bool = True,  # Default ON as per user spec
    include_csv_summary: bool = False,  # Default OFF
    include_script: bool = False,  # Default OFF
    include_plots: bool = False,  # Default OFF
) -> Dict[str, Any]:
    """
    Generates and saves various report outputs (summaries, script, plots)
    for a single processed item based on the provided segments and flags.

    Args:
        segments: The final, relabeled list of segment dictionaries.
        item_identifier: A unique identifier for this item (e.g., 'item_1', 'video_id').
        item_output_dir: The specific directory to save outputs for this item.
        config: The application configuration dictionary.
        log_file_handle: The main log file handle for logging progress/errors.
        include_json_summary: Flag to generate detailed JSON emotion summary.
        include_csv_summary: Flag to generate high-level CSV emotion summary.
        include_script: Flag to generate the plain text script file.
        include_plots: Flag to generate emotion plots.

    Returns:
        A dictionary containing the paths to the generated files. Keys might include:
        'json_summary_path', 'csv_summary_path', 'script_path', 'plot_paths'.
        Values will be None if the corresponding output was not requested or failed.
    """

    # Helper for logging within this function context
    def report_log(level, message):
        full_message = f"[{item_identifier}] {message}"
        log_func = log_info  # Default to info
        if level == "warning":
            log_func = log_warning
        elif level == "error":
            log_func = log_error
        log_func(full_message)  # Log to main application logger
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(
                    f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {full_message}\n"
                )
                log_file_handle.flush()
            except Exception as log_e:
                # Fallback print if log writing fails
                print(
                    f"WARN: Failed to write to item log file: {log_e} - Message: {message}"
                )

    report_log("info", f"Generating report outputs in: {item_output_dir}")
    generated_files = {
        "json_summary_path": None,
        "csv_summary_path": None,
        "script_path": None,
        "plot_paths": [],  # Can be multiple plots
    }

    if not segments:
        report_log("warning", "No segments provided, cannot generate report outputs.")
        return generated_files  # Return empty paths

    # --- 1. Calculate Emotion Summary (always needed if saving summary or plots) ---
    summary_data: Optional[EmotionSummary] = None
    if include_json_summary or include_csv_summary or include_plots:
        report_log("info", "Calculating emotion summary...")
        try:
            # Calculate with timeline details, as JSON needs it, and plots benefit from it.
            summary_data = calculate_emotion_summary(segments, include_timeline=True)
            if not summary_data:
                report_log(
                    "warning", "Emotion summary calculation resulted in empty data."
                )
        except Exception as e:
            report_log("error", f"Failed to calculate emotion summary: {e}")
            report_log("error", traceback.format_exc())
            summary_data = None  # Ensure it's None on error

    # --- 2. Save Summaries (conditionally) ---
    if summary_data:
        try:
            json_path, csv_path = save_emotion_summary(
                summary_data,
                item_output_dir,
                item_identifier,
                save_json=include_json_summary,
                save_csv=include_csv_summary,
            )
            generated_files["json_summary_path"] = json_path
            generated_files["csv_summary_path"] = csv_path
            if include_json_summary and not json_path:
                report_log("warning", "JSON summary saving failed.")
            if include_csv_summary and not csv_path:
                report_log("warning", "CSV summary saving failed.")
        except Exception as e:
            report_log("error", f"Error occurred during summary saving: {e}")
            report_log("error", traceback.format_exc())
    elif include_json_summary or include_csv_summary:
        report_log(
            "warning",
            "Skipping summary saving because calculation failed or yielded no data.",
        )

    # --- 3. Save Script Transcript (conditionally) ---
    if include_script:
        report_log("info", "Generating script transcript...")
        try:
            script_path = save_script_transcript(
                segments, item_output_dir, item_identifier
            )
            generated_files["script_path"] = script_path
            if not script_path:
                report_log("warning", "Script transcript saving failed.")
        except Exception as e:
            report_log("error", f"Failed to save script transcript: {e}")
            report_log("error", traceback.format_exc())

    # --- 4. Generate Plots (conditionally) ---
    if include_plots:
        if summary_data:
            report_log("info", "Generating plots...")
            try:
                # generate_all_plots expects output dir string and suffix
                plot_paths_str = generate_all_plots(
                    summary_data, str(item_output_dir), item_identifier
                )
                generated_files["plot_paths"] = [
                    Path(p) for p in plot_paths_str
                ]  # Convert back to Path objects
                report_log(
                    "info", f"Generated {len(generated_files['plot_paths'])} plot(s)."
                )
            except Exception as e:
                report_log("error", f"Plot generation failed: {e}")
                report_log("error", traceback.format_exc())
        else:
            report_log(
                "warning",
                "Skipping plot generation because summary data is unavailable.",
            )

    report_log("info", "Finished generating report outputs for item.")
    return generated_files
