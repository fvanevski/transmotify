# Project Export


## Folder Structure

``
.gitignore
.pytest_cache
  .gitignore
  CACHEDIR.TAG
  README.md
  v
    cache
      lastfailed
      nodeids
      stepwise
.repoignore
analysis
  emotion_fusion.py
  visualization.py
asr
  asr.py
config
  config.py
core
  logging.py
  orchestrator.py
emotion
  audio_model.py
  metrics.py
  text_model.py
  visual_model.py
environment.yml
LICENSE
main.py
README.md
requirements.txt
speaker_id
  id_mapping.py
  vid_preview_id.py
ui
  init.py
  webapp.py
utils
  file_manager.py
  transcripts.py
  wrapper.py
yt
  converter.py
  downloader.py
  metadata.py

``

### .gitignore

*(Unsupported file type)*

### .pytest_cache\.gitignore

*(Unsupported file type)*

### .pytest_cache\CACHEDIR.TAG

*(Unsupported file type)*

### .pytest_cache\README.md

```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

### .pytest_cache\v\cache\lastfailed

*(Unsupported file type)*

### .pytest_cache\v\cache\nodeids

*(Unsupported file type)*

### .pytest_cache\v\cache\stepwise

*(Unsupported file type)*

### .repoignore

*(Unsupported file type)*

### analysis\emotion_fusion.py

```py
# analysis/emotion_fusion.py
"""
Performs late fusion of emotion predictions from different modalities (text, audio).
"""

from typing import Dict, List, Any, Optional, Tuple

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Define standard aligned labels and mappings (moved from MultimodalAnalysis.__init__)
# These could potentially be loaded from config as well in the future.
ALIGNED_LABELS = ["anger", "joy", "sadness", "neutral"]
TEXT_TO_ALIGNED_MAP = {
    "anger": "anger",
    "joy": "joy",
    "sadness": "sadness",
    "neutral": "neutral",
    # Map other text labels if applicable, e.g.:
    "disgust": "anger",  # Example mapping
    "fear": "sadness",  # Example mapping
    "surprise": "joy",  # Example mapping
}
AUDIO_TO_ALIGNED_MAP = {
    "ang": "anger",
    "hap": "joy",
    "sad": "sadness",
    "neu": "neutral",
    # Add mappings for other audio labels if the audio model provides them
}
# Identify text labels that *don't* map to the core aligned set
ALL_TEXT_LABELS_EXAMPLE = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]  # Example list
TEXT_ONLY_LABELS = [
    label for label in ALL_TEXT_LABELS_EXAMPLE if label not in TEXT_TO_ALIGNED_MAP
]


# Extracted logic from core/multimodal_analysis.py::analyze
def fuse_emotions(
    text_emotion_scores: List[Dict[str, Any]],  # List of {'label': str, 'score': float}
    audio_emotion_scores: List[
        Dict[str, Any]
    ],  # List of {'label': str, 'score': float}
    text_fusion_weight: float = 0.6,
    audio_fusion_weight: float = 0.4,
    log_prefix: str = "[Fusion]",
) -> Tuple[str, float, Dict[str, float]]:
    """
    Performs weighted averaging late fusion on aligned text and audio emotion scores.
    Also identifies significant text-only emotions.

    Args:
        text_emotion_scores: Raw output from the text emotion model.
        audio_emotion_scores: Raw output from the audio emotion model.
        text_fusion_weight: Weight for the text modality in fusion.
        audio_fusion_weight: Weight for the audio modality in fusion.
        log_prefix: Prefix for log messages.

    Returns:
        A tuple containing:
            - fused_emotion (str): The final dominant emotion label after fusion.
            - fused_confidence (float): The confidence score of the fused emotion.
            - significant_text_emotions (Dict[str, float]): Text-only emotions with scores
              higher than any individual audio score.
    """
    # Initialize probability dictionaries for aligned labels
    aligned_text_probs = {label: 0.0 for label in ALIGNED_LABELS}
    aligned_audio_probs = {label: 0.0 for label in ALIGNED_LABELS}
    text_only_raw_probs = {label: 0.0 for label in TEXT_ONLY_LABELS}

    # --- Populate probabilities from text model output ---
    max_individual_audio_prob = 0.0  # Track max audio score for text-only check
    valid_text_scores = (
        isinstance(text_emotion_scores, list) and len(text_emotion_scores) > 0
    )
    if valid_text_scores:
        for item in text_emotion_scores:
            label = item.get("label")
            score = item.get("score", 0.0)
            if not isinstance(label, str) or not isinstance(score, (int, float)):
                continue

            # Map to aligned label if possible
            if label in TEXT_TO_ALIGNED_MAP:
                aligned_text_probs[
                    TEXT_TO_ALIGNED_MAP[label]
                ] += score  # Sum scores if multiple map to same aligned label
            elif label in TEXT_ONLY_LABELS:
                text_only_raw_probs[label] = score  # Store raw score for text-only

    # --- Populate probabilities from audio model output ---
    valid_audio_scores = (
        isinstance(audio_emotion_scores, list) and len(audio_emotion_scores) > 0
    )
    if valid_audio_scores:
        # Assuming audio scores are probabilities that sum to 1 for the audio model's native labels
        for item in audio_emotion_scores:
            label = item.get("label")
            score = item.get("score", 0.0)
            if not isinstance(label, str) or not isinstance(score, (int, float)):
                continue

            max_individual_audio_prob = max(
                max_individual_audio_prob, score
            )  # Update max audio score

            # Map to aligned label if possible
            if label in AUDIO_TO_ALIGNED_MAP:
                aligned_audio_probs[AUDIO_TO_ALIGNED_MAP[label]] += score  # Sum scores

    # --- Normalize Aligned Text Probabilities ---
    # Normalize scores within the aligned set so they sum to 1 (or 0 if none exist)
    sum_aligned_text_probs = sum(aligned_text_probs.values())
    if sum_aligned_text_probs > 1e-6:  # Use tolerance for floating point sum
        normalized_aligned_text_probs = {
            label: score / sum_aligned_text_probs
            for label, score in aligned_text_probs.items()
        }
    else:
        normalized_aligned_text_probs = {label: 0.0 for label in ALIGNED_LABELS}

    # --- Normalize Aligned Audio Probabilities ---
    # Do the same for audio probabilities mapped to the aligned set
    sum_aligned_audio_probs = sum(aligned_audio_probs.values())
    if sum_aligned_audio_probs > 1e-6:
        normalized_aligned_audio_probs = {
            label: score / sum_aligned_audio_probs
            for label, score in aligned_audio_probs.items()
        }
    else:
        normalized_aligned_audio_probs = {label: 0.0 for label in ALIGNED_LABELS}

    # --- Weighted Averaging for Aligned Labels ---
    # Ensure weights sum to 1 (simple normalization)
    total_weight = text_fusion_weight + audio_fusion_weight
    if total_weight <= 1e-6:
        log_warning(
            f"{log_prefix} Fusion weights sum to zero. Using equal weights (0.5, 0.5)."
        )
        norm_text_weight = 0.5
        norm_audio_weight = 0.5
    else:
        norm_text_weight = text_fusion_weight / total_weight
        norm_audio_weight = audio_fusion_weight / total_weight

    fused_probs = {label: 0.0 for label in ALIGNED_LABELS}
    for label in ALIGNED_LABELS:
        fused_probs[label] = (
            norm_text_weight * normalized_aligned_text_probs[label]
            + norm_audio_weight * normalized_aligned_audio_probs[label]
        )

    # --- Determine Fused Emotion and Confidence ---
    fused_emotion = "unknown"
    fused_confidence = 0.0
    if fused_probs and sum(fused_probs.values()) > 1e-6:
        # Find the label with the highest fused probability
        fused_emotion = max(fused_probs, key=fused_probs.get)
        fused_confidence = fused_probs[fused_emotion]
    # else: # Keep default 'unknown', 0.0

    # --- Identify Significant Text-Only Emotions ---
    # An emotion is significant if its text score is higher than the max *individual* audio score
    significant_text_emotions = {}
    for text_only_label, text_only_prob in text_only_raw_probs.items():
        # Exclude placeholder/error labels
        if text_only_label in ["no_text", "analysis_failed", "analysis_skipped"]:
            continue

        if text_only_prob > max_individual_audio_prob:
            significant_text_emotions[text_only_label] = text_only_prob
            log_info(
                f"{log_prefix} Significant text-only emotion '{text_only_label}' ({text_only_prob:.2f}) detected (>{max_individual_audio_prob:.2f} max audio score)."
            )

    # log_info(f"{log_prefix} Fusion result: Emotion='{fused_emotion}', Confidence={fused_confidence:.3f}, SignificantText={significant_text_emotions}")
    return fused_emotion, fused_confidence, significant_text_emotions

```

### analysis\visualization.py

```py
# analysis/visualization.py
"""
Generates various plots and visualizations based on emotion analysis results.
Moved from legacy core/plotting.py
"""

import traceback
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Attempt to import matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")  # Use Agg backend for non-interactive plotting
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    log_error("Matplotlib library not found. Plot generation will be unavailable.")
    MATPLOTLIB_AVAILABLE = False
    # Define dummy types if import fails
    Figure = Any
    Axes = Any

# Define types based on expected input structure (previously in core/plotting)
EmotionTimelinePoint = Dict[str, Any]  # Expects 'time', 'emotion', potentially others
SpeakerStats = Dict[
    str, Any
]  # Expects keys like 'emotion_timeline', 'emotion_volatility', etc.
EmotionSummary = Dict[str, SpeakerStats]  # Speaker ID -> SpeakerStats map

# Default Emotion Colors (can be overridden via config if needed later)
DEFAULT_EMOTION_COLORS: Dict[str, str] = {
    "joy": "gold",
    "neutral": "gray",
    "sadness": "blue",
    "anger": "red",
    "surprise": "orange",
    "fear": "purple",
    "disgust": "brown",
    "love": "pink",
    "unknown": "black",
    "analysis_skipped": "lightgrey",
    "analysis_failed": "darkred",
    "no_text": "whitesmoke",
    # Add any other expected emotion labels here
}


def _save_plot(figure: Figure, output_path: Path, log_prefix: str = "[Vis]") -> bool:
    """Helper function to save a matplotlib figure."""
    if not MATPLOTLIB_AVAILABLE:
        log_error(
            f"{log_prefix} Matplotlib not available, cannot save plot {output_path.name}"
        )
        return False
    try:
        # Ensure parent directory exists (use utility function eventually)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")  # Use tight bbox
        log_info(f"{log_prefix} Saved plot: {output_path}")
        return True
    except Exception as e:
        log_error(f"{log_prefix} Failed to save plot {output_path}: {e}")
        log_error(traceback.format_exc())
        return False
    finally:
        # Ensure plot is closed to free memory, even if saving failed
        if MATPLOTLIB_AVAILABLE:
            plt.close(figure)


# --- Plotting Functions ---
# (Moved from core/plotting.py)


def plot_emotion_trajectory(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_prefix: str,
    emotion_value_map: Dict[str, float],  # Needed for sorting y-axis
    emotion_colors: Dict[str, str] = DEFAULT_EMOTION_COLORS,
    log_prefix: str = "[Vis Traj]",
) -> List[Path]:
    """Generates emotion trajectory plots for each speaker."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    plot_files: List[Path] = []

    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2:
            log_warning(
                f"{log_prefix} Skipping trajectory plot for speaker '{speaker}': insufficient timeline data ({len(timeline)} points)."
            )
            continue

        times: List[float] = [
            point.get("time", float(i)) for i, point in enumerate(timeline)
        ]
        emotions: List[str] = [point.get("emotion", "unknown") for point in timeline]

        # Ensure times and emotions have same length after filtering Nones etc.
        if len(times) != len(emotions):
            log_warning(
                f"{log_prefix} Mismatch in time/emotion points for speaker '{speaker}'. Skipping trajectory."
            )
            continue

        try:
            fig: Figure
            ax: Axes
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                times,
                emotions,
                marker="o",
                linestyle="-",
                markersize=4,
                drawstyle="steps-post",
            )  # Use steps-post

            # Sort y-axis labels based on emotion value map
            unique_emotions = sorted(
                list(set(emotions)), key=lambda e: emotion_value_map.get(e, 0.0)
            )
            ax.set_yticks(unique_emotions)  # Set y-ticks to emotion labels

            ax.set_title(f"Emotion Trajectory for {speaker}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Detected Emotion")
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            fig.autofmt_xdate(rotation=30)  # Format x-axis dates/times nicely
            plt.tight_layout()

            # Sanitize speaker name for filename
            safe_speaker_name = re.sub(r"[^\w\-]+", "_", speaker)
            filename: str = f"{file_prefix}_{safe_speaker_name}_emotion_trajectory.png"
            output_path: Path = output_dir / filename
            if _save_plot(fig, output_path, log_prefix):
                plot_files.append(output_path)
        except Exception as e:
            log_error(
                f"{log_prefix} Failed generating trajectory plot for speaker '{speaker}': {e}"
            )
            log_error(traceback.format_exc())
            # Ensure figure is closed even if error occurred before _save_plot
            if "fig" in locals() and MATPLOTLIB_AVAILABLE:
                plt.close(fig)

    return plot_files


def plot_emotion_distribution(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_prefix: str,
    emotion_colors: Dict[str, str] = DEFAULT_EMOTION_COLORS,
    log_prefix: str = "[Vis Dist]",
) -> List[Path]:
    """Generates emotion distribution pie charts for each speaker."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    plot_files: List[Path] = []

    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline:
            log_warning(
                f"{log_prefix} Skipping distribution plot for speaker '{speaker}': missing timeline data."
            )
            continue

        emotions_list: List[str] = [
            point.get("emotion", "unknown") for point in timeline
        ]
        counts: Counter = Counter(emotions_list)
        if not counts:
            log_warning(
                f"{log_prefix} Skipping distribution plot for speaker '{speaker}': no emotions counted."
            )
            continue

        # Prepare data for pie chart
        sorted_items = sorted(counts.items())
        if not sorted_items:
            continue  # Should not happen if counts is not empty

        labels: Tuple[str, ...]
        values: Tuple[int, ...]
        labels, values = zip(*sorted_items)
        colors: List[str] = [
            emotion_colors.get(label, "grey") for label in labels
        ]  # Use grey for unknown colors

        try:
            fig: Figure
            ax: Axes
            fig, ax = plt.subplots(figsize=(8, 8))  # Keep it square
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",  # Format percentages
                colors=colors,
                startangle=90,  # Start first slice at 90 degrees (top)
                pctdistance=0.85,  # Distance of percentage labels from center
            )
            # Improve label readability
            plt.setp(autotexts, size=8, weight="bold", color="white")
            # Draw circle in center to make it a donut chart (optional aesthetic)
            # centre_circle = plt.Circle((0,0),0.70,fc='white')
            # fig.gca().add_artist(centre_circle)

            ax.set_title(f"Emotion Distribution for {speaker}")
            ax.axis("equal")  # Equal aspect ratio ensures a circular pie chart
            plt.tight_layout()  # Adjust layout

            safe_speaker_name = re.sub(r"[^\w\-]+", "_", speaker)
            filename: str = (
                f"{file_prefix}_{safe_speaker_name}_emotion_distribution.png"
            )
            output_path: Path = output_dir / filename
            if _save_plot(fig, output_path, log_prefix):
                plot_files.append(output_path)
        except Exception as e:
            log_error(
                f"{log_prefix} Failed generating distribution plot for speaker '{speaker}': {e}"
            )
            log_error(traceback.format_exc())
            if "fig" in locals() and MATPLOTLIB_AVAILABLE:
                plt.close(fig)

    return plot_files


def plot_emotion_volatility(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_prefix: str,
    log_prefix: str = "[Vis Vol]",
) -> List[Path]:
    """Generates a bar chart comparing emotion volatility across speakers."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    speakers: List[str] = list(summary_data.keys())
    # Ensure volatility is float, default to 0.0 if missing or invalid
    volatility: List[float] = []
    for spk in speakers:
        try:
            vol = float(summary_data[spk].get("emotion_volatility", 0.0))
            volatility.append(vol)
        except (ValueError, TypeError):
            log_warning(
                f"{log_prefix} Invalid volatility value for speaker '{spk}'. Using 0.0."
            )
            volatility.append(0.0)

    if not speakers or not any(
        v > 0 for v in volatility
    ):  # Check if any non-zero volatility
        log_warning(
            f"{log_prefix} Skipping volatility plot: No speakers with non-zero volatility data."
        )
        return []

    # Sort speakers by volatility descending
    sorted_pairs = sorted(
        zip(speakers, volatility), key=lambda item: item[1], reverse=True
    )
    sorted_speakers, sorted_volatility = (
        zip(*sorted_pairs) if sorted_pairs else ([], [])
    )

    try:
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=(max(6, len(sorted_speakers) * 0.8), 6)
        )  # Adjust width based on # speakers
        bars = ax.bar(sorted_speakers, sorted_volatility, color="skyblue")

        # Add value labels on top of bars
        ax.bar_label(bars, fmt="%.2f", padding=3)

        plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
        ax.set_ylabel("Emotion Volatility (StdDev of Scores)")
        ax.set_title("Emotion Volatility by Speaker")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        # Adjust y-limit slightly above max value
        ax.set_ylim(
            bottom=0, top=max(sorted_volatility) * 1.1 if sorted_volatility else 1
        )
        plt.tight_layout()

        filename: str = f"{file_prefix}_emotion_volatility.png"
        output_path: Path = output_dir / filename
        if _save_plot(fig, output_path, log_prefix):
            return [output_path]  # Only one file for this plot type
        else:
            return []
    except Exception as e:
        log_error(f"{log_prefix} Failed generating volatility plot: {e}")
        log_error(traceback.format_exc())
        if "fig" in locals() and MATPLOTLIB_AVAILABLE:
            plt.close(fig)
        return []


def plot_emotion_score_timeline(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_prefix: str,
    emotion_value_map: Dict[str, float],  # Needed to map emotion labels to scores
    emotion_colors: Dict[
        str, str
    ] = DEFAULT_EMOTION_COLORS,  # For coloring points (optional)
    log_prefix: str = "[Vis Score]",
) -> List[Path]:
    """Generates emotion intensity score timeline plots for each speaker."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    plot_files: List[Path] = []

    if not emotion_value_map:
        log_error(
            f"{log_prefix} Emotion value map not provided. Cannot generate score timeline."
        )
        return []

    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2:  # Need at least 2 points to plot a line
            log_warning(
                f"{log_prefix} Skipping score plot for speaker '{speaker}': insufficient timeline data ({len(timeline)} points)."
            )
            continue

        times: List[float] = []
        scores: List[float] = []
        point_colors: List[str] = []  # Optional: color points by emotion

        for i, point in enumerate(timeline):
            time_val = point.get("time")
            emotion_label = point.get("emotion", "unknown")
            if time_val is None:
                time_val = float(i)  # Fallback to index if time is missing
            score = emotion_value_map.get(emotion_label, 0.0)  # Map emotion to score

            times.append(time_val)
            scores.append(score)
            point_colors.append(emotion_colors.get(emotion_label, "grey"))

        if len(times) != len(scores):
            log_warning(
                f"{log_prefix} Mismatch in time/score points for speaker '{speaker}'. Skipping score plot."
            )
            continue

        try:
            fig: Figure
            ax: Axes
            fig, ax = plt.subplots(figsize=(12, 5))
            # Plot line connecting scores
            ax.plot(
                times,
                scores,
                linestyle="--",
                color="darkgrey",
                alpha=0.8,
                drawstyle="steps-post",
            )
            # Scatter plot points, potentially colored by emotion
            ax.scatter(
                times,
                scores,
                marker="o",
                s=25,
                c=point_colors,
                edgecolors="black",
                zorder=3,
            )

            ax.set_title(f"Emotion Intensity Score Timeline for {speaker}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Emotion Intensity Score")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.axhline(
                0, color="black", linewidth=0.8, linestyle="-"
            )  # Zero line for reference
            fig.autofmt_xdate(rotation=30)
            plt.tight_layout()

            safe_speaker_name = re.sub(r"[^\w\-]+", "_", speaker)
            filename: str = (
                f"{file_prefix}_{safe_speaker_name}_emotion_intensity_score.png"
            )
            output_path: Path = output_dir / filename
            if _save_plot(fig, output_path, log_prefix):
                plot_files.append(output_path)
        except Exception as e:
            log_error(
                f"{log_prefix} Failed generating score timeline plot for speaker '{speaker}': {e}"
            )
            log_error(traceback.format_exc())
            if "fig" in locals() and MATPLOTLIB_AVAILABLE:
                plt.close(fig)

    return plot_files


# Moved from core/plotting.py
def generate_all_plots(
    summary_data: EmotionSummary,
    output_dir: Path,
    file_prefix: str,
    # Pass necessary configs/maps required by individual plot functions
    emotion_value_map: Dict[str, float],
    emotion_colors: Dict[str, str] = DEFAULT_EMOTION_COLORS,
    log_prefix: str = "[Vis All]",
) -> List[Path]:
    """
    Generates all standard emotion plots based on the summary data.

    Args:
        summary_data: The calculated emotion summary dictionary.
        output_dir: Directory to save the plot image files.
        file_prefix: A prefix for all generated plot filenames (e.g., 'item_001').
        emotion_value_map: Dictionary mapping emotion labels to numeric scores.
        emotion_colors: Dictionary mapping emotion labels to colors.
        log_prefix: Prefix for log messages.

    Returns:
        A list of Path objects for the generated plot files.
    """
    all_plot_files_paths: List[Path] = []

    if not MATPLOTLIB_AVAILABLE:
        log_error(
            f"{log_prefix} Matplotlib not available. Skipping all plot generation."
        )
        return all_plot_files_paths

    log_info(
        f"{log_prefix} Generating plots with prefix '{file_prefix}' in directory: {output_dir}"
    )

    if not summary_data:
        log_warning(f"{log_prefix} No summary data provided. Skipping plot generation.")
        return all_plot_files_paths
    if not emotion_value_map:
        log_error(
            f"{log_prefix} Emotion value map not provided. Some plots cannot be generated."
        )
        # Decide if we should proceed with plots that don't need it, or stop. Stopping is safer.
        return all_plot_files_paths

    # List of plot functions to call
    # Each function should ideally return List[Path]
    plot_functions_to_run = [
        (
            plot_emotion_trajectory,
            {"emotion_value_map": emotion_value_map, "emotion_colors": emotion_colors},
        ),
        (plot_emotion_distribution, {"emotion_colors": emotion_colors}),
        (plot_emotion_volatility, {}),  # Doesn't need extra maps directly
        (
            plot_emotion_score_timeline,
            {"emotion_value_map": emotion_value_map, "emotion_colors": emotion_colors},
        ),
    ]

    for plot_func, kwargs in plot_functions_to_run:
        try:
            log_info(f"{log_prefix} Running plot function: {plot_func.__name__}")
            # Pass common args + specific kwargs
            plot_paths: List[Path] = plot_func(
                summary_data=summary_data,
                output_dir=output_dir,
                file_prefix=file_prefix,
                log_prefix=f"{log_prefix} [{plot_func.__name__}]",
                **kwargs,
            )
            all_plot_files_paths.extend(plot_paths)
        except Exception as e:
            # Log error but continue to try other plot types
            log_error(
                f"{log_prefix} Failed during {plot_func.__name__} generation: {e}\n{traceback.format_exc()}"
            )

    log_info(
        f"{log_prefix} Plot generation finished. Generated {len(all_plot_files_paths)} plot file(s)."
    )
    return all_plot_files_paths

```

### asr\asr.py

```py
# asr/asr.py
"""
Handles Automatic Speech Recognition (ASR) using the WhisperX tool.
Provides functions to run the transcription and diarization pipeline.
"""

import re
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
    from core.logging import log_info, log_warning, log_error
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
    log_info(f"{log_prefix} Starting WhisperX process for: {audio_path.name}")

    if not audio_path.is_file():
        log_error(f"{log_prefix} Input audio file not found: {audio_path}")
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(f"{log_prefix} Failed to create output directory {output_dir}: {e}")
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
        log_warning(
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
            log_info(f"{log_prefix} Loading model...")
        elif "Detected language:" in line:
            log_info(f"{log_prefix} {line.strip()}")
        elif re.search(r"Transcribing \d+ segments \|", line):  # Transcription progress
            log_info(f"{log_prefix} Progress: {line.strip()}")
        elif re.search(r"Aligning \d+ segments \|", line):  # Alignment progress
            log_info(f"{log_prefix} Progress: {line.strip()}")
        elif "Performing diarization" in line:
            log_info(f"{log_prefix} Starting diarization...")
        elif "Diarization complete" in line:
            log_info(f"{log_prefix} Diarization finished.")
        elif "Saving transcriptions to" in line:
            log_info(f"{log_prefix} {line.strip()}")
        # Catch potential Torch/WhisperX warnings/errors
        elif re.search(r"(warning|error|traceback)", line.lower()) and (
            "torch" in line.lower() or "whisper" in line.lower()
        ):
            # Downgrade severity slightly, as some warnings are common
            log_warning(f"{log_prefix} WhisperX/Torch Log: {line.strip()}")

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
    log_info(f"{log_prefix} Executing command: {' '.join(masked_command_log)}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=whisperx_output_callback,
        )
        log_info(f"{log_prefix} WhisperX command finished execution attempt.")

    except RuntimeError as e:
        # safe_run already logged the error details
        log_error(f"{log_prefix} WhisperX command failed execution. See logs above.")
        # Re-raise the error to indicate failure to the caller
        raise
    except Exception as e:
        log_error(f"{log_prefix} Unexpected error during WhisperX execution: {e}")
        raise RuntimeError(f"Unexpected error running WhisperX: {e}") from e

    # --- Find WhisperX Output JSON ---
    # (Integrates logic from legacy run_whisperx)
    log_info(f"{log_prefix} Locating WhisperX output JSON in: {output_dir}")
    # Expected name based on WhisperX default naming convention
    expected_json_path = output_dir / f"{audio_path.stem}.json"

    if expected_json_path.is_file():
        log_info(
            f"{log_prefix} Found WhisperX output at expected path: {expected_json_path}"
        )
        return expected_json_path
    else:
        log_warning(
            f"{log_prefix} Expected WhisperX output '{expected_json_path.name}' not found."
        )
        log_info(
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
            log_error(
                f"{log_prefix} Error iterating through output directory {output_dir}: {e}"
            )
            # Fall through to the error below

        if found_json_files:
            # Sort by modification time (most recent first) as a heuristic? Or just take first?
            # Taking the first found seems simplest based on original logic.
            selected_file = found_json_files[0]
            if len(found_json_files) > 1:
                log_warning(
                    f"{log_prefix} Multiple potential WhisperX JSON files found in {output_dir} (excluding standard names). "
                    f"Using the first one found: {selected_file.name}"
                )
            else:
                log_info(
                    f"{log_prefix} Found WhisperX output by searching: {selected_file}"
                )
            return selected_file
        else:
            # If no file is found after checking expected path and searching
            error_msg = f"{log_prefix} Could not locate WhisperX JSON output in {output_dir} after command execution."
            log_error(error_msg)
            # Use FileNotFoundError to be consistent with original intent if expected file is missing
            raise FileNotFoundError(error_msg)

```

### config\config.py

```py
# config/config.py
# Phase 2 Update: Integrates constants from legacy core/constants.py into defaults
# Source: Based on legacy codebase.txt and core/constants.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Use relative import for logging within the same top-level package
try:
    # Assumes core.logging is now established from Phase 1
    from core.logging import log_error, log_warning, log_info

    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


class Config:
    """
    Loads, validates, and provides access to application configuration.
    Manages default settings, loads overrides from a JSON file and
    environment variables.
    """

    def __init__(self, config_file: str = "config.json"):
        """
        Initializes the Config object by loading defaults, config file, and env vars.

        Args:
            config_file: The path to the JSON configuration file. Defaults to "config.json".
        """
        log_info("Initializing configuration...")
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        log_info("Configuration loaded and validated.")

    def _load_defaults(self) -> Dict[str, Any]:
        """Returns a dictionary of default configuration values, including constants."""
        log_info("Loading configuration defaults...")
        return {
            # --- Directory/File Settings ---
            "output_dir": "output",
            "temp_dir": "temp",
            "log_level": "INFO",
            # --- Core Processing Settings ---
            "hf_token": None,  # Recommended to set via HF_TOKEN env var
            "device": "cpu",  # Can be overridden by DEVICE env var ('cuda' or 'cpu')
            # --- Model Configurations ---
            "whisper_model_size": "large-v3",
            "whisper_language": "auto",  # Set to specific language code (e.g., "en") if needed
            "whisper_batch_size": 16,
            "whisper_compute_type": "float16",  # e.g., "float16", "int8_float16", "int8"
            "audio_emotion_model": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            "pyannote_diarization_model": "pyannote/speaker-diarization-3.1",  # Requires hf_token
            "deepface_detector_backend": "opencv",  # e.g., 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
            # --- Processing Parameters ---
            "min_diarization_duration": 5.0,  # Minimum audio duration for diarization attempt
            "visual_frame_rate": 1,  # Frames per second for visual analysis
            "text_fusion_weight": 0.6,  # Weight for text emotion in fusion
            "audio_fusion_weight": 0.4,  # Weight for audio emotion in fusion
            # --- Report Flags ---
            "include_json_summary": True,  # Generate detailed emotion_summary.json
            "include_csv_summary": False,  # Generate high-level emotion_summary.csv
            "include_script_transcript": False,  # Generate simple script_transcript.txt
            "include_plots": False,  # Generate emotion plots
            "include_source_audio": True,  # Include original audio in final zip
            # --- Cleanup Flag ---
            "cleanup_temp_on_success": True,  # Delete temp folder after successful run
            # --- Interactive Speaker Labeling ---
            "enable_interactive_labeling": False,  # Enable/disable the labeling UI flow
            "speaker_labeling_min_total_time": 15.0,  # Min total secs speaker must talk
            "speaker_labeling_min_block_time": 10.0,  # Min secs for one continuous block
            "speaker_labeling_preview_duration": 5.0,  # Duration of preview clips (approx)
            # --- Constants moved from core/constants.py ---
            "emotion_value_map": {  # Used for scoring/volatility calculations [cite: 26]
                "joy": 1.0,
                "love": 0.8,
                "surprise": 0.5,
                "neutral": 0.0,
                "fear": -1.5,
                "sadness": -1.0,
                "disgust": -1.8,
                "anger": -2.0,
                "unknown": 0.0,
                "analysis_skipped": 0.0,
                "analysis_failed": 0.0,
                "no_text": 0.0,
            },
            "log_file_name": "process_log.txt",  # Name for batch process logs [cite: 27]
            "intermediate_structured_transcript_name": "structured_transcript_intermediate.json",  # Internal use [cite: 27]
            "final_structured_transcript_name": "structured_transcript_final.json",  # Output per item [cite: 27]
            "emotion_summary_csv_name": "emotion_summary.csv",  # Base name for CSV summary [cite: 27]
            "emotion_summary_json_name": "emotion_summary.json",  # Base name for JSON summary [cite: 27]
            "script_transcript_name": "script_transcript.txt",  # Base name for text script [cite: 27]
            "final_zip_suffix": "_final_bundle.zip",  # Suffix for final batch zip [cite: 27]
            "default_snippet_match_threshold": 0.80,  # Default fuzzy match score (0.0-1.0) [cite: 27]
            # --- Deprecated/Potentially Unused (kept for reference) ---
            "batch_size": 10,  # Seems unused in favor of whisper_batch_size [cite: 8]
            "snippet_match_threshold": 0.80,  # Deprecated in favor of default_snippet_match_threshold [cite: 8]
        }

    def _load_config(self):
        """Loads configuration from defaults, JSON file, and environment variables."""
        defaults = self._load_defaults()
        loaded_config = {}

        # Load from JSON file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                log_info(f"Loaded configuration overrides from {self.config_file}")
            except json.JSONDecodeError:
                log_warning(
                    f"Error decoding JSON from {self.config_file}. Using defaults/env vars only."
                )  # Changed log level
            except Exception as e:
                log_warning(
                    f"Error loading {self.config_file}: {e}. Using defaults/env vars only."
                )  # Changed log level
        else:
            log_info(
                f"Configuration file {self.config_file} not found. Using default configuration and environment variables."
            )

        # Merge defaults and loaded config
        self.config = {**defaults, **loaded_config}
        log_info("Merged defaults and file configuration.")

        # Override specific critical settings with environment variables
        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.config["hf_token"] = env_hf_token
            log_info("Overriding 'hf_token' with environment variable HF_TOKEN.")
        elif not self.config.get("hf_token"):
            # Warning issued during validation if needed
            pass

        env_device = os.getenv("DEVICE")
        if env_device and env_device in ["cuda", "cpu"]:
            self.config["device"] = env_device
            log_info(
                f"Overriding 'device' with environment variable DEVICE: {env_device}"
            )
        elif env_device:
            log_warning(
                f"Environment variable DEVICE ('{env_device}') is invalid. Use 'cuda' or 'cpu'. Using config value ('{self.config.get('device')}')."
            )

    def _validate_config(self):
        """Performs basic validation on critical configuration settings."""
        log_info("Validating configuration...")
        hf_token = self.config.get("hf_token")
        # Allow hf_token to be None, but warn if diarization models are selected later?
        # For now, only warn if it's missing *and* diarization model looks like default pyannote
        is_pyannote_default = "pyannote/speaker-diarization" in self.config.get(
            "pyannote_diarization_model", ""
        )
        if not hf_token and is_pyannote_default:
            log_warning(
                "Hugging Face token ('hf_token') is missing (checked config and HF_TOKEN env var). "
                "The default Pyannote diarization model requires a token. Diarization may fail."
            )
        elif hf_token:
            log_info("Hugging Face token is configured.")

        device = self.config.get("device")
        if device not in ["cuda", "cpu"]:
            original_device = device
            self.config["device"] = "cpu"  # Fallback to CPU
            log_warning(
                f"Invalid device '{original_device}' specified. Falling back to 'cpu'."
            )
        else:
            log_info(f"Device configured to '{device}'.")

        # Validate fusion weights
        w_text = self.config.get("text_fusion_weight", 0.0)
        w_audio = self.config.get("audio_fusion_weight", 0.0)
        # Use tolerance for floating point comparison
        if not (0.99 <= (w_text + w_audio) <= 1.01):
            log_warning(
                f"Fusion weights (text: {w_text}, audio: {w_audio}) do not sum close to 1. Normalization might occur later."
            )

        # Validate speaker labeling parameters (basic type/range check)
        try:
            float(self.config.get("speaker_labeling_min_total_time", 0.0))
            float(self.config.get("speaker_labeling_min_block_time", 0.0))
            preview_duration = float(
                self.config.get("speaker_labeling_preview_duration", 0.0)
            )
            if preview_duration <= 0:
                log_warning(
                    "'speaker_labeling_preview_duration' must be positive. Check config."
                )
        except (ValueError, TypeError):
            log_warning(
                "Speaker labeling time parameters (min_total_time, min_block_time, preview_duration) must be numbers. Check config."
            )

        # Validate default snippet threshold
        try:
            thresh = float(self.config.get("default_snippet_match_threshold", 0.8))
            if not (0.0 <= thresh <= 1.0):
                log_warning(
                    "'default_snippet_match_threshold' must be between 0.0 and 1.0. Check config."
                )
                # Optionally clamp the value here if desired
        except (ValueError, TypeError):
            log_warning(
                "'default_snippet_match_threshold' must be a number. Check config."
            )

    def save_config(self):
        """Saves the current non-default configuration back to the config file."""
        log_info(f"Attempting to save configuration to {self.config_file}...")
        try:
            config_dir = self.config_file.parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Only save keys that are different from defaults or commonly overridden
            defaults = self._load_defaults()
            config_to_save = {}
            for key, value in self.config.items():
                # Always save hf_token as null if it came from env var
                if key == "hf_token" and os.getenv("HF_TOKEN"):
                    config_to_save[key] = None
                    continue
                # Save if the key is not in defaults or the value differs
                if key not in defaults or defaults[key] != value:
                    # Basic check for complex types (like dicts) that might not be intended for saving
                    if not isinstance(value, (dict, list)) or key in [
                        "emotion_value_map"
                    ]:  # Explicitly allow saving emotion map override
                        config_to_save[key] = value

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(
                    config_to_save, f, indent=2, ensure_ascii=False, sort_keys=True
                )
            log_info(f"Configuration saved successfully to {self.config_file}")
        except Exception as e:
            log_error(f"Failed to save configuration to {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key, returning default if not found."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a configuration value and saves the config file."""
        log_info(f"Setting configuration key '{key}' and saving.")
        self.config[key] = value
        self.save_config()

    # Helper to get nested dictionary items easily
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Gets a potentially nested configuration value using multiple keys."""
        data = self.config
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return default
        return data

```

### core\logging.py

```py
# core/logging.py
# Source: Based on legacy codebase.txt
"""
Configures and provides basic logging functionality for the application.
"""

import logging
import os
from pathlib import Path  # Import pathlib
from typing import Dict, Any  # Import typing helpers


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up basic file logging based on the provided configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary, expected to
                                 contain 'log_level' and 'output_dir'.
    """
    log_level_str: str = config.get(
        "log_level", "INFO"
    ).upper()  # Get level, default INFO, ensure uppercase
    log_dir_str: str = config.get("output_dir", "output")  # Get dir, default "output"
    log_filename: str = "app.log"  # Keep filename simple, could be made configurable

    log_dir_path = Path(log_dir_str)
    log_file_path = log_dir_path / log_filename

    # Ensure the log directory exists
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Cannot log this error to file yet, print to console
        print(f"ERROR: Failed to create log directory {log_dir_path}: {e}")
        # Optionally raise error or exit? For now, continue, logging might fail.

    # Define logging format
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Get numeric log level
    log_level: int = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    try:
        logging.basicConfig(
            filename=str(log_file_path),  # basicConfig expects string path
            level=log_level,
            format=log_format,
            datefmt=date_format,
            # Using force=True to allow reconfiguration if basicConfig was called elsewhere
            force=True,
        )
        # Optional: Prevent libraries (like transformers) from flooding logs below a certain level
        # logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.info("Logging setup complete.")  # Log initial message
    except Exception as e:
        # If basicConfig fails, print error to console
        print(f"ERROR: Failed to configure logging to file {log_file_path}: {e}")


def log_info(message: str) -> None:
    """Logs an informational message."""
    # Get logger for the module calling this function (helps identify source)
    # Note: Using __name__ might show 'core.logging' if called directly,
    # consider using inspect.stack()[1].filename for caller module if needed.
    logger = logging.getLogger(__name__)
    logger.info(message)


def log_error(message: str) -> None:
    """Logs an error message."""
    logger = logging.getLogger(__name__)
    logger.error(message)


def log_warning(message: str) -> None:
    """Logs a warning message."""
    logger = logging.getLogger(__name__)
    logger.warning(message)

```

### core\orchestrator.py

```py
# core/orchestrator.py
# REVISED: Skips visual analysis call for YouTube URLs based on user clarification.
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

# --- Core Components ---
from core.logging import log_error, log_info, log_warning  # Moved outside try

try:
    from config.config import Config

    # from core.logging import log_error, log_info, log_warning # Removed from here
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
    from asr.asr import run_whisperx
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
    log_error(f"Orchestrator failed to import core components: {e}")
    raise RuntimeError(f"Orchestrator failed to import core components: {e}") from e

# Type Aliases (Consider moving to a types file if complex)
# LabelingState = Dict[str, Dict[str, Any]]  # Removed duplicate definition
BatchOutputFiles = Dict[
    str, Dict[str, Union[str, Path]]
]  # batch_job_id -> {file_type: path} # Match create_zip_archive
# Type Hints
EmotionSummary = Dict[str, Dict[str, Any]]
SpeakerLabels = Optional[Dict[str, str]]
LabelingItemState = Dict[str, Any]
LabelingBatchState = Dict[str, Union[LabelingItemState, List[str], Dict[str, bool]]]
LabelingState = Dict[str, LabelingBatchState]


class Orchestrator:
    """Manages the speech analysis workflow, state, and interactions."""

    def __init__(self, config: Config):
        self.config = config
        # self.config: Dict[str, Any] = config.config # Removed redundant assignment
        self.labeling_state: LabelingState = {}
        self.batch_output_files: Dict[str, Dict[str, Path]] = {}
        log_info("Orchestrator: Initializing emotion models...")
        self._init_emotion_models()
        log_info("Orchestrator initialized.")

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
        log_info(
            f"{log_prefix} Attempting to save detailed emotion summary JSON to: {output_path}"
        )
        try:
            summary_serializable = convert_floats(summary_data)
            if save_text_file(
                json.dumps(summary_serializable, indent=2, ensure_ascii=False),
                output_path,
            ):
                log_info(
                    f"{log_prefix} Detailed emotion summary JSON saved successfully."
                )
                return output_path
            else:
                log_error(
                    f"{log_prefix} Failed to save emotion summary JSON using file manager."
                )
                return None
        except Exception as e:
            log_error(
                f"{log_prefix} Failed to serialize or save emotion summary JSON to {output_path}: {e}"
            )
            log_error(traceback.format_exc())
            return None

    def _save_csv_summary(
        self, summary_data: EmotionSummary, output_path: Path, log_prefix: str
    ) -> Optional[Path]:
        log_info(
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
                log_info(
                    f"{log_prefix} High-level emotion summary CSV saved successfully."
                )
                return output_path
            else:
                log_error(
                    f"{log_prefix} Failed to save emotion summary CSV using file manager."
                )
                return None
        except Exception as e:
            log_error(
                f"{log_prefix} Failed to generate or save emotion summary CSV to {output_path}: {e}"
            )
            log_error(traceback.format_exc())
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
        log_info(f"{log_prefix} Starting batch processing for: {input_source}")

        # Initialize state, paths, counters... (same as before)
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
        return_batch_id: Optional[str] = (
            None  # Initialize for potential labeling return
        )
        total_processed_or_pending: int = 0  # Initialize counter
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
            # Create work dir, setup logging... (same as before)
            if not create_directory(batch_work_path):
                raise RuntimeError(
                    f"Failed to create batch work directory: {batch_work_path}"
                )
            batch_log_path = batch_work_path / self.config.get(
                "log_file_name", "process_log.txt"
            )
            log_file_handle = open(batch_log_path, "w", encoding="utf-8")
            log_info(f"{log_prefix} Batch log file created at: {batch_log_path}")
            self.batch_output_files[batch_job_id][batch_log_path.name] = batch_log_path

            # Read Input Source (XLSX assumed for now)... (same as before)
            if not Path(input_source).is_file() or not str(input_source).endswith(
                ".xlsx"
            ):
                raise ValueError(
                    f"Invalid input source. Expecting an XLSX file path: {input_source}"
                )
            log_info(f"{log_prefix} Reading batch file: {input_source}")
            try:
                df = pd.read_excel(input_source, sheet_name=0)
                total_items = len(df)
                url_col_name = df.columns[0]
                if df.empty:
                    raise ValueError("Excel file contains no data rows.")
                log_info(
                    f"{log_prefix} Read {total_items} rows. Using column '{url_col_name}'."
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to read or parse Excel file {input_source}: {e}"
                ) from e

            # Get labeling config... (same as before)
            enable_labeling = self.config.get("enable_interactive_labeling", False)
            labeling_min_total_time = float(
                self.config.get("speaker_labeling_min_total_time", 15.0)
            )
            labeling_min_block_time = float(
                self.config.get("speaker_labeling_min_block_time", 10.0)
            )
            log_info(f"{log_prefix} Interactive Labeling Enabled: {enable_labeling}")
            items_requiring_labeling_list = []

            # --- Item Processing Loop ---
            # Use enumerate for a reliable 1-based index
            for item_index, (sequential_index, row) in enumerate(
                df.iterrows(), start=1
            ):
                # item_index = int(sequential_index) + 1 # Removed old index calculation
                item_identifier = (
                    f"item_{item_index:03d}"  # Use enumerate index directly
                )
                item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
                log_info(
                    f"{item_log_prefix} --- Processing item {item_index}/{total_items} ---"
                )

                # Get source URL/Path... (same as before)
                source_url_or_path = row.get(url_col_name)
                if (
                    not isinstance(source_url_or_path, str)
                    or not source_url_or_path.strip()
                ):
                    log_warning(
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

                # Create item work dir... (same as before)
                item_work_path = batch_work_path / item_identifier
                if not create_directory(item_work_path):
                    log_error(
                        f"{item_log_prefix} Failed to create item work directory. Skipping."
                    )
                    failed_count += 1
                    batch_results_list.append(
                        f"[{item_identifier}] Failed: Cannot create work directory."
                    )
                    continue

                # --- Run Pipeline Stages ---
                audio_input_path: Optional[Path] = None
                metadata: Optional[Dict[str, Any]] = None
                segments: Optional[SegmentsList] = None
                processed_segments: Optional[SegmentsList] = None

                try:
                    # 1. Prepare Audio & Metadata (Download/Copy, Convert)... (same logic as before)
                    temp_audio_dir = item_work_path / "audio_temp"
                    create_directory(temp_audio_dir)
                    raw_dl_path = temp_audio_dir / f"raw_download_{item_identifier}"
                    final_wav_path = item_work_path / f"audio_{item_identifier}.wav"
                    if is_youtube_url:
                        log_info(f"{item_log_prefix} Downloading YouTube stream...")
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
                        log_info(f"{item_log_prefix} Converting to WAV...")
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
                        log_info(f"{item_log_prefix} Fetching metadata...")
                        metadata = fetch_youtube_metadata(
                            youtube_url=source_url_or_path, log_prefix=item_log_prefix
                        )
                        if metadata:
                            metadata["prepared_audio_path"] = str(audio_input_path)
                        else:
                            log_warning(f"{item_log_prefix} Failed to fetch metadata.")
                    elif is_local_file:
                        log_info(f"{item_log_prefix} Processing local file...")
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

                    # 2. Duration Check... (same as before)
                    min_duration = float(
                        self.config.get("min_diarization_duration", 5.0)
                    )
                    if not check_audio_duration(
                        audio_input_path, min_duration, item_log_prefix
                    ):
                        log_warning(f"{item_log_prefix} Audio duration too short.")

                    # 3. Run ASR (WhisperX)... (same as before)
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
                    whisperx_json_path = run_whisperx(
                        audio_path=audio_input_path,
                        output_dir=asr_output_dir,
                        model_size=self.config.get("whisper_model_size", "large-v3"),
                        device=self.config.get("device", "cpu"),
                        compute_type=self.config.get("whisper_compute_type", "float16"),
                        language=self.config.get("whisper_language"),
                        batch_size=self.config.get("whisper_batch_size", 16),
                        hf_token=self.config.get("hf_token"),
                        min_speakers=self.config.get("diarization_min_speakers"),
                        max_speakers=self.config.get("diarization_max_speakers"),
                        output_filename_exclusions=[
                            name for name in excluded_asr_outputs if name
                        ],
                        log_file_handle=log_file_handle,
                        log_prefix=item_log_prefix,
                    )
                    if not whisperx_json_path:
                        raise RuntimeError("ASR (WhisperX) execution failed.")

                    # 4. Structure Transcript... (same as before)
                    segments = convert_json_to_structured(whisperx_json_path)
                    if not segments:
                        log_warning(
                            f"{item_log_prefix} No segments found after structuring ASR output."
                        )

                    # 5. Run Emotion Analysis & Fusion (if segments exist)
                    if segments:
                        log_info(
                            f"{item_log_prefix} Running emotion analysis for {len(segments)} segments..."
                        )
                        processed_segments = []
                        visual_emotion_map: Optional[Dict[int, Optional[str]]] = (
                            None  # Store visual results per segment index
                        )

                        # *** REVISED VISUAL ANALYSIS CALL ***
                        video_path_for_visual: Optional[Path] = None
                        if is_local_file:
                            # Basic check if local file might be video
                            vid_extensions = [
                                ".mp4",
                                ".avi",
                                ".mov",
                                ".mkv",
                                ".wmv",
                            ]  # Add more if needed
                            local_path_obj = Path(source_url_or_path)
                            if local_path_obj.suffix.lower() in vid_extensions:
                                video_path_for_visual = local_path_obj

                        if video_path_for_visual:
                            # Only run visual analysis if we have a local video file path
                            log_info(
                                f"{item_log_prefix} Running visual emotion analysis on local file: {video_path_for_visual.name}"
                            )
                            visual_emotion_map = (
                                self.visual_emotion_model.predict_video_segments(
                                    video_path_for_visual, segments
                                )
                            )
                        elif is_youtube_url:
                            log_info(
                                f"{item_log_prefix} Skipping visual emotion analysis for YouTube URL (no local video file)."
                            )
                            visual_emotion_map = (
                                {}
                            )  # Ensure it's an empty dict, not None
                        # else: # Local file but not identified as video
                        # log_info(f"{item_log_prefix} Skipping visual emotion analysis for local file (not identified as video).")
                        # visual_emotion_map = {}

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
                            # Get visual result from pre-calculated map (will be None if YT or no map)
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
                        log_info(
                            f"{item_log_prefix} Emotion analysis and fusion complete."
                        )
                    else:  # No segments
                        log_warning(
                            f"{item_log_prefix} Skipping emotion analysis due to missing segments."
                        )
                        processed_segments = (
                            segments  # Pass original (empty) segments list
                        )

                    segments = processed_segments  # Use processed segments

                except Exception as e:  # Catch errors in the pipeline steps
                    err_msg = (
                        f"{item_log_prefix} ERROR during item processing pipeline: {e}"
                    )
                    log_error(err_msg)
                    log_error(traceback.format_exc())
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
                            pass  # Ignore errors during cleanup
                    continue  # Next item

                # Check Labeling Needed... (same logic as before)
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
                        log_info(
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
                        log_info(
                            f"{item_log_prefix} Labeling enabled, but no eligible speakers."
                        )
                elif enable_labeling:
                    log_info(
                        f"{item_log_prefix} Labeling enabled, but not YT URL or no segments."
                    )

                # Finalize Immediately or Defer... (same logic as before)
                if not needs_labeling:
                    log_info(f"{item_log_prefix} Finalizing item immediately.")
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
                    except Exception as final_e:
                        log_error(
                            f"{item_log_prefix} Immediate finalization failed: {final_e}"
                        )
                        log_error(traceback.format_exc())
                        batch_results_list.append(
                            f"[{item_identifier}] Failed: Finalization error."
                        )
                        failed_count += 1
                        if item_work_path.exists():
                            try:
                                shutil.rmtree(item_work_path)
                            except Exception:
                                pass  # Ignore errors during cleanup

                log_info(
                    f"{item_log_prefix} --- Finished item {item_index}/{total_items} ---"
                )
            # --- End of Item Loop ---

            # Store labeling order, check overall status... (same as before)
            if labeling_is_required_overall:
                self.labeling_state[batch_job_id][
                    "items_requiring_labeling_order"
                ] = items_requiring_labeling_list
            total_processed_or_pending = (
                processed_immediately_count + pending_labeling_count
            )
            log_info(
                f"{log_prefix} Batch loop complete. Finalized: {processed_immediately_count}, Pending: {pending_labeling_count}, Failed: {failed_count}"
            )
            if total_processed_or_pending == 0 and failed_count > 0:
                raise RuntimeError("No items processed successfully or queued.")

            # Determine final status message & return ID if labeling needed... (same as before)
            if labeling_is_required_overall:
                batch_status_message = f"{log_prefix} Initial processing complete. {pending_labeling_count} item(s) require labeling."
                return_batch_id: Optional[str] = batch_job_id
            else:
                log_info(f"{log_prefix} No labeling needed. Creating final ZIP.")
                permanent_output_dir = Path(self.config.get("output_dir", "./output"))
                zip_suffix = self.config.get("final_zip_suffix", "_final_bundle.zip")
                master_zip_name = f"{batch_job_id}_batch_results{zip_suffix}"
                master_zip_path = permanent_output_dir / master_zip_name
                # Cast the files dict to match the function signature
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
            log_info(f"{log_prefix} Batch Status: {batch_status_message}")

        # Exception Handling... (same general structure as before)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            err_msg = f"{log_prefix} BATCH ERROR: {e}"
            log_error(err_msg)
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
            log_error(err_msg + "\n" + traceback.format_exc())
            if log_file_handle:
                log_file_handle.write(f"{err_msg}\n{traceback.format_exc()}\n")
            batch_status_message = err_msg
            return_batch_id = None
            if batch_job_id in self.batch_output_files:
                del self.batch_output_files[batch_job_id]
            if batch_job_id in self.labeling_state:
                del self.labeling_state[batch_job_id]

        finally:
            # Close log, batch cleanup... (same logic as before)
            should_close_log = (
                log_file_handle
                and not log_file_handle.closed
                and not labeling_is_required_overall
            )
            if should_close_log:
                log_info(f"{log_prefix} Closing batch log file.")
                if log_file_handle:  # Add extra check to satisfy linter
                    log_file_handle.close()
            elif (
                log_file_handle
                and not log_file_handle.closed
                and labeling_is_required_overall
            ):
                log_info(f"{log_prefix} Keeping batch log open.")
            cleanup_temp = self.config.get("cleanup_temp_on_success", True)
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
                    log_info(
                        f"{log_prefix} Cleaning up batch temp dir: {batch_work_path}"
                    )
                    if not cleanup_directory(batch_work_path, recreate=False):
                        log_warning(f"{log_prefix} Failed remove batch temp dir.")
                elif labeling_is_required_overall:
                    log_info(f"{log_prefix} Keeping batch temp dir: {batch_work_path}")
                else:
                    log_warning(
                        f"{log_prefix} Skipping cleanup of batch temp dir: {batch_work_path}"
                    )

        # Final Summary String... (same as before)
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

        return batch_status_message, results_summary_string, return_batch_id

    # --- Internal Helper for Finalizing Items ---
    # (No changes needed in this method based on the clarification)
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
        # ... (previous implementation of _finalize_batch_item remains the same) ...
        item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(f"{item_log_prefix} Starting finalization...")
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
                log_error(
                    f"{item_log_prefix} Failed to save final structured transcript."
                )
            # Explicitly cast to Dict[str, bool] to help Pylance with the Union type
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
                log_info(f"{item_log_prefix} Calculating emotion summary...")
                summary_data = calculate_emotion_summary(
                    segments=relabeled_segments,
                    emotion_value_map=self.config.get("emotion_value_map", {}),
                    include_timeline=True,
                    log_prefix=item_log_prefix,
                )
                if not summary_data:
                    log_warning(
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
                log_info(f"{item_log_prefix} Generating plots...")
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
            log_info(f"{item_log_prefix} Finalization complete.")
            return generated_files
        except Exception as e:
            log_error(
                f"{item_log_prefix} Unexpected error during finalization: {e}\n{traceback.format_exc()}"
            )
            return None

    # --- Methods for Interactive Labeling Workflow ---
    # (No changes needed in these methods based on the clarification)
    def start_labeling_item(
        self, batch_job_id: str, item_identifier: str
    ) -> Optional[Tuple[str, str, List[int]]]:
        # ... (previous implementation) ...
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(f"{log_prefix} UI Request: Start labeling item.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            log_error(f"{log_prefix} Item state invalid.")
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
        # ... (previous implementation) ...
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(f"{log_prefix} UI Request: Store label {speaker_id} = '{user_label}'.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            log_error(f"{log_prefix} Item state invalid.")
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
        # ... (previous implementation) ...
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(
            f"{log_prefix} UI Request: Get next speaker after index {current_speaker_index}."
        )
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            log_error(f"{log_prefix} Item state invalid.")
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
        # ... (previous implementation) ...
        log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(f"{log_prefix} UI Request: Skip remaining speakers.")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            log_error(f"{log_prefix} Item state invalid.")
            return False
        success = skip_labeling_for_item(item_state=item_state, log_prefix=log_prefix)
        if success:
            log_info(f"{log_prefix} Triggering finalization after skip.")
            finalized_ok = self.finalize_item(batch_job_id, item_identifier)
            return finalized_ok
        else:
            log_error(f"{log_prefix} skip_labeling_for_item returned False.")
            return False

    def finalize_item(self, batch_job_id: str, item_identifier: str) -> bool:
        # ... (previous implementation) ...
        item_log_prefix = self._get_log_prefix(batch_job_id, item_identifier)
        log_info(f"{item_log_prefix} Orchestrator finalizing item...")
        item_state = self.labeling_state.get(batch_job_id, {}).get(item_identifier)
        if not item_state or not isinstance(item_state, dict):
            log_error(f"{item_log_prefix} Cannot finalize - item state invalid.")
            return False
        segments = item_state.get("segments")
        # Cast retrieved values to help Pylance
        collected_labels = cast(SpeakerLabelMap, item_state.get("collected_labels", {}))
        metadata = cast(Dict[str, Any], item_state.get("metadata", {}))
        audio_path = item_state.get("audio_path")
        item_work_path = item_state.get("item_work_path")
        if not segments or not audio_path or not item_work_path:
            log_error(f"{item_log_prefix} Cannot finalize - missing data in state.")
            self._remove_item_state(batch_job_id, item_identifier)
            return False
        # Cast non-None values after check
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
            log_error(
                f"{item_log_prefix} Unexpected error during item finalization: {e}\n{traceback.format_exc()}"
            )
            finalization_success = False
        finally:
            if log_handle and not log_handle.closed:
                log_handle.close()
            self._remove_item_state(batch_job_id, item_identifier)  # Clean up state
        return finalization_success

    def check_completion_and_zip(self, batch_job_id: str) -> Optional[Path]:
        # ... (previous implementation) ...
        log_prefix = self._get_log_prefix(batch_job_id)
        log_info(f"{log_prefix} UI Request: Check batch completion and ZIP.")
        batch_state = self.labeling_state.get(batch_job_id)
        items_remaining = False
        if batch_state:
            item_keys_present = [k for k in batch_state.keys() if k.startswith("item_")]
            items_remaining = bool(item_keys_present)
        if items_remaining:
            log_info(f"{log_prefix} Batch labeling not complete.")
            return None
        log_info(f"{log_prefix} All items finalized. Proceeding with ZIP creation.")
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
            # Cast the files dict to match the function signature
            files_for_zip_cast = cast(Dict[str, Union[str, Path]], files_for_zip)
            if log_file_path.exists():
                log_handle = open(log_file_path, "a", encoding="utf-8")
                # Ensure the added log file path also fits the cast type
                files_for_zip_cast[log_file_path.name] = log_file_path
            zip_path = create_zip_archive(
                zip_path=master_zip_path,
                files_to_add=files_for_zip_cast,
                log_prefix=log_prefix,
            )
            cleanup_temp = self.config.get("cleanup_temp_on_success", True)
            if zip_path and cleanup_temp and batch_work_path.exists():
                log_info(
                    f"{log_prefix} Cleaning up batch temp dir after ZIP: {batch_work_path}"
                )
                cleanup_directory(batch_work_path, recreate=False)
            elif zip_path and not cleanup_temp:
                log_info(
                    f"{log_prefix} Skipping cleanup of temp dir: {batch_work_path}"
                )
            elif not zip_path and batch_work_path.exists():
                log_warning(
                    f"{log_prefix} Keeping temp dir due to ZIP failure: {batch_work_path}"
                )
        except Exception as e:
            log_error(
                f"{log_prefix} Error during final zip/cleanup: {e}\n{traceback.format_exc()}"
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
        # ... (previous implementation) ...
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
                    log_info(f"{log_prefix} Removed labeling state for item.")
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
                    log_warning(
                        f"{log_prefix} Attempted to remove non-item state for '{item_identifier}'."
                    )
                remaining_item_keys = [
                    k for k in batch_state.keys() if k.startswith("item_")
                ]
                if not remaining_item_keys:
                    log_info(
                        f"{self._get_log_prefix(batch_job_id)} No items left. Removing batch state."
                    )
                    del self.labeling_state[batch_job_id]

```

### emotion\audio_model.py

```py
# emotion/audio_model.py
"""
Handles loading and running audio-based emotion classification models (e.g., SpeechBrain).
"""

import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Attempt imports needed for SpeechBrain and audio handling
try:
    import torch
    import torchaudio
    from speechbrain.inference.interfaces import foreign_class

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    log_error(
        "SpeechBrain, torch, or torchaudio library not found. Audio emotion analysis will be unavailable."
    )
    SPEECHBRAIN_AVAILABLE = False
    # Define dummy types if import fails
    foreign_class = Any
    torch = Any
    torchaudio = Any


class AudioEmotionModel:
    """
    Loads and runs a SpeechBrain (or similar) model for audio emotion classification.
    Designed to work with models loaded via SpeechBrain's foreign_class interface.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        saved_model_path: Optional[
            str
        ] = None,  # Alternative to model_source for local models
        pymodule_file: str = "custom_interface.py",  # May need adjustment based on model
        classname: str = "CustomEncoderWav2vec2Classifier",  # May need adjustment based on model
        device: str = "cpu",
        expected_sample_rate: int = 16000,
    ):
        """
        Initializes the audio emotion classifier using SpeechBrain's foreign_class.

        Args:
            model_source: Identifier for the model on Hugging Face Hub or local path.
            saved_model_path: Path to saved model files (overrides model_source if provided).
            pymodule_file: Python file defining the model interface class.
            classname: The name of the class within pymodule_file to load.
            device: Device to run inference on ('cpu' or 'cuda').
            expected_sample_rate: The sample rate the model expects (e.g., 16000 Hz).
        """
        self.model_source = model_source
        self.saved_path = saved_model_path
        self.pymodule_file = pymodule_file
        self.classname = classname
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.expected_sr = expected_sample_rate
        self.model: Optional[Any] = None  # Stores the loaded SpeechBrain model instance
        self.model_labels: Optional[List[str]] = (
            None  # Stores the labels reported by the model
        )

        if not SPEECHBRAIN_AVAILABLE:
            log_error(
                "Cannot initialize AudioEmotionModel: Required libraries not installed."
            )
            return

        log_info(
            f"Loading audio emotion model source='{model_source}' class='{classname}' onto device: {self.device}..."
        )

        try:
            load_params = {
                "source": self.model_source,
                "pymodule_file": self.pymodule_file,
                "classname": self.classname,
                # Use savedir if loading local model checkpoints
                "savedir": self.saved_path if self.saved_path else None,
                # run_opts are passed during inference, device is handled here
                "run_opts": {
                    "device": self.device
                },  # Ensure model runs on specified device
            }
            # Remove savedir if None, as foreign_class expects it not present or a valid path
            if load_params["savedir"] is None:
                del load_params["savedir"]

            self.model = foreign_class(**load_params)

            # Attempt to get labels from the loaded model (if available)
            if hasattr(self.model, "hparams") and "label_encoder" in self.model.hparams:
                self.model_labels = self.model.hparams.label_encoder.allowed_labels
                log_info(f"Audio model labels identified: {self.model_labels}")
            else:
                log_warning(
                    "Could not automatically determine audio model labels from hparams."
                )
                # Consider adding a manual way to set labels if needed

            log_info("Audio emotion model loaded successfully.")
        except FileNotFoundError as fnf_e:
            log_error(
                f"Failed to load audio model: Required file not found ({fnf_e}). Check model source, path, and pymodule file '{self.pymodule_file}'."
            )
            self.model = None
        except Exception as e:
            log_error(f"Failed to load audio emotion model using foreign_class: {e}")
            log_error(traceback.format_exc())
            self.model = None

    def predict_segment(
        self, audio_path: Path, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Analyzes the emotion of a single audio segment.

        Args:
            audio_path: Path to the full audio file.
            start_time: Start time of the segment in seconds.
            end_time: End time of the segment in seconds.

        Returns:
            A list of dictionaries, each containing 'label' and 'score', derived from
            the model's output probabilities for the segment.
            Returns a default list indicating failure if analysis cannot be performed.
        """
        if not self.model:
            log_warning("Audio emotion model not loaded. Returning 'analysis_failed'.")
            return [{"label": "analysis_failed", "score": 1.0}]

        if end_time <= start_time:
            log_warning(
                f"Invalid segment duration ({start_time=}, {end_time=}). Returning 'analysis_skipped'."
            )
            return [{"label": "analysis_skipped", "score": 1.0}]

        try:
            # Calculate frame offset and number of frames
            frame_offset = int(start_time * self.expected_sr)
            num_frames = int((end_time - start_time) * self.expected_sr)

            if num_frames <= 0:
                log_warning(
                    f"Segment duration resulted in non-positive num_frames ({num_frames=}). Skipping."
                )
                return [{"label": "analysis_skipped", "score": 1.0}]

            # Load the specific audio segment
            waveform, sample_rate = torchaudio.load(
                audio_path, frame_offset=frame_offset, num_frames=num_frames
            )

            # Resample if necessary (though input should ideally match)
            if sample_rate != self.expected_sr:
                log_warning(
                    f"Input sample rate ({sample_rate}Hz) differs from model expected ({self.expected_sr}Hz). Resampling..."
                )
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.expected_sr
                )
                waveform = resampler(waveform)

            # Ensure correct shape (Batch, Time) or (Time,) -> (1, Time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim > 2 or waveform.shape[0] > 1:
                # Handle potential multi-channel, take first channel
                log_warning(
                    f"Audio segment has unexpected shape: {waveform.shape}. Using first channel."
                )
                waveform = waveform[0, :].unsqueeze(0)

            # Move waveform to the correct device
            waveform = waveform.to(self.device)

            # Perform inference - output format depends heavily on the specific model's classify_batch
            # We expect probabilities or logits that can be converted to probabilities.
            with torch.no_grad():  # Ensure no gradients are computed
                out = self.model.classify_batch(waveform)

            # --- Extract Probabilities/Scores ---
            # This part needs careful adaptation based on the actual 'out' structure.
            # Example assumes 'out' might be (probabilities_tensor, ...) or just probabilities_tensor
            raw_scores_tensor = None
            if (
                isinstance(out, tuple)
                and len(out) > 0
                and isinstance(out[0], torch.Tensor)
            ):
                raw_scores_tensor = out[0]
            elif isinstance(out, torch.Tensor):
                raw_scores_tensor = out

            if raw_scores_tensor is None:
                log_warning(
                    f"Unexpected output format from audio model classify_batch: {type(out)}. Cannot extract scores."
                )
                return [{"label": "analysis_failed", "score": 1.0}]

            # Remove batch dimension if present (assuming batch size 1 for single segment)
            if raw_scores_tensor.ndim > 1:
                raw_scores_tensor = raw_scores_tensor.squeeze(0)

            # Apply softmax if the output looks like logits (not probabilities summing to ~1)
            # Simple check: are values outside [0, 1] or do they not sum near 1?
            is_logits = not torch.all(
                (raw_scores_tensor >= -0.01) & (raw_scores_tensor <= 1.01)
            ) or not (0.95 <= torch.sum(raw_scores_tensor).item() <= 1.05)

            if is_logits:
                log_info("Applying softmax to audio model output (assumed logits).")
                probabilities = torch.softmax(raw_scores_tensor, dim=-1)
            else:
                probabilities = raw_scores_tensor  # Assume already probabilities

            # Convert probabilities to numpy array
            scores = probabilities.detach().cpu().numpy()

            # --- Map scores to labels ---
            if self.model_labels and len(scores) == len(self.model_labels):
                result_list = [
                    {"label": self.model_labels[i], "score": float(scores[i])}
                    for i in range(len(scores))
                ]
                # Sort by score descending for potential use later
                result_list.sort(key=lambda x: x["score"], reverse=True)
                return result_list
            elif self.model_labels:
                log_warning(
                    f"Mismatch between number of scores ({len(scores)}) and labels ({len(self.model_labels)}). Returning raw scores."
                )
                # Fallback: return scores with generic labels
                return [
                    {"label": f"score_{i}", "score": float(scores[i])}
                    for i in range(len(scores))
                ]
            else:
                log_warning("Audio model labels unknown. Returning raw scores.")
                # Fallback: return scores with generic labels
                return [
                    {"label": f"score_{i}", "score": float(scores[i])}
                    for i in range(len(scores))
                ]

        except FileNotFoundError:
            log_error(
                f"Audio file not found at {audio_path} during segment prediction."
            )
            return [
                {
                    "label": "analysis_failed",
                    "score": 1.0,
                    "error": "Audio file not found",
                }
            ]
        except Exception as e:
            log_error(
                f"Error during audio emotion analysis for segment {start_time:.2f}-{end_time:.2f}s: {e}"
            )
            log_error(traceback.format_exc())
            return [{"label": "analysis_failed", "score": 1.0}]

```

### emotion\metrics.py

```py
# emotion/metrics.py
"""
Calculates summary metrics and statistics based on emotion analysis results.
"""

import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Type hints (assuming Segment structure is defined elsewhere or basic dict)
Segment = Dict[str, Any]
SegmentsList = List[Segment]
EmotionTimelinePoint = Dict[str, Any]  # Expects 'time', 'emotion', etc.
SpeakerStats = Dict[str, Any]
EmotionSummary = Dict[str, SpeakerStats]  # Speaker ID -> SpeakerStats map


# Moved from core/reporting.py
def calculate_emotion_summary(
    segments: SegmentsList,
    emotion_value_map: Dict[str, float],  # Map from emotion label to numeric score
    include_timeline: bool = True,  # Usually needed for plotting later
    log_prefix: str = "[Metrics]",
) -> EmotionSummary:
    """
    Calculates emotion statistics per speaker from a list of segments.

    Args:
        segments: List of segment dictionaries, must have 'speaker' and 'emotion' keys.
                  'emotion' key should contain the *final* emotion label for the segment.
                  Optionally uses 'start', 'fused_emotion_confidence',
                  'significant_text_emotions' if include_timeline is True.
        emotion_value_map: Dictionary mapping emotion labels to numeric scores for
                           volatility and mean score calculation.
        include_timeline: If True, adds a detailed timestamped emotion timeline to the output.
        log_prefix: Prefix for log messages.

    Returns:
        A dictionary where keys are speaker IDs and values are dictionaries
        containing calculated statistics (counts, dominant, volatility, score_mean, etc.).
    """
    # Use defaultdict to easily append emotions per speaker
    speaker_emotions = defaultdict(list)
    # Use defaultdict for timeline if requested
    speaker_timeline = defaultdict(list) if include_timeline else None

    log_info(
        f"{log_prefix} Calculating emotion summary for {len(segments)} segments..."
    )

    if not emotion_value_map:
        log_error(
            f"{log_prefix} Emotion value map is empty. Cannot calculate score-based metrics (volatility, mean)."
        )
        # Decide how to proceed - return empty summary or summary without score metrics?
        # Returning partial summary seems more informative.
        # return {} # Option 1: Fail completely
        calculate_score_metrics = False
    else:
        calculate_score_metrics = True

    for i, seg in enumerate(segments):
        # Ensure speaker is treated as a string
        spk = str(
            seg.get("speaker", f"unknown_speaker_{i}")
        )  # Use index if speaker missing
        # Use the final 'emotion' key (assumed to be added by fusion or earlier step)
        emo = seg.get("emotion", "unknown")  # Default to 'unknown' if missing

        speaker_emotions[spk].append(emo)

        # If timeline requested, capture emotion and time, plus other details if available
        if include_timeline and speaker_timeline is not None:
            timeline_entry: Dict[str, Any] = {
                "time": seg.get("start"),  # Use segment start time
                "emotion": emo,
                # Include other potentially useful info from the segment if present
                "fused_confidence": seg.get("fused_emotion_confidence"),
                "significant_text": seg.get("significant_text_emotions"),
                "text_raw": seg.get("text_emotion"),  # Raw text model output
                "audio_raw": seg.get("audio_emotion"),  # Raw audio model output
                "visual_raw": seg.get(
                    "visual_emotion"
                ),  # Raw visual model output (often just label)
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
        # Use keys from emotion_value_map to define indicative vs non-indicative?
        # Or use a hardcoded list? Using hardcoded for now based on legacy.
        non_indicative = ["unknown", "analysis_skipped", "analysis_failed", "no_text"]
        filterable_emotions = [e for e in emotions_list if e not in non_indicative]

        if filterable_emotions:
            dominant_emotion = Counter(filterable_emotions).most_common(1)[0][0]
        else:
            # If only non-indicative emotions, check if 'neutral' is most common among them
            dominant_emotion = (
                "neutral" if emotion_counts.get("neutral", 0) > 0 else "unknown"
            )

        # --- Calculate Volatility and Mean Score (if possible) ---
        volatility = None
        mean_score = None
        if calculate_score_metrics:
            emotion_values = [
                emotion_value_map.get(e, 0.0) for e in emotions_list
            ]  # Default to 0.0 if emotion not in map
            # Check if any non-zero scores exist before calculating stats
            if any(abs(v) > 1e-6 for v in emotion_values):
                try:
                    volatility = (
                        statistics.stdev(emotion_values)
                        if len(emotion_values) > 1
                        else 0.0
                    )
                    mean_score = statistics.mean(emotion_values)
                except statistics.StatisticsError as stat_err:
                    log_warning(
                        f"{log_prefix} Statistics error for speaker '{spk}' (e.g., only one data point): {stat_err}. Setting volatility/mean to None/0.0."
                    )
                    volatility = 0.0 if len(emotion_values) == 1 else None
                    mean_score = emotion_values[0] if len(emotion_values) == 1 else None
            else:
                # Handle case where all mapped scores are zero
                volatility = 0.0
                mean_score = 0.0

        # Build the summary entry for this speaker
        speaker_summary_entry: Dict[str, Any] = {
            "total_segments": len(emotions_list),
            "dominant_emotion": dominant_emotion,
            "emotion_counts": dict(emotion_counts),  # Convert Counter to dict for JSON
            "emotion_transitions": transitions,
        }
        # Add score metrics only if calculated
        if volatility is not None:
            speaker_summary_entry["emotion_volatility"] = round(volatility, 4)
        if mean_score is not None:
            speaker_summary_entry["emotion_score_mean"] = round(mean_score, 4)

        # Add timeline if requested and available
        if (
            include_timeline
            and speaker_timeline is not None
            and spk in speaker_timeline
        ):
            # Sort timeline by time before adding
            speaker_summary_entry["emotion_timeline"] = sorted(
                speaker_timeline[spk],
                key=lambda x: x.get(
                    "time", float("inf")
                ),  # Use inf for robust sorting if time missing
            )

        summary[spk] = speaker_summary_entry

    log_info(f"{log_prefix} Emotion summary calculated for {len(summary)} speakers.")
    return summary

```

### emotion\text_model.py

```py
# emotion/text_model.py
"""
Handles loading and running text-based emotion classification models.
"""

from typing import Optional, List, Dict, Any, Union

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

try:
    from transformers.pipelines import pipeline
    from transformers.pipelines.base import Pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    log_error(
        "Transformers library not found. Text emotion analysis will be unavailable."
    )
    TRANSFORMERS_AVAILABLE = False
    # Define dummy Pipeline type hint if import fails
    Pipeline = Any


class TextEmotionModel:
    """
    Loads and runs a transformers pipeline for text emotion classification.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[Union[str, int]] = None,
    ):
        """
        Initializes the text emotion classifier.

        Args:
            model_name: The name of the text classification model on Hugging Face Hub.
            device: The device to run the model on (e.g., "cpu", "cuda", 0, 1).
                    If None, transformers pipeline will attempt auto-detection.
        """
        self.model_name = model_name
        self.device_arg = device  # Store device arg for pipeline
        self.classifier: Optional[Pipeline] = None

        if not TRANSFORMERS_AVAILABLE:
            log_error(
                "Cannot initialize TextEmotionModel: Transformers library not installed."
            )
            return

        log_info(f"Loading text emotion classifier model: {self.model_name}...")
        try:
            # Ensure all scores are returned for detailed analysis later
            # Pass device explicitly if provided, otherwise let pipeline handle it
            pipeline_args = {"model": self.model_name, "return_all_scores": True}
            if self.device_arg is not None:
                pipeline_args["device"] = self.device_arg

            if TRANSFORMERS_AVAILABLE:
                self.classifier = pipeline("text-classification", **pipeline_args)
            else:
                self.classifier = None

            # Check if classifier was successfully initialized before accessing device
            if self.classifier:
                resolved_device = (
                    self.classifier.device
                )  # Get actual device used by pipeline
                log_info(
                    f"Text emotion classifier loaded successfully on device: {resolved_device}."
                )
            # No else needed here as the except block handles initialization failure

        except Exception as e:
            log_error(
                f"Failed to load text emotion classifier '{self.model_name}': {e}"
            )
            self.classifier = None

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyzes the emotion of a given text snippet.

        Args:
            text: The input text string.

        Returns:
            A list of dictionaries, each containing 'label' and 'score' for an emotion,
            e.g., [{'label': 'joy', 'score': 0.9}, ...].
            Returns a default list indicating failure or no text if analysis cannot be performed.
        """
        if not self.classifier:
            log_warning(
                "Text emotion classifier not loaded. Returning 'analysis_failed'."
            )
            return [{"label": "analysis_failed", "score": 1.0}]

        if not text or not isinstance(text, str) or not text.strip():
            # Return a default structure indicating no text was provided
            return [{"label": "no_text", "score": 1.0}]

        try:
            # The pipeline returns a list containing one list of dictionaries
            # e.g., [[{'label': 'joy', 'score': 0.9}, ...]]
            result: List[List[Dict[str, float]]] = self.classifier(text)
            if (
                result
                and isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], list)
            ):
                # Return the inner list of score dictionaries
                return result[0]
            else:
                log_warning(
                    f"Unexpected output format from text classifier for text: '{text[:50]}...'. Result: {result}"
                )
                return [{"label": "analysis_failed", "score": 1.0}]
        except Exception as e:
            log_error(
                f"Error during text emotion analysis for text snippet: '{text[:50]}...': {e}"
            )
            # Return a structure indicating analysis failure
            return [{"label": "analysis_failed", "score": 1.0}]

```

### emotion\visual_model.py

```py
# emotion/visual_model.py
"""
Handles loading and running visual-based emotion classification models (e.g., DeepFace).
Analyzes video frames corresponding to speech segments.
"""

import traceback
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any, Union

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Attempt imports needed for DeepFace and video handling
try:
    import cv2
    import torch  # DeepFace uses torch backend selection implicitly sometimes
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    log_error(
        "DeepFace or OpenCV (cv2) library not found. Visual emotion analysis will be unavailable."
    )
    DEEPFACE_AVAILABLE = False
    # Define dummy types if import fails
    cv2 = Any
    DeepFace = Any


class VisualEmotionModel:
    """
    Uses DeepFace to analyze dominant emotions in video frames corresponding to
    given time segments.
    """

    def __init__(
        self,
        detector_backend: str = "opencv",
        analysis_frame_rate: int = 1,  # Target FPS for analysis
        device: str = "cpu",  # Device hint ('cpu' or 'cuda')
    ):
        """
        Initializes the Visual Emotion Model wrapper.

        Args:
            detector_backend: Face detector backend for DeepFace
                              (e.g., 'opencv', 'ssd', 'mtcnn', 'retinaface').
            analysis_frame_rate: Target frames per second to analyze.
            device: Target device ('cpu' or 'cuda'). Note: DeepFace device handling
                    can be complex; this acts as a hint.
        """
        self.detector_backend = detector_backend
        self.frame_rate = max(1, analysis_frame_rate)  # Ensure at least 1 FPS
        # DeepFace uses device IDs (-1 for CPU, 0+ for GPU) or backend-specific settings
        self.device_hint = device  # Store the general device hint

        if not DEEPFACE_AVAILABLE:
            log_error(
                "Cannot initialize VisualEmotionModel: Required libraries not installed."
            )
            return

        log_info(
            f"VisualEmotionModel initialized. Backend: '{self.detector_backend}', Target FPS: {self.frame_rate}, Device Hint: '{self.device_hint}'."
        )
        # Note: DeepFace models are often loaded on first use, not necessarily here.

    def _get_deepface_device_param(self) -> Optional[Union[str, int]]:
        """Determine the device parameter format DeepFace might expect."""
        # This is heuristic. DeepFace backends might handle devices differently.
        # Some might expect 'cuda', some 'cuda:0', some just 0.
        # Returning None often lets DeepFace try its default/auto-detection.
        if self.device_hint == "cuda" and torch.cuda.is_available():
            # Returning None might be safer, let DeepFace/backend decide GPU ID?
            # Or try common formats:
            # return "cuda"
            return 0  # Often works for first GPU
        else:
            # return "cpu"
            return -1  # Often works for CPU

    def predict_video_segments(
        self, video_path: Path, segments: List[Dict[str, Any]]
    ) -> Dict[int, Optional[str]]:
        """
        Analyzes dominant visual emotion for time ranges corresponding to input segments.

        Args:
            video_path: Path to the video file.
            segments: A list of segment dictionaries, each requiring 'start' and 'end' keys
                      (in seconds) to define the time range for analysis.

        Returns:
            A dictionary mapping the original index of each input segment to the
            dominant visual emotion label (str) found within its time range, or None
            if no dominant emotion could be determined for that segment.
            Returns empty dict if analysis fails globally.
        """
        segment_visual_emotions: Dict[int, Optional[str]] = {
            i: None for i in range(len(segments))
        }

        if not DEEPFACE_AVAILABLE:
            log_error("DeepFace not available. Skipping visual emotion analysis.")
            return segment_visual_emotions  # Return dict with all None

        if not video_path.is_file():
            log_error(f"Video file not found for visual analysis: {video_path}")
            return segment_visual_emotions  # Return dict with all None

        log_info(
            f"Analyzing visual emotions for {len(segments)} segments in video: {video_path.name}"
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log_error(f"Failed to open video file: {video_path}")
            return segment_visual_emotions  # Return dict with all None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                log_warning(
                    f"Could not determine valid FPS for {video_path.name}. Assuming 30 FPS."
                )
                fps = 30.0  # Assume standard FPS if detection fails

            frame_interval = max(
                1, int(round(fps / self.frame_rate))
            )  # Analyze every Nth frame
            log_info(
                f"Video FPS: {fps:.2f}. Analyzing frames at interval: {frame_interval} (target ~{self.frame_rate} analysis FPS)"
            )

            # Store dominant emotion per timestamp {timestamp_sec: dominant_emotion_str}
            frame_emotions: Dict[float, str] = {}
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Check if it was the end or an error
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # Allow a small tolerance for frame count mismatch
                    if total_frames > 0 and current_pos < total_frames - 2:
                        log_warning(
                            f"Error reading frame {frame_idx} / {int(total_frames)} from video file."
                        )
                    break  # End of video or error

                # Process frame at the desired interval
                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / fps
                    try:
                        # Run DeepFace analysis on the frame
                        analysis_results = DeepFace.analyze(
                            img_path=frame,
                            actions=["emotion"],
                            detector_backend=self.detector_backend,
                            enforce_detection=False,  # Don't fail if no face found
                            prog_bar=False,  # Disable internal progress bar
                            # device = self._get_deepface_device_param() # Pass device hint - careful! might cause issues
                        )

                        # DeepFace returns list of dicts, one per face
                        if (
                            isinstance(analysis_results, list)
                            and len(analysis_results) > 0
                        ):
                            # Find the dominant emotion among all detected faces in this frame
                            emotions_in_frame = [
                                res.get("dominant_emotion")
                                for res in analysis_results
                                if res.get("dominant_emotion")
                            ]
                            if emotions_in_frame:
                                # Simple majority vote for dominant emotion in the frame
                                most_common_emotion = Counter(
                                    emotions_in_frame
                                ).most_common(1)[0][0]
                                frame_emotions[timestamp_sec] = most_common_emotion
                                # log_info(f"  Frame {frame_idx} ({timestamp_sec:.2f}s): Detected emotion - {most_common_emotion}") # Verbose

                    except ValueError as ve:
                        # Catch errors like "Face could not be detected." if enforce_detection=True
                        # log_warning(f"DeepFace analysis skipped for frame {frame_idx} ({timestamp_sec:.2f}s): {ve}")
                        pass  # Ignore frames where analysis fails or no face detected
                    except Exception as e:
                        log_warning(
                            f"DeepFace analysis error at frame {frame_idx} ({timestamp_sec:.2f}s): {e}"
                        )
                        # Avoid logging full traceback for every frame error unless debugging
                        # log_warning(traceback.format_exc())
                        pass  # Continue to next frame

                frame_idx += 1

            log_info(
                f"Finished analyzing {frame_idx} video frames. Found emotions at {len(frame_emotions)} timestamps."
            )

            # --- Assign frame emotions to segments ---
            log_info("Assigning detected visual emotions to segments...")
            for i, seg in enumerate(segments):
                seg_start = seg.get("start")
                seg_end = seg.get("end")

                if seg_start is None or seg_end is None:
                    log_warning(
                        f"Segment {i} missing start/end time. Cannot assign visual emotion."
                    )
                    continue

                # Collect emotions from frames within the segment's time range
                emotions_in_segment_range = [
                    emo
                    for ts, emo in frame_emotions.items()
                    if seg_start <= ts < seg_end
                ]

                if emotions_in_segment_range:
                    # Assign the most frequent dominant emotion found within the segment's time range
                    dominant_seg_emotion = Counter(
                        emotions_in_segment_range
                    ).most_common(1)[0][0]
                    segment_visual_emotions[i] = dominant_seg_emotion
                    # log_info(f"  Segment {i} ({seg_start:.2f}-{seg_end:.2f}s): Assigned visual emotion - {dominant_seg_emotion}") # Verbose
                # else: # Keep as None if no visual emotion found in range
                #    log_info(f"  Segment {i} ({seg_start:.2f}-{seg_end:.2f}s): No visual emotion detected in range.") # Verbose

        except Exception as e:
            log_error(f"Error during video processing loop: {e}")
            log_error(traceback.format_exc())
            # Return potentially partially filled dict in case of error mid-processing
            return segment_visual_emotions
        finally:
            if cap and cap.isOpened():
                cap.release()
                # log_info("Video capture released.") # Verbose

        log_info("Finished assigning visual emotions to segments.")
        return segment_visual_emotions

```

### environment.yml

```yml
name: speech-pipeline
channels:
  - nvidia          # for the CUDA runtime (via pytorch-cuda)
  - pytorch         # for PyTorch, torchvision, torchaudio
  - conda-forge     # for pandas, matplotlib, OpenCV, and other conda‑friendly deps
  - defaults

dependencies:
  # Base Python
  - python=3.10

  # Conda‑managed CUDA + PyTorch stack
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1

  # Data/frame + plotting libraries
  - pandas
  - matplotlib

  # Use pip for the rest of the stack
  - pip
  - pip:
    - tensorflow==2.12           # for DeepFace
    - opencv-python              # cv2
    - deepface                   # facial emotion
    - whisperx                   # WhisperX transcription + diarization
    - pyannote.audio             # speaker diarization
    - transformers               # text emotion with HuggingFace
    - speechbrain                # audio emotion
    - gradio                     # UI
    - yt-dlp                     # YouTube downloads
```

### LICENSE

*(Unsupported file type)*

### main.py

```py
# main.py
# Revised entry point using the refactored Orchestrator and UI components.

import sys
import traceback
import torch

# Import refactored components
try:
    from config.config import Config
    from core.logging import setup_logging, log_error, log_info, log_warning
    from core.orchestrator import Orchestrator
    from ui.webapp import UI  # Import the UI class from the new location
except ImportError as e:
    # Use basic print for errors before logging is set up
    print(f"FATAL ERROR: Failed to import core components: {e}")
    print(
        "Ensure you are running from the project root directory and all dependencies are installed."
    )
    sys.exit(1)


def main():
    # Initial environment setup (like torch settings) before logging
    try:
        if torch.cuda.is_available():
            # Check if TF32 is supported (Ampere GPUs onwards)
            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("INFO: TF32 optimizations enabled for CUDA.")
            else:
                print("INFO: TF32 optimizations not supported on this GPU.")
        else:
            print("INFO: CUDA not available, running on CPU.")
    except Exception as e:
        print(f"WARN: Could not set Torch backend settings: {e}")

    # --- Configuration and Logging Setup ---
    try:
        # 1. Initialize Configuration
        config_manager = Config()  # Uses config.json by default

        # 2. Setup Logging (using the loaded config)
        # setup_logging expects the config *dictionary*
        setup_logging(config_manager.config)

        # Optional: Log config source for debugging
        log_info(
            f"Configuration initialized using: {config_manager.config_file.resolve()}"
        )

        # 3. Perform Pre-computation Checks (like HF Token) after logging is ready
        hf_token = config_manager.get("hf_token")
        if not hf_token:
            # This warning is now more effective as logging is configured
            log_warning(
                "Hugging Face token ('hf_token') is missing or null in config/environment. "
                "Diarization may fail if using models requiring authentication (e.g., Pyannote)."
            )
        else:
            # Sensitive info, avoid logging the token itself
            log_info("Hugging Face token is configured.")

    except Exception as e:
        # Catch errors during config/logging setup
        initialization_error = (
            f"FATAL ERROR during initialization: {e}\n{traceback.format_exc()}"
        )
        # Try logging, but also print as logging might have failed
        try:
            log_error(initialization_error)
        except:
            pass
        print(initialization_error)
        sys.exit(1)

    # --- Application Setup and Launch ---
    try:
        # 4. Initialize the Orchestrator (passing the Config instance)
        orchestrator = Orchestrator(config=config_manager)

        # 5. Initialize the UI (passing the Orchestrator instance)
        webapp = UI(orchestrator=orchestrator)

        # 6. Launch the UI
        log_info("Launching Gradio UI...")
        # Check if running in an interactive environment (like IPython/Jupyter)
        # Gradio might not launch correctly or block in such environments.
        if hasattr(sys, "ps1") and sys.ps1:
            log_warning(
                "Running in an interactive shell (e.g., IPython/Jupyter). "
                "Gradio UI might not launch as expected or may block. "
                "Run as a standard Python script for best results."
            )
            # Optionally, you might prevent launch here or change launch parameters
            # webapp.launch(prevent_blocking=True) # Example, check Gradio docs

        # Launch for standard execution (bind to all interfaces)
        # Add other Gradio launch options as needed (e.g., share=True for public link)
        webapp.launch(
            server_name="0.0.0.0"
        )  # Binds to 0.0.0.0 to be accessible on network

        log_info("Gradio UI closed.")

    except ImportError as ie:
        # Catch potential import errors within Orchestrator or UI if missed earlier
        critical_error = f"FATAL ERROR: Missing dependency required by Orchestrator or UI: {ie}\n{traceback.format_exc()}"
        try:
            log_error(critical_error)
        except:
            pass
        print(critical_error)
        sys.exit(1)
    except Exception as e:
        runtime_error = (
            f"FATAL ERROR during application runtime: {e}\n{traceback.format_exc()}"
        )
        try:
            log_error(runtime_error)
        except:
            pass
        print(runtime_error)
        sys.exit(1)


if __name__ == "__main__":
    main()

```

### README.md

```md
# Speech Transcription and Emotion Analysis

    ## Overview
    This project provides a unified pipeline for speech transcription, diarization, and emotion analysis using WhisperX and BERT. It includes a Gradio-based UI for user interaction.

    ## Features
    - Transcription using WhisperX
    - Diarization to identify speakers
    - Emotion analysis using BERT
    - Post-processing for speaker labeling and emotion summary
    - Batch processing support
    - Robust error handling and logging

    ## Installation
    1. Clone the repository:
       ```sh
       git clone https://github.com/your-username/speech-transcription-system.git
       cd speech-transcription-system
       ```
    2. Create and activate a virtual environment:
       ```sh
       virtualenv venv
       .\venv\Scripts\activate  # On Windows
       source venv/bin/activate  # On macOS and Linux
       ```
    3. Install dependencies:
       ```sh
       pip install -r requirements.txt
       ```

    ## Usage
    1. Run the application:
       ```sh
       python main.py
       ```
    2. Open the Gradio UI in your web browser and follow the instructions to upload files and perform tasks.

    ## Directory Structure
    ```
    speech_transcription_system/
    ├── config/
    │   └── config.py
    ├── core/
    │   ├── __init__.py
    │   ├── transcription.py
    │   ├── diarization.py
    │   ├── emotion_analysis.py
    │   ├── file_management.py
    │   └── utils.py
    ├── ui/
    │   ├── __init__.py
    │   ├── main_gui.py
    │   └── postprocess_gui.py
    ├── logs/
    ├── temp/
    ├── output/
    ├── requirements.txt
    └── main.py
    ```

    ## Configuration
    - `config.json`: Configuration file for output directories, batch size, and log level.

    ## Contributing
    Contributions are welcome! Please open an issue or submit a pull request.

    ## License
    This project is licensed under the MIT License.
    ```

#### 4. **Create a Pull Request**

- **Push Your Changes:**
  - Ensure your changes are committed and pushed to the `refactor-core` branch:
    ```sh
    git add .
    git commit -m "Refactor core modules and improve directory structure"
    git push origin refactor-core
    ```

- **Create a Pull Request:**
  - Go to your GitHub repository.
  - Click on the "New pull request" button.
  - Select the `refactor-core` branch as the base branch and the `main` (or `master`) branch as the compare branch.
  - Write a detailed description of the changes you made.
  - Request a code review from your team or peers.

### Additional Tips

- **Continuous Integration (CI):**
  - Consider setting up a CI/CD pipeline to automate testing and deployment.
  - Tools like GitHub Actions, GitLab CI, or Jenkins can be used for this purpose.

- **Code Linting and Formatting:**
  - Use tools like `flake8`, `black`, or `pylint` to ensure your code is well-formatted and follows best practices.

- **Documentation:**
  - Consider using tools like Sphinx or MkDocs to generate comprehensive documentation for your project.

By following these steps, you can ensure that your refactored code is well-organized, thoroughly tested, and ready for integration into the main branch. If you have any more questions or need further assistance, feel free to ask!
```

### requirements.txt

```txt
# requirements.txt

# PyTorch and related packages will be installed via:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126


# Core dependencies
whisperx==3.3.2
transformers==4.51.3
gradio==5.25.1
matplotlib==3.10.1
pandas==2.2.3
yt-dlp==2025.3.31

# Diarization dependency (pinned to specific commit from your environment)
pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git@240a7f3ef60bc613169df860b536b10e338dbf3c

# Add other direct dependencies below if any
```

### speaker_id\id_mapping.py

```py
# speaker_id/id_mapping.py
"""
Handles applying collected speaker labels to transcript segments.
"""

from typing import List, Dict, Any, Optional

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Type Hinting
Segment = Dict[str, Any]
SegmentsList = List[Segment]
SpeakerLabelMap = Dict[str, str]  # Maps Original ID (e.g., SPEAKER_XX) -> User Label


# Extracted logic from core/pipeline.py::finalize_labeled_item
def apply_speaker_labels(
    segments: SegmentsList,
    speaker_labels: Optional[SpeakerLabelMap],
    log_prefix: str = "[ID Map]",
) -> SegmentsList:
    """
    Applies speaker labels from a mapping to a list of segments.
    Modifies the 'speaker' field in the segment dictionaries in-place OR
    returns a new list (currently modifies in-place for efficiency).

    Args:
        segments: The list of segment dictionaries to modify.
        speaker_labels: A dictionary mapping original speaker IDs (e.g., "SPEAKER_00")
                        to user-provided labels (e.g., "Alice"). If None or empty,
                        no relabeling occurs.
        log_prefix: Prefix for log messages.

    Returns:
        The list of segments with updated speaker labels.
    """
    if not segments:
        log_info(f"{log_prefix} No segments provided for relabeling.")
        return []
    if not speaker_labels:
        log_info(f"{log_prefix} No speaker label map provided. Skipping relabeling.")
        return segments  # Return original segments

    log_info(f"{log_prefix} Applying speaker labels based on mapping: {speaker_labels}")
    segments_relabeled_count = 0
    unique_original_ids = set(
        seg.get("speaker") for seg in segments if seg.get("speaker")
    )

    for seg in segments:
        original_speaker_id = str(seg.get("speaker", "unknown"))

        # Check if this original ID has a mapping provided by the user
        if original_speaker_id in speaker_labels:
            final_label = speaker_labels[original_speaker_id]
            # Apply the label only if it's not blank/empty
            if final_label and isinstance(final_label, str) and final_label.strip():
                new_label = final_label.strip()
                if seg["speaker"] != new_label:  # Check if change occurs
                    seg["speaker"] = new_label
                    segments_relabeled_count += 1
            else:
                # If user provided a blank label for this ID, keep the original ID
                # log_info(f"{log_prefix} Keeping original ID for {original_speaker_id} due to blank user label.")
                pass  # Keep original seg["speaker"]

    # Log summary of changes
    applied_ids = {k for k, v in speaker_labels.items() if v and v.strip()}
    unmapped_original_ids = unique_original_ids - set(speaker_labels.keys())
    blank_mapped_ids = {k for k, v in speaker_labels.items() if not v or not v.strip()}

    log_info(
        f"{log_prefix} Applied {segments_relabeled_count} non-blank speaker label instances."
    )
    if unmapped_original_ids:
        log_info(
            f"{log_prefix} Original IDs without user mapping: {sorted(list(unmapped_original_ids))}"
        )
    if blank_mapped_ids:
        log_info(
            f"{log_prefix} Original IDs mapped to blank labels (kept original ID): {sorted(list(blank_mapped_ids))}"
        )

    return segments

```

### speaker_id\vid_preview_id.py

```py
# speaker_id/vid_preview_id.py
"""
Handles identifying speakers eligible for labeling, selecting video preview segments,
and managing the state transitions for the interactive labeling workflow.
"""

import math
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, TextIO

# Assuming utils.transcripts and core.logging are available
try:
    # group_segments_by_speaker is needed for identify_eligible_speakers
    from utils.transcripts import group_segments_by_speaker
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

    # Dummy group_segments_by_speaker if import fails
    def group_segments_by_speaker(segments):
        return []


# Type Hinting
Segment = Dict[str, Any]
SegmentsList = List[Segment]
DialogueBlock = Dict[str, Any]
# Define structure for labeling state items (as used in legacy core/pipeline)
# This state will need to be passed into functions like store_speaker_label
# by the future orchestrator.
LabelingItemState = Dict[str, Any]


# Moved from core/speaker_labeling.py
def identify_eligible_speakers(
    segments: SegmentsList, min_total_time: float, min_block_time: float
) -> List[str]:
    """
    Identifies speakers who meet minimum total speaking time and minimum
    continuous block time thresholds.

    Args:
        segments: List of segment dictionaries from transcription.
        min_total_time: Minimum total seconds a speaker must talk across all segments.
        min_block_time: Minimum seconds for at least one continuous dialogue block.

    Returns:
        A list of unique speaker IDs (e.g., 'SPEAKER_00') eligible for labeling,
        SORTED BY TOTAL SPEAKING TIME in descending order.
    """
    log_info(
        f"Identifying speakers eligible for labeling (min_total_time={min_total_time}s, min_block_time={min_block_time}s)..."
    )
    if not segments:
        log_warning("No segments provided to identify_eligible_speakers.")
        return []

    speaker_total_time = defaultdict(float)
    for seg in segments:
        speaker = seg.get("speaker")
        start = seg.get("start")
        end = seg.get("end")
        if (
            speaker
            and start is not None
            and end is not None
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end > start
        ):
            speaker_total_time[str(speaker)] += end - start

    eligible_speakers_with_time = []
    # Use group_segments_by_speaker (now imported from utils.transcripts)
    all_speaker_blocks: List[DialogueBlock] = group_segments_by_speaker(segments)
    speaker_to_blocks_map = defaultdict(list)
    for block in all_speaker_blocks:
        speaker_id = block.get("speaker")
        if speaker_id:
            speaker_to_blocks_map[str(speaker_id)].append(block)

    # Check eligibility for all speakers found
    unique_speaker_ids = list(speaker_total_time.keys())

    for speaker_id in unique_speaker_ids:
        total_time = speaker_total_time[speaker_id]
        if total_time < min_total_time:
            # log_info(f"Speaker {speaker_id} ineligible: Total time {total_time:.2f}s < {min_total_time}s") # Verbose
            continue  # Skip if total time is too low

        # Check for at least one long enough block
        has_long_block = False
        speaker_blocks = speaker_to_blocks_map.get(speaker_id, [])
        for block in speaker_blocks:
            block_start = block.get("start")
            block_end = block.get("end")
            if (
                block_start is not None
                and block_end is not None
                and isinstance(block_start, (int, float))
                and isinstance(block_end, (int, float))
                and (block_end - block_start) >= min_block_time
            ):
                has_long_block = True
                break

        if not has_long_block:
            # log_info(f"Speaker {speaker_id} ineligible: No single block >= {min_block_time}s") # Verbose
            continue  # Skip if no suitable block found

        # If both conditions met, add to list with time for sorting
        eligible_speakers_with_time.append({"id": speaker_id, "time": total_time})

    # Sort eligible speakers by total speaking time (descending)
    eligible_speakers_with_time.sort(key=lambda x: x["time"], reverse=True)

    # Extract just the sorted IDs
    sorted_eligible_speaker_ids = [spk["id"] for spk in eligible_speakers_with_time]

    log_info(
        f"Found {len(sorted_eligible_speaker_ids)} eligible speakers (ordered by speaking time): {sorted_eligible_speaker_ids}"
    )
    return sorted_eligible_speaker_ids


# Moved from core/speaker_labeling.py
def select_preview_time_segments(
    speaker_id: str,
    segments: SegmentsList,
    preview_duration: float,
    min_block_time: float,
) -> List[int]:  # Returns list of integer start times in seconds
    """
    Selects up to 3 start times (in seconds) for video previews for a given speaker.
    Prioritizes the start of the first 3 blocks longer than min_block_time.
    Uses fallback logic based on longest block if fewer than 3 suitable blocks exist.

    Args:
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') to select segments for.
        segments: The full list of segments for the entire audio.
        preview_duration: The desired duration of each preview clip (used for fallback calc).
        min_block_time: Minimum block duration to be considered for primary selection.

    Returns:
        A list of unique integers, each representing a start time in seconds for a preview clip.
        Returns an empty list if no suitable segments can be found.
    """
    log_info(
        f"Selecting preview start times for {speaker_id} (preview_duration={preview_duration}s, min_block_time={min_block_time}s)..."
    )
    preview_start_times_float = []
    selected_start_times_set = set()  # To ensure uniqueness

    # Filter segments for the target speaker
    speaker_segments = [
        seg for seg in segments if str(seg.get("speaker", "")) == speaker_id
    ]
    if not speaker_segments:
        log_warning(f"No segments found for speaker {speaker_id}.")
        return []

    # Group the speaker's segments into dialogue blocks
    dialogue_blocks = group_segments_by_speaker(speaker_segments)
    if not dialogue_blocks:
        log_warning(f"Could not group segments into blocks for speaker {speaker_id}.")
        return []

    # Sort blocks by start time
    dialogue_blocks.sort(key=lambda b: b.get("start", float("inf")))

    # Identify blocks long enough for primary selection
    long_blocks = []
    for block in dialogue_blocks:
        start = block.get("start")
        end = block.get("end")
        if (
            start is not None
            and end is not None
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and (end - start) >= min_block_time
        ):
            long_blocks.append(block)

    # --- Select primary clips from the start of long blocks ---
    for block in long_blocks:
        if len(preview_start_times_float) >= 3:
            break
        start = block.get("start")
        if start is not None and isinstance(start, (int, float)):
            # Ensure start time is non-negative and floor it for integer seconds
            valid_start = max(0.0, start)
            start_sec_int = math.floor(valid_start)
            if start_sec_int not in selected_start_times_set:
                preview_start_times_float.append(
                    valid_start
                )  # Keep float for potential sorting
                selected_start_times_set.add(
                    start_sec_int
                )  # Add int to set for uniqueness check

    # --- Fallback: Use time near end of the longest block if needed ---
    if len(preview_start_times_float) < 3 and dialogue_blocks:
        # Find the block with the longest duration
        longest_block = max(
            dialogue_blocks,
            key=lambda b: (b.get("end", 0.0) or 0.0) - (b.get("start", 0.0) or 0.0),
        )
        l_start = longest_block.get("start")
        l_end = longest_block.get("end")

        # Check if the longest block is valid and long enough for a preview from its end
        if (
            l_start is not None
            and l_end is not None
            and isinstance(l_start, (int, float))
            and isinstance(l_end, (int, float))
            and (l_end - l_start) >= preview_duration
        ):
            # Calculate start time for the *last* N seconds of the block
            fallback_start_float = max(
                0.0, l_start, l_end - preview_duration
            )  # Ensure non-negative
            fallback_start_int = math.floor(fallback_start_float)

            # Add only if it's distinct (based on integer seconds)
            if fallback_start_int not in selected_start_times_set:
                log_info(
                    f"Using fallback start time near end of longest block ({l_start:.2f}-{l_end:.2f}s) -> Start at {fallback_start_int}s"
                )
                preview_start_times_float.append(fallback_start_float)
                selected_start_times_set.add(fallback_start_int)

    # --- Ensure at least two distinct clips if possible ---
    # If only one preview time was found, try adding the start of the longest block as a second distinct time
    if len(preview_start_times_float) == 1 and dialogue_blocks:
        longest_block = max(
            dialogue_blocks,
            key=lambda b: (b.get("end", 0.0) or 0.0) - (b.get("start", 0.0) or 0.0),
        )
        l_start = longest_block.get("start")
        if l_start is not None and isinstance(l_start, (int, float)):
            valid_l_start_float = max(0.0, l_start)
            valid_l_start_int = math.floor(valid_l_start_float)
            if valid_l_start_int not in selected_start_times_set:
                log_info(
                    f"Adding second start time from start of longest block ({l_start:.2f}s -> {valid_l_start_int}s)"
                )
                preview_start_times_float.append(valid_l_start_float)
                selected_start_times_set.add(valid_l_start_int)

    # Final conversion to unique integer seconds, sorted
    final_start_times_int = sorted(list(selected_start_times_set))

    log_info(
        f"Selected {len(final_start_times_int)} unique preview start times (seconds) for {speaker_id}: {final_start_times_int}"
    )
    return final_start_times_int


# --- Functions below moved from core/pipeline.py ---
# --- IMPORTANT: These functions rely on external state management ---
# --- (e.g., labeling_state dict) which needs to be provided ---
# --- by the calling context (likely the Phase 9 orchestrator). ---


# Moved from core/pipeline.py
def start_interactive_labeling_for_item(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    labeling_config: Dict[str, Any],  # Expects relevant config values
    log_prefix: str = "[Labeling Start]",
) -> Optional[
    Tuple[str, str, List[int]]
]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
    """
    Prepares and returns data for the first speaker to be labeled for an item.
    Depends on item_state containing 'eligible_speakers', 'youtube_url', 'segments'.

    Args:
        item_state: The dictionary containing the state for this specific item.
        labeling_config: Dictionary with relevant config keys like
                         'speaker_labeling_preview_duration', 'speaker_labeling_min_block_time'.
        log_prefix: Prefix for log messages.

    Returns:
        Tuple (speaker_id, youtube_url, start_times_list) or None if setup fails.
    """
    log_info(f"{log_prefix} Starting interactive labeling...")

    eligible_speakers = item_state.get(
        "eligible_speakers", []
    )  # Already sorted by time
    youtube_url = item_state.get("youtube_url")
    segments = item_state.get("segments", [])

    if not eligible_speakers:
        log_warning(
            f"{log_prefix} No eligible speakers found in state. Cannot start labeling."
        )
        # Orchestrator should handle this - maybe finalize immediately?
        return None
    if not youtube_url:
        log_error(f"{log_prefix} YouTube URL missing from item state.")
        return None
    if not segments:
        log_warning(f"{log_prefix} Segments missing from item state.")
        # Continue, preview selection will just return empty list

    first_speaker_id = eligible_speakers[0]
    log_info(f"{log_prefix} First speaker to label: {first_speaker_id}")

    # Get preview config from passed dict
    preview_duration = float(
        labeling_config.get("speaker_labeling_preview_duration", 5.0)
    )
    min_block_time = float(labeling_config.get("speaker_labeling_min_block_time", 10.0))

    start_times = select_preview_time_segments(
        speaker_id=first_speaker_id,
        segments=segments,
        preview_duration=preview_duration,
        min_block_time=min_block_time,
    )  # Returns List[int]

    if not start_times:
        log_warning(
            f"{log_prefix} Could not select preview start times for {first_speaker_id}."
        )
        # Return speaker ID and URL, but empty times list
        return first_speaker_id, youtube_url, []

    return first_speaker_id, youtube_url, start_times


# Moved from core/pipeline.py
def store_speaker_label(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    speaker_id: str,
    user_label: str,
    log_prefix: str = "[Label Store]",
) -> bool:
    """
    Stores the user-provided label for a speaker within the item's state dict.
    The caller (orchestrator) is responsible for managing and persisting item_state.

    Args:
        item_state: The dictionary containing the state for this specific item.
                    This dictionary will be modified in place.
        speaker_id: The original speaker ID (e.g., SPEAKER_XX).
        user_label: The label provided by the user.
        log_prefix: Prefix for log messages.

    Returns:
        True if label was stored (dictionary updated), False otherwise (e.g., state invalid).
    """
    if not isinstance(item_state, dict):
        log_error(
            f"{log_prefix} Invalid item_state provided (not a dict). Cannot store label."
        )
        return False

    if "collected_labels" not in item_state or not isinstance(
        item_state["collected_labels"], dict
    ):
        item_state["collected_labels"] = {}  # Initialize if missing or wrong type

    cleaned_label = (
        user_label.strip() if user_label else ""
    )  # Store blank if user entered nothing
    item_state["collected_labels"][speaker_id] = cleaned_label
    log_info(
        f"{log_prefix} Stored label for {speaker_id}: '{cleaned_label}' (in provided item_state dict)"
    )
    return True


# Moved from core/pipeline.py
def get_next_speaker_for_labeling(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    current_speaker_index: int,  # Index of the speaker *just* labeled
    labeling_config: Dict[str, Any],  # Expects relevant config values
    log_prefix: str = "[Label Next]",
) -> Optional[
    Tuple[str, str, List[int]]
]:  # Returns SpeakerID, YouTubeURL, List[StartTimes]
    """
    Gets the ID, URL, and start times for the next speaker in the item's eligible list.
    Depends on item_state containing 'eligible_speakers', 'youtube_url', 'segments'.

    Args:
        item_state: The dictionary containing the state for this specific item.
        current_speaker_index: The index (in eligible_speakers list) of the speaker
                               that was just processed or labeled.
        labeling_config: Dictionary with relevant config keys like
                         'speaker_labeling_preview_duration', 'speaker_labeling_min_block_time'.
        log_prefix: Prefix for log messages.

    Returns:
        Tuple (speaker_id, youtube_url, start_times_list) for the next speaker,
        or None if all eligible speakers for this item have been processed.
    """

    eligible_speakers = item_state.get("eligible_speakers", [])  # Sorted list
    youtube_url = item_state.get("youtube_url")
    segments = item_state.get("segments", [])

    if not youtube_url:
        log_error(f"{log_prefix} YouTube URL missing from item state for next speaker.")
        return None  # Cannot proceed

    next_speaker_index = current_speaker_index + 1

    if next_speaker_index < len(eligible_speakers):
        next_speaker_id = eligible_speakers[next_speaker_index]
        log_info(
            f"{log_prefix} Getting data for next speaker (Index {next_speaker_index}): {next_speaker_id}"
        )

        preview_duration = float(
            labeling_config.get("speaker_labeling_preview_duration", 5.0)
        )
        min_block_time = float(
            labeling_config.get("speaker_labeling_min_block_time", 10.0)
        )

        start_times = select_preview_time_segments(
            speaker_id=next_speaker_id,
            segments=segments,
            preview_duration=preview_duration,
            min_block_time=min_block_time,
        )

        if not start_times:
            log_warning(
                f"{log_prefix} Could not select preview start times for {next_speaker_id}."
            )
            return next_speaker_id, youtube_url, []

        return next_speaker_id, youtube_url, start_times
    else:
        log_info(f"{log_prefix} All eligible speakers processed for this item.")
        return None  # Signal item completion


# Moved from core/pipeline.py
def skip_labeling_for_item(
    item_state: LabelingItemState,  # Expects the state dict for the specific item
    log_prefix: str = "[Label Skip]",
) -> bool:
    """
    Handles the logic when a user skips labeling for remaining speakers in an item.
    Currently, this function primarily serves as a signal; the actual finalization
    (which uses collected labels so far) is handled separately by the orchestrator
    calling the id_mapping functionality.

    Args:
        item_state: The dictionary containing the state for this specific item.
        log_prefix: Prefix for log messages.

    Returns:
        True (as the skip action itself is always considered successful).
    """
    # This function might manipulate item_state in the future if needed,
    # e.g., setting a flag item_state['skipped'] = True
    log_warning(f"{log_prefix} User requested to skip remaining speakers for item.")
    # The orchestrator will see that the next speaker returned is None (implicitly)
    # after this is called and should then trigger finalization using the
    # labels collected so far in item_state['collected_labels'].
    return True

```

### ui\init.py

```py
# ui/init.py
from .webapp import UI

# Define what gets imported with 'from ui import *'
__all__ = ["UI"]

```

### ui\webapp.py

```py
# ui/webapp.py
"""
Defines the Gradio web interface for the speech analysis application.
Interacts with the core.orchestrator module to run processing pipelines
and manage interactive speaker labeling.
Moved from legacy ui/main_gui.py
"""

import os
import traceback
import re
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator, Union, TYPE_CHECKING

# Imports for type checking only (avoids circular imports)
if TYPE_CHECKING:
    import gradio as gr
    from gradio import themes
    from core.orchestrator import Orchestrator

# Gradio and Pandas are core dependencies for the UI
# Declare GRADIO_AVAILABLE and component placeholders outside the try block
GRADIO_AVAILABLE = False
gr = themes = pd = Any  # type: ignore # Use Any as placeholders initially

try:
    import gradio as gr  # type: ignore
    from gradio import themes  # type: ignore
    import pandas as pd  # type: ignore

    GRADIO_AVAILABLE = True
except ImportError:
    print("ERROR: Gradio or Pandas library not found. UI cannot be launched.")
    # UI cannot function without Gradio, so exiting or raising might be appropriate
    # depending on how main.py handles this. For now, the flag is set.

# Import the backend orchestrator and logging
try:
    # Make Orchestrator import essential. If it fails, the app likely can't run.
    from core.orchestrator import Orchestrator
    from core.logging import log_error, log_warning, log_info
except ImportError as e:
    # Define fallback loggers ONLY if logging import failed
    # If Orchestrator failed, it's a more critical error usually handled in main.py
    print(
        f"ERROR: Failed to import core modules (Orchestrator, logging): {e}. Ensure PYTHONPATH is correct."
    )

    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")  # type: ignore

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")  # type: ignore

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")  # type: ignore

    # Raising here might be better if Orchestrator is critical, or let main.py handle it.
    # raise RuntimeError("Critical component Orchestrator failed to import.") from e
    Orchestrator = (
        Any  # Use Any as a placeholder if needed for type checking downstream
    )


# --- Helper to generate YouTube embed HTML ---
# Moved from legacy ui/main_gui.py
def get_youtube_embed_html(youtube_url: str, start_time_seconds: int = 0) -> str:
    """Creates HTML for embedding a YouTube video starting at a specific time."""
    if (
        not youtube_url
        or not isinstance(youtube_url, str)
        or not youtube_url.startswith("http")
    ):
        log_warning(f"Invalid YouTube URL for embed: {youtube_url}")
        return "<p>Invalid YouTube URL</p>"

    video_id = None
    # Handle common YouTube URL formats (including potential youtube.com/watch?v=... )
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",  # Standard watch URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})",  # Shortened URL
        r"embed/([a-zA-Z0-9_-]{11})",  # Embed URL
        # Added more robust handling for shorts URLs
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        # User content URL pattern might be less common or stable
        # r"googleusercontent\.com/youtube\.com/\d+/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        log_warning(f"Could not extract Video ID from URL: {youtube_url}")
        return f"<p>Could not extract Video ID from URL: {youtube_url}</p>"  # Show URL in error

    start_param = max(
        0, int(math.floor(start_time_seconds))
    )  # Ensure non-negative integer
    # Use standard YouTube embed URL
    embed_url = (
        f"https://www.youtube.com/embed/{video_id}?start={start_param}&controls=1"
    )

    # Use standard iframe attributes
    return (
        f'<iframe width="560" height="315" src="{embed_url}" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        f'referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
    )


# --- Main UI Class ---
# ***** CORRECTED: Class definition moved to top level *****
class WebApp:
    """Manages the Gradio interface and its interactions with the Orchestrator."""

    # --- MODIFIED: Accepts Orchestrator instance ---
    def __init__(self, orchestrator: "Orchestrator"):
        """
        Initializes the UI.

        Args:
           orchestrator: An instance of the core.orchestrator.Orchestrator class.
        """
        if not GRADIO_AVAILABLE:
            # This check prevents initialization if Gradio isn't installed.
            raise ImportError("Gradio or Pandas not found. UI cannot be initialized.")

        # Type check orchestrator if possible (will be Any if import failed, but useful if it succeeded)
        if not isinstance(orchestrator, Orchestrator) and Orchestrator is not Any:
            raise TypeError("Orchestrator instance is required.")

        self.orchestrator = orchestrator
        # Access config via the orchestrator's config object's get method for safety
        self.config_data = (
            orchestrator.config.config
        )  # Get the raw config dict if needed
        log_info("UI Initialized with Orchestrator instance.")

    def create_ui(self) -> "gr.Blocks":
        """Creates the Gradio Blocks interface."""
        # Ensure Gradio components are available before using them
        if not GRADIO_AVAILABLE or not gr or not themes:
            raise RuntimeError("Gradio components unavailable for UI creation.")

        default_theme = themes.Default()

        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Speech Analysis Pipeline")

            # --- UI States ---
            ui_mode_state = gr.State(
                "idle"
            )  # idle, processing, labeling, finished, error
            batch_job_id_state = gr.State(None)
            items_to_label_state = gr.State(
                []
            )  # List of item_identifiers requiring labeling
            current_item_index_state = gr.State(0)  # Index into items_to_label_state
            current_youtube_url_state = gr.State("")  # URL for the item being labeled
            eligible_speakers_state = gr.State(
                []
            )  # List of SPEAKER_IDs for the current item
            current_speaker_index_state = gr.State(
                0
            )  # Index into eligible_speakers_state
            current_start_times_state = gr.State(
                []
            )  # List of preview start times (int seconds)
            current_clip_index_state = gr.State(
                0
            )  # Index into current_start_times_state

            # --- Layout ---
            with gr.Row():
                with gr.Column(scale=1):
                    # --- BATCH INPUT GROUP ---
                    with gr.Group() as batch_input_group:
                        gr.Markdown("## 1. Batch Processing Input")
                        gr.Markdown(
                            "Upload an Excel file (.xlsx). The first column should contain YouTube URLs or local file paths."
                        )
                        batch_input_file = gr.File(
                            label="Upload Batch File",
                            type="filepath",
                            file_types=[".xlsx"],
                        )
                        gr.Markdown("### Output Options")
                        # Use orchestrator.config.get() for robust access to config values
                        include_source_audio_checkbox = gr.Checkbox(
                            label="Include Source Audio in ZIP",
                            value=self.orchestrator.config.get(
                                "include_source_audio", True
                            ),
                        )
                        include_json_summary_checkbox = gr.Checkbox(
                            label="Include Detailed JSON Summary",
                            value=self.orchestrator.config.get(
                                "include_json_summary", True
                            ),
                        )
                        include_csv_summary_checkbox = gr.Checkbox(
                            label="Include High-Level CSV Summary",
                            value=self.orchestrator.config.get(
                                "include_csv_summary", False
                            ),
                        )
                        include_script_transcript_checkbox = gr.Checkbox(
                            label="Include Simple Text Transcript",
                            value=self.orchestrator.config.get(
                                "include_script_transcript", False
                            ),
                        )
                        include_plots_checkbox = gr.Checkbox(
                            label="Include Emotion Plots",
                            value=self.orchestrator.config.get("include_plots", False),
                        )
                        batch_process_btn = gr.Button(
                            "Start Batch Processing ▶️", variant="primary"
                        )

                with gr.Column(scale=1):
                    # --- STATUS OUTPUT GROUP ---
                    with gr.Group() as status_output_group:
                        gr.Markdown("## 2. Processing Status & Results")
                        batch_status_output = gr.Textbox(
                            label="Status Log",
                            interactive=False,
                            lines=15,
                            max_lines=20,
                        )
                        batch_download_output = gr.Textbox(
                            label="Final Output ZIP Path",
                            interactive=False,
                            lines=1,
                            placeholder="Path to ZIP bundle will appear here...",
                        )

            # --- INTERACTIVE LABELING GROUP (Initially Hidden) ---
            with gr.Column(visible=False) as labeling_ui_group:
                gr.Markdown("## 3. Interactive Speaker Labeling")
                labeling_progress_md = gr.Markdown("Labeling Speaker: ---")
                with gr.Row():
                    with gr.Column(scale=2):
                        video_player_html = gr.HTML(label="Speaker Preview")
                    with gr.Column(scale=1):
                        gr.Markdown("### Clip Navigation")
                        current_clip_display = gr.Markdown("Preview 1 of X")
                        with gr.Row():
                            prev_clip_btn = gr.Button("⬅️ Previous")
                            next_clip_btn = gr.Button("Next ➡️")
                        gr.Markdown("### Enter Label")
                        speaker_label_input = gr.Textbox(
                            label="Enter Speaker Name/Label",
                            placeholder="e.g., Alice (leave blank to keep original ID)",
                        )
                        submit_label_btn = gr.Button(
                            "Submit Label & Next Speaker ▶️", variant="primary"
                        )
                        skip_item_btn = gr.Button("Skip Rest of Item ⏭️", variant="stop")

            # --- Helper Functions for UI Logic ---

            def change_ui_mode(mode: str) -> Dict["gr.UIComponent", Dict[Any, Any]]:
                """Updates visibility of UI groups based on the current mode."""
                log_info(f"Changing UI mode to: {mode}")
                is_labeling = mode == "labeling"
                is_idle_or_finished = mode in ["idle", "finished", "error"]
                return {
                    labeling_ui_group: gr.update(visible=is_labeling),
                    batch_input_group: gr.update(visible=True),  # Always visible
                    batch_process_btn: gr.update(interactive=is_idle_or_finished),
                    status_output_group: gr.update(visible=True),  # Always visible
                }

            def process_batch_wrapper(
                xlsx_file_path: Optional[
                    str
                ],  # Gradio File component gives filepath string
                include_audio: bool,
                include_json: bool,
                include_csv: bool,
                include_script: bool,
                include_plots: bool,
            ) -> Generator[Dict, Any, Any]:
                """Wrapper to handle batch processing initiation and UI updates."""
                if not xlsx_file_path:  # Check if path is None or empty
                    yield {
                        batch_status_output: "ERROR: Please upload an Excel file.",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }
                    return

                # Reset states for new batch run
                yield {
                    batch_status_output: "Starting batch processing...",
                    batch_download_output: "",
                    ui_mode_state: "processing",
                    batch_job_id_state: None,
                    items_to_label_state: [],
                    current_item_index_state: 0,
                    current_youtube_url_state: "",
                    eligible_speakers_state: [],
                    current_speaker_index_state: 0,
                    current_start_times_state: [],
                    current_clip_index_state: 0,
                    video_player_html: "",
                    **change_ui_mode("processing"),
                }

                try:
                    # --- CALL ORCHESTRATOR ---
                    status_msg, results_summary, returned_batch_id = (
                        self.orchestrator.process_batch(
                            input_source=xlsx_file_path,  # Pass the path string
                            include_source_audio=include_audio,
                            include_json_summary=include_json,
                            include_csv_summary=include_csv,
                            include_script_transcript=include_script,
                            include_plots=include_plots,
                        )
                    )
                    current_status_text = f"{status_msg}\n\n{results_summary}"
                    yield {batch_status_output: current_status_text}

                    if returned_batch_id:
                        # Batch requires labeling
                        current_batch_job_id = returned_batch_id
                        log_info(
                            f"Batch [{current_batch_job_id}] requires labeling. Initializing UI."
                        )
                        # Get the ordered list of items directly from orchestrator state
                        batch_state = self.orchestrator.labeling_state.get(
                            current_batch_job_id, {}
                        )
                        items_requiring_labeling = batch_state.get(
                            "items_requiring_labeling_order", []
                        )

                        if not items_requiring_labeling:
                            log_error(
                                f"Batch [{current_batch_job_id}] needs labeling, but no items found in state."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{current_status_text}\n\nERROR: Internal state error finding items."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                            return

                        first_item_id = items_requiring_labeling[0]
                        log_info(f"Starting labeling UI with item: {first_item_id}")

                        # --- CALL ORCHESTRATOR to get first speaker data ---
                        first_speaker_data = self.orchestrator.start_labeling_item(
                            current_batch_job_id, first_item_id
                        )

                        if first_speaker_data:
                            first_speaker_id, yt_url, first_start_times = (
                                first_speaker_data
                            )
                            # Get eligible speakers for this specific item
                            item_state = batch_state.get(first_item_id, {})
                            eligible_speakers_list = item_state.get(
                                "eligible_speakers", []
                            )

                            initial_start_time = (
                                first_start_times[0] if first_start_times else 0
                            )
                            clip_count = len(first_start_times)
                            initial_html = get_youtube_embed_html(
                                yt_url, initial_start_time
                            )
                            total_labeling_items = len(items_requiring_labeling)

                            yield {
                                batch_job_id_state: current_batch_job_id,
                                items_to_label_state: items_requiring_labeling,
                                current_item_index_state: 0,
                                current_youtube_url_state: yt_url,
                                eligible_speakers_state: eligible_speakers_list,
                                current_speaker_index_state: 0,
                                current_start_times_state: first_start_times,
                                current_clip_index_state: 0,
                                ui_mode_state: "labeling",
                                labeling_progress_md: f"Labeling Speaker: **{first_speaker_id}** (Speaker 1/{len(eligible_speakers_list)}, Item 1/{total_labeling_items})",
                                current_clip_display: f"Preview 1 of {clip_count}",
                                video_player_html: initial_html,
                                speaker_label_input: gr.update(value=""),  # Clear input
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                **change_ui_mode("labeling"),
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for item {first_item_id} from orchestrator."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{current_status_text}\n\nERROR: Failed to start labeling for {first_item_id}."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    else:
                        # Batch finished without needing labeling
                        log_info(
                            "Batch processing finished. No interactive labeling required."
                        )
                        # Determine final status based on status message content
                        final_mode = (
                            "finished"
                            if "✅" in status_msg or "Batch complete" in status_msg
                            else "error"
                        )
                        final_zip_path_str = ""
                        # Try to extract zip path if finished successfully
                        if final_mode == "finished":
                            # Make regex more general for different success messages
                            match = re.search(
                                r"(?:Output|Download):\s*(.+\.zip)", status_msg
                            )
                            if match:
                                final_zip_path_str = match.group(1).strip()
                            else:
                                log_warning(
                                    f"Could not extract ZIP path from success message: {status_msg}"
                                )

                        yield {
                            batch_download_output: final_zip_path_str,
                            ui_mode_state: final_mode,
                            **change_ui_mode(final_mode),
                        }

                except Exception as e:
                    error_trace = traceback.format_exc()
                    log_error(f"Error in process_batch_wrapper: {e}\n{error_trace}")
                    yield {
                        batch_status_output: f"An unexpected error occurred during batch processing: {e}\n\n{error_trace}",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }

            def handle_change_clip(
                direction: int,
                current_clip_idx: int,
                start_times: list,
                youtube_url: str,
            ) -> Dict[str, Any]:  # Return type hint for clarity
                """Handles changing the preview clip."""
                if not start_times:
                    return {}  # No clips to change
                num_clips = len(start_times)
                # Basic bounds check
                if num_clips <= 1:
                    return {}  # Only one clip, nothing to change

                new_clip_idx = (
                    current_clip_idx + direction
                ) % num_clips  # Use modulo for wrapping
                # Simple calculation without complex min/max:
                # new_clip_idx = max(0, min(current_clip_idx + direction, num_clips - 1))

                # Avoid update if index didn't actually change (e.g., at boundaries with min/max)
                # if new_clip_idx == current_clip_idx:
                #     return {}

                new_start_time = start_times[new_clip_idx]
                new_html = get_youtube_embed_html(youtube_url, new_start_time)
                return {
                    current_clip_index_state: new_clip_idx,
                    video_player_html: new_html,
                    current_clip_display: f"Preview {new_clip_idx + 1} of {num_clips}",
                }

            def move_to_next_labeling_state(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,  # Index of the item JUST completed/skipped
                current_status_text: str,  # Pass the current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles UI transitions AFTER an item is finalized."""
                log_info(
                    f"[{batch_id}] UI moving state after finalizing item index {item_idx}"
                )

                next_item_idx = item_idx + 1

                if next_item_idx < len(items_to_label):
                    # --- Move UI to the NEXT item ---
                    next_item_id = items_to_label[next_item_idx]
                    log_info(
                        f"[{batch_id}] Moving UI to label next item: {next_item_id} (index {next_item_idx})"
                    )

                    # --- CALL ORCHESTRATOR to get data for the next item ---
                    next_speaker_data = self.orchestrator.start_labeling_item(
                        batch_id, next_item_id
                    )

                    if next_speaker_data:
                        next_speaker_id, next_yt_url, next_start_times = (
                            next_speaker_data
                        )
                        # Retrieve state info for the new item from orchestrator state
                        batch_state = self.orchestrator.labeling_state.get(batch_id, {})
                        next_item_state = batch_state.get(next_item_id, {})
                        next_eligible_spkrs = next_item_state.get(
                            "eligible_speakers", []
                        )

                        initial_start_time = (
                            next_start_times[0] if next_start_times else 0
                        )
                        clip_count = len(next_start_times)
                        total_items_count = len(items_to_label)
                        new_html = get_youtube_embed_html(
                            next_yt_url, initial_start_time
                        )

                        yield {
                            current_item_index_state: next_item_idx,
                            current_youtube_url_state: next_yt_url,
                            eligible_speakers_state: next_eligible_spkrs,
                            current_speaker_index_state: 0,  # Reset speaker index for new item
                            current_start_times_state: next_start_times,
                            current_clip_index_state: 0,  # Reset clip index
                            labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker 1/{len(next_eligible_spkrs)}, Item {next_item_idx + 1}/{total_items_count})",
                            current_clip_display: f"Preview 1 of {clip_count}",
                            video_player_html: new_html,
                            speaker_label_input: gr.update(value=""),  # Clear input
                            prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                            next_clip_btn: gr.update(interactive=(clip_count > 1)),
                            ui_mode_state: "labeling",  # Ensure UI stays in labeling mode
                            batch_status_output: current_status_text,  # Pass status through
                            **change_ui_mode("labeling"),
                        }
                    else:
                        # Error starting the next item
                        log_error(
                            f"Failed to get initial speaker data for next item {next_item_id}."
                        )
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\nERROR: Failed start labeling next item {next_item_id}."
                            ),
                            ui_mode_state: "error",
                            **change_ui_mode("error"),
                        }
                else:
                    # --- Finished all items in the batch ---
                    log_info(
                        f"[{batch_id}] UI: Finished labeling/skipping all items. Checking completion..."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nLabeling complete for all items. Finalizing batch and creating ZIP..."
                        )
                    }
                    # Update internal status text (not strictly necessary if only yielding once more)
                    current_status_text = f"{current_status_text}\n\nLabeling complete for all items. Finalizing batch and creating ZIP..."

                    # --- CALL ORCHESTRATOR to check completion and create ZIP ---
                    final_zip_path = self.orchestrator.check_completion_and_zip(
                        batch_id
                    )

                    # --- CORRECTED: Redundant assignment removed ---
                    final_status_msg = (
                        f"✅ Batch complete. Output: {final_zip_path}"
                        if final_zip_path
                        else "❗️ Batch complete, but ZIP creation failed."
                    )
                    final_mode = "finished" if final_zip_path else "error"

                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\n{final_status_msg}"
                        ),  # Append final message
                        batch_download_output: (
                            str(final_zip_path) if final_zip_path else ""
                        ),
                        ui_mode_state: final_mode,
                        **change_ui_mode(final_mode),
                    }

            def handle_submit_label_wrapper(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,
                speakers_to_label: List[str],
                speaker_idx: int,
                current_label_input: str,
                current_status_text: str,  # Pass current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles storing the label and moving to the next speaker or item."""
                # Basic state validation
                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                    or not speakers_to_label
                    or speaker_idx >= len(speakers_to_label)
                ):
                    log_error(
                        f"Submit label called with invalid state: {batch_id=}, {item_idx=}, {len(items_to_label)=}, {speaker_idx=}, {len(speakers_to_label)=}"
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: Invalid state during label submission."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                speaker_id = speakers_to_label[speaker_idx]
                log_info(
                    f"[{batch_id}-{item_id}] UI submitting label for {speaker_id}: '{current_label_input}'"
                )

                # --- CALL ORCHESTRATOR to store label ---
                success = self.orchestrator.store_label(
                    batch_id, item_id, speaker_id, current_label_input
                )
                if not success:
                    log_warning(
                        f"Failed to store label for {speaker_id} in orchestrator state."
                    )
                    # Update status but continue
                    current_status_text = f"{current_status_text}\n\nWARN: Failed to store label for {speaker_id}."
                    yield {batch_status_output: gr.update(value=current_status_text)}
                    # Proceed to next speaker anyway? Or halt? Current logic proceeds.

                # --- CALL ORCHESTRATOR to get next speaker in *same* item ---
                next_speaker_data = self.orchestrator.get_next_labeling_speaker(
                    batch_id, item_id, speaker_idx
                )

                if next_speaker_data:
                    # --- Still more speakers in the CURRENT item ---
                    next_speaker_id, yt_url, next_start_times = next_speaker_data
                    next_speaker_idx = speaker_idx + 1
                    initial_start_time = next_start_times[0] if next_start_times else 0
                    clip_count = len(next_start_times)
                    total_speakers = len(speakers_to_label)
                    total_items_count = len(items_to_label)
                    new_html = get_youtube_embed_html(yt_url, initial_start_time)

                    yield {  # Update UI for next speaker in same item
                        current_speaker_index_state: next_speaker_idx,
                        current_start_times_state: next_start_times,
                        current_clip_index_state: 0,
                        labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker {next_speaker_idx + 1}/{total_speakers}, Item {item_idx + 1}/{total_items_count})",
                        current_clip_display: f"Preview 1 of {clip_count}",
                        video_player_html: new_html,
                        speaker_label_input: gr.update(value=""),  # Clear input
                        prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                        next_clip_btn: gr.update(interactive=(clip_count > 1)),
                        batch_status_output: current_status_text,  # Pass status through
                    }
                else:
                    # --- Finished speakers for CURRENT item -> Finalize item & Transition UI ---
                    log_info(
                        f"[{batch_id}-{item_id}] Finished labeling speakers for item {item_id}. Triggering finalization."
                    )
                    current_status_text = (
                        f"{current_status_text}\n\nFinalizing item {item_id}..."
                    )
                    yield {batch_status_output: gr.update(value=current_status_text)}

                    # --- CALL ORCHESTRATOR to finalize item ---
                    finalization_success = self.orchestrator.finalize_item(
                        batch_id, item_id
                    )

                    if not finalization_success:
                        log_error(
                            f"[{batch_id}-{item_id}] Finalization failed after submitting last label."
                        )
                        current_status_text = f"{current_status_text}\n\nERROR: Failed to finalize item {item_id}. Attempting to proceed..."
                        yield {
                            batch_status_output: gr.update(value=current_status_text)
                        }
                    else:
                        current_status_text = f"{current_status_text}\n\nItem {item_id} finalized successfully."
                        yield {
                            batch_status_output: gr.update(value=current_status_text)
                        }

                    # Now call the UI transition generator to move to next item or finish batch
                    yield from move_to_next_labeling_state(
                        batch_id, items_to_label, item_idx, current_status_text
                    )

            def handle_skip_item_wrapper(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,
                current_status_text: str,  # Pass current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles skipping remaining speakers and finalizing the item."""
                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                ):
                    log_error(
                        f"Skip item called with invalid state: {batch_id=}, {item_idx=}, {len(items_to_label)=}"
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: Invalid state during skip."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                log_info(
                    f"[{batch_id}-{item_id}] UI skipping rest of speakers for item."
                )
                current_status_text = (
                    f"{current_status_text}\n\nSkipping & finalizing item {item_id}..."
                )
                yield {batch_status_output: gr.update(value=current_status_text)}

                # --- CALL ORCHESTRATOR to handle skip (which includes finalize) ---
                success = self.orchestrator.skip_item_labeling(batch_id, item_id)

                if not success:
                    log_warning(
                        f"[{batch_id}-{item_id}] Orchestrator finalize/skip returned failure, but attempting UI transition."
                    )
                    current_status_text = f"{current_status_text}\n\nWARN: Finalization during skip failed for item {item_id}."
                    yield {batch_status_output: gr.update(value=current_status_text)}
                else:
                    current_status_text = f"{current_status_text}\n\nItem {item_id} skipped and finalized."
                    yield {batch_status_output: gr.update(value=current_status_text)}

                # Call the common UI transition logic
                yield from move_to_next_labeling_state(
                    batch_id, items_to_label, item_idx, current_status_text
                )

            # --- Event Listeners ---
            # Define outputs list once for state components often updated together
            state_outputs_labeling = [
                ui_mode_state,
                batch_job_id_state,  # Only set initially
                items_to_label_state,
                current_item_index_state,
                current_youtube_url_state,
                eligible_speakers_state,
                current_speaker_index_state,
                current_start_times_state,
                current_clip_index_state,
            ]
            ui_outputs_labeling = [
                batch_status_output,
                batch_download_output,
                video_player_html,
                speaker_label_input,
                labeling_progress_md,
                current_clip_display,
                prev_clip_btn,
                next_clip_btn,
                labeling_ui_group,
                batch_input_group,
                status_output_group,
                batch_process_btn,
            ]

            # Batch processing button
            batch_process_btn.click(
                fn=process_batch_wrapper,
                inputs=[
                    batch_input_file,
                    include_source_audio_checkbox,
                    include_json_summary_checkbox,
                    include_csv_summary_checkbox,
                    include_script_transcript_checkbox,
                    include_plots_checkbox,
                ],
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Combine UI and state outputs
            )

            # Previous clip button
            prev_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(-1),  # Pass direction implicitly
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[  # Only updates these specific UI elements
                    current_clip_index_state,
                    video_player_html,
                    current_clip_display,
                ],
            )

            # Next clip button
            next_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(1),  # Pass direction implicitly
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[  # Only updates these specific UI elements
                    current_clip_index_state,
                    video_player_html,
                    current_clip_display,
                ],
            )

            # Submit label button
            submit_label_btn.click(
                fn=handle_submit_label_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    speaker_label_input,
                    batch_status_output,  # Pass current status text
                ],
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Updates UI and state
            )

            # Skip item button
            skip_item_btn.click(
                fn=handle_skip_item_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    batch_status_output,  # Pass current status text
                ],
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Updates UI and state
            )

            return demo

    # --- Launch Method ---
    def launch(self, **kwargs):
        """Creates and launches the Gradio interface."""
        if not GRADIO_AVAILABLE:
            log_error("Cannot launch UI: Gradio or Pandas not available.")
            # Potentially raise here to prevent main.py from continuing?
            # raise ImportError("Gradio or Pandas not found, UI cannot be launched.")
            return  # Or just return if main.py handles this

        try:
            log_info("Creating Gradio UI...")
            gradio_app = self.create_ui()
            log_info("Launching Gradio UI...")
            # Pass launch kwargs (e.g., server_name, share)
            gradio_app.launch(**kwargs)
        except Exception as e:
            # Catch errors specifically during UI creation or launch
            log_error(f"FATAL ERROR during Gradio UI creation or launch: {e}")
            log_error(traceback.format_exc())
            print(
                f"\nFATAL ERROR: Could not launch Gradio UI. Check logs. Error: {e}\n"
            )
            # Re-raise the exception so the main script knows launch failed
            raise

```

### utils\file_manager.py

```py
# utils/file_manager.py
"""
Utility functions for file and directory management, including temporary files,
directory creation, cleanup, and archive creation.
"""

import os
import shutil
import tempfile
import zipfile
import traceback
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any
from config.config import (
    Config,
)  # Assuming config.py is in the same directory as this module

# Importing Config from config.py, assuming it contains necessary configuration settings

# Assuming core.logging is established from Phase 1
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


def __init__(self, config: Config):
    self.config = config


# Moved from core/utils.py
def create_directory(path: Union[str, Path]) -> bool:
    """Creates a directory if it doesn't exist. Returns True on success."""
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        log_info(f"Ensured directory exists: {dir_path}")
        return True
    except OSError as e:
        log_warning(f"Could not create directory {dir_path}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error creating directory {dir_path}: {e}")
        return False


# Refactored from core/utils.py:get_temp_file
# Renamed for clarity
def get_temp_file_path(suffix: str = "") -> Optional[Path]:
    """Creates a temporary file and returns its Path object, or None on failure."""
    try:
        # Note: This creates a file in the system's default temp location,
        # NOT necessarily within the project's configured temp_dir.
        # Consider adding a version that takes a base directory if needed.
        config = Config()  # Instantiate Config
        config_temp_dir = Path(config.get("temp_dir", "./temp"))
        log_info(f"Using temporary directory: {config_temp_dir}")
        # Ensure the temp directory exists
        if not config_temp_dir.exists():
            log_warning(
                f"Temporary directory does not exist: {config_temp_dir}. Creating it."
            )
            create_directory(config_temp_dir)
        temp_file = tempfile.mkstemp(dir=config_temp_dir, suffix=suffix)
        temp_file_path: Path = Path(temp_file.name)
        temp_file.close()
        log_info(f"Created temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        log_error(f"Failed to create temporary file: {e}")
        return None


# Refactored from core/file_management.py:FileManager.save_file
def save_text_file(content: str, file_path: Union[str, Path]) -> bool:
    """Saves string content to a file. Ensures parent directory exists."""
    path_obj = Path(file_path)
    log_info(f"Attempting to save text file to: {path_obj}")
    if not create_directory(path_obj.parent):
        log_error(f"Failed to create parent directory for {path_obj}. Aborting save.")
        return False
    try:
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        log_info(f"File saved successfully: {path_obj}")
        return True
    except IOError as e:
        log_error(f"Failed to save file {path_obj}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error saving file {path_obj}: {e}")
        return False


# Refactored from core/file_management.py:FileManager.cleanup_temp_files
# Made more generic: cleans ANY directory.
def cleanup_directory(dir_path: Union[str, Path], recreate: bool = True) -> bool:
    """Removes and optionally recreates a directory."""
    path_obj = Path(dir_path)
    log_info(f"Attempting cleanup for directory: {path_obj} (recreate={recreate})")
    try:
        if path_obj.exists():
            shutil.rmtree(path_obj)
            log_info(f"Removed directory: {path_obj}")
        else:
            log_info(f"Directory does not exist, no removal needed: {path_obj}")

        if recreate:
            return create_directory(path_obj)
        return True
    except OSError as e:
        log_error(f"Failed cleanup for directory {path_obj}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error during cleanup for {path_obj}: {e}")
        return False


# Refactored from core/pipeline.py:create_final_zip
def create_zip_archive(
    zip_path: Union[str, Path],
    files_to_add: Dict[str, Union[str, Path]],  # Arcname -> Source Path
    log_prefix: str = "",
) -> Optional[Path]:
    """Creates a ZIP archive from a dictionary of files."""
    zip_path_obj = Path(zip_path)
    log_info(f"{log_prefix}Attempting to create ZIP archive: {zip_path_obj}")

    if not files_to_add:
        log_warning(
            f"{log_prefix}No files provided to add to the zip archive. Skipping zip creation."
        )
        return None

    # Ensure output directory exists
    if not create_directory(zip_path_obj.parent):
        log_error(
            f"{log_prefix}Failed to create output directory {zip_path_obj.parent}. Aborting zip."
        )
        return None

    # Use a temporary zip file during creation
    temp_zip_path = zip_path_obj.with_suffix(f".{os.getpid()}.temp.zip")
    files_added_count = 0
    files_skipped_count = 0

    try:
        with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for arc_name, local_path_src in files_to_add.items():
                try:
                    local_path = Path(local_path_src)  # Ensure it's a Path object
                except TypeError:
                    log_warning(
                        f"{log_prefix}Invalid path type for archive name '{arc_name}', skipping: {local_path_src}"
                    )
                    files_skipped_count += 1
                    continue

                if local_path.is_file():
                    try:
                        zf.write(local_path, arcname=arc_name)
                        files_added_count += 1
                        log_info(
                            f"{log_prefix}Added to zip: '{local_path}' as '{arc_name}'"
                        )
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
                temp_zip_path.unlink(missing_ok=True)  # Use missing_ok=True
            return None

        # Move temporary zip to final location
        shutil.move(str(temp_zip_path), str(zip_path_obj))
        log_info(f"{log_prefix}Successfully created final ZIP: {zip_path_obj}")
        log_info(
            f"{log_prefix}Files added: {files_added_count}, Files skipped: {files_skipped_count}"
        )
        return zip_path_obj

    except Exception as e:
        log_error(
            f"{log_prefix}Failed to create ZIP file {zip_path_obj}: {e}\n{traceback.format_exc()}"
        )
        if temp_zip_path.exists():
            temp_zip_path.unlink(missing_ok=True)
            log_info(f"{log_prefix}Cleaned up temporary zip file.")
        return None


# Extracted from core/pipeline.py:_prepare_audio_input
def copy_local_file(
    source_path: Union[str, Path],
    destination_dir: Union[str, Path],
    destination_filename: Optional[str] = None,
    log_prefix: str = "",
) -> Optional[Path]:
    """Copies a local file to a destination directory."""
    src_path = Path(source_path)
    dest_dir = Path(destination_dir)

    if not src_path.is_file():
        log_error(f"{log_prefix}Source file not found or is not a file: {src_path}")
        return None

    if not create_directory(dest_dir):
        log_error(
            f"{log_prefix}Failed to ensure destination directory exists: {dest_dir}"
        )
        return None

    dest_filename = destination_filename or src_path.name
    dest_path = dest_dir / dest_filename

    try:
        log_info(f"{log_prefix}Copying local file {src_path} to {dest_path}")
        shutil.copy(str(src_path), str(dest_path))
        log_info(f"{log_prefix}File copied successfully.")
        return dest_path
    except Exception as e:
        log_error(
            f"{log_prefix}Failed to copy file from {src_path} to {dest_path}: {e}"
        )
        log_error(traceback.format_exc())
        return None

```

### utils\transcripts.py

```py
# utils/transcripts.py
"""
Utility functions for processing, structuring, and manipulating transcript data,
including segment grouping, snippet matching, and saving formats.
"""

import json
import re
import string
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, TextIO, Union
from collections import defaultdict
from datetime import datetime

# Use rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz

    FUZZ_AVAILABLE = True
except ImportError:
    FUZZ_AVAILABLE = False

# Use pandas for reading excel and NaN handling
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Assuming core.logging is established from Phase 1
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


if not FUZZ_AVAILABLE:
    log_warning("Rapidfuzz library not found. Snippet matching will be unavailable.")
if not PANDAS_AVAILABLE:
    log_warning("Pandas library not found. Reading XLSX snippets will be unavailable.")

# Type hints
Segment = Dict[str, Any]
SegmentsList = List[Segment]


# Moved from core/utils.py
def parse_xlsx_snippets(snippet_string: Any) -> Dict[str, str]:
    """
    Parses a string (potentially from an XLSX cell) into a Dict[SpeakerName, SnippetText].
    Handles None, NaN, and non-string types gracefully. Requires pandas.
    """
    if not PANDAS_AVAILABLE:
        log_error("Pandas not available, cannot parse XLSX snippets.")
        return {}

    mapping = {}
    # Check for pandas NaN or None explicitly, or if not a string
    if (
        pd.isna(snippet_string)
        or not isinstance(snippet_string, str)
        or not snippet_string.strip()
    ):
        return mapping  # Return empty if input is invalid

    lines = snippet_string.strip().split("\n")
    for line in lines:
        # Regex to capture "Speaker Name: Snippet Text" potentially with extra whitespace
        match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
        if match:
            name = match.group(1).strip()
            snippet = match.group(2).strip()
            if name and snippet:
                mapping[name] = snippet
                log_info(
                    f"Parsed snippet - Name: '{name}', Snippet: '{snippet[:50]}...'"
                )
            else:
                log_warning(
                    f"Could not parse snippet line effectively (empty name or snippet): '{line}'"
                )
        else:
            log_warning(f"Ignoring invalid snippet line format: '{line}'")
    return mapping


# Moved from core/utils.py
def group_segments_by_speaker(segments: SegmentsList) -> List[Dict[str, Any]]:
    """Groups consecutive segments by the same speaker into dialogue blocks."""
    blocks = []
    if not segments:
        log_info("No segments provided for grouping.")
        return blocks

    log_info(f"Grouping {len(segments)} segments by speaker...")

    # Use an iterator to handle the first segment initialization cleanly
    segment_iter = iter(segments)
    try:
        first_seg = next(segment_iter)
    except StopIteration:
        log_info("Segment list was empty after iterator check.")
        return blocks  # Empty list

    # Initialize current block with the first segment
    current_speaker = str(first_seg.get("speaker", "unknown"))
    current_text = (first_seg.get("text") or "").strip()
    current_start = first_seg.get("start")
    current_end = first_seg.get("end")
    current_indices = [0]  # Store original index of segments in the block
    # Add other segment keys if needed in the block (e.g., emotion, words)
    # current_words = first_seg.get("words", [])

    # Iterate through the rest of the segments
    for i, seg in enumerate(segment_iter, start=1):
        speaker = str(seg.get("speaker", "unknown"))
        text = (seg.get("text") or "").strip()
        start = seg.get("start")
        end = seg.get("end")
        # words = seg.get("words", []) # Get words if needed

        # If the speaker is the same, append text and update end time
        if speaker == current_speaker:
            if text:
                current_text += (
                    (" " + text) if current_text else text
                )  # Add space only if needed
            # Update the end time to the end time of the current segment
            if end is not None:
                current_end = end  # Take the latest end time
            current_indices.append(i)
            # current_words.extend(words) # Append words if tracking them per block
        else:
            # Speaker changed: finalize the previous block
            if (
                current_start is not None and current_end is not None
            ):  # Only add blocks with valid times
                blocks.append(
                    {
                        "speaker": current_speaker,
                        "text": current_text,
                        "start": current_start,
                        "end": current_end,
                        "indices": current_indices,
                        # "words": current_words, # Add words if needed
                    }
                )
            else:
                log_warning(
                    f"Skipping block for speaker '{current_speaker}' due to invalid start/end times."
                )

            # Start a new block with the current segment
            current_speaker = speaker
            current_text = text
            current_start = start
            current_end = end
            current_indices = [i]
            # current_words = words

    # Add the last accumulated block after the loop finishes
    if (
        current_start is not None and current_end is not None
    ):  # Check times for the last block
        blocks.append(
            {
                "speaker": current_speaker,
                "text": current_text,
                "start": current_start,
                "end": current_end,
                "indices": current_indices,
                # "words": current_words,
            }
        )
    else:
        log_warning(
            f"Skipping final block for speaker '{current_speaker}' due to invalid start/end times."
        )

    log_info(f"Grouped into {len(blocks)} speaker blocks.")
    return blocks


# Moved from core/utils.py
def match_snippets_to_speakers(
    segments: SegmentsList,
    speaker_snippet_map: Dict[str, str],  # User Name -> Snippet Text
    match_threshold: float = 0.80,  # Default threshold (0.0-1.0)
) -> Dict[str, str]:  # Returns WhisperX ID ('SPEAKER_XX') -> User Name
    """
    Fuzzy-match user snippets to speaker dialogue blocks using rapidfuzz.
    Requires rapidfuzz library.

    Args:
        segments: List of segment dictionaries from transcription.
        speaker_snippet_map: Dictionary mapping user-provided names to text snippets.
        match_threshold: Minimum similarity score (0.0 to 1.0) for a match.

    Returns:
        A dictionary mapping original WhisperX speaker IDs (e.g., 'SPEAKER_00')
        to the user-provided name (e.g., 'Alice') based on the best match above threshold.
    """
    if not FUZZ_AVAILABLE:
        log_error("Rapidfuzz library not available. Cannot perform snippet matching.")
        return {}

    log_info(
        f"Attempting to match {len(speaker_snippet_map)} snippets to speakers (threshold: {match_threshold:.2f})..."
    )
    if not speaker_snippet_map or not segments:
        log_info("No speaker snippets provided or no segments to match against.")
        return {}

    # Group segments into blocks for more context
    blocks = group_segments_by_speaker(segments)
    if not blocks:
        log_warning(
            "Segment grouping resulted in zero blocks. Cannot perform snippet matching."
        )
        return {}

    speaker_id_to_name_mapping: Dict[str, str] = {}
    # Track best score per original speaker ID to avoid weaker matches overwriting strong ones
    best_match_scores: Dict[str, float] = defaultdict(float)

    # Convert threshold 0-1 range to 0-100 for rapidfuzz partial_ratio
    fuzz_threshold = match_threshold * 100.0

    # Helper for normalization (lowercase, remove punctuation, collapse whitespace)
    def normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Iterate through each user-provided snippet
    for user_name, snippet in speaker_snippet_map.items():
        if not snippet or len(snippet) < 5:  # Basic validation
            log_warning(
                f"Ignoring short/empty snippet for '{user_name}'. Snippet: '{snippet}'"
            )
            continue

        normalized_snippet = normalize_text(snippet)
        if not normalized_snippet:
            log_warning(f"Snippet for '{user_name}' became empty after normalization.")
            continue

        best_match_for_this_snippet = {"score": -1.0, "speaker_id": None}
        found_match_above_threshold = False

        log_info(f"Matching snippet for '{user_name}': '{normalized_snippet[:100]}...'")

        # Compare snippet against each speaker block
        for blk in blocks:
            original_speaker_id = blk.get("speaker")  # This is SPEAKER_XX or unknown
            block_text = blk.get("text", "")
            if not original_speaker_id or not block_text:
                continue

            normalized_block_text = normalize_text(block_text)
            if not normalized_block_text:
                continue

            # Use partial_ratio for finding snippet within larger block
            ratio = fuzz.partial_ratio(normalized_snippet, normalized_block_text)

            # Track the best block match for the current snippet (for logging/debugging)
            if ratio > best_match_for_this_snippet["score"]:
                best_match_for_this_snippet = {
                    "score": ratio,
                    "speaker_id": original_speaker_id,
                }

            # Check if score meets threshold AND is better than any previous match for this speaker_id
            if (
                ratio >= fuzz_threshold
                and ratio > best_match_scores[original_speaker_id]
            ):
                # Assign mapping: original_speaker_id -> user_name
                speaker_id_to_name_mapping[original_speaker_id] = user_name
                best_match_scores[original_speaker_id] = (
                    ratio  # Update best score for this ID
                )
                log_info(
                    f"  ✅ Match FOUND: Snippet '{user_name}' (score {ratio:.1f}) assigned to speaker '{original_speaker_id}' "
                    f"(overwriting score {best_match_scores.get(original_speaker_id, 0.0):.1f})"
                )
                found_match_above_threshold = True
                # Continue checking other blocks, this snippet might match another speaker even better

        # Log if no match above threshold was found for the current snippet after checking all blocks
        if not found_match_above_threshold:
            best_score = best_match_for_this_snippet["score"]
            best_spk = best_match_for_this_snippet["speaker_id"]
            log_info(
                f"  ❌ No match >= threshold ({fuzz_threshold:.1f}) found for snippet '{user_name}'. Best was {best_score:.1f} vs '{best_spk}'."
            )

    log_info(
        f"Final snippet mapping (WhisperX ID -> User Name): {speaker_id_to_name_mapping}"
    )
    return speaker_id_to_name_mapping


# Moved from core/utils.py
def convert_floats(obj: Any) -> Any:
    """
    Recursively converts numpy float types (e.g., float32) and other potentially
    non-JSON-serializable number types within nested structures to standard Python floats.
    """
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(elem) for elem in obj]
    # Check if it quacks like a numpy float (more robust than specific type checks)
    elif hasattr(obj, "item") and callable(obj.item) and isinstance(obj.item(), float):
        return float(obj)
    # Add checks for other types if necessary (e.g., Decimal)
    # elif isinstance(obj, decimal.Decimal):
    #     return float(obj)
    else:
        # Return object as is if it's not a dict, list, or convertible number type
        return obj


# Moved from core/transcription.py
def convert_json_to_structured(json_path: Path) -> SegmentsList:
    """
    Reads a WhisperX JSON output file and converts it into a structured list of segments.

    Args:
        json_path: Path to the WhisperX output JSON file.

    Returns:
        A list of dictionaries, where each represents a segment containing keys like
        'start', 'end', 'text', 'speaker', 'words'. Returns empty list on failure.

    Raises:
        FileNotFoundError: If the json_path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the 'segments' key in the JSON is not a list.
    """
    log_info(f"Reading and structuring WhisperX JSON output from: {json_path}")
    if not json_path.is_file():
        log_error(f"WhisperX output JSON file not found at {json_path}")
        raise FileNotFoundError(f"WhisperX output JSON file not found at {json_path}")

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
    segments_raw: Any = data.get("segments", [])  # Default to empty list

    if not isinstance(segments_raw, list):
        log_error(
            f"'segments' key in {json_path} is not a list (type: {type(segments_raw)}). Cannot process."
        )
        raise TypeError(f"'segments' key in {json_path} is not a list.")

    log_info(f"Structuring {len(segments_raw)} segments from WhisperX output...")

    for i, segment in enumerate(segments_raw):
        if not isinstance(segment, dict):
            log_warning(f"Item #{i} in 'segments' is not a dictionary. Skipping.")
            continue

        # Extract data with defaults
        text: str = segment.get("text", "").strip()
        start_time: Optional[float] = segment.get("start")
        end_time: Optional[float] = segment.get("end")
        speaker: str = str(segment.get("speaker", "unknown"))  # Ensure string type
        words_raw: Any = segment.get("words", [])
        words: List[Dict[str, Any]] = words_raw if isinstance(words_raw, list) else []

        # Basic validation for timing
        if (
            start_time is None
            or end_time is None
            or not isinstance(start_time, (int, float))
            or not isinstance(end_time, (int, float))
            or start_time > end_time
        ):
            log_warning(
                f"Segment {i} has invalid/missing time: start={start_time}, end={end_time}. Using raw values."
            )
            # Decide how to handle: skip segment, use None, use 0? Using raw values for now.

        segment_output: Segment = {
            "start": start_time,
            "end": end_time,
            "text": text,
            "speaker": speaker,
            "words": words,  # Ensure words is always a list
        }
        structured.append(segment_output)

    log_info(f"Finished structuring segments. Returning {len(structured)} segments.")
    return structured


# Moved from core/utils.py
# Assumes group_segments_by_speaker is available in this module
def save_script_transcript(
    segments: SegmentsList, output_path: Path, log_prefix: str = "[Transcript]"
) -> Optional[Path]:
    """
    Saves a simple text transcript grouped by speaker with HH:MM:SS timestamps.

    Args:
        segments: The list of segment dictionaries.
        output_path: The Path object where the transcript file should be saved.
        log_prefix: Prefix for log messages.

    Returns:
        The output Path object if successful, None otherwise.
    """
    log_info(f"{log_prefix} Attempting to save script transcript to: {output_path}")
    try:
        # Ensure parent directory exists
        if not create_directory(output_path.parent):
            log_error(
                f"{log_prefix} Failed to create parent directory for {output_path}. Aborting save."
            )
            return None

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {output_path.stem}\n")
            f.write("=" * (len(output_path.stem) + 12) + "\n\n")

            speaker_blocks = group_segments_by_speaker(segments)

            def format_time(seconds: Optional[float]) -> str:
                """Formats seconds into [HH:MM:SS.ss] or [MM:SS.ss]"""
                if (
                    seconds is None
                    or not isinstance(seconds, (int, float))
                    or seconds < 0
                ):
                    return "[??:??.??]"
                try:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = seconds % 60
                    if hours > 0:
                        return (
                            f"[{hours:02d}:{minutes:02d}:{secs:05.2f}]"  # HH:MM:SS.ss
                        )
                    else:
                        return f"[{minutes:02d}:{secs:05.2f}]"  # MM:SS.ss
                except Exception:
                    return "[??:??.??]"  # Fallback

            for block in speaker_blocks:
                speaker = block.get("speaker", "unknown")
                text = block.get("text", "").strip()
                start_time = block.get("start")
                end_time = block.get("end")

                if text:  # Only write blocks with actual text content
                    time_str = f"{format_time(start_time)} -> {format_time(end_time)}"
                    f.write(
                        f"{time_str} {speaker}:\n{text}\n\n"
                    )  # Add newline after speaker for readability

        log_info(f"{log_prefix} Script transcript saved successfully to {output_path}")
        return output_path

    except Exception as e:
        error_msg = (
            f"{log_prefix} Failed to save script transcript to {output_path}: {e}"
        )
        log_error(error_msg)
        log_error(traceback.format_exc())
        return None


# Moved from core/utils.py
def run_ffprobe_duration_check(audio_path: Path, min_duration: float = 5.0) -> bool:
    """
    Checks the duration of an audio file using ffprobe. Logs warnings.

    Args:
        audio_path: Path to the audio file.
        min_duration: Minimum duration in seconds required for diarization.

    Returns:
        True if duration check passes (or ffprobe fails safely),
        False if audio is shorter than min_duration.
    """
    log_info(f"Checking audio duration for {audio_path.name} (min: {min_duration}s)...")
    try:
        command = [
            "ffprobe",
            "-v",
            "error",  # Only show errors from ffprobe itself
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",  # Output only the value
            str(audio_path),
        ]
        log_info(f"Running ffprobe check: {' '.join(command)}")  # Restored log
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raise exception on non-zero exit code from ffprobe
            timeout=30,  # Add a timeout to prevent hanging
        )
        duration_str = result.stdout.strip()

        if not duration_str or duration_str == "N/A":
            log_warning(
                f"ffprobe could not determine duration for {audio_path.name}. Proceeding, but diarization might be skipped or fail."
            )
            return True  # Proceed cautiously if duration is unknown

        duration = float(duration_str)
        log_info(
            f"Detected audio duration: {duration:.2f} seconds for {audio_path.name}"
        )  # Restored log

        if duration < min_duration:
            log_warning(
                f"Audio duration ({duration:.1f}s) is less than the minimum "
                f"threshold ({min_duration}s) required for robust diarization. Quality may be affected."
            )
            return False  # Indicate check failed (too short)
        else:
            return True  # Indicate check passed (long enough)

    except FileNotFoundError:
        log_warning(
            "ffprobe command not found. Cannot perform audio duration check. Diarization quality check skipped."
        )
        return True  # Proceed if ffprobe is not available, assume long enough
    except subprocess.TimeoutExpired:
        log_warning(
            f"ffprobe timed out while checking duration for {audio_path.name}. Proceeding without check."
        )
        return True
    except subprocess.CalledProcessError as e:
        log_warning(
            f"ffprobe returned an error (exit code {e.returncode}) while checking duration for {audio_path.name}. Output: {e.stderr or e.stdout}. Proceeding without check."
        )
        return True
    except ValueError as e:
        log_warning(
            f"Could not convert ffprobe duration output ('{duration_str}') to float for {audio_path.name}: {e}. Proceeding without check."
        )
        return True
    except Exception as e:
        # Catch any other unexpected errors during the check
        log_warning(
            f"Unexpected error running ffprobe duration check for {audio_path.name}: {e}. Proceeding without check."
        )
        log_warning(traceback.format_exc())  # Log traceback for unexpected errors
        return True  # Proceed cautiously on other errors

```

### utils\wrapper.py

```py
# utils/wrapper.py
"""
Provides utility functions for wrapping and safely executing external processes.
"""

import os
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any, Callable

# Assuming core.logging is established from Phase 1
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


# Moved from core/utils.py
def safe_run(
    command: List[str],
    log_file_handle: Optional[TextIO],
    log_prefix: str = "[PROC]",
    env: Optional[Dict[str, str]] = None,
    output_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
) -> Optional[str]:
    """
    Runs an external command safely, logging output and handling errors.

    Args:
        command: The command and arguments as a list of strings.
        log_file_handle: An open file handle for writing process logs (optional).
        log_prefix: A prefix string for log messages (e.g., session ID).
        env: Optional dictionary of environment variables for the subprocess.
        output_callback: Optional function to process command output lines in real-time.
        capture_output: If True, captures and returns the standard output as a string.
                        Note: If True, output_callback might receive lines twice if
                        it also prints them. Best used when callback doesn't print.

    Returns:
        The captured standard output as a single string if capture_output is True,
        otherwise None.

    Raises:
        FileNotFoundError: If the command executable is not found.
        RuntimeError: If the command returns a non-zero exit code or another
                      error occurs during execution.
    """
    safe_command: List[str] = [str(item) for item in command]
    process: Optional[subprocess.Popen] = None
    captured_output_lines: List[str] = [] if capture_output else None

    # Prepare the environment for the subprocess
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    log_info(
        f"{log_prefix} Executing command: {' '.join(safe_command)}"
    )  # Log command execution

    try:
        # Start the subprocess
        process = subprocess.Popen(
            safe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 chars
            env=subprocess_env,
            bufsize=1,  # Line buffered
        )

        # Read output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                stripped_line = line.strip()  # Strip leading/trailing whitespace
                log_line: str = (
                    f"{log_prefix} {stripped_line}\n"  # Add prefix for log file
                )

                # Write raw line (with prefix) to the dedicated process log file
                if log_file_handle and not log_file_handle.closed:
                    try:
                        log_file_handle.write(log_line)
                        log_file_handle.flush()
                    except Exception as log_e:
                        # Fallback print if writing to file fails
                        print(
                            f"WARNING: Failed to write log line to file handle: {log_e} - Line: {stripped_line}"
                        )

                # Capture output line if requested
                if capture_output:
                    captured_output_lines.append(
                        line
                    )  # Append original line with newline

                # Pass the stripped line to the output callback
                if output_callback:
                    try:
                        output_callback(stripped_line)
                    except Exception as cb_e:
                        cb_error_msg = f"{log_prefix} ERROR in output_callback: {cb_e} - Line: {stripped_line}"
                        log_error(cb_error_msg)  # Log callback error using main logger
                        # Try to write callback error to the file handle
                        if log_file_handle and not log_file_handle.closed:
                            try:
                                log_file_handle.write(f"{cb_error_msg}\n")
                                log_file_handle.flush()
                            except Exception as log_e2:
                                print(
                                    f"WARNING: Failed to write callback error to file handle: {log_e2}"
                                )

            process.stdout.close()

        # Wait for the process and check return code
        return_code: int = process.wait()

        if return_code != 0:
            error_msg = (
                f"Command failed with exit code {return_code}: {' '.join(safe_command)}"
            )
            # Write error to process log file
            if log_file_handle and not log_file_handle.closed:
                try:
                    log_file_handle.write(f"{log_prefix} ERROR: {error_msg}\n")
                except Exception as log_e:
                    print(
                        f"WARNING: Failed to write command error to log file handle: {log_e}"
                    )
            # Log error using main logger
            log_error(error_msg)
            raise RuntimeError(error_msg)
        else:
            log_info(f"{log_prefix} Command finished successfully.")

        # Return captured output if requested
        return "".join(captured_output_lines) if capture_output else None

    except FileNotFoundError:
        error_msg = (
            f"Command not found: {safe_command[0]}. Ensure it is installed and in PATH."
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(f"{log_prefix} ERROR: {error_msg}\n")
            except Exception as log_e:
                print(
                    f"WARN: Failed to write FileNotFoundError to log file handle: {log_e}"
                )
        log_error(error_msg)
        raise  # Re-raise the original FileNotFoundError

    except Exception as e:
        # Catch other potential errors (e.g., permission issues)
        error_msg = (
            f"An error occurred while running command {' '.join(safe_command)}: {e}"
        )
        if log_file_handle and not log_file_handle.closed:
            try:
                log_file_handle.write(
                    f"{log_prefix} ERROR: {error_msg}\n{traceback.format_exc()}\n"
                )
            except Exception as log_e:
                print(
                    f"WARN: Failed to write other exception to log file handle: {log_e}"
                )
        log_error(error_msg)
        log_error(traceback.format_exc())  # Log full traceback

        # Attempt to terminate the process if it's still running
        if process and process.poll() is None:
            log_warning(
                f"{log_prefix} Attempting to terminate process {process.pid}..."
            )
            try:
                process.terminate()
                process.wait(timeout=5)  # Wait briefly for graceful termination
                log_warning(f"{log_prefix} Process terminated.")
            except subprocess.TimeoutExpired:
                log_warning(
                    f"{log_prefix} Process did not terminate gracefully, killing."
                )
                process.kill()  # Force kill
            except Exception as term_err:
                log_warning(
                    f"{log_prefix} Error terminating process after failure: {term_err}"
                )

        raise RuntimeError(error_msg) from e

```

### yt\converter.py

```py
# yt/converter.py
"""
Handles converting downloaded media streams to WAV format using ffmpeg
and checking audio duration using ffprobe.
"""

import re
import subprocess  # For duration check fallback if safe_run fails initally
from pathlib import Path
from typing import List, Optional, TextIO, Union

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
    from core.logging import log_info, log_warning, log_error
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


# Moved from core/utils.py (via utils/transcripts.py)
# Renamed for clarity within this module
def check_audio_duration(
    audio_path: Path, min_duration_sec: float = 5.0, log_prefix: str = "[FFPROBE]"
) -> bool:
    """
    Checks the duration of an audio file using ffprobe. Logs warnings.

    Args:
        audio_path: Path to the audio file.
        min_duration_sec: Minimum duration in seconds required for processing (e.g., diarization).
        log_prefix: Prefix for log messages.

    Returns:
        True if duration check passes (>= min_duration_sec) or ffprobe fails safely.
        False if audio is shorter than min_duration_sec.
    """
    if not audio_path.is_file():
        log_warning(
            f"{log_prefix} Audio file not found at {audio_path}, cannot check duration."
        )
        return False  # Cannot proceed if file doesn't exist

    log_info(
        f"{log_prefix} Checking audio duration for {audio_path.name} (min: {min_duration_sec}s)..."
    )
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        # Use safe_run for consistency, though direct subprocess call is also possible
        # If safe_run isn't available, this will raise RuntimeError from the fallback above
        duration_str = safe_run(
            command=command,
            log_file_handle=None,  # Don't need separate log for this usually
            log_prefix=log_prefix,
            capture_output=True,
            output_callback=lambda line: None,  # Suppress callback logging for ffprobe
        )

        if duration_str is None:
            # safe_run already logs the error if the command fails
            log_warning(
                f"{log_prefix} ffprobe command failed or returned no output for {audio_path.name}. Proceeding without duration check."
            )
            return True  # Proceed cautiously

        duration_str = duration_str.strip()
        if not duration_str or duration_str == "N/A":
            log_warning(
                f"{log_prefix} ffprobe could not determine duration for {audio_path.name}. Proceeding, quality checks skipped."
            )
            return True  # Proceed if duration is unknown

        duration = float(duration_str)
        log_info(
            f"{log_prefix} Detected audio duration: {duration:.2f} seconds for {audio_path.name}"
        )

        if duration < min_duration_sec:
            log_warning(
                f"{log_prefix} Audio duration ({duration:.1f}s) is less than the minimum "
                f"threshold ({min_duration_sec}s). Downstream processing (e.g., diarization) may be affected or skipped."
            )
            return False  # Indicate check failed (too short)
        else:
            return True  # Indicate check passed (long enough)

    except FileNotFoundError:
        log_warning(
            f"{log_prefix} ffprobe command not found. Cannot perform audio duration check."
        )
        return True  # Proceed if ffprobe is not available
    except ValueError as e:
        log_warning(
            f"{log_prefix} Could not convert ffprobe duration output ('{duration_str}') to float for {audio_path.name}: {e}. Proceeding without check."
        )
        return True
    except Exception as e:
        # Catch any other unexpected errors during the check (e.g., RuntimeError from safe_run)
        log_warning(
            f"{log_prefix} Unexpected error running ffprobe duration check for {audio_path.name}: {e}. Proceeding without check."
        )
        # Consider logging traceback for unexpected errors if needed:
        # import traceback
        # log_warning(traceback.format_exc())
        return True  # Proceed cautiously on other errors


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_channels: int = 1,
    audio_samplerate: int = 16000,
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[FFMPEG]",
) -> Optional[Path]:
    """
    Converts an audio/video file to a standardized WAV format using ffmpeg.

    Args:
        input_path: Path to the input media file.
        output_path: The desired path for the output WAV file.
        audio_channels: Number of audio channels for the output (default: 1 for mono).
        audio_samplerate: Sample rate for the output audio (default: 16000 Hz).
        log_file_handle: Optional file handle for logging subprocess output.
        log_prefix: Prefix for log messages.

    Returns:
        The Path object to the output WAV file if successful, None otherwise.
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    log_info(
        f"{log_prefix} Converting {input_path_obj.name} to WAV at {output_path_obj} (Channels: {audio_channels}, Rate: {audio_samplerate} Hz)"
    )

    if not input_path_obj.is_file():
        log_error(f"{log_prefix} Input file not found: {input_path_obj}")
        return None

    # Ensure output directory exists
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(
            f"{log_prefix} Failed to create parent directory {output_path_obj.parent}: {e}"
        )
        return None

    command: List[str] = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i",
        str(input_path_obj),  # Input file
        "-vn",  # No video output
        "-ac",
        str(audio_channels),  # Set audio channels
        "-ar",
        str(audio_samplerate),  # Set audio sample rate
        "-acodec",
        "pcm_s16le",  # Standard WAV codec (signed 16-bit little-endian PCM)
        "-nostdin",  # Disable interaction
        str(output_path_obj),  # Output file path
    ]

    def ffmpeg_output_callback(line: str):
        """Parses ffmpeg output for logging."""
        # Regex for progress: time=HH:MM:SS.ms bitrate=... speed=X.Yx
        progress_match = re.search(
            r"time=\s*(\d{2}:\d{2}:\d{2}\.\d+).*?speed=\s*([\d.]+)x", line
        )
        size_match = re.search(r"size=\s*(\S+)", line)  # Capture size info if present

        if progress_match:
            time_str = progress_match.group(1)
            speed_str = progress_match.group(2)
            size_str = f" (size: {size_match.group(1)})" if size_match else ""
            log_info(
                f"{log_prefix} Progress: time={time_str}, speed={speed_str}x{size_str}"
            )
        elif "error" in line.lower() or "failed" in line.lower():
            log_error(f"{log_prefix} ffmpeg Error Logged: {line.strip()}")
        # elif "warning" in line.lower(): # Often too verbose
        #    log_warning(f"{log_prefix} ffmpeg Warning: {line.strip()}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=ffmpeg_output_callback,
        )

        if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
            log_error(
                f"{log_prefix} Conversion finished, but output file is missing or empty: {output_path_obj}"
            )
            raise FileNotFoundError(
                f"ffmpeg failed to create a valid output file at {output_path_obj}"
            )

        log_info(
            f"{log_prefix} Conversion to WAV completed successfully: {output_path_obj}"
        )
        return output_path_obj

    except (RuntimeError, FileNotFoundError) as e:
        log_error(f"{log_prefix} Conversion failed for {input_path_obj.name}: {e}")
        output_path_obj.unlink(missing_ok=True)  # Cleanup failed output
        return None
    except Exception as e:
        log_error(
            f"{log_prefix} Unexpected error during conversion for {input_path_obj.name}: {e}"
        )
        output_path_obj.unlink(missing_ok=True)
        return None

```

### yt\downloader.py

```py
# yt/downloader.py
"""
Handles downloading audio/video streams from YouTube using yt-dlp.
"""

import re
from pathlib import Path
from typing import List, Optional, TextIO, Union

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
    from core.logging import log_info, log_warning, log_error
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


def download_youtube_stream(
    youtube_url: str,
    output_path: Union[str, Path],
    youtube_dl_format: str = "bestaudio/best",
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[YT DL]",
) -> Optional[Path]:
    """
    Downloads the best audio stream from a YouTube URL using yt-dlp.

    Args:
        youtube_url: The URL of the YouTube video.
        output_path: The desired path for the downloaded stream file (e.g., .webm, .m4a).
        youtube_dl_format: The format string for yt-dlp (default: 'bestaudio/best').
        log_file_handle: Optional file handle for logging subprocess output.
        log_prefix: Prefix for log messages.

    Returns:
        The Path object to the downloaded file if successful, None otherwise.
    """
    output_path_obj = Path(output_path)
    log_info(f"{log_prefix} Starting download for {youtube_url} to {output_path_obj}")

    # Ensure output directory exists (using file_manager utility eventually)
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(
            f"{log_prefix} Failed to create parent directory {output_path_obj.parent}: {e}"
        )
        return None

    command: List[str] = [
        "yt-dlp",
        "-f",
        youtube_dl_format,
        "--no-playlist",  # Ensure only single video is downloaded if URL is part of playlist
        "--no-abort-on-error",  # Try to continue if parts fail (e.g. subtitles)
        "-o",
        str(output_path_obj),  # Specify exact output path
        "--",  # End of options, ensures URL is treated as positional argument
        youtube_url,
    ]

    def yt_dlp_output_callback(line: str):
        """Parses yt-dlp output for logging."""
        progress_match = re.search(
            r"\[download\]\s+([\d.]+%) of\s+~?([\d.]+\s*\w+)\s+at\s+([\d.]+\s*\w+/s)\s+ETA\s+([\d:]+)",
            line,
        )
        if progress_match:
            log_info(
                f"{log_prefix} DL Progress: {progress_match.group(1)} at {progress_match.group(3)} ETA {progress_match.group(4)}"
            )
        elif "[download] Destination:" in line:
            # Logged outside if needed, this line is less informative with -o
            pass
        elif "[info]" in line and "Downloading" not in line:
            log_info(f"{log_prefix} Info: {line.split(':', 1)[-1].strip()}")
        elif "[ExtractAudio]" in line:
            log_info(
                f"{log_prefix} Extracting audio..."
            )  # yt-dlp might do internal conversion
        elif "ERROR:" in line:
            # Log error, but don't raise immediately, let safe_run handle exit code
            log_error(f"{log_prefix} yt-dlp Error Logged: {line.strip()}")

    try:
        safe_run(
            command=command,
            log_file_handle=log_file_handle,
            log_prefix=log_prefix,
            output_callback=yt_dlp_output_callback,
        )

        if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
            log_error(
                f"{log_prefix} Download finished, but output file is missing or empty: {output_path_obj}"
            )
            raise FileNotFoundError(
                f"yt-dlp failed to create a valid output file at {output_path_obj}"
            )

        log_info(f"{log_prefix} Download completed successfully: {output_path_obj}")
        return output_path_obj

    except (RuntimeError, FileNotFoundError) as e:
        log_error(f"{log_prefix} Download failed for {youtube_url}: {e}")
        # Attempt cleanup of potentially incomplete file
        output_path_obj.unlink(missing_ok=True)
        return None
    except Exception as e:
        log_error(
            f"{log_prefix} Unexpected error during download for {youtube_url}: {e}"
        )
        output_path_obj.unlink(missing_ok=True)
        return None

```

### yt\metadata.py

```py
# yt/metadata.py
"""
Handles fetching metadata (title, description, etc.) for YouTube videos using yt-dlp.
"""

import json
import traceback
from typing import Dict, Any, Optional, List

# Assuming utils.wrapper and core.logging are available from previous phases
try:
    from utils.wrapper import safe_run
    from core.logging import log_info, log_warning, log_error
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


def fetch_youtube_metadata(
    youtube_url: str, log_prefix: str = "[YT Meta]"
) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for a YouTube video as a JSON object using yt-dlp.

    Args:
        youtube_url: The URL of the YouTube video.
        log_prefix: Prefix for log messages.

    Returns:
        A dictionary containing video metadata if successful, None otherwise.
        Includes standard yt-dlp fields like 'title', 'description', 'uploader', etc.
    """
    log_info(f"{log_prefix} Fetching metadata for {youtube_url}...")
    command: List[str] = [
        "yt-dlp",
        "-j",  # Output JSON
        "--no-playlist",  # Ensure metadata for single video
        "--",
        youtube_url,
    ]

    try:
        # Use safe_run to capture the JSON output string
        # Do not pass log_file_handle here to avoid flooding log with JSON
        metadata_output = safe_run(
            command=command,
            log_file_handle=None,  # Avoid writing large JSON to main log
            log_prefix=log_prefix,
            capture_output=True,
            output_callback=lambda line: None,  # Suppress callback logging for this
        )

        if not metadata_output:
            log_warning(
                f"{log_prefix} No metadata output captured from yt-dlp for {youtube_url}."
            )
            return None  # Indicate failure if no output

        try:
            # Log that we received *some* output before parsing
            # log_info(f"{log_prefix} Received metadata output (length: {len(metadata_output)}). Parsing...") # Too verbose
            metadata = json.loads(metadata_output)
            log_info(
                f"{log_prefix} Metadata fetched and parsed successfully for {youtube_url}."
            )
            # Add the original URL for reference
            metadata["input_youtube_url"] = youtube_url
            return metadata

        except json.JSONDecodeError as e:
            log_error(
                f"{log_prefix} Failed to decode yt-dlp metadata JSON for {youtube_url}: {e}"
            )
            # Log the beginning of the problematic output for debugging
            log_error(
                f"{log_prefix} Start of problematic metadata output: {metadata_output[:500]}..."
            )
            return None  # Indicate failure
        except Exception as e:
            # Catch other potential errors during parsing
            log_error(
                f"{log_prefix} Unexpected error processing metadata for {youtube_url}: {e}"
            )
            log_error(traceback.format_exc())
            return None

    except RuntimeError as e:
        # safe_run already logged the command failure
        log_error(
            f"{log_prefix} Metadata fetch command failed for {youtube_url}. See previous logs."
        )
        return None
    except Exception as e:
        # Catch unexpected errors during safe_run execution itself
        log_error(
            f"{log_prefix} Unexpected error during metadata fetch execution for {youtube_url}: {e}"
        )
        return None

```
