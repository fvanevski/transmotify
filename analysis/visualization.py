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
