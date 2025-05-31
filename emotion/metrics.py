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
