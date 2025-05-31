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
