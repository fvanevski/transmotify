# core/multimodal_analysis.py
import cv2
import torch
import torchaudio
from collections import defaultdict, Counter
from pathlib import Path
from deepface import DeepFace
import traceback

# Import necessary types for type hinting
from typing import Dict, List, Optional, Any

# Import foreign_class for loading custom SpeechBrain models
from speechbrain.inference.interfaces import foreign_class

# Import Segment and SegmentsList from transcription
from .transcription import SegmentsList, Segment

from .emotion_analysis import EmotionAnalysis
from .logging import log_info, log_warning, log_error

# ADDED numpy for probability handling
import numpy as np


class MultimodalAnalysis:
    """
    Provides intermediate multimodal emotion and deception analysis for segments.
    Input: list of segments with 'text', 'start', 'end', 'speaker', and 'source_id';
           maps from source_id to corresponding audio and video paths.
    Outputs: updates segments in-place with keys:
      - text_emotion (list of dicts: [{'label': 'joy', 'score': 0.9}, ...])
      - audio_emotion (list of dicts: [{'label': 'hap', 'score': 0.7}, ...])
      - visual_emotion (dominant label or None)
      - fused_emotion (primary fused label based on aligned modalities)
      - fused_emotion_confidence (confidence score for fused_emotion)
      - significant_text_emotions (dict of significant text-only labels and their scores)
      - deception_flag (Placeholder)
      - emotion (mapped from fused_emotion for summary/plotting)
    """

    def __init__(self, config):  # Removed default device argument here, get from config
        self.config = config
        # Get device from config
        device_config = config.get("device", "cuda")
        self.device_sb = (
            "cuda" if torch.cuda.is_available() and device_config == "cuda" else "cpu"
        )
        # DeepFace uses GPU ID, 0 for first GPU, -1 for CPU
        self.device_other = (
            0 if torch.cuda.is_available() and device_config == "cuda" else -1
        )
        log_info("Initializing MultimodalAnalysis...")

        try:
            log_info("Loading text emotion analyzer...")
            # Pass config to EmotionAnalysis
            self.text_analyzer = EmotionAnalysis(config)
            if (
                hasattr(self.text_analyzer, "emotion_classifier")
                and self.text_analyzer.emotion_classifier
            ):
                log_info("Text emotion analyzer loaded successfully.")
            else:
                log_warning(
                    "Text emotion analyzer failed to load or classifier not initialized."
                )
                self.text_analyzer = None  # Ensure it's None if loading failed
        except Exception as e:
            log_error(f"Failed to initialize text emotion analyzer: {e}")
            self.text_analyzer = None

        log_info(
            f"Loading SpeechBrain audio emotion model onto device: {self.device_sb}..."
        )
        self.audio_model = None
        try:
            # Use the model source from config with a default
            audio_model_source = config.get(
                "audio_emotion_model",
                "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            )
            self.audio_model = foreign_class(
                source=audio_model_source,
                pymodule_file="custom_interface.py",  # This might need adjustment based on the specific SpeechBrain model
                classname="CustomEncoderWav2vec2Classifier",  # This might need adjustment
                run_opts={"device": self.device_sb},
            )
            log_info(
                "SpeechBrain audio emotion model loaded successfully using foreign_class."
            )
        except Exception as e:
            log_error(
                f"Failed to load SpeechBrain audio emotion model using foreign_class from source '{audio_model_source}': {e}"
            )
            self.audio_model = None

        # Define emotion label mappings and aligned labels
        # MODIFIED: Define mappings and aligned labels here
        self.audio_labels = [
            "hap",
            "sad",
            "ang",
            "neu",
        ]  # Example order, verify with model output
        # Assuming text_emotion_analysis output labels are consistent
        self.text_labels = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]

        # Mapping text labels to aligned labels (using text labels as reference for fusion)
        self.aligned_labels = [
            "anger",
            "joy",
            "sadness",
            "neutral",
        ]  # Define the common set of labels for fusion
        self.text_to_aligned = {
            "anger": "anger",
            "joy": "joy",
            "sadness": "sadness",
            "neutral": "neutral",
        }
        # Mapping audio labels to aligned labels (assuming 1-to-1 for aligned)
        self.audio_to_aligned = {
            "ang": "anger",
            "hap": "joy",
            "sad": "sadness",
            "neu": "neutral",
        }
        # Text labels that do not have a direct mapping in the aligned set
        self.text_only_labels = [
            label for label in self.text_labels if label not in self.text_to_aligned
        ]

        self.frame_rate = config.get("visual_frame_rate", 1)
        # Get fusion weights from config with defaults
        self.text_fusion_weight = config.get(
            "text_fusion_weight", 0.6
        )  # Default weights, can be tuned
        self.audio_fusion_weight = config.get("audio_fusion_weight", 0.4)
        # Ensure weights sum to 1 (simple normalization if configured weights don't)
        total_weight = self.text_fusion_weight + self.audio_fusion_weight
        if total_weight == 0:  # Avoid division by zero
            log_warning("Fusion weights sum to zero. Using equal weights.")
            self.text_fusion_weight = 0.5
            self.audio_fusion_weight = 0.5
        else:
            self.text_fusion_weight /= total_weight
            self.audio_fusion_weight /= total_weight

        log_info(
            f"Fusion weights: Text={self.text_fusion_weight:.2f}, Audio={self.audio_fusion_weight:.2f}"
        )

        # Initialize DeepFace components if visual analysis is expected
        self.deepface_initialized = False
        if config.get(
            "enable_visual_analysis", True
        ):  # Assume visual analysis is enabled by default if not in config
            try:
                # Perform a dummy analysis to trigger model loading
                log_info("Initializing DeepFace for visual analysis...")
                dummy_image = np.zeros(
                    (100, 100, 3), dtype=np.uint8
                )  # Create a dummy blank image
                # MODIFIED: Use device and backend/models from config for dummy analysis
                deepface_backend = self.config.get(
                    "deepface_detector_backend", "opencv"
                )
                deepface_models = self.config.get("deepface_models", ["Emotion"])
                DeepFace.analyze(
                    dummy_image,
                    actions=["emotion"],
                    enforce_detection=False,
                    prog_bar=False,
                    detector_backend=deepface_backend,
                    models=deepface_models,
                    device=f"cuda:{self.device_other}"
                    if self.device_other != -1
                    else "cpu",  # Use f-string for device
                )
                self.deepface_initialized = True
                log_info("DeepFace initialized successfully.")
            except Exception as e:
                log_error(f"Failed to initialize DeepFace for visual analysis: {e}")
                self.deepface_initialized = False
                log_warning(
                    "Visual emotion analysis will be skipped due to DeepFace initialization failure."
                )

    # MODIFIED: analyze method signature to accept segments, source_audio_map, source_video_map
    def analyze(
        self,
        segments: SegmentsList,
        source_audio_map: Dict[str, Path],
        source_video_map: Dict[str, Path],
    ) -> SegmentsList:
        """
        Analyzes emotion and deception for a list of segments from potentially multiple sources.
        Args:
            segments: List of segment dictionaries (must include 'source_id', 'text', 'start', 'end').
            source_audio_map: Dict mapping source_id (str) to Path object of the prepared audio file.
            source_video_map: Dict mapping source_id (str) to Path object of the video file.

        Returns:
            The updated list of segment dictionaries with analysis results.
        """
        sr = 16000  # Assuming 16kHz sample rate for audio model

        # --- Text Emotion Analysis (Can be done segment by segment) ---
        if self.text_analyzer is None:
            log_warning(
                "Text emotion analyzer not loaded. Skipping text emotion analysis."
            )
            for seg in segments:
                seg["text_emotion"] = []  # Store as empty list if skipped
        else:
            log_info("Analyzing text emotion for each segment...")
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                # analyze_emotion now returns list of score dicts or failure indicator
                seg["text_emotion"] = self.text_analyzer.analyze_emotion(
                    text
                )  # Store the full result list of dicts

        # --- Audio Emotion Analysis (Needs audio file per segment's source) ---
        if self.audio_model is None:
            log_warning(
                "SpeechBrain audio model not loaded. Skipping audio emotion analysis."
            )
            for seg in segments:
                seg["audio_emotion"] = []  # Store as empty list if skipped
        else:
            log_info("Analyzing audio emotion for each segment...")
            for i, seg in enumerate(segments):
                start, end = seg.get("start", 0), seg.get("end", 0)
                source_id = seg.get("source_id")

                if source_id is None:
                    log_warning(
                        f"Segment {i} ({start:.2f}-{end:.2f}s) is missing 'source_id'. Skipping audio emotion analysis."
                    )
                    seg["audio_emotion"] = []
                    continue

                audio_path = source_audio_map.get(source_id)

                if audio_path is None or not audio_path.exists():
                    log_warning(
                        f"Audio file not found for source '{source_id}' associated with segment {i}. Skipping audio emotion analysis."
                    )
                    seg["audio_emotion"] = []
                    continue

                if end <= start:
                    log_warning(
                        f"Segment {i} ({start:.2f}-{end:.2f}s) from source '{source_id}' has invalid duration. Skipping audio emotion analysis."
                    )
                    seg["audio_emotion"] = []
                    continue

                try:
                    # Load audio segment using the correct audio_path
                    waveform, sample_rate = torchaudio.load(
                        str(audio_path),  # Ensure path is string
                        frame_offset=int(start * sr),
                        num_frames=int((end - start) * sr),
                    )
                    waveform = waveform.to(self.device_sb)

                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim > 2:
                        log_warning(
                            f"Audio segment {i} ({start:.2f}-{end:.2f}s) from source '{source_id}' has unexpected waveform shape: {waveform.shape}. Skipping audio emotion."
                        )
                        seg["audio_emotion"] = []
                        continue

                    # Get raw model output, hopefully including probabilities/logits
                    out = self.audio_model.classify_batch(waveform)

                    # --- Extract Probabilities from SpeechBrain Output ---
                    audio_probs_raw = None
                    audio_labels_returned = (
                        None  # Sometimes labels are returned with probs
                    )

                    if isinstance(out, tuple):
                        if len(out) > 0 and isinstance(out[0], torch.Tensor):
                            audio_probs_raw = out[
                                0
                            ]  # Assuming first element is probabilities tensor
                            if audio_probs_raw.ndim > 1:
                                audio_probs_raw = audio_probs_raw.squeeze(
                                    0
                                )  # Remove batch dim if present
                            # Apply softmax if the model outputs logits instead of probabilities
                            if not torch.all(
                                (audio_probs_raw >= 0) & (audio_probs_raw <= 1)
                            ):
                                # It looks like logits, apply softmax
                                audio_probs_raw = torch.softmax(audio_probs_raw, dim=-1)
                                # log_info(f"Applied softmax to audio model output for segment {i} from source '{source_id}'.") # Can be verbose
                            # Convert to numpy for easier handling
                            audio_probs_raw = audio_probs_raw.detach().cpu().numpy()
                        if len(out) > 3 and isinstance(out[3], list):
                            audio_labels_returned = out[
                                3
                            ]  # Assuming labels are the fourth element

                    if audio_probs_raw is not None and len(audio_probs_raw) == len(
                        self.audio_labels
                    ):
                        # Create list of dictionaries for audio emotions
                        seg["audio_emotion"] = [
                            {
                                "label": self.audio_labels[j],
                                "score": float(audio_probs_raw[j]),
                            }
                            for j in range(len(self.audio_labels))
                        ]  # Ensure float
                    elif audio_labels_returned and len(audio_labels_returned) > 0:
                        # Fallback if probabilities not clear, just use the predicted label with score 1.0
                        log_warning(
                            f"Could not extract probabilities from SpeechBrain output for segment {i} from source '{source_id}'. Using top label only."
                        )
                        seg["audio_emotion"] = [
                            {"label": audio_labels_returned[0], "score": 1.0}
                        ]  # Store as list for consistency
                    else:
                        log_warning(
                            f"Unexpected output format from SpeechBrain classify_batch for segment {i} from source '{source_id}': {out}. Skipping audio emotion."
                        )
                        seg["audio_emotion"] = []  # Store as empty list on failure

                except Exception as e:
                    log_warning(
                        f"Audio emotion failed for segment {i} ({start:.2f}-{end:.2f}s) from source '{source_id}': {e}"
                    )
                    seg["audio_emotion"] = []  # Store as empty list on error

        # --- Visual Emotion Analysis (Needs video file per segment's source) ---
        if not self.deepface_initialized:
            log_warning("DeepFace not initialized. Skipping visual emotion analysis.")
            for seg in segments:
                seg["visual_emotion"] = None  # Store as None if skipped
        else:
            log_info("Analyzing visual emotion for each segment's source video...")
            # Group segments by source_id to process each video once
            segments_by_source = defaultdict(list)
            for seg in segments:
                source_id = seg.get("source_id")
                if source_id:
                    segments_by_source[source_id].append(seg)
                else:
                    seg["visual_emotion"] = (
                        None  # Cannot perform visual analysis without source_id
                    )
                    log_warning(
                        f"Segment starting at {seg.get('start', 0):.2f}s is missing 'source_id'. Cannot perform visual analysis."
                    )

            for source_id, source_segments in segments_by_source.items():
                video_path = source_video_map.get(source_id)
                if video_path is None or not video_path.exists():
                    log_warning(
                        f"Video file not found for source '{source_id}'. Skipping visual emotion analysis for these segments."
                    )
                    for seg in source_segments:
                        seg["visual_emotion"] = None
                    continue

                log_info(
                    f"Analyzing video emotion via DeepFace for source: {source_id} ({video_path})..."
                )
                cap = cv2.VideoCapture(str(video_path))  # Ensure path is string
                if not cap.isOpened():
                    log_error(
                        f"Failed to open video file at {video_path} for visual analysis for source '{source_id}'."
                    )
                    for seg in source_segments:
                        seg["visual_emotion"] = None  # Store as None on failure
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                interval = max(int(fps / self.frame_rate), 1)
                visuals = defaultdict(list)  # {timestamp: [list of dominant emotions]}
                idx = 0
                log_info(
                    f"Analyzing video frames for source '{source_id}' at {self.frame_rate} fps ({interval} frame interval)..."
                )
                deepface_backend = self.config.get(
                    "deepface_detector_backend", "opencv"
                )
                deepface_models = self.config.get(
                    "deepface_models", ["Emotion"]
                )  # Ensure 'Emotion' action is included

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        if not cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        ):
                            log_warning(
                                f"Error reading frame {idx} from video file for source '{source_id}'."
                            )
                        break

                    if idx % interval == 0:
                        ts = idx / fps
                        try:
                            res = DeepFace.analyze(
                                frame,
                                actions=["emotion"],
                                enforce_detection=False,
                                prog_bar=False,
                                detector_backend=deepface_backend,
                                # models=deepface_models, # DeepFace uses default models for actions unless specified otherwise
                                device=f"cuda:{self.device_other}"
                                if self.device_other != -1
                                else "cpu",
                            )
                            detected_emotions_in_frame = [
                                face_res.get("dominant_emotion")
                                for face_res in res
                                if "dominant_emotion" in face_res
                            ]

                            if detected_emotions_in_frame:
                                # Count occurrences of each detected emotion
                                emotion_counts = Counter(detected_emotions_in_frame)
                                # Find the emotion(s) with the highest count
                                max_count = max(emotion_counts.values())
                                most_common_emotions = [
                                    emo
                                    for emo, count in emotion_counts.items()
                                    if count == max_count
                                ]
                                # In case of a tie, pick the first one
                                visuals[ts].append(most_common_emotions[0])

                        except Exception as e:
                            # log_warning(f"DeepFace analyze error at {ts:.2f}s for source '{source_id}': {e}") # Too verbose
                            pass

                    idx += 1
                cap.release()
                log_info(
                    f"Finished analyzing {idx} video frames for source '{source_id}'. Found {len(visuals)} timestamps with visual detections."
                )

                log_info(
                    f"Assigning visual emotion to segments for source '{source_id}'..."
                )
                for seg in source_segments:
                    start, end = seg.get("start", 0), seg.get("end", 0)
                    segment_emotions = [
                        emo
                        for t, ems in visuals.items()
                        if start <= t < end
                        for emo in ems
                        if emo  # Filter out None or empty strings
                    ]
                    if segment_emotions:
                        # Assign the most frequent dominant emotion from frames within the segment
                        seg["visual_emotion"] = max(
                            set(segment_emotions), key=segment_emotions.count
                        )
                    else:
                        seg["visual_emotion"] = (
                            None  # Store as None if no visual detection in segment
                        )

        # --- Late Fusion with Weighted Averaging and Confidence ---
        log_info(
            "Performing weighted averaging late fusion and calculating confidence..."
        )
        for i, seg in enumerate(segments):
            text_emotion_scores = seg.get(
                "text_emotion", []
            )  # List of {'label': label, 'score': score}
            audio_emotion_scores = seg.get(
                "audio_emotion", []
            )  # List of {'label': label, 'score': score}
            # Visual emotion is a single label or None, not used in this probability fusion step

            # Initialize probability dictionaries for aligned labels
            aligned_text_probs = {label: 0.0 for label in self.aligned_labels}
            aligned_audio_probs = {label: 0.0 for label in self.aligned_labels}
            text_only_raw_probs = {
                label: 0.0 for label in self.text_only_labels
            }  # Probabilities for text-only labels

            # Populate raw probabilities for aligned text labels and text-only labels
            if text_emotion_scores:
                for item in text_emotion_scores:
                    label = item.get("label")
                    score = item.get("score", 0.0)
                    if label in self.text_to_aligned:
                        aligned_text_probs[self.text_to_aligned[label]] = score
                    elif label in self.text_only_labels:
                        text_only_raw_probs[label] = score

            # Populate raw probabilities for aligned audio labels
            if audio_emotion_scores:
                for item in audio_emotion_scores:
                    label = item.get("label")
                    score = item.get("score", 0.0)
                    if label in self.audio_to_aligned:
                        aligned_audio_probs[self.audio_to_aligned[label]] = score
                    # Assuming audio_emotion_scores contain probabilities that sum to 1 for audio_labels

            # --- Identify Significant Text-Only Emotions ---
            significant_text_emotions = {}
            # Find max probability among the audio labels for direct comparison
            max_individual_audio_prob = 0.0
            if audio_emotion_scores:
                max_individual_audio_prob = max(
                    [item.get("score", 0.0) for item in audio_emotion_scores]
                )

            for text_only_label in self.text_only_labels:
                text_only_prob = text_only_raw_probs.get(text_only_label, 0.0)
                # Check if this text-only probability is higher than ANY individual audio probability threshold (e.g., 0.5 or the max audio prob)
                # Using a simple threshold (e.g., 0.5) or relative comparison might be needed based on model outputs
                # Let's use a config threshold, fallback to comparing against max individual audio prob
                sig_text_threshold = self.config.get(
                    "significant_text_emotion_threshold",
                    max_individual_audio_prob * 1.1,
                )  # Example: 10% higher than max audio prob

                if text_only_prob > sig_text_threshold:  # Compare against threshold
                    # Also ensure it's not just a default/placeholder score
                    if text_only_prob > 0 and not (
                        text_only_label in ["no_text", "analysis_failed"]
                        and text_only_prob == 1.0
                    ):
                        significant_text_emotions[text_only_label] = float(
                            text_only_prob
                        )  # Ensure float

            seg["significant_text_emotions"] = (
                significant_text_emotions  # Store significant text-only emotions
            )

            # --- Normalize Aligned Text Probabilities ---
            # Only normalize if there are valid text probabilities for the aligned labels
            sum_aligned_text_probs = sum(aligned_text_probs.values())
            if sum_aligned_text_probs > 0:
                normalized_aligned_text_probs = {
                    label: score / sum_aligned_text_probs
                    for label, score in aligned_text_probs.items()
                }
            else:
                normalized_aligned_text_probs = {
                    label: 0.0 for label in self.aligned_labels
                }  # All zeros if no aligned text probs

            # --- Weighted Averaging for Aligned Labels ---
            fused_probs = {label: 0.0 for label in self.aligned_labels}
            for label in self.aligned_labels:
                # Ensure aligned_audio_probs.get(label, 0.0) is used to handle cases where an audio label mapping doesn't exist for an aligned label
                fused_probs[label] = (
                    self.text_fusion_weight * normalized_aligned_text_probs[label]
                    + self.audio_fusion_weight * aligned_audio_probs.get(label, 0.0)
                )

            # --- Determine Fused Emotion and Confidence ---
            if not fused_probs or sum(fused_probs.values()) == 0:
                # Case where no valid probabilities were available for aligned labels
                seg["fused_emotion"] = "unknown"
                seg["fused_emotion_confidence"] = 0.0
            else:
                max_fused_prob = max(fused_probs.values())
                # Find all labels with the max probability (handle ties)
                best_labels = [
                    label
                    for label, score in fused_probs.items()
                    if score == max_fused_prob
                ]
                # Use the first label in case of a tie, or add more sophisticated tie-breaking if needed
                seg["fused_emotion"] = best_labels[0] if best_labels else "unknown"
                seg["fused_emotion_confidence"] = float(max_fused_prob)  # Ensure float

            # --- Assign main 'emotion' key for summary/plotting (based on fused result) ---
            # MODIFIED: Base main emotion on the fused result, with fallback to significant text or visual
            main_emotion = seg.get("fused_emotion")

            # Fallback logic if fused emotion is 'unknown' or not meaningful
            if not isinstance(main_emotion, str) or main_emotion in ["unknown", None]:
                # If fused is unknown, check for significant text-only emotions
                sig_text_emotions = seg.get("significant_text_emotions", {})
                if sig_text_emotions:
                    # Use the dominant significant text emotion (by score)
                    dominant_sig_text = max(
                        sig_text_emotions, key=sig_text_emotions.get
                    )
                    main_emotion = dominant_sig_text
                else:
                    # If no significant text, check for visual emotion
                    visual_emotion = seg.get("visual_emotion")
                    if isinstance(visual_emotion, str) and visual_emotion not in [
                        "unknown",
                        None,
                    ]:
                        main_emotion = visual_emotion
                    else:
                        # Final fallback
                        main_emotion = (
                            "unknown"  # Or "neutral" depending on desired default
                        )

            # Ensure the final emotion is a string and not empty/None
            if not isinstance(main_emotion, str) or not main_emotion.strip():
                main_emotion = "unknown"

            seg["emotion"] = main_emotion

            # Optional: Keep raw scores for detailed inspection
            # seg['text_emotion_scores_raw'] = text_emotion_scores
            # seg['audio_emotion_scores_raw'] = audio_emotion_scores

        log_info(
            "Multimodal analysis complete. Segments updated with fused emotion, confidence, and significant text emotions."
        )
        return segments
