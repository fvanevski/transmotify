# core/multimodal_analysis.py
import cv2
import torch
import torchaudio
from collections import defaultdict, Counter
from pathlib import Path
from deepface import DeepFace

# Import foreign_class for loading custom SpeechBrain models
from speechbrain.inference.interfaces import foreign_class

from .emotion_analysis import EmotionAnalysis
from .logging import log_info, log_warning, log_error
# ADDED numpy for probability handling
import numpy as np

class MultimodalAnalysis:
    """
    Provides intermediate multimodal emotion and deception analysis for segments.
    Input: list of segments with 'text', 'start', 'end', 'speaker'; audio_path and video_path.
    Outputs: updates segments in-place with keys:
      - text_emotion (list of dicts: [{'label': 'joy', 'score': 0.9}, ...])
      - audio_emotion (list of dicts: [{'label': 'hap', 'score': 0.7}, ...])
      - visual_emotion (dominant label or None)
      - fused_emotion (primary fused label based on aligned modalities)
      - fused_emotion_confidence (confidence score for fused_emotion)
      - significant_text_emotions (dict of significant text-only labels and their scores)
      - deception_flag
      - emotion (mapped from fused_emotion for summary/plotting)
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device_sb = "cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu"
        # DeepFace uses GPU ID, 0 for first GPU, -1 for CPU
        self.device_other = 0 if torch.cuda.is_available() and device == 'cuda' else -1
        log_info("Initializing MultimodalAnalysis...")

        try:
            log_info("Loading text emotion analyzer...")
            # Pass config to EmotionAnalysis
            self.text_analyzer = EmotionAnalysis(config)
            if self.text_analyzer.emotion_classifier:
                 log_info("Text emotion analyzer loaded successfully.")
            else:
                 log_warning("Text emotion analyzer failed to load.")
                 self.text_analyzer = None # Ensure it's None if loading failed
        except Exception as e:
            log_error(f"Failed to initialize text emotion analyzer: {e}")
            self.text_analyzer = None


        log_info(f"Loading SpeechBrain audio emotion model onto device: {self.device_sb}...")
        self.audio_model = None
        try:
            # Use the model source from config with a default
            audio_model_source = config.get("audio_emotion_model", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
            self.audio_model = foreign_class(
                source=audio_model_source,
                pymodule_file="custom_interface.py", # This might need adjustment based on the specific SpeechBrain model
                classname="CustomEncoderWav2vec2Classifier", # This might need adjustment
                run_opts={"device": self.device_sb}
            )
            log_info("SpeechBrain audio emotion model loaded successfully using foreign_class.")
        except Exception as e:
            log_error(f"Failed to load SpeechBrain audio emotion model using foreign_class from source '{audio_model_source}': {e}")
            self.audio_model = None

        # Define emotion label mappings and aligned labels
        # MODIFIED: Define mappings and aligned labels here
        self.audio_labels = ['hap', 'sad', 'ang', 'neu'] # Example order, verify with model output
        self.text_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

        # Mapping text labels to aligned labels (using text labels as reference)
        self.aligned_labels = ['anger', 'joy', 'sadness', 'neutral']
        self.text_to_aligned = {
            'anger': 'anger', 'joy': 'joy', 'sadness': 'sadness', 'neutral': 'neutral'
        }
        # Mapping audio labels to aligned labels (assuming 1-to-1 for aligned)
        self.audio_to_aligned = {
            'ang': 'anger', 'hap': 'joy', 'sad': 'sadness', 'neu': 'neutral'
        }
        # Text labels that do not have a direct mapping in the aligned set
        self.text_only_labels = [label for label in self.text_labels if label not in self.text_to_aligned]


        self.frame_rate = config.get("visual_frame_rate", 1)
        # Get fusion weights from config with defaults
        self.text_fusion_weight = config.get("text_fusion_weight", 0.6) # Default weights, can be tuned
        self.audio_fusion_weight = config.get("audio_fusion_weight", 0.4)
        # Ensure weights sum to 1 (simple normalization if configured weights don't)
        total_weight = self.text_fusion_weight + self.audio_fusion_weight
        if total_weight == 0: # Avoid division by zero
            log_warning("Fusion weights sum to zero. Using equal weights.")
            self.text_fusion_weight = 0.5
            self.audio_fusion_weight = 0.5
        else:
            self.text_fusion_weight /= total_weight
            self.audio_fusion_weight /= total_weight

        log_info(f"Fusion weights: Text={self.text_fusion_weight:.2f}, Audio={self.audio_fusion_weight:.2f}")


    def analyze(self, segments, audio_path: str, video_path: str):
        sr = 16000 # Assuming 16kHz sample rate for audio model

        # --- Audio Emotion Analysis ---
        if self.audio_model is None:
            log_warning("SpeechBrain audio model not loaded. Skipping audio emotion analysis.")
            for seg in segments:
                 seg['audio_emotion'] = [] # Store as empty list if skipped
        else:
            log_info("Analyzing audio emotion for each segment...")
            for i, seg in enumerate(segments):
                start, end = seg.get('start', 0), seg.get('end', 0)
                if end <= start:
                    log_warning(f"Segment {i} ({start:.2f}-{end:.2f}s) has invalid duration. Skipping audio emotion analysis.")
                    seg['audio_emotion'] = []
                    continue

                try:
                    # Load audio segment
                    waveform, sample_rate = torchaudio.load(
                        audio_path,
                        frame_offset=int(start * sr),
                        num_frames=int((end - start) * sr)
                    )
                    waveform = waveform.to(self.device_sb)

                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim > 2:
                        log_warning(f"Audio segment {i} ({start:.2f}-{end:.2f}s) has unexpected waveform shape: {waveform.shape}. Skipping audio emotion.")
                        seg['audio_emotion'] = []
                        continue

                    # Get raw model output, hopefully including probabilities/logits
                    # The exact output format depends on the SpeechBrain model's classify_batch
                    # We need the probabilities or logits to calculate probabilities if not provided directly
                    out = self.audio_model.classify_batch(waveform)

                    # --- Extract Probabilities from SpeechBrain Output ---
                    # This part is highly dependent on the exact structure of `out`
                    # Based on typical SpeechBrain outputs, probabilities might be the first element
                    # If `out` is (probabilities, embeddings, lengths, predicted_labels) or similar
                    audio_probs_raw = None
                    audio_labels_returned = None # Sometimes labels are returned with probs

                    if isinstance(out, tuple):
                         if len(out) > 0 and isinstance(out[0], torch.Tensor):
                              audio_probs_raw = out[0] # Assuming first element is probabilities tensor
                              if audio_probs_raw.ndim > 1:
                                   audio_probs_raw = audio_probs_raw.squeeze(0) # Remove batch dim if present
                              # Apply softmax if the model outputs logits instead of probabilities
                              if not torch.all((audio_probs_raw >= 0) & (audio_probs_raw <= 1)):
                                  # It looks like logits, apply softmax
                                  audio_probs_raw = torch.softmax(audio_probs_raw, dim=-1)
                                  log_info(f"Applied softmax to audio model output for segment {i}.")
                              # Convert to numpy for easier handling
                              audio_probs_raw = audio_probs_raw.detach().cpu().numpy()
                         if len(out) > 3 and isinstance(out[3], list):
                             audio_labels_returned = out[3] # Assuming labels are the fourth element

                    if audio_probs_raw is not None and len(audio_probs_raw) == len(self.audio_labels):
                         # Create list of dictionaries for audio emotions
                         seg['audio_emotion'] = [{'label': self.audio_labels[j], 'score': audio_probs_raw[j]} for j in range(len(self.audio_labels))]
                    elif audio_labels_returned and len(audio_labels_returned) > 0:
                        # Fallback if probabilities not clear, just use the predicted label with score 1.0
                         log_warning(f"Could not extract probabilities from SpeechBrain output for segment {i}. Using top label only.")
                         seg['audio_emotion'] = [{'label': audio_labels_returned[0], 'score': 1.0}] # Store as list for consistency
                    else:
                         log_warning(f"Unexpected output format from SpeechBrain classify_batch for segment {i}: {out}. Skipping audio emotion.")
                         seg['audio_emotion'] = [] # Store as empty list on failure


                except Exception as e:
                    log_warning(f"Audio emotion failed for segment {i} ({start:.2f}-{end:.2f}s): {e}")
                    seg['audio_emotion'] = [] # Store as empty list on error

        # --- Text Emotion Analysis ---
        if self.text_analyzer is None:
             log_warning("Text emotion analyzer not loaded. Skipping text emotion analysis.")
             for seg in segments:
                  seg['text_emotion'] = [] # Store as empty list if skipped
        else:
            log_info("Analyzing text emotion for each segment...")
            for i, seg in enumerate(segments):
                text = seg.get('text', '').strip()
                # analyze_emotion now returns list of score dicts or failure indicator
                seg['text_emotion'] = self.text_analyzer.analyze_emotion(text) # Store the full result list of dicts


        # --- Visual Emotion Analysis (Placeholder - No changes as per user request) ---
        if not video_path or not Path(video_path).exists():
             log_warning(f"Video file not provided or does not exist at {video_path}. Skipping visual emotion analysis.")
             for seg in segments:
                  seg['visual_emotion'] = None # Store as None if skipped
             visuals = {} # Keep visuals dict empty
        else:
            log_info(f"Analyzing visual emotion via DeepFace for video: {video_path}...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                 log_error(f"Failed to open video file at {video_path} for visual analysis.")
                 for seg in segments:
                      seg['visual_emotion'] = None # Store as None on failure
                 visuals = {} # Keep visuals dict empty
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                interval = max(int(fps / self.frame_rate), 1)
                visuals = defaultdict(list) # {timestamp: [list of dominant emotions]}
                idx = 0
                log_info(f"Analyzing video frames at {self.frame_rate} fps ({interval} frame interval)...")
                # MODIFIED: Use self.device_other for DeepFace
                deepface_backend = self.config.get("deepface_detector_backend", 'opencv') # Example backend config
                deepface_models = self.config.get("deepface_models", ['Emotion']) # Example models config

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        if not cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            log_warning(f"Error reading frame {idx} from video file.")
                        break

                    if idx % interval == 0:
                        ts = idx / fps
                        try:
                            # Pass device and backend/models from config
                            res = DeepFace.analyze(
                                frame,
                                actions=['emotion'],
                                enforce_detection=False,
                                prog_bar=False,
                                detector_backend=deepface_backend,
                                models=deepface_models, # DeepFace models for 'emotion' action are not explicitly set here, default is used.
                                device=f"cuda:{self.device_other}" if self.device_other != -1 else "cpu" # Use f-string for device
                            )
                            detected_emotions_in_frame = [face_res.get('dominant_emotion') for face_res in res if 'dominant_emotion' in face_res]

                            if detected_emotions_in_frame:
                                most_common_emotion = max(set(detected_emotions_in_frame), key=detected_emotions_in_frame.count)
                                visuals[ts].append(most_common_emotion)

                        except Exception as e:
                            # log_warning(f"DeepFace analyze error at {ts:.2f}s: {e}") # Too verbose
                            pass

                    idx += 1
                cap.release()
                log_info(f"Finished analyzing {idx} video frames. Found {len(visuals)} timestamps with visual detections.")

            log_info("Assigning visual emotion to segments...")
            for i, seg in enumerate(segments):
                start, end = seg.get('start', 0), seg.get('end', 0)
                segment_emotions = [
                    emo for t, ems in visuals.items()
                    if start <= t < end
                    for emo in ems if emo
                ]
                if segment_emotions:
                    # Just assign the most frequent dominant emotion from frames within the segment
                    seg['visual_emotion'] = max(set(segment_emotions), key=segment_emotions.count)
                else:
                    seg['visual_emotion'] = None # Store as None if no visual detection in segment


        # --- Late Fusion with Weighted Averaging and Confidence ---
        log_info("Performing weighted averaging late fusion and calculating confidence...")
        for i, seg in enumerate(segments):
            text_emotion_scores = seg.get('text_emotion', []) # List of {'label': label, 'score': score}
            audio_emotion_scores = seg.get('audio_emotion', []) # List of {'label': label, 'score': score}
            # Visual emotion is a single label or None, not used in this probability fusion step

            # Initialize probability dictionaries for aligned labels
            aligned_text_probs = {label: 0.0 for label in self.aligned_labels}
            aligned_audio_probs = {label: 0.0 for label in self.aligned_labels}
            text_only_raw_probs = {label: 0.0 for label in self.text_only_labels} # Probabilities for text-only labels

            # Populate raw probabilities for aligned text labels and text-only labels
            if text_emotion_scores:
                 for item in text_emotion_scores:
                     label = item.get('label')
                     score = item.get('score', 0.0)
                     if label in self.text_to_aligned:
                         aligned_text_probs[self.text_to_aligned[label]] = score
                     elif label in self.text_only_labels:
                          text_only_raw_probs[label] = score


            # Populate raw probabilities for aligned audio labels
            if audio_emotion_scores:
                 for item in audio_emotion_scores:
                     label = item.get('label')
                     score = item.get('score', 0.0)
                     if label in self.audio_to_aligned:
                         aligned_audio_probs[self.audio_to_aligned[label]] = score
                     # Assuming audio_emotion_scores contain probabilities that sum to 1 for audio_labels


            # --- Identify Significant Text-Only Emotions ---
            significant_text_emotions = {}
            max_audio_prob_among_aligned = max(aligned_audio_probs.values()) if aligned_audio_probs else 0.0 # Max prob among the 4 audio labels

            # Find max probability among the audio labels for direct comparison
            max_individual_audio_prob = 0.0
            if audio_emotion_scores:
                 max_individual_audio_prob = max([item.get('score', 0.0) for item in audio_emotion_scores])


            for text_only_label in self.text_only_labels:
                 text_only_prob = text_only_raw_probs.get(text_only_label, 0.0)
                 # Check if this text-only probability is higher than ANY individual audio probability
                 if text_only_prob > max_individual_audio_prob:
                      # Also ensure it's not just a default/placeholder score
                      if text_only_prob > 0 and not (text_only_label in ["no_text", "analysis_failed"] and text_only_prob == 1.0):
                           significant_text_emotions[text_only_label] = text_only_prob


            seg['significant_text_emotions'] = significant_text_emotions # Store significant text-only emotions


            # --- Normalize Aligned Text Probabilities ---
            # Only normalize if there are valid text probabilities for the aligned labels
            sum_aligned_text_probs = sum(aligned_text_probs.values())
            if sum_aligned_text_probs > 0:
                normalized_aligned_text_probs = {
                    label: score / sum_aligned_text_probs for label, score in aligned_text_probs.items()
                }
            else:
                normalized_aligned_text_probs = {label: 0.0 for label in self.aligned_labels} # All zeros if no aligned text probs


            # --- Weighted Averaging for Aligned Labels ---
            fused_probs = {label: 0.0 for label in self.aligned_labels}
            for label in self.aligned_labels:
                 fused_probs[label] = (self.text_fusion_weight * normalized_aligned_text_probs[label] +
                                        self.audio_fusion_weight * aligned_audio_probs.get(label, 0.0)) # Use .get for safety

            # --- Determine Fused Emotion and Confidence ---
            if not fused_probs or sum(fused_probs.values()) == 0:
                # Case where no valid probabilities were available for aligned labels
                seg['fused_emotion'] = "unknown"
                seg['fused_emotion_confidence'] = 0.0
            else:
                 max_fused_prob = max(fused_probs.values())
                 # Find all labels with the max probability (handle ties)
                 best_labels = [label for label, score in fused_probs.items() if score == max_fused_prob]
                 # Use the first label in case of a tie, or add more sophisticated tie-breaking if needed
                 seg['fused_emotion'] = best_labels[0] if best_labels else "unknown"
                 seg['fused_emotion_confidence'] = max_fused_prob


            # --- Assign main 'emotion' key for summary/plotting (based on fused result) ---
            # MODIFIED: Base main emotion on the fused result
            main_emotion = seg.get('fused_emotion')
            if not isinstance(main_emotion, str) or main_emotion in ["unknown", "analysis_skipped", "analysis_failed", "no_text", None]:
                # Fallback to a default if fused is still unknown/invalid
                main_emotion = "unknown" # Or "neutral" depending on desired default

            seg['emotion'] = main_emotion

            # Optional: Keep raw scores for detailed inspection
            # seg['text_emotion_scores_raw'] = text_emotion_scores
            # seg['audio_emotion_scores_raw'] = audio_emotion_scores


        log_info("Multimodal analysis complete. Segments updated with fused emotion, confidence, and significant text emotions.")
        return segments