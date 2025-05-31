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
            # self.model = foreign_class(
            #     source=self.model_source,
            #     pymodule_file=self.pymodule_file, # This might need adjustment based on the specific SpeechBrain model
            #     classname=self.classname, # This might need adjustment
            #     run_opts={"device": self.device}
            # )

            # Attempt to get labels from the loaded model (if available)
            # if hasattr(self.model, "hparams") in self.model.hparams:
            #     self.model_labels = self.model.hparams.label_encoder.allowed_labels
            #     log_info(f"Audio model labels identified: {self.model_labels}")
            # else:
            #     self.model_labels = ['hap', 'sad', 'ang', 'neu'] # Example order, verify with model output
            #     self.text_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            #     log_warning(
            #         "Could not automatically determine audio model labels from hparams."
            #     )
                # Consider adding a manual way to set labels if needed
            self.model_labels = ['hap', 'sad', 'ang', 'neu'] # Example order, verify with model output
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
