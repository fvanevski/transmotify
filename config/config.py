# config/config.py
# Phase 2 Update: Integrates constants from legacy core/constants.py into defaults
# Source: Based on legacy codebase.txt and core/constants.py
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

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
        logger.info("Initializing configuration...")
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        logger.info("Configuration loaded and validated.")

    def _load_defaults(self) -> Dict[str, Any]:
        """Returns a dictionary of default configuration values, including constants."""
        logger.info("Loading configuration defaults...")
        return {
            # --- Directory/File Settings ---
            "output_dir": "output",
            "temp_dir": "temp",
            "log_level": "INFO",
            # --- Core Processing Settings ---
            "hf_token": None,  # Recommended to set via HF_TOKEN env var
            "device": "cpu",  # Can be overridden by DEVICE env var ('cuda' or 'cpu')
            # --- Model Configurations ---
            # "whisper_model_size": "large-v3-turbo", # WhisperX specific
            # "whisper_language": "auto",  # WhisperX specific
            # "whisper_batch_size": 32,  # WhisperX specific
            # "whisper_compute_type": "float16",  # WhisperX specific
            "audio_emotion_model": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            # "pyannote_diarization_model": "pyannote/speaker-diarization-3.1",  # Diarization now part of Riva ASR
            "deepface_detector_backend": "opencv",  # e.g., 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
            # --- Riva ASR Configuration ---
            "riva_server_uri": "localhost:50051",
            "riva_asr_language_code": "en-US",
            "riva_max_speakers_diarization": 2, # Default max speakers for Riva diarization (None to disable explicit setting)
            "riva_enable_automatic_punctuation": True,
            "riva_request_timeout_seconds": 120, # Client-side timeout for Riva ASR requests
            # --- Processing Parameters ---
            "min_diarization_duration": 5.0,  # Minimum audio duration for diarization attempt (may still be relevant for deciding to call ASR with diarization)
            "visual_frame_rate": 1,  # Frames per second for visual analysis
            "text_fusion_weight": 0.6,  # Weight for text emotion in fusion
            "audio_fusion_weight": 0.4,  # Weight for audio emotion in fusion
            # --- Report Flags ---
            "include_json_summary": True,  # Generate detailed emotion_summary.json
            "include_csv_summary": False,  # Generate high-level emotion_summary.csv
            "include_script_transcript": True,  # Generate simple script_transcript.txt
            "include_plots": False,  # Generate emotion plots
            "include_source_audio": False,  # Include original audio in final zip
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
            "snippet_match_threshold": 0.80,  # Deprecated in favor of default_snippet_match_threshold [cite: 8]
        }

    def _load_config(self):
        """Loads configuration from defaults, JSON file, and environment variables.
        If a key in the JSON file has a 'null' (None) value, the hardcoded default for that key is used.
        """
        defaults = self._load_defaults()
        
        # Start with a copy of the defaults
        self.config = defaults.copy()
        logger.info("Loaded configuration defaults.")

        # Load from JSON file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config_from_json = json.load(f)
                logger.info(f"Loaded configuration overrides from {self.config_file}")

                # Iterate through items loaded from JSON.
                # Only override a default if the value in config.json is not None.
                for key, value in loaded_config_from_json.items():
                    if key in self.config: # Ensure the key is a known config parameter
                        if value is not None:
                            self.config[key] = value
                            logger.debug(f"Overriding '{key}' with value from config.json: {value}")
                        else:
                            logger.debug(f"Key '{key}' is null in config.json, using default: {self.config[key]}")
                    else:
                        logger.warning(f"Unknown key '{key}' found in {self.config_file}. Ignoring.")
                        
            except json.JSONDecodeError:
                logger.warning(
                    f"Error decoding JSON from {self.config_file}. Using defaults/env vars only."
                )
            except Exception as e:
                logger.warning(
                    f"Error loading {self.config_file}: {e}. Using defaults/env vars only."
                )
        else:
            logger.info(
                f"Configuration file {self.config_file} not found. Using default configuration and environment variables."
            )

        # Override specific critical settings with environment variables
        # This logic remains the same and will apply after defaults and JSON processing.
        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.config["hf_token"] = env_hf_token
            logger.info("Overriding 'hf_token' with environment variable HF_TOKEN.")
        
        env_device = os.getenv("DEVICE")
        if env_device and env_device in ["cuda", "cpu"]:
            self.config["device"] = env_device
            logger.info(
                f"Overriding 'device' with environment variable DEVICE: {env_device}"
            )
        elif env_device:
            logger.warning(
                f"Environment variable DEVICE ('{env_device}') is invalid. Use 'cuda' or 'cpu'. Using config value ('{self.config.get('device')}')."
            )

    def _validate_config(self):
        """Performs basic validation on critical configuration settings."""
        logger.info("Validating configuration...")
        hf_token = self.config.get("hf_token")
        # Allow hf_token to be None, but warn if diarization models are selected later?
        # For now, only warn if it's missing. The specific check for Pyannote model is removed.
        # If hf_token is needed for other models in the future, this warning remains generally useful.
        if not hf_token:
            logger.warning(
                "Hugging Face token ('hf_token') is missing (checked config and HF_TOKEN env var). "
                "Some Hugging Face models may require a token for download or use."
            )
        elif hf_token:
            logger.info("Hugging Face token is configured.")

        device = self.config.get("device")
        if device not in ["cuda", "cpu"]:
            original_device = device
            self.config["device"] = "cpu"  # Fallback to CPU
            logger.warning(
                f"Invalid device '{original_device}' specified. Falling back to 'cpu'."
            )
        else:
            logger.info(f"Device configured to '{device}'.")

        # Validate fusion weights
        w_text = self.config.get("text_fusion_weight", 0.0)
        w_audio = self.config.get("audio_fusion_weight", 0.0)
        # Use tolerance for floating point comparison
        if not (0.99 <= (w_text + w_audio) <= 1.01):
            logger.warning(
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
                logger.warning(
                    "'speaker_labeling_preview_duration' must be positive. Check config."
                )
        except (ValueError, TypeError):
            logger.warning(
                "Speaker labeling time parameters (min_total_time, min_block_time, preview_duration) must be numbers. Check config."
            )

        # Validate default snippet threshold
        try:
            thresh = float(self.config.get("default_snippet_match_threshold", 0.8))
            if not (0.0 <= thresh <= 1.0):
                logger.warning(
                    "'default_snippet_match_threshold' must be between 0.0 and 1.0. Check config."
                )
                # Optionally clamp the value here if desired
        except (ValueError, TypeError):
            logger.warning(
                "'default_snippet_match_threshold' must be a number. Check config."
            )

    def save_config(self):
        """Saves the current non-default configuration back to the config file."""
        logger.info(f"Attempting to save configuration to {self.config_file}...")
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
            logger.info(f"Configuration saved successfully to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key, returning default if not found."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a configuration value and saves the config file."""
        logger.info(f"Setting configuration key '{key}' and saving.")
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
