# config/config.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional  # Added Optional

# Use relative import for logging within the same top-level package
try:
    from core.logging import (
        log_error,
        log_warning,
        log_info,
    )  # Added log_warning, log_info

    LOGGING_AVAILABLE = True
except ImportError:
    # Fallback if core.logging is not available during isolated config use or setup race conditions
    LOGGING_AVAILABLE = False

    def log_error(message):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message):
        print(f"INFO (logging unavailable): {message}")


class Config:
    """Loads, validates, and provides access to application configuration."""

    def __init__(self, config_file: str = "config.json"):
        """
        Initializes the Config object by loading defaults, config file, and env vars.

        Args:
            config_file: The path to the JSON configuration file.
        """
        # Use print statements here because logging setup depends on Config being initialized first.
        print("INFO: Initializing configuration...")
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        print("INFO: Configuration loaded and validated.")

    def _load_defaults(self) -> Dict[str, Any]:
        """Returns a dictionary of default configuration values."""
        print("INFO: Loading configuration defaults...")
        return {
            # Directory/File Settings
            "output_dir": "output",
            "temp_dir": "temp",
            "log_level": "INFO",
            # Core Processing Settings
            "hf_token": None,  # Default to None, expect environment variable
            "device": "cpu",
            # Model Configurations
            "whisper_model_size": "large-v3",
            "whisper_language": "auto",
            "whisper_batch_size": 16,
            "whisper_compute_type": "float16",  # Good balance for many GPUs
            "audio_emotion_model": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            "pyannote_diarization_model": "pyannote/speaker-diarization-3.1",
            "deepface_detector_backend": "opencv",
            # Processing Parameters
            "min_diarization_duration": 5.0,
            "visual_frame_rate": 1,
            "text_fusion_weight": 0.6,
            "audio_fusion_weight": 0.4,
            # --- ADDED Default Report Flags ---
            "include_json_summary": True,
            "include_csv_summary": False,
            "include_script_transcript": False,
            "include_plots": False,
            "include_source_audio": True,
            # --- ADDED Default Cleanup Flag ---
            "cleanup_temp_on_success": True,
            # --- Deprecated/Unused ---
            "batch_size": 10,  # Seems unused in current pipeline logic
            "snippet_match_threshold": 0.80,  # Deprecated by user request
        }

    def _load_config(self):
        """Loads configuration from JSON file and environment variables, merging over defaults."""
        defaults = self._load_defaults()
        loaded_config = {}

        # Load from JSON file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                print(f"INFO: Loaded configuration from {self.config_file}")
            except json.JSONDecodeError:
                print(
                    f"WARN: Error decoding JSON from {self.config_file}. Using defaults for file values."
                )
                # Keep defaults, loaded_config remains empty
            except Exception as e:
                print(
                    f"WARN: Error loading {self.config_file}: {e}. Using defaults for file values."
                )
                # Keep defaults, loaded_config remains empty
        else:
            print(
                f"INFO: Configuration file {self.config_file} not found. Using default configuration."
            )

        # Merge defaults and loaded config
        self.config = {**defaults, **loaded_config}
        print("INFO: Merged defaults and file configuration.")

        # Override specific critical settings with environment variables
        # Hugging Face Token (Highest Priority)
        env_hf_token = os.getenv("HF_TOKEN")
        if env_hf_token:
            self.config["hf_token"] = env_hf_token
            print("INFO: Overriding 'hf_token' with environment variable HF_TOKEN.")
        elif not self.config.get("hf_token"):  # Check if still None after loading file
            print(
                "WARN: 'hf_token' not found in config file or HF_TOKEN environment variable."
            )

        # Device Preference
        env_device = os.getenv("DEVICE")
        if env_device and env_device in ["cuda", "cpu"]:
            self.config["device"] = env_device
            print(
                f"INFO: Overriding 'device' with environment variable DEVICE: {env_device}"
            )
        elif env_device:
            print(
                f"WARN: Environment variable DEVICE ('{env_device}') is invalid. Use 'cuda' or 'cpu'. Using value from config file or default ('{self.config.get('device')}')."
            )

    def _validate_config(self):
        """Performs basic validation on critical configuration settings."""
        print("INFO: Validating configuration...")
        # Validate Hugging Face Token (Crucial for Pyannote)
        hf_token = self.config.get("hf_token")
        if not hf_token:
            # This remains a critical startup warning, potentially error in future
            print(
                "CRITICAL WARNING: Hugging Face token ('hf_token') is missing. "
                "Set the HF_TOKEN environment variable or add 'hf_token' to config.json. "
                "Pyannote diarization models usually require a token."
            )
        # Allowing startup for now, but diarization might fail later.
        # Consider raising ValueError here if token is strictly required:
        # raise ValueError("Hugging Face token ('hf_token') is missing.")
        else:
            print("INFO: Hugging Face token is configured.")

        # Validate Device (ensure it's 'cuda' or 'cpu')
        device = self.config.get("device")
        if device not in ["cuda", "cpu"]:
            original_device = device
            self.config["device"] = "cpu"  # Fallback to CPU
            print(
                f"WARN: Invalid device '{original_device}' specified. Falling back to 'cpu'."
            )
        else:
            print(f"INFO: Device configured to '{device}'.")

        # Validate fusion weights (should sum close to 1)
        w_text = self.config.get("text_fusion_weight", 0.0)
        w_audio = self.config.get("audio_fusion_weight", 0.0)
        if not (0.99 <= (w_text + w_audio) <= 1.01):  # Allow slight float inaccuracy
            print(
                f"WARN: Fusion weights (text: {w_text}, audio: {w_audio}) do not sum close to 1. Normalization might occur in MultimodalAnalysis."
            )

    def save_config(self):
        """Saves the current configuration back to the config file."""
        # Use logging here if available, otherwise print
        log_func = log_info if LOGGING_AVAILABLE else print

        log_func(f"Attempting to save configuration to {self.config_file}...")
        try:
            config_dir = self.config_file.parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Avoid saving sensitive info like hf_token back to file if loaded from env
            config_to_save = self.config.copy()
            if os.getenv("HF_TOKEN") and "hf_token" in config_to_save:
                config_to_save["hf_token"] = (
                    None  # Or remove key: del config_to_save["hf_token"]
                )

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            log_func(f"Configuration saved successfully to {self.config_file}")
        except Exception as e:
            # Use log_error if possible
            error_func = log_error if LOGGING_AVAILABLE else print
            error_func(f"Failed to save configuration to {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key, returning default if not found."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a configuration value and saves the config file."""
        log_func = log_info if LOGGING_AVAILABLE else print
        log_func(f"Setting configuration key '{key}' and saving.")
        self.config[key] = value
        self.save_config()
