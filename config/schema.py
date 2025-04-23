 """speech_analysis.config.schema
--------------------------------
Pydantic settings model that holds **all user‑tunable parameters** for the
speech‑analysis pipeline.  Environment variables automatically override the
defaults (thanks to BaseSettings).

* If `device` is set to "auto" (default) the validator detects CUDA
  availability at runtime and flips it to either "cuda" or "cpu".
* Deprecated legacy fields are intentionally **omitted**; a migration shim in
  `core.config` falls back to constants if an old module requests them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseSettings, Field, validator

try:
    import torch
except ImportError:  # torch might not be installed in all environments
    torch = None  # type: ignore


class Settings(BaseSettings):
    # ---------------------------------------------------------------------
    # Directories & logging
    # ---------------------------------------------------------------------
    output_dir: Path = Field(Path("output"), description="Where all artifacts are written.")
    temp_dir: Path = Field(Path("temp"), description="Scratch space for intermediate files.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_filename: str = Field("app.log", description="Name of the log file (inside output_dir)")

    # ------------------------------------------------------------------
    # Hardware / platform
    # ------------------------------------------------------------------
    device: Literal["auto", "cpu", "cuda"] = Field(
        "auto", description="Execution device. 'auto' → choose cuda if available else cpu."
    )
    hf_token: Optional[str] = Field(
        None, description="HuggingFace auth token (optional unless private models are used)."
    )

    # ------------------------------------------------------------------
    # WhisperX ASR parameters
    # ------------------------------------------------------------------
    whisper_model_size: Literal[
        "tiny", "base", "small", "medium", "large-v2", "large-v3"
    ] = "large-v3"
    whisper_language: str = "auto"
    whisper_batch_size: int = 16
    whisper_compute_type: Literal["float16", "float32", "int8"] = "float16"

    # ------------------------------------------------------------------
    # Models – emotion & diarization
    # ------------------------------------------------------------------
    audio_emotion_model: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    pyannote_diarization_model: str = "pyannote/speaker-diarization-3.1"
    deepface_detector_backend: Literal[
        "opencv", "retinaface", "mediapipe", "ssd"
    ] = "opencv"

    # ------------------------------------------------------------------
    # Processing parameters
    # ------------------------------------------------------------------
    min_diarization_duration: float = 5.0
    visual_frame_rate: int = 1

    text_fusion_weight: float = 0.6
    audio_fusion_weight: float = 0.4

    # ------------------------------------------------------------------
    # Output / report flags
    # ------------------------------------------------------------------
    include_json_summary: bool = True
    include_csv_summary: bool = False
    include_script_transcript: bool = False
    include_plots: bool = False
    include_source_audio: bool = True
    cleanup_temp_on_success: bool = True

    # ------------------------------------------------------------------
    # Interactive speaker labeling
    # ------------------------------------------------------------------
    enable_interactive_labeling: bool = False
    speaker_labeling_min_total_time: float = 15.0
    speaker_labeling_min_block_time: float = 10.0
    speaker_labeling_preview_duration: float = 5.0

    # ------------------------------------------------------------------
    # Validators & derived defaults
    # ------------------------------------------------------------------
    @validator("device", pre=True)  # runs before type validation so str accepted
    def _auto_device(cls, v: str):  # noqa: N805
        if v == "auto":
            cuda_ok = torch and getattr(torch, "cuda", None) and torch.cuda.is_available()
            return "cuda" if cuda_ok else "cpu"
        return v

    @validator("audio_fusion_weight")
    def _fusion_sum_to_one(cls, v: float, values):  # noqa: N805
        text_w = values.get("text_fusion_weight", 0.0)
        if abs(text_w + v - 1.0) > 1e-3:
            # Renormalise weights to sum to 1 while preserving ratio.
            total = text_w + v or 1.0
            values["text_fusion_weight"] = text_w / total
            return v / total
        return v

    class Config:
        env_prefix = "SA_"  # All env vars start with SA_ (e.g. SA_OUTPUT_DIR)
        env_file = ".env"  # Optional dotenv file
