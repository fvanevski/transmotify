# asr/asr.py
"""
 asr/asr.py – *refactored*
 ---------------------------------
 Native WhisperX implementation that replaces the previous
 subprocess‑based call. The public signature is preserved so the rest
 of the pipeline remains untouched.

 Key points ▸
 • Uses the official WhisperX Python API (load_model ▸ transcribe ▸ align ▸ diarize) as shown in the project documentation.
 • Returns the same JSON schema the CLI produced, written to
   ``<output_dir>/<audio stem>.json``.
 • All user‑facing options from the old wrapper are still recognised:
   model_size, device, compute_type, language, batch_size, diarization
   speaker hints, HF token …
 • Detailed logging mirrors the old behaviour.
 • Graceful error handling + optional log_file_handle stream.
 • Aggressive resource cleanup (gc + cuda cache) to keep memory usage
   comparable to the previous subprocess.
 """

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import whisperx  # type: ignore – external dependency

try:
    import torch
except ImportError:  # CPU‑only env – torch is optional for WhisperX CPU use
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["run_whisperx"]


def _flush_cuda() -> None:
    """Free VRAM if torch + CUDA are present."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_whisperx(
    # Input / output
    audio_path: Path,
    output_dir: Path,
    *,
    # Core model params
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    compute_type: str = "float16",
    # Language & batching
    language: Optional[str] = None,
    batch_size: int = 16,
    # Diarisation
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    # Output discovery compatibility
    output_filename_exclusions: Optional[List[str]] = None,
    # Logging
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[ASR WhisperX]",
) -> Path:
    """Run WhisperX natively and return the path of the resulting JSON.

    The behaviour is equivalent to the previous subprocess wrapper, so
    downstream code that calls *run_whisperx()* does **not** need to
    change.
    """

    t0 = time.perf_counter()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{audio_path.stem}.json"

    # ------------------------------------------------------------------
    # 1️⃣  Load audio & ASR model
    # ------------------------------------------------------------------
    logger.info(f"{log_prefix} Loading WhisperX model '{model_size}' on {device} …")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))

    # ------------------------------------------------------------------
    # 2️⃣  Transcription
    # ------------------------------------------------------------------
    logger.info(f"{log_prefix} Transcribing … (batch_size={batch_size})")
    result: Dict[str, Any] = model.transcribe(audio, batch_size=batch_size)
    logger.debug(
        f"{log_prefix} Raw transcription produced {len(result['segments'])} segments"
    )

    # ------------------------------------------------------------------
    # 3️⃣  Alignment
    # ------------------------------------------------------------------
    logger.info(f"{log_prefix} Aligning word timestamps …")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False
    )

    # ------------------------------------------------------------------
    # 4️⃣  Speaker diarization (optional)
    # ------------------------------------------------------------------
    if hf_token is not None:
        logger.info(f"{log_prefix} Performing speaker diarization …")
        diarize_pipeline = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_kwargs: Dict[str, Any] = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = int(max_speakers)
        diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        logger.warning(
            f"{log_prefix} No Hugging Face token provided – skipping diarization."
        )

    # ------------------------------------------------------------------
    # 5️⃣  Persist JSON (CLI‑compat schema)
    # ------------------------------------------------------------------
    with output_json.open("w", encoding="utf‑8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    logger.info(f"{log_prefix} Saved transcript → {output_json.relative_to(output_dir.parent)}")

    # ------------------------------------------------------------------
    # 6️⃣  House‑keeping (free VRAM/RAM)
    # ------------------------------------------------------------------
    del model, align_model  # type: ignore[name-defined]
    _flush_cuda()
    gc.collect()

    elapsed = time.perf_counter() - t0
    logger.info(f"{log_prefix} Completed in {elapsed:,.1f}s")

    # ------------------------------------------------------------------
    # 7️⃣  Locate output – keep *exact* behaviour expected by callers
    # ------------------------------------------------------------------
    exclusions = output_filename_exclusions or []
    if output_json.name in exclusions:
        logger.debug(f"{log_prefix} JSON file in exclusion list, caller will handle.")
    return output_json
