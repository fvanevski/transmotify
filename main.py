 """Package entry‑point.

Initialises configuration, logging, and launches the Gradio UI defined in
`ui.interface`.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

from speech_analysis.core.config import Config
from speech_analysis.core.logging import setup_logging, get_logger

from ui.interface import build_ui  # NEW


def main() -> None:
    # ---------------------------------------------------------
    # Environment tweaks (CUDA TF32 etc.) – optional perf boost
    # ---------------------------------------------------------
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("INFO: Enabled TF32 for faster CUDA ops.")
    except Exception as e:  # pragma: no cover – perf hint only
        print(f"WARN: Could not set Torch backend TF32 flags: {e}")

    # ---------------------------------------------------------
    # Config + logging
    # ---------------------------------------------------------
    cfg = Config()
    setup_logging(cfg.config)
    logger = get_logger(__name__)

    if not cfg.get("hf_token"):
        logger.error("Hugging Face token missing – diarisation may fail.")
    else:
        logger.info("Hugging Face token configured.")

    # ---------------------------------------------------------
    # Build and launch Gradio interface
    # ---------------------------------------------------------
    try:
        ui = build_ui(cfg)
        logger.info("Launching Gradio UI …")
        ui.launch(server_name="0.0.0.0")
    except Exception as exc:
        logger.exception("Fatal: could not launch UI")
        print("FATAL: UI launch failed. Check logs.")
        traceback.print_exc()


if __name__ == "__main__":  # pragma: no cover
    main()
