# pipeline/manager.py

"""pipeline.manager
------------------------------------
Stateless, linear orchestrator that wires together the core sub‑packages into
a single *run_pipeline* function.  It deliberately avoids UI concerns, Excel
reading, and interactive state – those live in the labeling layer and the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

from core.config import Config
from core.logging import get_logger

from transmotify_io.downloader import download_youtube
from transmotify_io.converter import convert_to_wav
from transcription.whisperx_wrapper import transcribe
# NOTE: Assuming 'Segment' type hint is available via SegmentsList or directly
from transcription.segments import load_segments, SegmentsList
from emotion.analyzer import MultimodalAnalyzer
from reporting.generator import generate_all
from labeling import selector as labeling_selector

logger = get_logger(__name__)

__all__ = ["run_pipeline"]


# ---------------------------------------------------------------------------\n# Public entry‑point\n# ---------------------------------------------------------------------------


def run_pipeline(
    sources: Iterable[str],
    cfg: Config,
    *,
    out_root: Path | str | None = None,
    interactive: bool | None = None,
) -> List[Dict[str, Any]]:
    """Process each *source* (YouTube URL or local audio file) through the
    complete speech‑analysis pipeline.

    Parameters
    ----------
    sources
        Iterable of input locations.  YouTube URLs must start with http/https;
        Anything else is treated as a local audio/video file.
    cfg
        A :class:`core.config.Config` instance.
    out_root
        Folder in which per‑item sub‑directories will be created.  Defaults to
        ``cfg.output_dir``.
    interactive
        Whether to compute data needed for interactive labeling.
        If *None* (default) the value is read from ``cfg.enable_interactive_labeling``.

    Returns
    -------\n    List[Dict[str, Any]]
        One manifest dictionary per input source, containing necessary data including
        segments, speaker data for labeling (if interactive), report paths etc.
    """
    out_root = Path(out_root or cfg.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    interactive = (
        cfg.enable_interactive_labeling if interactive is None else interactive
    )
    # Conditionally initialize analyzer only if needed?
    # For now, initialize always as it might be used even if labeling is off.
    analyzer = MultimodalAnalyzer(cfg)

    manifests: List[Dict[str, Any]] = []

    for idx, src in enumerate(sources, start=1):
        # --- Setup Item ---
        # Use a more descriptive name if possible, e.g., from filename or URL part
        source_name = Path(src).stem if not str(src).startswith(("http:", "https:")) else f"youtube_{idx:03d}"
        item_id = f"{source_name}_{idx:03d}" # Combine name and index
        item_dir = out_root / item_id
        item_dir.mkdir(exist_ok=True)
        logger.info(f"[{item_id}] Processing source: {src}")

        wav_path: Path | None = None # Initialize wav_path
        meta: Dict[str, Any] = {}
        segments: SegmentsList = []
        eligible: List[str] = []
        report_manifest: Dict[str, Any] = {}
        speaker_labeling_data: Dict[str, Dict[str, Any]] = {}

        try:
            # 1) Acquire / convert audio -------------------------------------------------
            logger.info(f"[{item_id}] Acquiring audio...")
            if str(src).startswith(("http://", "https://")):
                # Assume download_youtube returns Path and Dict[str, Any]
                wav_path, meta = download_youtube(str(src), cfg, tmp=item_dir / "tmp")
            else:
                local_path = Path(src)
                if not local_path.exists():
                    logger.error(f"[{item_id}] Local source file not found: {src}")
                    continue # Skip to next source
                # Assume convert_to_wav returns Path
                wav_path = convert_to_wav(local_path, dst=item_dir / f"{local_path.stem}.wav", cfg=cfg)
                meta = {"source_path": str(src), "source_type": "local_file"}

            if not wav_path or not wav_path.exists():
                 logger.error(f"[{item_id}] Failed to obtain valid WAV file. Skipping.")
                 continue

            # 2) ASR + diarization ------------------------------------------------------
            logger.info(f"[{item_id}] Transcribing and diarizing...")
            # Assume transcribe returns Path to JSON
            json_path = transcribe(wav_path, cfg, out_dir=item_dir / "whisperx")
            if not json_path or not json_path.exists():
                 logger.error(f"[{item_id}] Transcription failed or produced no output. Skipping.")
                 continue
            # Assume load_segments returns SegmentsList
            segments = load_segments(json_path)
            if not segments:
                 logger.warning(f"[{item_id}] Transcription output yielded no segments.")
                 # Continue processing? Or skip? For now, continue, reporting might handle empty lists.

            # 3) Multimodal emotion -----------------------------------------------------\n            logger.info(f"[{item_id}] Analyzing emotions...")
            # Assume analyzer.run modifies segments in-place or returns modified list
            segments = analyzer.run(segments, audio_path=wav_path, video_path=None) # video_path is None for now

            # 4) Interactive speaker eligibility (optional) ----------------------------
            if interactive:
                logger.info(f"[{item_id}] Identifying eligible speakers for labeling...")
                min_total_time = getattr(cfg, "speaker_labeling_min_total_time", 10.0)
                min_block_time = getattr(cfg, "speaker_labeling_min_block_time", 2.0)
                eligible = labeling_selector.identify_eligible_speakers(
                    segments, min_total_time=min_total_time, min_block_time=min_block_time
                )
                logger.info(f"[{item_id}] Found {len(eligible)} eligible speakers: {eligible}")


            # 5) Reporting & artifact generation (including snippets) -----------------
            logger.info(f"[{item_id}] Generating reports and artifacts...")
            # Call the updated generate_all, passing wav_path and eligible
            report_manifest, speaker_labeling_data = generate_all(
                 segments,
                 cfg,
                 artifact_root=item_dir,
                 wav_path=wav_path, # Pass the WAV path
                 eligible_speakers=eligible # Pass the list of eligible speakers
            )

            # --- Assemble Final Item Manifest ---
            item_manifest = {
                "id": item_id, # Add the item ID
                "name": source_name, # Add a display name
                "source": src,
                "output_directory": str(item_dir), # Use output_directory consistently
                "segments": segments, # Use "segments" key
                "speakers": speaker_labeling_data, # Use "speakers" key for labeling data
                "report_manifest": report_manifest,
                "metadata": meta,
                # Add eligible list if useful downstream, though LabelingSession recalculates
                # "eligible_speakers_list": eligible,
            }
            manifests.append(item_manifest)
            logger.info(f"[{item_id}] Processing complete.")

        except Exception as e:
            logger.error(f"[{item_id}] UNHANDLED EXCEPTION during processing of '{src}': {e}", exc_info=True)
            # Optionally append an error manifest?
            manifests.append({
                 "id": item_id,
                 "name": source_name,
                 "source": src,
                 "output_directory": str(item_dir),
                 "error": str(e),
                 "traceback": traceback.format_exc() if cfg.debug else "Enable debug mode for traceback",
            })

    logger.info(f"Pipeline finished. Processed {len(sources)} sources. Returning {len(manifests)} manifests.")
    return manifests

# Need to import traceback if adding error traceback logging
import traceback
