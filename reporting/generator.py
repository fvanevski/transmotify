# reporting/generator.py

\"\"\"reporting.generator
--------------------------------------
Generate all reporting artifacts for a single analysis run.

The function `generate_all` is designed to be called by the pipeline manager
once the segment list has been fully annotated (fused emotions, etc.). It
creates the artifact directory if needed, delegates work to specialised
modules, and returns a manifest mapping logical names to output paths, plus
data required for interactive speaker labeling.
\"\"\"

from __future__ import annotations

import json
import subprocess
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from core.config import Config
from core.logging import get_logger
from constants import (
    EMOTION_SUMMARY_JSON_NAME, EMOTION_SUMMARY_CSV_NAME
)
from reporting import summaries as summ
from reporting import plotting as pltg
from reporting import export as exp
# Need Segment type hint and grouping function
from transcription.segments import Segment, SegmentsList, group_segments_by_speaker
from utils.subprocess import run as run_subprocess, SubprocessError # Use the project's subprocess runner

logger = get_logger(__name__)

__all__ = ["generate_all"]


# Helper function for audio snippet extraction
def _extract_audio_snippet(
    input_wav: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    cfg: Config # Pass config for potential ffmpeg path etc.
) -> Path | None:
    """Extracts an audio snippet using ffmpeg."""
    end_time = start_time + duration
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use libmp3lame for MP3 encoding, reasonable quality preset
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", str(input_wav),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-vn", # No video
        "-acodec", "libmp3lame",
        "-q:a", "5", # Quality preset (0-9, lower is better)
        # "-ac", "1", # Force mono? Optional.
        # "-ar", "16000", # Force sample rate? Optional.
        str(output_path),
    ]
    try:
        logger.info(f"Running ffmpeg to extract snippet: {' '.join(cmd)}")
        run_subprocess(cmd, capture_output=True) # Use capture_output to hide ffmpeg verbose logs unless error
        if output_path.exists() and output_path.stat().st_size > 0:
             logger.info(f"Successfully created snippet: {output_path.name}")
             return output_path
        else:
             logger.error(f"ffmpeg ran but output snippet {output_path.name} is missing or empty.")
             return None
    except SubprocessError as e:
        logger.error(f"ffmpeg failed to extract snippet {output_path.name}: {e}", exc_info=False)
        logger.error(f"ffmpeg command: {' '.join(cmd)}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting snippet {output_path.name}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------\n# Public API\n# ---------------------------------------------------------------------------


def generate_all(
    segments: SegmentsList,
    cfg: Config,
    artifact_root: Path,
    *, # Make new args keyword-only for clarity
    wav_path: Path | None = None, # Path to original WAV needed for snippets
    eligible_speakers: List[str] | None = None # List of speakers needing snippets/transcripts
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]: # Return (report_manifest, speaker_labeling_data)
    \"\"\"Create all reporting artifacts and data for labeling.

    Parameters
    ----------
    segments
        The list of segments with fused emotions already populated.
    cfg
        Validated runtime configuration.
    artifact_root
        Directory where artifacts should be written.
    wav_path
        Path to the full source WAV file for this item. Required if interactive labeling is enabled.
    eligible_speakers
        List of speaker IDs identified as eligible for labeling. Required if interactive labeling is enabled.

    Returns
    -------\n    Tuple[Dict, Dict]
        1. `report_manifest`: Mapping from logical artifact keys (``summary_json``, ``plots`` …)
           to their respective file paths.
        2. `speaker_labeling_data`: Mapping from eligible speaker IDs to dictionaries containing
           `audio_snippet_path` and `example_transcript`. Empty if labeling not enabled/possible.
    \"\"\"

    artifact_root.mkdir(parents=True, exist_ok=True)

    report_manifest: Dict[str, Path | Dict[str, Path]] = {}
    speaker_labeling_data: Dict[str, Dict[str, Any]] = {}

    # --- Ensure inputs are valid for labeling snippet generation ---
    can_generate_snippets = (
         cfg.enable_interactive_labeling and
         eligible_speakers and
         wav_path and
         wav_path.exists() and
         shutil.which("ffmpeg") # Check if ffmpeg command exists
    )
    if cfg.enable_interactive_labeling and not can_generate_snippets:
         logger.warning("Interactive labeling enabled, but cannot generate speaker snippets.")
         if not eligible_speakers: logger.warning("- No eligible speakers identified.")
         if not wav_path or not wav_path.exists(): logger.warning(f"- Source WAV path invalid or missing: {wav_path}")
         if not shutil.which("ffmpeg"): logger.warning("- ffmpeg command not found in PATH.")

    # ------------------------------------------------------------------
    # 0. Speaker Snippets & Transcripts (for Labeling)
    # ------------------------------------------------------------------
    if can_generate_snippets:
        logger.info(f"Generating snippets/transcripts for {len(eligible_speakers)} speakers...")
        snippets_dir = artifact_root / "labeling_snippets"
        snippets_dir.mkdir(exist_ok=True)
        # Default duration unless overridden in config
        snippet_duration = getattr(cfg, 'speaker_labeling_snippet_duration', 5.0)

        # Group segments by speaker once for efficiency
        speaker_to_segments: Dict[str, List[Segment]] = defaultdict(list)
        for seg in segments:
            spk = seg.get("speaker")
            if spk:
                speaker_to_segments[str(spk)].append(seg)

        for speaker_id in eligible_speakers:
            spk_segments = speaker_to_segments.get(speaker_id)
            if not spk_segments:
                logger.warning(f"Speaker {speaker_id} is eligible but has no segments? Skipping snippet gen.")
                continue

            # Find the longest continuous block for this speaker
            blocks = group_segments_by_speaker(spk_segments)
            if not blocks:
                logger.warning(f"Could not group segments into blocks for speaker {speaker_id}. Skipping snippet gen.")
                continue

            longest_block = max(blocks, key=lambda b: b.get("end", 0) - b.get("start", 0))
            start_time = longest_block.get("start")
            example_transcript = longest_block.get("text", "...")

            if start_time is None:
                 logger.warning(f"Longest block for speaker {speaker_id} has no start time. Skipping snippet gen.")
                 continue

             # Define output path for the snippet (use MP3 for web compatibility)
            snippet_path = snippets_dir / f"{speaker_id}_snippet.mp3"

            # Extract snippet using ffmpeg
            extracted_path = _extract_audio_snippet(
                input_wav=wav_path,
                output_path=snippet_path,
                start_time=start_time,
                duration=snippet_duration,
                cfg=cfg
            )

            if extracted_path:
                speaker_labeling_data[speaker_id] = {
                    "audio_snippet_path": str(extracted_path), # Store as string for JSON etc.
                    "example_transcript": example_transcript.strip()
                }
            else:
                 logger.warning(f"Failed to generate audio snippet for speaker {speaker_id}.")
                 # Store placeholder or skip? Skipping means this speaker won't appear in labeling UI.
                 # Let's skip for now, LabelingSession init should handle missing keys.


    # ------------------------------------------------------------------
    # 1. Summaries (JSON / CSV)
    # ------------------------------------------------------------------
    # (Keep existing summary logic)
    stats = summ.build_summary(segments)

    if cfg.include_json_summary:
        json_path = artifact_root / EMOTION_SUMMARY_JSON_NAME
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info("Saved JSON summary → %s", json_path.name)
            report_manifest["summary_json"] = json_path
        except Exception as e:
            logger.error(f"Failed to save JSON summary: {e}", exc_info=True)


    if cfg.include_csv_summary:
        try:
            csv_path = artifact_root / EMOTION_SUMMARY_CSV_NAME
            summ.save_csv(stats, csv_path)
            logger.info("Saved CSV summary → %s", csv_path.name)
            report_manifest.setdefault("summary_csv", csv_path)
        except ImportError:
            logger.warning("pandas not installed – skipping CSV summary export")
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {e}", exc_info=True)


    # ------------------------------------------------------------------
    # 2. Plots
    # ------------------------------------------------------------------
    # (Keep existing plot logic)
    if cfg.include_plots:
        plots_dir = artifact_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        plots_manifest: Dict[str, Path] = {}
        plot_errors = False

        try:
            traj = pltg.plot_trajectory(segments, plots_dir / "trajectory.png")
            if traj: plots_manifest["trajectory"] = traj
            else: plot_errors=True
        except Exception as e: logger.error(f"Failed to generate trajectory plot: {e}", exc_info=True); plot_errors=True
        try:
            dist = pltg.plot_distribution(segments, plots_dir / "distribution.png")
            if dist: plots_manifest["distribution"] = dist
            else: plot_errors=True
        except Exception as e: logger.error(f"Failed to generate distribution plot: {e}", exc_info=True); plot_errors=True
        try:
            # Ensure stats['per_speaker'] exists before passing
            if stats and "per_speaker" in stats:
                 vol = pltg.plot_volatility(stats["per_speaker"], plots_dir / "volatility.png")
                 if vol: plots_manifest["volatility"] = vol
                 else: plot_errors=True
            else:
                 logger.warning("Skipping volatility plot: 'per_speaker' data not found in summary stats.")
        except Exception as e: logger.error(f"Failed to generate volatility plot: {e}", exc_info=True); plot_errors=True
        try:
            inten = pltg.plot_intensity_timeline(segments, plots_dir / "intensity.png")
            if inten: plots_manifest["intensity"] = inten
            else: plot_errors=True
        except Exception as e: logger.error(f"Failed to generate intensity plot: {e}", exc_info=True); plot_errors=True


        if plots_manifest:
             report_manifest["plots"] = plots_manifest
             logger.info(f"Generated {len(plots_manifest)} plots in {plots_dir.name}/")
        if plot_errors:
             logger.warning("Some plots failed to generate (see errors above).")


    # ------------------------------------------------------------------
    # 3. Script‑like transcript
    # ------------------------------------------------------------------
    # (Keep existing script logic)
    if cfg.include_script_transcript:
        try:
            txt_path = exp.save_script_transcript(
                segments, artifact_root, artifact_root.name
            )
            if txt_path:
                 report_manifest["script_txt"] = txt_path
                 logger.info(f"Generated script transcript: {txt_path.name}")
        except Exception as e:
            logger.error(f"Failed to generate script transcript: {e}", exc_info=True)


    # ------------------------------------------------------------------
    # 4. Bundle ZIP (if any artifact warrants it) - Now consider snippets dir?
    # ------------------------------------------------------------------
    # Existing logic only considers reports. If snippets should be included,
    # the exp.package_results might need adjustment or snippets should be manually added.
    # For now, keep existing logic.
    # if report_manifest: # Only zip if reports were generated
    #     try:
    #         zip_path = exp.package_results(artifact_root, cfg)
    #         if zip_path:
    #             report_manifest["bundle_zip"] = zip_path
    #             logger.info(f"Created results bundle: {zip_path.name}")
    #     except Exception as e:
    #         logger.error(f"Failed to create results bundle: {e}", exc_info=True)

    # Return both the report manifest and the speaker labeling data
    return report_manifest, speaker_labeling_data
