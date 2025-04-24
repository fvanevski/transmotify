# reporting/generator.py

"""reporting.generator
--------------------------------------
Generate all reporting artifacts for a single analysis run.

The function `generate_all` is designed to be called by the pipeline manager
once the segment list has been fully annotated (fused emotions, etc.).  It
creates the artifact directory if needed, delegates work to specialised
modules, and returns a manifest mapping logical names to output paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from core.config import Config
from core.logging import get_logger
from constants import (
    EMOTION_SUMMARY_JSON_NAME, EMOTION_SUMMARY_CSV_NAME
)
from reporting import summaries as summ
from reporting import plotting as pltg
from reporting import export as exp

logger = get_logger(__name__)

__all__ = ["generate_all"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all(
    segments: List[dict],
    cfg: Config,
    artifact_root: Path,
) -> Dict[str, Path]:
    """Create all reporting artifacts and return a manifest.

    Parameters
    ----------
    segments
        The list of segments with fused emotions already populated.
    cfg
        Validated runtime configuration.
    artifact_root
        Directory where artifacts should be written.

    Returns
    -------
    dict
        A mapping from logical artifact keys (``summary_json``, ``plots`` …)
        to their respective file paths. Keys absent from the dict were not
        generated, e.g. because the corresponding *include_* flag was ``False``.
    """

    artifact_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Path | Dict[str, Path]] = {}

    # ------------------------------------------------------------------
    # 1. Summaries (JSON / CSV)
    # ------------------------------------------------------------------
    stats = summ.build_summary(segments)

    if cfg.include_json_summary:
        json_path = artifact_root / EMOTION_SUMMARY_JSON_NAME
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON summary → %s", json_path.name)
        manifest["summary_json"] = json_path

    if cfg.include_csv_summary:
        try:
            csv_path = artifact_root / EMOTION_SUMMARY_CSV_NAME
            summ.save_csv(stats, csv_path)
            logger.info("Saved CSV summary → %s", csv_path.name)
            manifest.setdefault("summary_csv", csv_path)
        except ImportError:
            logger.warning("pandas not installed – skipping CSV summary export")

    # ------------------------------------------------------------------
    # 2. Plots
    # ------------------------------------------------------------------
    if cfg.include_plots:
        plots_dir = artifact_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        plots_manifest: Dict[str, Path] = {}

        traj = pltg.plot_trajectory(segments, plots_dir / "trajectory.png")
        plots_manifest["trajectory"] = traj
        dist = pltg.plot_distribution(segments, plots_dir / "distribution.png")
        plots_manifest["distribution"] = dist
        vol = pltg.plot_volatility(stats["per_speaker"], plots_dir / "volatility.png")
        plots_manifest["volatility"] = vol
        inten = pltg.plot_intensity_timeline(segments, plots_dir / "intensity.png")
        plots_manifest["intensity"] = inten

        manifest["plots"] = plots_manifest

    # ------------------------------------------------------------------
    # 3. Script‑like transcript
    # ------------------------------------------------------------------
    if cfg.include_script_transcript:
        txt_path = exp.save_script_transcript(
            segments, artifact_root, artifact_root.name
        )
        manifest["script_txt"] = txt_path

    # ------------------------------------------------------------------
    # 4. Bundle ZIP (if any artifact warrants it)
    # ------------------------------------------------------------------
    if (
        cfg.include_json_summary
        or cfg.include_csv_summary
        or cfg.include_plots
        or cfg.include_script_transcript
    ):
        zip_path = exp.package_results(artifact_root, cfg)
        manifest["bundle_zip"] = zip_path

    return manifest
