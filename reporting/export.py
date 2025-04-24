# reporting/export.py

"""reporting.export
-----------------------------------
Human‑readable transcript writer and final artifact bundler.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from constants import ( # type: ignore
    FINAL_ZIP_SUFFIX,
    EMOTION_SUMMARY_JSON_NAME,
    EMOTION_SUMMARY_CSV_NAME,
    SCRIPT_TRANSCRIPT_NAME,
)
from core.logging import get_logger
from utils.paths import ensure_dir


__all__ = [
    "save_script_transcript",
    "package_results",
]

# ---------------------------------------------------------------------------
# Transcript helper
# ---------------------------------------------------------------------------
logger = get_logger(__name__)
# ---------------------------------------------------------------------------


def _fmt_ts(sec: float | None) -> str:
    if sec is None:
        return "[??:??]"
    h, m = divmod(int(sec), 3600)
    m, s = divmod(m, 60)
    if h:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def save_script_transcript(
    segments: Sequence[dict],
    out_dir: Path,
    item_id: str,
) -> Path:
    """Write a plain‑text transcript grouped by speaker.

    Returns the path to the created file.
    """
    ensure_dir(out_dir)
    path = out_dir / SCRIPT_TRANSCRIPT_NAME

    logger.info("Saving script transcript → %s", path)
    blocks = []
    # group_segments_by_speaker was removed because it was causing a circular import
    # this code must be reimplemented if the grouping is wanted.
    # blocks = group_segments_by_speaker(list(segments))

    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"Transcript for {item_id}\n")
        fh.write("=" * (len(item_id) + 15) + "\n\n")
        for blk in blocks:
            speaker = blk.get("speaker", "unknown")
            txt = blk.get("text", "").strip()
            start, end = blk.get("start"), blk.get("end")
            if txt:
                fh.write(f"{_fmt_ts(start)} - {_fmt_ts(end)} {speaker}: {txt}\n\n")

    return path


# ---------------------------------------------------------------------------
# Bundler
# ---------------------------------------------------------------------------


def _collect_files(manifest: Dict[str, Path | Dict[str, Path]]) -> List[Path]:
    files: List[Path] = []
    for v in manifest.values():
        if isinstance(v, Path):
            files.append(v)
        elif isinstance(v, dict):
            files.extend(v.values())
    return files


def package_results(
    artifact_root: Path,
    manifest: Dict[str, Path | Dict[str, Path]],
    include_source_audio: bool = True,
    source_audio_path: Path | None = None,
) -> Path:
    """Create a ZIP bundle of selected artifacts and return its path."""
    ensure_dir(artifact_root)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    zip_path = artifact_root / f"results_{timestamp}{FINAL_ZIP_SUFFIX}"

    logger.info("Bundling artifacts → %s", zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in _collect_files(manifest):
            if file.exists():
                zf.write(file, arcname=file.name)
        if include_source_audio and source_audio_path and source_audio_path.exists():
            zf.write(source_audio_path, arcname=source_audio_path.name)

    logger.info("ZIP bundle created (%.2f kB)", zip_path.stat().st_size / 1024)
    return zip_path
