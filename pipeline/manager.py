 """speech_analysis.pipeline.manager
------------------------------------
Stateless, linear orchestrator that wires together the core sub‑packages into
a single *run_pipeline* function.  It deliberately avoids UI concerns, Excel
reading, and interactive state – those live in the labeling layer and the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

from speech_analysis.core.config import Config
from speech_analysis.core.logging import get_logger

from speech_analysis.io.downloader import download_youtube
from speech_analysis.io.converter import convert_to_wav
from speech_analysis.transcription.whisperx_wrapper import transcribe
from speech_analysis.transcription.segments import load_segments
from speech_analysis.emotion.analyzer import MultimodalAnalyzer
from speech_analysis.reporting.generator import generate_all
from speech_analysis.labeling import selector as labeling_selector

logger = get_logger(__name__)

__all__ = ["run_pipeline"]


# ---------------------------------------------------------------------------
# Public entry‑point
# ---------------------------------------------------------------------------

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
        A :class:`speech_analysis.core.config.Config` instance.
    out_root
        Folder in which per‑item sub‑directories will be created.  Defaults to
        ``cfg.output_dir``.
    interactive
        Whether to compute *eligible speakers* for later interactive labeling.
        If *None* (default) the value is read from ``cfg.enable_interactive_labeling``.

    Returns
    -------
    List[Dict[str, Any]]
        One manifest dictionary per input source, containing at minimum:

        * ``source`` – the original input string
        * ``artifact_dir`` – the directory created for this item
        * ``report_manifest`` – return value of
          :func:`speech_analysis.reporting.generator.generate_all`
        * ``eligible_speakers`` – list of IDs (present only when *interactive*)
    """
    out_root = Path(out_root or cfg.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    interactive = cfg.enable_interactive_labeling if interactive is None else interactive
    analyzer = MultimodalAnalyzer(cfg)

    manifests: List[Dict[str, Any]] = []

    for idx, src in enumerate(sources, start=1):
        item_id = f"item_{idx:03d}"
        item_dir = out_root / item_id
        item_dir.mkdir(exist_ok=True)

        logger.info(f"[{item_id}] Processing {src}")

        # 1) Acquire / convert audio -------------------------------------------------
        if str(src).startswith(("http://", "https://")):
            wav_path, meta = download_youtube(str(src), cfg, tmp=item_dir / "tmp")
        else:
            wav_path = convert_to_wav(Path(src), dst=item_dir / Path(src).name, cfg=cfg)
            meta = {"source_path": str(src), "source_type": "local_file"}

        # 2) ASR + diarization ------------------------------------------------------
        json_path = transcribe(wav_path, cfg, out_dir=item_dir / "whisperx")
        segments = load_segments(json_path)

        # 3) Multimodal emotion -----------------------------------------------------
        segments = analyzer.run(segments, audio_path=wav_path, video_path="")

        # 4) Interactive speaker eligibility (optional) ----------------------------
        eligible: List[str] = []
        if interactive:
            eligible = labeling_selector.identify_eligible_speakers(
                segments,
                cfg.speaker_labeling_min_total_time,
                cfg.speaker_labeling_min_block_time,
            )

        # 5) Reporting & artifact generation ---------------------------------------
        report_manifest = generate_all(segments, cfg, artifact_root=item_dir)

        manifests.append(
            {
                "source": src,
                "artifact_dir": item_dir,
                "report_manifest": report_manifest,
                "eligible_speakers": eligible,
                "metadata": meta,
            }
        )

    return manifests
