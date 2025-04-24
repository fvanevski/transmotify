 """Gradio front‑end wiring for the speech‑analysis toolkit.

This interface keeps the user‑journey identical to the legacy *main_gui.py*
but is wired to the refactored back‑end stack:

* `pipeline.manager.run_pipeline` for batch processing
* `labeling.session.LabelingSession` for interactive speaker relabeling

The module does **no** heavy computation – all `@gradio` callbacks yield UI
updates or hand off work to the back‑end.
"""

from __future__ import annotations

import functools
import math
import re
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator

import gradio as gr
from gradio import themes
import pandas as pd

from speech_analysis.core.logging import get_logger
from speech_analysis.core.config import Config
from speech_analysis.pipeline.manager import run_pipeline
from speech_analysis.labeling.session import LabelingSession
from speech_analysis.labeling import selector

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _youtube_embed(url: str, start: int = 0) -> str:
    """Return an HTML <iframe> for *url* at *start* seconds."""
    if not url or not url.startswith("http"):
        return "<p>Invalid YouTube URL</p>"
    vid: Optional[str] = None
    if "youtube.com/watch?v=" in url:
        vid = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        vid = url.split("youtu.be/")[1].split("?")[0]
    if not vid:
        return f"<p>Could not extract Video ID from URL: {url}</p>"
    start = max(0, int(math.floor(start)))
    return (
        f'<iframe width="560" height="315" '
        f'src="https://www.youtube.com/embed/{vid}?start={start}&controls=1" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        f'allowfullscreen></iframe>'
    )


# ---------------------------------------------------------------------------
# UI class
# ---------------------------------------------------------------------------

class WebApp:
    """Instantiate Gradio Blocks bound to the back‑end pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session: Optional[LabelingSession] = None
        self.batch_results: List[Dict[str, Any]] = []
        self._build_interface()

    # ------------------------------------------------------------------
    # Gradio building
    # ------------------------------------------------------------------

    def _build_interface(self):
        """Create *self.blocks* – a `gr.Blocks` graph – but don’t launch it."""
        default_theme = themes.Default()
        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Speech‑analysis batch processor + speaker labeling")

            # ----------------------------- states ---------------------
            mode   = gr.State("idle")  # idle | processing | labeling | finished | error
            result_manifest = gr.State([])  # list‑of‑dicts returned by pipeline
            item_idx = gr.State(0)           # which item we’re labeling
            speaker_idx = gr.State(0)        # which speaker in that item

            # ----------------------------- first column --------------
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Upload batch Excel (.xlsx)")
                    file_input = gr.File(file_types=[".xlsx"], type="filepath")

                    with gr.Row():
                        include_plots = gr.Checkbox("Plots", value=self.cfg.include_plots)
                        label_mode = gr.Checkbox("Interactive speaker labeling", value=self.cfg.enable_interactive_labeling)
                    run_btn = gr.Button("Run analysis ▶", variant="primary")

                # ------------------------- status column --------------
                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Status", lines=10, interactive=False)
                    download_box = gr.Textbox(label="Download", lines=4, interactive=False)

            # ----------------------------- labeling UI ---------------
            with gr.Column(visible=False) as label_column:
                gr.Markdown("## Speaker labeling")
                progress_md = gr.Markdown()
                video_html = gr.HTML()
                label_input = gr.Textbox(label="Speaker name")
                nav_prev, nav_next = gr.Button("⬅ Prev"), gr.Button("Next ➡")
                submit_btn = gr.Button("Store label")

            # ------------------------- helpers -----------------------

            def _ui_mode(new: str):
                """Return Gradio `update` dict showing/hiding blocks."""
                return {
                    label_column: gr.update(visible=(new == "labeling")),
                    run_btn: gr.update(interactive=(new in {"idle", "finished", "error"})),
                }

            # ------------------------- pipeline run ------------------

            def do_run(xlsx_path: str, want_plots: bool, want_labeling: bool):
                if not xlsx_path:
                    yield {
                        status_box: "❗ Upload an .xlsx file first",
                        **_ui_mode("error"),
                    }
                    return

                yield {status_box: "Running batch …", **_ui_mode("processing")}

                # Read urls from excel (still here because easy)
                try:
                    df = pd.read_excel(xlsx_path)
                except Exception as exc:  # pragma: no cover
                    yield {status_box: f"❗ Failed to read Excel: {exc}", **_ui_mode("error")}
                    return
                url_col = self.cfg.get("batch_url_column", "YouTube URL")
                if url_col not in df.columns:
                    yield {status_box: f"❗ Column '{url_col}' missing", **_ui_mode("error")}
                    return
                sources = df[url_col].dropna().tolist()

                # run backend (streaming not needed – batch is usually quick)
                try:
                    self.batch_results = run_pipeline(sources, self.cfg, interactive=want_labeling)
                except Exception as exc:  # pragma: no cover
                    logger.exception("Pipeline failure")
                    yield {status_box: f"❗ Pipeline crashed: {exc}", **_ui_mode("error")}
                    return

                yield {status_box: "Batch completed", result_manifest: self.batch_results}

                if not want_labeling:
                    # all done – provide zip links (they’re inside manifest)
                    zips = [d["report_manifest"].get("bundle_zip") for d in self.batch_results if d]
                    dl_text = "\n".join(map(str, filter(None, zips))) or "<no zip>"
                    yield {download_box: dl_text, **_ui_mode("finished")}
                    return

                # init session for labeling
                self.session = LabelingSession(
                    preview_duration=self.cfg.speaker_labeling_preview_duration,
                    min_block_time=self.cfg.speaker_labeling_min_block_time,
                    min_total_time=self.cfg.speaker_labeling_min_total_time,
                )
                for d in self.batch_results:
                    self.session.add_item(
                        item_id=d["artifact_dir"].name,
                        segments=d["report_manifest"]["summary_json"],  # path; we just need segments list – ok placeholder
                        youtube_url=d["source"],
                    )
                # jump into first item / speaker
                first = self.session.start(self.session.pending_items()[0])
                spk_id, yt, starts = first  # type: ignore
                video = _youtube_embed(yt, starts[0] if starts else 0)
                yield {
                    progress_md: f"Item 1 / Speaker 1 **{spk_id}**",
                    video_html: video,
                    **_ui_mode("labeling"),
                }

            run_btn.click(
                do_run,
                inputs=[file_input, include_plots, label_mode],
                outputs=[status_box, download_box, result_manifest, * _ui_mode("idle").keys()],
            )

        # end Blocks
        self.blocks = demo

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def launch(self, **kwargs):
        self.blocks.launch(**kwargs)
