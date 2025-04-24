# ui/interface.py

"""ui.interface
-----------------------------------
Main Gradio application entry point that defines the UI layout and orchestrates
calls to the processing pipeline and labeling session manager.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
from gradio import themes

from core.config import Config
from core.logging import get_logger
from labeling.session import LabelingSession
from pipeline.manager import run_pipeline
from transmotify_io.converter import convert_to_wav
from utils.paths import ensure_dir, find_unique_filename

logger = get_logger(__name__)

__all__ = ["App"]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

POLL_INTERVAL_S = 1


# --------------------------------------------------------------------------
# Gradio app class
# --------------------------------------------------------------------------


class App:
    """Encapsulates Gradio UI state and interactions."""

    def __init__(self, cfg: Config):
        """Initialise UI state."""
        self.cfg: Config = cfg
        self.batch_results: List[Dict[str, Any]] = []
        self.labeling_session: LabelingSession | None = None
        self.temp_dir: Path = ensure_dir(Path(tempfile.gettempdir()) / "transmotify-ui")
        self.blocks: gr.Blocks | None = None

        # Build interface definition
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
            mode = gr.State("idle")  # idle | processing | labeling | finished | error
            result_manifest = gr.State([])  # list‑of‑dicts returned by pipeline
            item_idx = gr.State(0)  # which item we’re labeling
            speaker_idx = gr.State(0)  # which speaker in that item

            # ----------------------------- first column --------------
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Upload batch Excel (.xlsx)")
                    file_input = gr.File(file_types=[".xlsx"], type="filepath")

                    with gr.Group():  # Group output options
                        gr.Markdown("### Output Options")
                        include_source_audio_checkbox = gr.Checkbox(
                            label="Include Source Audio",
                            value=getattr(
                                self.cfg, "include_source_audio", True
                            ),  # Use getattr for safety
                        )
                        include_json_summary_checkbox = gr.Checkbox(
                            label="Include Granular Emotion Summary JSON",
                            value=getattr(self.cfg, "include_json_summary", True),
                        )
                        include_csv_summary_checkbox = gr.Checkbox(
                            label="Include Overall Emotion Summary CSV",
                            value=getattr(self.cfg, "include_csv_summary", False),
                        )
                        include_script_transcript_checkbox = gr.Checkbox(
                            label="Include Simple Script",
                            value=getattr(self.cfg, "include_script_transcript", False),
                        )
                        include_plots_checkbox = (
                            gr.Checkbox(  # Renamed from include_plots
                                label="Include Plots",
                                value=getattr(self.cfg, "include_plots", False),
                            )
                        )

                    # Interactive labeling toggle (kept separate)
                    label_mode = gr.Checkbox(
                        label="Interactive speaker labeling",
                        value=self.cfg.enable_interactive_labeling,
                    )
                    run_btn = gr.Button("Run analysis ▶", variant="primary")

                # ------------------------- status column --------------
                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Status", lines=10, interactive=False)
                    download_box = gr.Textbox(
                        label="Download", lines=4, interactive=False
                    )

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
                is_labeling = new == "labeling"
                return {
                    mode: new,
                    label_column: gr.update(visible=is_labeling),
                    run_btn: gr.update(visible=not is_labeling),
                    file_input: gr.update(visible=not is_labeling),
                }

            def _status(text: str):
                """Update status box."""
                return {status_box: gr.update(value=text)}

            def _download_link(path: Path | None):
                """Update download box with link if available."""
                if path and path.exists():
                    # Copy to temp dir to ensure Gradio can serve it
                    serve_path = self.temp_dir / path.name
                    shutil.copy(path, serve_path)
                    logger.info("Copied final output to %s for serving", serve_path)
                    return {
                        download_box: gr.update(
                            value=f"[{path.name}](/file={serve_path})", visible=True
                        )
                    }
                return {download_box: gr.update(value="", visible=False)}

            # ------------------------- main pipeline runner ----------

            def do_run(
                file_obj: tempfile.TemporaryFile,
                labeling_enabled: bool,
                # Pass checkbox values
                incl_source_audio: bool,
                incl_json_summary: bool,
                incl_csv_summary: bool,
                incl_script: bool,
                incl_plots: bool,
                progress=gr.Progress(track_tqdm=True),
            ):
                """Entry point triggered by 'Run analysis' button."""
                if not file_obj:
                    return {**_ui_mode("idle"), **_status("Error: No file uploaded.")}

                yield {
                    **_ui_mode("processing"),
                    **_status("Starting batch processing..."),
                    **_download_link(None),  # Clear download link
                }

                xlsx_path = Path(file_obj.name)
                try:
                    # Override config values based on checkboxes
                    # We modify a *copy* so the original cfg isn't changed
                    run_cfg = self.cfg.model_copy(
                        update={
                            "enable_interactive_labeling": labeling_enabled,
                            "include_source_audio": incl_source_audio,
                            "include_json_summary": incl_json_summary,
                            "include_csv_summary": incl_csv_summary,
                            "include_script_transcript": incl_script,
                            "include_plots": incl_plots,
                        }
                    )

                    logger.info(
                        "Run config overrides: %s",
                        run_cfg.model_dump(exclude={"hf_token", "openai_api_key"}),
                    )

                    # Read sources from Excel file
                    try:
                        df = pd.read_excel(xlsx_path)
                        sources = df.iloc[
                            :, 0
                        ].tolist()  # Assuming sources are in the first column
                        logger.info(f"Read {len(sources)} sources from {xlsx_path}")
                    except Exception as e:
                        logger.error(f"Error reading Excel file {xlsx_path}: {e}")
                        yield {
                            **_status(f"Error reading Excel file: {e}"),
                            **_ui_mode("error"),
                        }
                        return  # Stop processing on error

                    # Run the main pipeline
                    self.batch_results = run_pipeline(sources=sources, cfg=run_cfg)

                    # If labeling enabled, prepare session
                    if labeling_enabled:
                        self.labeling_session = LabelingSession(
                            self.batch_results, cfg=run_cfg
                        )
                        self.labeling_session.start()
                        yield _ui_mode("labeling")
                        yield from do_labeling_update(0, 0)  # Show first speaker
                    else:
                        # Package non-labeling results
                        final_zip = self._package_results(self.batch_results, run_cfg)
                        yield {
                            **_ui_mode("finished"),
                            **_status(f"Analysis complete. Results packaged."),
                            **_download_link(final_zip),
                        }

                except Exception as e:
                    logger.error("Pipeline failure", exc_info=True)
                    yield {
                        **_ui_mode("error"),
                        **_status(f"Error: {e}\n{traceback.format_exc()}"),
                        **_download_link(None),
                    }

            # ------------------------- labeling interactions ---------

            def do_labeling_update(
                current_item_idx: int, current_speaker_idx: int
            ) -> Dict[str, Any]:
                """Update labeling UI elements for the current speaker."""
                if not self.labeling_session:
                    return _status("Error: Labeling session not initialised.")

                item_info, speaker_info, progress = self.labeling_session.get_state(
                    current_item_idx, current_speaker_idx
                )

                progress_str = (
                    f"Item {progress['item_num']}/{progress['total_items']} "
                    f"(Speaker {progress['speaker_num']}/{progress['total_speakers']})"
                )

                # Prepare audio player HTML
                audio_path_str = str(speaker_info["audio_snippet_path"])
                # Convert to MP3 for browser compatibility if needed
                mp3_path = self.temp_dir / f"{Path(audio_path_str).stem}.mp3"

                try:
                    # Only convert if MP3 doesn't exist or source is newer
                    if (
                        not mp3_path.exists()
                        or Path(audio_path_str).stat().st_mtime
                        > mp3_path.stat().st_mtime
                    ):
                        # Ensure converter uses a config where output_dir is temp_dir
                        conv_cfg = self.cfg.model_copy(
                            update={"output_dir": self.temp_dir}
                        )
                        converted = convert_to_wav(
                            Path(audio_path_str),
                            dst_format="mp3",
                            cfg=conv_cfg,
                            output_dir=self.temp_dir,
                        )
                        # Note: convert_audio_format might place it directly in temp_dir or a sub-folder
                        # Assuming it returns the correct path or places it predictably
                        if converted and converted.exists():
                            mp3_path = converted  # Use the path returned by converter
                        elif (
                            self.temp_dir / f"{Path(audio_path_str).stem}.mp3"
                        ).exists():
                            mp3_path = (
                                self.temp_dir / f"{Path(audio_path_str).stem}.mp3"
                            )
                        else:
                            logger.warning(
                                "MP3 conversion failed or file not found for %s",
                                audio_path_str,
                            )
                            # Fallback? Serve original? For now, log warning.

                except Exception as conv_err:
                    logger.error(
                        "Error converting snippet %s to MP3: %s",
                        audio_path_str,
                        conv_err,
                    )
                    # Fallback or indicate error in UI?

                # Use file= for Gradio server; depends on where conversion saves it
                audio_url = f"/file={mp3_path}" if mp3_path.exists() else ""

                html_content = f"""
                <h4>{item_info["name"]} - Speaker {speaker_info["id"]}</h4>
                <p>Saying: <i>"{speaker_info["example_transcript"]}"</i></p>
                """
                if audio_url:
                    html_content += f"""
                     <audio controls preload="auto">
                         <source src="{audio_url}" type="audio/mpeg">
                         Your browser does not support the audio element.
                     </audio>
                     """
                else:
                    html_content += "<p><i>Audio preview unavailable.</i></p>"

                return {
                    item_idx: current_item_idx,
                    speaker_idx: current_speaker_idx,
                    progress_md: gr.update(value=progress_str),
                    video_html: gr.update(value=html_content),
                    label_input: gr.update(value=speaker_info.get("label", "")),
                    nav_prev: gr.update(
                        interactive=progress["item_num"] > 1
                        or progress["speaker_num"] > 1
                    ),
                    nav_next: gr.update(
                        interactive=progress["item_num"] < progress["total_items"]
                        or progress["speaker_num"] < progress["total_speakers"]
                    ),
                }

            def do_labeling_nav(
                current_item_idx: int, current_speaker_idx: int, direction: str
            ) -> Dict[str, Any]:
                """Handle Prev/Next button clicks."""
                if not self.labeling_session:
                    return {}
                new_item_idx, new_speaker_idx = self.labeling_session.navigate(
                    current_item_idx, current_speaker_idx, direction
                )
                # Use yield to update state *before* yielding UI updates
                yield {item_idx: new_item_idx, speaker_idx: new_speaker_idx}
                yield from do_labeling_update(new_item_idx, new_speaker_idx)

            def do_labeling_submit(
                current_item_idx: int, current_speaker_idx: int, label_text: str
            ) -> Dict[str, Any]:
                """Handle 'Store label' button click."""
                if not self.labeling_session:
                    return _status("Error: Labeling session not active.")
                self.labeling_session.store_label(
                    current_item_idx, current_speaker_idx, label_text
                )

                # Check if finished
                if self.labeling_session.is_complete():
                    final_zip = self._package_results(
                        self.labeling_session.get_labeled_results(), self.cfg
                    )
                    # Yield intermediate status before final UI mode change
                    yield _status("Labeling complete. Packaging results...")
                    yield {
                        **_ui_mode("finished"),
                        **_status("Labeling complete. Results packaged."),
                        **_download_link(final_zip),
                    }
                else:
                    # Auto-advance to next speaker
                    yield from do_labeling_nav(
                        current_item_idx, current_speaker_idx, "next"
                    )

            # ------------------------- wiring ------------------------
            run_btn.click(
                fn=do_run,
                inputs=[
                    file_input,
                    label_mode,
                    include_source_audio_checkbox,
                    include_json_summary_checkbox,
                    include_csv_summary_checkbox,
                    include_script_transcript_checkbox,
                    include_plots_checkbox,
                ],
                outputs=[
                    mode,
                    status_box,
                    label_column,
                    run_btn,
                    file_input,
                    download_box,
                    # Labeling UI outputs (for initial population if labeling starts)
                    item_idx,
                    speaker_idx,
                    progress_md,
                    video_html,
                    label_input,
                    nav_prev,
                    nav_next,
                ],
            )

            nav_prev.click(
                fn=lambda iidx, sidx: do_labeling_nav(iidx, sidx, "prev"),
                inputs=[item_idx, speaker_idx],
                outputs=[
                    item_idx,
                    speaker_idx,
                    progress_md,
                    video_html,
                    label_input,
                    nav_prev,
                    nav_next,
                ],
                # queuing=False # Faster updates for navigation?
            )
            nav_next.click(
                fn=lambda iidx, sidx: do_labeling_nav(iidx, sidx, "next"),
                inputs=[item_idx, speaker_idx],
                outputs=[
                    item_idx,
                    speaker_idx,
                    progress_md,
                    video_html,
                    label_input,
                    nav_prev,
                    nav_next,
                ],
                # queuing=False
            )
            submit_btn.click(
                fn=do_labeling_submit,
                inputs=[item_idx, speaker_idx, label_input],
                outputs=[
                    mode,
                    status_box,
                    label_column,
                    run_btn,
                    file_input,
                    download_box,
                    # Labeling UI outputs (for next speaker or finished state)
                    item_idx,
                    speaker_idx,
                    progress_md,
                    video_html,
                    label_input,
                    nav_prev,
                    nav_next,
                ],
            )

        self.blocks = demo

    def launch(self, *args, **kwargs):
        """Launch the Gradio interface."""
        if not self.blocks:
            self._build_interface()

        if self.blocks:
            self.blocks.queue()  # Enable queuing for handling multiple requests
            self.blocks.launch(*args, **kwargs)
        else:
            logger.error("Failed to build Gradio interface.")

    # --------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------

    def _package_results(
        self, results: List[Dict[str, Any]], cfg: Config
    ) -> Path | None:
        """Create final ZIP archive from pipeline results."""
        if not results:
            logger.warning("No results to package.")
            return None

        # Determine base output dir from first result (assuming consistency)
        first_output_dir = Path(results[0].get("output_directory", "."))
        package_dir = first_output_dir.parent
        zip_name_base = f"{package_dir.name}_outputs"

        # Create ZIP file
        zip_filename = find_unique_filename(package_dir, f"{zip_name_base}.zip")
        zip_path = package_dir / zip_filename

        logger.info("Packaging results into %s", zip_path)

        try:
            shutil.make_archive(
                base_name=str(
                    zip_path.with_suffix("")
                ),  # make_archive wants path without extension
                format="zip",
                root_dir=package_dir,
                base_dir=first_output_dir.name,  # Zip only the specific output folder(s)
                logger=logger,
            )

            # Optionally, clean up individual result folders after zipping?
            # if cfg.cleanup_intermediate:
            #     for item in results:
            #         item_dir = Path(item.get("output_directory"))
            #         if item_dir.exists() and item_dir.is_relative_to(package_dir):
            #              logger.info("Cleaning up intermediate directory: %s", item_dir)
            #              shutil.rmtree(item_dir)

            logger.info("Successfully created package: %s", zip_path)
            return zip_path

        except Exception as zip_err:
            logger.error("Failed to create results package: %s", zip_err, exc_info=True)
            return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config("config.json")  # Load config from default path
    app = App(cfg=config)
    logger.info("Launching Gradio UI …")
    app.launch(server_name="0.0.0.0", share=False)  # Run on local network
