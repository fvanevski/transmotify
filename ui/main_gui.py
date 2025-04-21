# ui/main_gui.py
import os
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator

import gradio as gr
from gradio import themes
import pandas as pd

from core.logging import log_error, log_warning, log_info

# Import Pipeline - we will add a batch processing method to this later
from core.pipeline import Pipeline, SpeakerPreviewsList


DATAFRAME_HEADERS = ["Speaker ID", "Dialogue Preview", "Enter Label Here"]


class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        with gr.Blocks(theme=themes.Soft()) as demo:
            gr.Markdown("# Speech Transcription, Labeling, and Analysis")

            with gr.Tabs() as tabs:
                with gr.TabItem("Batch Process", id="batch_process_tab"):
                    gr.Markdown(
                        "Upload an Excel file (.xlsx) containing YouTube links and optional speaker:dialogue snippet pairs for batch processing. "
                        "Speaker labels will be assigned based on snippet matching only. Unmatched speakers will retain their original IDs."
                    )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Batch Processing Input")
                            batch_input_file = gr.File(
                                label="Upload Excel File (.xlsx)",
                                type="filepath",
                                file_types=[".xlsx"],
                            )
                            batch_process_btn = gr.Button(
                                "Start Batch Processing ▶️", variant="primary"
                            )

                        with gr.Column():
                            gr.Markdown("## Batch Processing Status & Results")
                            batch_status_output = gr.Textbox(
                                label="Current Batch Status", interactive=False, lines=5
                            )
                            # We might list individual results or provide a combined download later
                            batch_download_output = gr.Textbox(
                                label="Batch Results", interactive=False, lines=5
                            )

                with gr.TabItem("Speaker Labeling", id="speaker_labeling_tab"):
                    gr.Markdown(
                        "Upload a structured transcript JSON file to manually label speakers. "
                        "Download the relabeled JSON and other analysis outputs."
                    )

            # --- Wrapper Functions for Button Clicks ---

            def parse_speaker_snippets(snippet_text: Optional[str]) -> Dict[str, str]:
                """Parses the multiline text input into a Dict[Name, Snippet]."""
                mapping = {}
                if not snippet_text or not snippet_text.strip():
                    return mapping
                lines = snippet_text.strip().split("\n")
                for line in lines:
                    match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
                    if match:
                        name = match.group(1).strip()
                        snippet = match.group(2).strip()
                        if name and snippet:
                            mapping[name] = snippet
                        else:
                            log_warning(
                                f"Could not parse speaker snippet line effectively: '{line}'"
                            )
                    else:
                        log_warning(
                            f"Ignoring invalid speaker snippet line format: '{line}'"
                        )
                log_info(f"Parsed speaker snippets: {mapping}")
                return mapping

            # ADDED: Placeholder wrapper function for batch processing
            def process_batch_wrapper(
                xlsx_file_obj: Optional[str],
            ) -> Generator[Tuple[str, str], Any, Any]:
                """Handles the batch processing from an Excel file."""
                batch_status = "Starting batch processing..."
                batch_results = ""

                if xlsx_file_obj is None:
                    batch_status = "ERROR: Please upload an Excel file."
                    yield batch_status, batch_results
                    return

                try:
                    # --- Call the actual pipeline's batch processing method ---
                    # This will read the xlsx, process each item, and generate results
                    batch_status_message, batch_results_summary = (
                        self.pipeline.process_batch_xlsx(xlsx_file_obj)
                    )

                    batch_status = batch_status_message  # Use the status message returned by the pipeline
                    batch_results = batch_results_summary  # Use the results summary returned by the pipeline

                except FileNotFoundError as e:
                    batch_status = f"ERROR: File not found: {e}"
                    log_error(f"Batch processing error: {e}")
                except pd.errors.EmptyDataError:
                    batch_status = "ERROR: The uploaded Excel file is empty."
                    log_error("Batch processing error: Empty Excel file.")
                except Exception as e:
                    batch_status = f"ERROR: An unexpected error occurred during batch processing: {e}"
                    log_error(f"Batch processing error: {e}\n{traceback.format_exc()}")

                yield batch_status, batch_results

            # --- Button Clicks (linking UI events to wrapper functions) ---

            # ADDED: Batch Process Button Click
            batch_process_btn.click(
                fn=process_batch_wrapper,
                inputs=[batch_input_file],
                outputs=[batch_status_output, batch_download_output],
            )

        return demo

    def launch(self, **kwargs):
        """Launches the Gradio interface."""
        self.interface_ui().launch(**kwargs)
