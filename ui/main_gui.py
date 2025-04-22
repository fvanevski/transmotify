# ui/main_gui.py
import os
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator  # Import Generator

import gradio as gr
from gradio import themes  # Import themes explicitly
import pandas as pd

from core.logging import log_error, log_warning, log_info

# Import Pipeline - we will add a batch processing method to this later
from core.pipeline import Pipeline


# Removed DATAFRAME_HEADERS as it's only used by the single-process tab


class UI:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline(config)

    def interface_ui(self):
        # Changed theme to Default and used the imported themes
        with gr.Blocks(theme=themes.Default()) as demo:
            gr.Markdown(
                "# Batch Processing for Speech Transcription, Labeling, and Analysis"
            )

            gr.Markdown(
                "Upload an Excel file (.xlsx) containing YouTube URLs and optional speaker:dialogue snippet pairs for batch processing. "
                "The Excel file should have columns for 'URL' and optionally 'Speaker Snippets'. "
                "Speaker labels will be assigned based on snippet matching provided in the Excel file. "
                "Future updates will include video preview labeling, with snippet matching now driven by the Excel input."
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

            # --- Wrapper Functions for Button Clicks ---

            # Removed parse_speaker_snippets as it's now handled in core/utils

            # Removed process_audio_wrapper and relabel_finalize_wrapper

            # ADDED: Placeholder wrapper function for batch processing
            # Updated type hint to Generator
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

            # Removed Single Process Button Clicks

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
