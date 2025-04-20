# ui/main_gui.py
import os
import traceback
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import gradio as gr
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
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Speech Transcription, Labeling, and Analysis")

            with gr.Tabs() as tabs:

                with gr.TabItem("Single Process", id="single_process_tab"):
                    gr.Markdown(
                        "Upload an audio file or provide a YouTube link to get a transcript with speaker diarization and emotion analysis. "
                        "Optionally, provide known speaker names and short dialogue snippets to improve automatic speaker labeling."
                    )

                    # State variables for Single Process
                    intermediate_json_path_state = gr.State(value=None)
                    job_work_path_state = gr.State(value=None)
                    speaker_snippet_map_state = gr.State(value=None)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Step 1: Process Audio File or URL")
                            with gr.Group():
                                input_file = gr.File(label="Upload Audio File (.wav, .mp3, etc.)", type="filepath")
                                youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                                speaker_snippets_input = gr.Textbox(
                                    label="Known Speakers & Dialogue Snippets (Optional)",
                                    placeholder='Format: Speaker Name: Dialogue snippet\nExample:\nAlice Smith: As I mentioned earlier today...\nBob Johnson: The budget numbers look correct.',
                                    lines=4,
                                    info="Provide one known speaker and a unique short phrase they said per line."
                                )
                                process_btn = gr.Button("1. Process Audio & Apply Snippets ▶️", variant="primary")

                        with gr.Column(scale=2):
                            gr.Markdown("## Step 2: Refine Speaker Labels (Optional)")
                            with gr.Column(visible=False) as relabel_section:
                                gr.Markdown("AI labels applied from snippets (if any). Edit the 'Enter Label Here' column below to correct or add missing labels.")
                                speaker_label_df = gr.DataFrame(
                                    headers=DATAFRAME_HEADERS, datatype=["str", "str", "str"],
                                    row_count=(1,"dynamic"), col_count=(len(DATAFRAME_HEADERS),"fixed"),
                                    interactive=[False, False, True], label="Speaker Labels",
                                )
                                relabel_btn = gr.Button("2. Apply Final Labels & Generate Output ✨", variant="primary")

                            gr.Markdown("## Status & Results")
                            status_output = gr.Textbox(label="Current Status", interactive=False, lines=2)
                            download_output = gr.File(label="Download Results (ZIP)", interactive=False)

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
                                 file_types=[".xlsx"]
                             )
                             batch_process_btn = gr.Button("Start Batch Processing ▶️", variant="primary")

                        with gr.Column():
                             gr.Markdown("## Batch Processing Status & Results")
                             batch_status_output = gr.Textbox(label="Current Batch Status", interactive=False, lines=5)
                             # We might list individual results or provide a combined download later
                             batch_download_output = gr.Textbox(label="Batch Results", interactive=False, lines=5)


            # --- Wrapper Functions for Button Clicks ---

            def parse_speaker_snippets(snippet_text: Optional[str]) -> Dict[str, str]:
                """Parses the multiline text input into a Dict[Name, Snippet]."""
                mapping = {}
                if not snippet_text or not snippet_text.strip():
                    return mapping
                lines = snippet_text.strip().split('\n')
                for line in lines:
                    match = re.match(r"^\s*([^:]+?)\s*:\s*(.+)\s*$", line)
                    if match:
                        name = match.group(1).strip()
                        snippet = match.group(2).strip()
                        if name and snippet:
                            mapping[name] = snippet
                        else:
                            log_warning(f"Could not parse speaker snippet line effectively: '{line}'")
                    else:
                        log_warning(f"Ignoring invalid speaker snippet line format: '{line}'")
                log_info(f"Parsed speaker snippets: {mapping}")
                return mapping

            # Corrected process_audio_wrapper to handle new return type and state
            # Updated type hints for clarity
            def process_audio_wrapper(
                input_file_obj: Optional[str],
                url: Optional[str],
                speaker_snippets_text: Optional[str]
            ) -> Tuple[str, Optional[str], pd.DataFrame, gr.update, Optional[Dict[str, str]], Optional[str], Optional[Path]]:
                """Handles Step 1: Process audio, apply snippet matching, update state and UI for Step 2."""
                input_source = None
                current_status_msg = "Starting audio processing..."
                json_path_update = None
                speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS)
                relabel_visible = False
                download_output_update = None
                work_path_update = None


                speaker_snippet_map = parse_speaker_snippets(speaker_snippets_text)
                snippet_map_state_update = speaker_snippet_map


                # --- Determine input source ---
                if input_file_obj is not None:
                    input_source = input_file_obj
                    current_status_msg = f"Processing uploaded file: {os.path.basename(input_source)}..."
                elif url:
                    input_source = url
                    current_status_msg = f"Processing YouTube URL: {url}..."
                else:
                    current_status_msg = "ERROR: Please provide an audio file or a YouTube URL."
                    yield current_status_msg, json_path_update, speaker_df_update, gr.update(visible=relabel_visible), snippet_map_state_update, download_output_update, work_path_update
                    return

                # Yield initial status and cleared outputs
                yield current_status_msg, json_path_update, speaker_df_update, gr.update(visible=relabel_visible), snippet_map_state_update, download_output_update, work_path_update

                # --- Call the pipeline ---
                if input_source:
                    try:
                        # Correctly unpack the four elements returned by process_audio
                        # path_returned is the JSON path string (or None on error)
                        # status_returned is the status message string
                        # speaker_previews is the list of previews
                        # work_path_returned is the Path object for the job's temp directory
                        path_returned, status_returned, speaker_previews, work_path_returned = self.pipeline.process_audio(
                            input_source,
                            speaker_snippet_map
                        )

                        # Update state variables with the correct returned values
                        json_path_update = path_returned
                        work_path_update = work_path_returned
                        current_status_msg = status_returned

                        if path_returned:
                            speaker_df_update = pd.DataFrame(speaker_previews)
                            relabel_visible = True
                        else:
                            relabel_visible = False

                    except Exception as e:
                        current_status_msg = f"ERROR: An unexpected error occurred during processing: {e}"
                        log_error(f"UI Error during process_audio_wrapper: {e}\n{traceback.format_exc()}")
                        json_path_update = None
                        work_path_update = None
                        speaker_df_update = pd.DataFrame(columns=DATAFRAME_HEADERS)
                        relabel_visible = False
                        download_output_update = None


                yield current_status_msg, json_path_update, speaker_df_update, gr.update(visible=relabel_visible), snippet_map_state_update, download_output_update, work_path_update


            # Modified relabel_finalize_wrapper to accept the job's work path state
            # Added type hints for clarity
            def relabel_finalize_wrapper(
                json_path: Optional[str],
                edited_dataframe: Any,
                work_path: Optional[Path] # Work path from the new state
            ) -> Tuple[str, Optional[str]]:
                """Handles Step 2: Apply final labels from DataFrame and finalize output."""
                status_msg = "Starting relabeling and finalization..."
                final_zip_path = None

                if not json_path or not work_path:
                    status_msg = "ERROR: Processing from Step 1 was not successful or completed. Cannot relabel."
                    if not json_path: log_error("Relabel/Finalize called without structured_json_path.")
                    if not work_path: log_error("Relabel/Finalize called without job_work_path.")
                    yield status_msg, None
                    return

                yield status_msg, None

                try:
                    speaker_mapping_data = None
                    if isinstance(edited_dataframe, pd.DataFrame):
                        speaker_mapping_data = edited_dataframe.to_dict('records')
                    elif isinstance(edited_dataframe, list):
                        speaker_mapping_data = edited_dataframe
                    else:
                        log_warning(f"Unexpected type for edited_dataframe: {type(edited_dataframe)}. Attempting to proceed.")
                        speaker_mapping_data = edited_dataframe

                    # Call the pipeline's second stage, passing the correct json_path, labels, AND work_path
                    final_zip_path, status_msg = self.pipeline.relabel_and_finalize(
                        json_path,
                        speaker_mapping_data,
                        work_path # Pass the job's working directory path
                    )
                except Exception as e:
                    status_msg = f"ERROR: An unexpected error occurred during finalization: {e}"
                    log_error(f"UI Error during relabel_finalize_wrapper: {e}\n{traceback.format_exc()}")
                    final_zip_path = None

                yield status_msg, final_zip_path

            # ADDED: Placeholder wrapper function for batch processing
            def process_batch_wrapper(xlsx_file_obj: Optional[str]) -> Tuple[str, str]:
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
                    batch_status_message, batch_results_summary = self.pipeline.process_batch_xlsx(xlsx_file_obj)

                    batch_status = batch_status_message # Use the status message returned by the pipeline
                    batch_results = batch_results_summary # Use the results summary returned by the pipeline

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

            # Single Process Button Clicks
            process_btn.click(
                fn=process_audio_wrapper,
                inputs=[input_file, youtube_url, speaker_snippets_input],
                outputs=[
                    status_output,
                    intermediate_json_path_state,
                    speaker_label_df,
                    relabel_section,
                    speaker_snippet_map_state,
                    download_output,
                    job_work_path_state
                ]
            )

            relabel_btn.click(
                fn=relabel_finalize_wrapper,
                inputs=[intermediate_json_path_state, speaker_label_df, job_work_path_state],
                outputs=[status_output, download_output]
            )

            # ADDED: Batch Process Button Click
            batch_process_btn.click(
                 fn=process_batch_wrapper,
                 inputs=[batch_input_file],
                 outputs=[batch_status_output, batch_download_output]
            )


        return demo

    def launch(self, **kwargs):
        """Launches the Gradio interface."""
        self.interface_ui().launch(**kwargs)