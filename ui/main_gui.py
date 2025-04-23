# ui/main_gui.py
import os
import traceback
import re
from pathlib import Path
from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
    List,
    Generator,
    Union,
)  # Added Generator, Union

import gradio as gr
from gradio import themes
import pandas as pd

# Assuming Pipeline class is accessible via from core.pipeline import Pipeline
# Make sure your project structure allows this import
try:
    from core.pipeline import Pipeline
    from core.logging import (
        log_error,
        log_warning,
        log_info,
    )  # Import logging functions
except ImportError:
    print(
        "ERROR: Failed to import core modules. Ensure main.py is run from the project root."
    )

    # Define dummy classes/functions if import fails to allow Gradio to load basic UI
    class Pipeline:
        def __init__(self, config):
            pass

        def process_batch_xlsx(self, *args, **kwargs):
            return "Error: Pipeline not loaded.", "", None

        def start_interactive_labeling_for_item(self, *args, **kwargs):
            return None

        def store_speaker_label(self, *args, **kwargs):
            return False

        def get_next_speaker_for_labeling(self, *args, **kwargs):
            return None

        def finalize_labeled_item(self, *args, **kwargs):
            return None

        def check_batch_completion_and_zip(self, *args, **kwargs):
            return None

    def log_error(msg):
        print(f"LOG_ERROR: {msg}")

    def log_warning(msg):
        print(f"LOG_WARNING: {msg}")

    def log_info(msg):
        print(f"LOG_INFO: {msg}")


class UI:
    def __init__(self, config):
        # Store config dictionary directly
        self.config_data = config.config if hasattr(config, "config") else config
        # Instantiate the pipeline here
        self.pipeline = Pipeline(self.config_data)
        log_info("UI Initialized with Pipeline instance.")

    def interface_ui(self):
        # Theme definition
        default_theme = themes.Default()

        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Batch Processing & Interactive Speaker Labeling")

            # --- UI Mode State ---
            # Controls visibility of different UI sections
            # Modes: 'idle', 'processing', 'labeling', 'finished', 'error'
            ui_mode_state = gr.State("idle")

            # --- Batch Processing State ---
            batch_job_id_state = gr.State(
                None
            )  # Stores the current batch job ID string
            # List of item identifiers (e.g., ['item_001', 'item_002']) needing labeling
            items_to_label_state = gr.State([])
            current_item_index_state = gr.State(0)  # Index into items_to_label_state

            # --- Speaker Labeling State (for the current item) ---
            # List of speaker IDs (e.g., ['SPEAKER_00', 'SPEAKER_03']) needing labeling for the current item
            eligible_speakers_state = gr.State([])
            # Index within eligible_speakers_state for the speaker currently displayed
            current_speaker_index_state = gr.State(0)
            # List of file paths (strings) for the current speaker's video clips
            current_clip_paths_state = gr.State([])
            # Index of the video clip currently displayed in the gr.Video component
            current_clip_index_state = gr.State(0)
            # Dictionary storing {'SPEAKER_XX': 'UserLabel'} for the current item
            collected_labels_state = gr.State({})

            # --- Main Layout ---
            with gr.Row():
                # --- Input Column ---
                with gr.Column(scale=1):
                    with gr.Group() as batch_input_group:
                        gr.Markdown("## Batch Processing Input")
                        gr.Markdown(
                            "Upload an Excel file (.xlsx) containing YouTube URLs or local file paths "
                            "in a column named 'YouTube URL' (configurable). "
                            "Optionally enable interactive speaker labeling below."
                        )
                        batch_input_file = gr.File(
                            label="Upload Excel File (.xlsx)",
                            type="filepath",
                            file_types=[".xlsx"],
                        )
                        gr.Markdown("### Output Options")
                        include_source_audio_checkbox = gr.Checkbox(
                            label="Include Source Audio",
                            value=self.config_data.get("include_source_audio", True),
                        )
                        include_json_summary_checkbox = gr.Checkbox(
                            label="Include Granular Emotion Summary JSON",
                            value=self.config_data.get("include_json_summary", True),
                        )
                        include_csv_summary_checkbox = gr.Checkbox(
                            label="Include Overall Emotion Summary CSV",
                            value=self.config_data.get("include_csv_summary", False),
                        )
                        include_script_transcript_checkbox = gr.Checkbox(
                            label="Include Simple Script",
                            value=self.config_data.get(
                                "include_script_transcript", False
                            ),
                        )
                        include_plots_checkbox = gr.Checkbox(
                            label="Include Plots",
                            value=self.config_data.get("include_plots", False),
                        )

                        batch_process_btn = gr.Button(
                            "Start Batch Processing ▶️", variant="primary"
                        )

                # --- Status/Results Column ---
                with gr.Column(scale=1):
                    with gr.Group() as status_output_group:
                        gr.Markdown("## Processing Status & Results")
                        batch_status_output = gr.Textbox(
                            label="Current Status", interactive=False, lines=10
                        )
                        batch_download_output = gr.Textbox(
                            label="Final Output",
                            interactive=False,
                            lines=5,
                            placeholder="Final ZIP path will appear here...",
                        )

            # --- Interactive Speaker Labeling UI (Initially Hidden) ---
            with gr.Column(visible=False) as labeling_ui_group:  # Start hidden
                gr.Markdown("## Interactive Speaker Labeling")
                labeling_progress_md = gr.Markdown(
                    "Labeling Speaker: ---"
                )  # Placeholder
                with gr.Row():
                    with gr.Column(scale=2):
                        video_player = gr.Video(
                            label="Speaker Preview Clip", interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Clip Navigation")
                        current_clip_display = gr.Markdown("Clip 1 of X")  # Placeholder
                        prev_clip_btn = gr.Button("⬅️ Previous Clip")
                        next_clip_btn = gr.Button("Next Clip ➡️")
                        gr.Markdown("### Enter Label")
                        speaker_label_input = gr.Textbox(
                            label="Enter Speaker Name/Label",
                            placeholder="e.g., Alice (or leave blank)",
                        )
                        submit_label_btn = gr.Button(
                            "Submit Label & Next Speaker ▶️", variant="primary"
                        )

            # --- Helper Functions for UI Logic ---

            def change_ui_mode(mode):
                """Updates visibility of UI groups based on the mode."""
                log_info(f"Changing UI mode to: {mode}")
                is_labeling = mode == "labeling"
                is_idle_or_finished = (
                    mode == "idle" or mode == "finished" or mode == "error"
                )
                return {
                    labeling_ui_group: gr.update(visible=is_labeling),
                    batch_input_group: gr.update(visible=is_idle_or_finished),
                    status_output_group: gr.update(visible=True),  # Always visible
                    # Disable batch start button while processing/labeling
                    batch_process_btn: gr.update(interactive=is_idle_or_finished),
                }

            # --- Main Batch Processing Wrapper ---
            def process_batch_wrapper(
                xlsx_file_obj: Optional[str],  # File path
                include_audio: bool,
                include_json: bool,
                include_csv: bool,
                include_script: bool,
                include_plots: bool,
            ) -> Generator[
                Dict, Any, Any
            ]:  # Yield dictionaries to update multiple components
                if xlsx_file_obj is None:
                    yield {
                        batch_status_output: "ERROR: Please upload an Excel file.",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),  # Update UI visibility
                    }
                    return

                # Reset states for new batch
                current_batch_job_id = None
                yield {
                    batch_status_output: "Starting batch processing...",
                    batch_download_output: "",
                    ui_mode_state: "processing",
                    batch_job_id_state: None,
                    items_to_label_state: [],
                    current_item_index_state: 0,
                    eligible_speakers_state: [],
                    current_speaker_index_state: 0,
                    current_clip_paths_state: [],
                    current_clip_index_state: 0,
                    collected_labels_state: {},
                    **change_ui_mode(
                        "processing"
                    ),  # Show processing, hide inputs/labeling
                }

                try:
                    # --- Call pipeline's initial processing ---
                    status_msg, results_summary, returned_batch_id = (
                        self.pipeline.process_batch_xlsx(
                            xlsx_filepath=xlsx_file_obj,
                            include_source_audio=include_audio,
                            include_json_summary=include_json,
                            include_csv_summary=include_csv,
                            include_script_transcript=include_script,
                            include_plots=include_plots,
                        )
                    )

                    yield {batch_status_output: f"{status_msg}\n\n{results_summary}"}

                    if returned_batch_id:
                        # --- Labeling is Required ---
                        current_batch_job_id = returned_batch_id
                        log_info(
                            f"Batch [{current_batch_job_id}] requires labeling. Initializing UI."
                        )

                        # Need to find the first item and first speaker
                        # Let's assume the pipeline state holds the list of items needing labeling.
                        # We need a way to get this list. Let's modify the backend assumption slightly:
                        # Assume self.pipeline.labeling_state[batch_id] exists now.
                        items_requiring_labeling = list(
                            self.pipeline.labeling_state.get(
                                current_batch_job_id, {}
                            ).keys()
                        )
                        items_requiring_labeling.sort()  # Process in order

                        if not items_requiring_labeling:
                            log_error(
                                f"Batch [{current_batch_job_id}] indicated labeling needed, but no items found in state."
                            )
                            yield {
                                batch_status_output: f"{status_msg}\n\n{results_summary}\n\nERROR: Internal state error during labeling setup.",
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                            return

                        first_item_id = items_requiring_labeling[0]
                        log_info(f"Starting labeling with item: {first_item_id}")

                        # Get data for the first speaker of the first item
                        first_speaker_data = (
                            self.pipeline.start_interactive_labeling_for_item(
                                current_batch_job_id, first_item_id
                            )
                        )

                        if first_speaker_data:
                            first_speaker_id, first_clip_paths = first_speaker_data
                            eligible_speakers_list = self.pipeline.labeling_state[
                                current_batch_job_id
                            ][first_item_id].get("eligible_speakers", [])
                            initial_clip_path = (
                                first_clip_paths[0] if first_clip_paths else None
                            )
                            clip_count = len(first_clip_paths)

                            # Update all states and UI elements for labeling mode
                            yield {
                                batch_job_id_state: current_batch_job_id,
                                items_to_label_state: items_requiring_labeling,
                                current_item_index_state: 0,  # Start with first item
                                eligible_speakers_state: eligible_speakers_list,
                                current_speaker_index_state: 0,  # Start with first speaker
                                current_clip_paths_state: first_clip_paths,
                                current_clip_index_state: 0,
                                collected_labels_state: {},  # Reset labels for item
                                ui_mode_state: "labeling",
                                labeling_progress_md: f"Labeling Speaker: **{first_speaker_id}** (Speaker 1 of {len(eligible_speakers_list)}, Item 1 of {len(items_requiring_labeling)})",
                                current_clip_display: f"Clip 1 of {clip_count}",
                                video_player: gr.update(
                                    value=initial_clip_path, interactive=False
                                ),  # Load first video
                                speaker_label_input: gr.update(value=""),  # Clear input
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                **change_ui_mode("labeling"),  # Show labeling UI
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for item {first_item_id} in batch {current_batch_job_id}."
                            )
                            yield {
                                batch_status_output: f"{status_msg}\n\n{results_summary}\n\nERROR: Failed to start labeling process.",
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                            # Clean up? Maybe finalize the failed item? Difficult state.

                    else:
                        # --- No Labeling Required or Error Occurred ---
                        log_info(
                            f"No interactive labeling needed for batch or initial processing failed."
                        )
                        yield {
                            # Status/Results already updated above
                            ui_mode_state: "finished"
                            if "✅" in status_msg
                            else "error",
                            **change_ui_mode(
                                "finished" if "✅" in status_msg else "error"
                            ),
                        }

                except Exception as e:
                    error_trace = traceback.format_exc()
                    log_error(f"Error in process_batch_wrapper: {e}\n{error_trace}")
                    yield {
                        batch_status_output: f"An unexpected error occurred: {e}\n\n{error_trace}",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }

            # --- Button Click Handlers ---

            # 1. Start Batch Button
            batch_process_btn.click(
                fn=process_batch_wrapper,
                inputs=[
                    batch_input_file,
                    include_source_audio_checkbox,
                    include_json_summary_checkbox,
                    include_csv_summary_checkbox,
                    include_script_transcript_checkbox,
                    include_plots_checkbox,
                ],
                outputs=[
                    # Components to update directly
                    batch_status_output,
                    batch_download_output,
                    video_player,
                    speaker_label_input,
                    labeling_progress_md,
                    current_clip_display,
                    prev_clip_btn,
                    next_clip_btn,
                    # Groups/Columns visibility
                    labeling_ui_group,
                    batch_input_group,
                    status_output_group,
                    batch_process_btn,
                    # State variables
                    ui_mode_state,
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_clip_paths_state,
                    current_clip_index_state,
                    collected_labels_state,
                ],
            )

            # 2. Change Video Clip Buttons
            def handle_change_clip(
                direction: int, current_clip_idx: int, clip_paths: list
            ) -> Dict:
                """Handles changing the displayed video clip."""
                if not clip_paths:
                    return {}  # No clips

                new_clip_idx = current_clip_idx + direction
                num_clips = len(clip_paths)

                # Clamp index
                new_clip_idx = max(0, min(new_clip_idx, num_clips - 1))

                if new_clip_idx == current_clip_idx:
                    return {}  # No change

                new_clip_path = clip_paths[new_clip_idx]
                return {
                    current_clip_index_state: new_clip_idx,
                    video_player: gr.update(value=new_clip_path, interactive=False),
                    current_clip_display: f"Clip {new_clip_idx + 1} of {num_clips}",
                }

            prev_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(-1),
                    current_clip_index_state,
                    current_clip_paths_state,
                ],
                outputs=[current_clip_index_state, video_player, current_clip_display],
            )
            next_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(1),
                    current_clip_index_state,
                    current_clip_paths_state,
                ],
                outputs=[current_clip_index_state, video_player, current_clip_display],
            )

            # 3. Submit Label Button
            def handle_submit_label(
                batch_id: str,
                items_to_label: list,  # List of item_ids
                item_idx: int,
                speakers_to_label: list,  # List of speaker_ids for current item
                speaker_idx: int,
                collected_labs: dict,  # Labels collected so far for current item
                current_label_input: str,  # Value from textbox
            ) -> Generator[Dict, Any, Any]:  # Yield updates
                """Handles submitting a label and moving to the next speaker or item."""

                if (
                    not batch_id
                    or item_idx >= len(items_to_label)
                    or speaker_idx >= len(speakers_to_label)
                ):
                    log_error("Submit label called with invalid state.")
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nERROR: Invalid state during label submission."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                speaker_id = speakers_to_label[speaker_idx]

                log_info(
                    f"[{batch_id}-{item_id}] Submitting label for {speaker_id}: '{current_label_input}'"
                )

                # Store the label via pipeline
                success = self.pipeline.store_speaker_label(
                    batch_id, item_id, speaker_id, current_label_input
                )
                if not success:
                    log_error(f"Failed to store label for {speaker_id} in backend.")
                    # Update status but try to continue? Or halt? Let's try to continue.
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nWARN: Failed to store label for {speaker_id} in backend."
                        )
                    }
                    # Update local copy of labels anyway for UI state consistency
                collected_labs[speaker_id] = (
                    current_label_input  # Update the dictionary directly
                )

                # Get next speaker for the CURRENT item
                next_speaker_data = self.pipeline.get_next_speaker_for_labeling(
                    batch_id, item_id, speaker_idx
                )

                if next_speaker_data:
                    # --- Still more speakers in the CURRENT item ---
                    next_speaker_id, next_clip_paths = next_speaker_data
                    next_speaker_idx = speaker_idx + 1
                    initial_clip_path = next_clip_paths[0] if next_clip_paths else None
                    clip_count = len(next_clip_paths)

                    yield {
                        current_speaker_index_state: next_speaker_idx,
                        current_clip_paths_state: next_clip_paths,
                        current_clip_index_state: 0,
                        collected_labels_state: collected_labs,  # Pass updated labels back
                        labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker {next_speaker_idx + 1} of {len(speakers_to_label)}, Item {item_idx + 1} of {len(items_to_label)})",
                        current_clip_display: f"Clip 1 of {clip_count}",
                        video_player: gr.update(
                            value=initial_clip_path, interactive=False
                        ),
                        speaker_label_input: gr.update(value=""),  # Clear input
                        prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                        next_clip_btn: gr.update(interactive=(clip_count > 1)),
                    }
                else:
                    # --- Finished speakers for the CURRENT item ---
                    log_info(
                        f"[{batch_id}-{item_id}] Finished labeling all speakers for this item."
                    )
                    # Finalize this item in the backend
                    self.pipeline.finalize_labeled_item(batch_id, item_id)
                    # Check if there are more items to label
                    next_item_idx = item_idx + 1
                    if next_item_idx < len(items_to_label):
                        # --- Move to the NEXT item ---
                        next_item_id = items_to_label[next_item_idx]
                        log_info(
                            f"[{batch_id}] Moving to label next item: {next_item_id}"
                        )

                        # Get data for the first speaker of the next item
                        first_speaker_data_next_item = (
                            self.pipeline.start_interactive_labeling_for_item(
                                batch_id, next_item_id
                            )
                        )

                        if first_speaker_data_next_item:
                            next_item_speaker_id, next_item_clip_paths = (
                                first_speaker_data_next_item
                            )
                            next_item_eligible_speakers = self.pipeline.labeling_state[
                                batch_id
                            ][next_item_id].get("eligible_speakers", [])
                            initial_clip_path = (
                                next_item_clip_paths[0]
                                if next_item_clip_paths
                                else None
                            )
                            clip_count = len(next_item_clip_paths)

                            yield {
                                current_item_index_state: next_item_idx,  # Move to next item
                                eligible_speakers_state: next_item_eligible_speakers,  # New list of speakers
                                current_speaker_index_state: 0,  # Start with first speaker
                                current_clip_paths_state: next_item_clip_paths,
                                current_clip_index_state: 0,
                                collected_labels_state: {},  # Reset collected labels for new item
                                labeling_progress_md: f"Labeling Speaker: **{next_item_speaker_id}** (Speaker 1 of {len(next_item_eligible_speakers)}, Item {next_item_idx + 1} of {len(items_to_label)})",
                                current_clip_display: f"Clip 1 of {clip_count}",
                                video_player: gr.update(
                                    value=initial_clip_path, interactive=False
                                ),
                                speaker_label_input: gr.update(value=""),
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for next item {next_item_id} in batch {batch_id}."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{batch_status_output.value}\n\nERROR: Failed to start labeling next item {next_item_id}."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    else:
                        # --- Finished all items in the batch ---
                        log_info(f"[{batch_id}] Finished labeling all items.")
                        yield {
                            batch_status_output: gr.update(
                                value=f"{batch_status_output.value}\n\nLabeling complete for all items. Finalizing batch..."
                            )
                        }

                        # Trigger final zip creation
                        final_zip_path = self.pipeline.check_batch_completion_and_zip(
                            batch_id
                        )

                        if final_zip_path:
                            final_status = f"✅ Batch processing and labeling complete. Final output: {final_zip_path}"
                            final_mode = "finished"
                        else:
                            final_status = f"❗️ Batch processing and labeling complete, but failed to create final ZIP."
                            final_mode = "error"

                        yield {
                            batch_status_output: gr.update(
                                value=f"{batch_status_output.value}\n\n{final_status}"
                            ),
                            batch_download_output: str(final_zip_path)
                            if final_zip_path
                            else "",
                            ui_mode_state: final_mode,
                            **change_ui_mode(
                                final_mode
                            ),  # Hide labeling UI, show inputs again
                        }

            submit_label_btn.click(
                fn=handle_submit_label,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    collected_labels_state,
                    speaker_label_input,
                ],
                outputs=[
                    # Components to update directly
                    batch_status_output,
                    batch_download_output,
                    video_player,
                    speaker_label_input,
                    labeling_progress_md,
                    current_clip_display,
                    prev_clip_btn,
                    next_clip_btn,
                    # Groups/Columns visibility
                    labeling_ui_group,
                    batch_input_group,
                    status_output_group,
                    batch_process_btn,
                    # State variables that might change
                    ui_mode_state,
                    current_item_index_state,
                    current_speaker_index_state,
                    eligible_speakers_state,
                    current_clip_paths_state,
                    current_clip_index_state,
                    collected_labels_state,
                ],
            )

        return demo

    def launch(self, **kwargs):
        """Launches the Gradio interface."""
        self.interface_ui().launch(**kwargs)
