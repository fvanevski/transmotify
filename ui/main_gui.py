# ui/main_gui.py
import os
import traceback
import re
import math  # Import math for floor function
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator, Union

import gradio as gr
from gradio import themes
import pandas as pd

# Assuming Pipeline class is accessible via from core.pipeline import Pipeline
try:
    from core.pipeline import Pipeline
    from core.logging import log_error, log_warning, log_info
except ImportError:
    print(
        "ERROR: Failed to import core modules. Ensure main.py is run from the project root."
    )

    # Define dummy classes/functions if import fails
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

        def skip_labeling_for_item(self, *args, **kwargs):
            return False  # Add dummy skip

        def check_batch_completion_and_zip(self, *args, **kwargs):
            return None

    def log_error(msg):
        print(f"LOG_ERROR: {msg}")

    def log_warning(msg):
        print(f"LOG_WARNING: {msg}")

    def log_info(msg):
        print(f"LOG_INFO: {msg}")


# --- Helper to generate YouTube embed HTML ---
def get_youtube_embed_html(youtube_url: str, start_time_seconds: int = 0) -> str:
    """Generates HTML iframe embed code for a YouTube video with a start time."""
    if not youtube_url or not youtube_url.startswith("http"):
        return "<p>Invalid YouTube URL</p>"

    # Extract video ID from various URL formats
    video_id = None
    if "youtube.com/watch?v=" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]

    if not video_id:
        return f"<p>Could not extract Video ID from URL: {youtube_url}</p>"

    # Note: Autoplay is often blocked by browsers, controls=1 enables player controls
    # Ensure start_time is an integer
    start_param = int(math.floor(start_time_seconds))
    embed_url = (
        f"https://www.youtube.com/embed/{video_id}?start={start_param}&controls=1"
    )

    # Adjust size as needed
    return f'<iframe width="560" height="315" src="{embed_url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'


class UI:
    def __init__(self, config):
        self.config_data = config.config if hasattr(config, "config") else config
        self.pipeline = Pipeline(self.config_data)
        log_info("UI Initialized with Pipeline instance.")

    def interface_ui(self):
        default_theme = themes.Default()

        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Batch Processing & Interactive Speaker Labeling")

            # --- UI Mode State ---
            ui_mode_state = gr.State(
                "idle"
            )  # 'idle', 'processing', 'labeling', 'finished', 'error'

            # --- Batch Processing State ---
            batch_job_id_state = gr.State(None)
            # Stores item_ids needing labeling IN ORDER determined by pipeline
            items_to_label_state = gr.State([])
            current_item_index_state = gr.State(0)

            # --- Speaker Labeling State (Current Item) ---
            current_youtube_url_state = gr.State("")  # YouTube URL for current item
            # Stores speaker_ids needing labeling IN ORDER for current item
            eligible_speakers_state = gr.State([])
            current_speaker_index_state = gr.State(0)
            # Stores list of start times (floats/ints in seconds) for current speaker
            current_start_times_state = gr.State([])
            # Index of the start time currently used for the embed
            current_clip_index_state = gr.State(0)
            # Stores {'SPEAKER_XX': 'UserLabel'} for the CURRENT item being labeled
            collected_labels_state = gr.State({})

            # --- Main Layout ---
            with gr.Row():
                # Input Column
                with gr.Column(scale=1):
                    with gr.Group() as batch_input_group:
                        gr.Markdown("## Batch Processing Input")
                        gr.Markdown(
                            "Upload Excel file with 'YouTube URL' column. Enable labeling in `config.json` if desired."
                        )
                        batch_input_file = gr.File(
                            label="Upload Excel File (.xlsx)",
                            type="filepath",
                            file_types=[".xlsx"],
                        )
                        gr.Markdown("### Output Options")
                        # Checkboxes remain the same...
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

                # Status/Results Column
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
            with gr.Column(visible=False) as labeling_ui_group:
                gr.Markdown("## Interactive Speaker Labeling")
                labeling_progress_md = gr.Markdown("Labeling Speaker: ---")
                with gr.Row():
                    with gr.Column(scale=2):
                        # --- Changed gr.Video to gr.HTML ---
                        video_player_html = gr.HTML(label="Speaker Preview")
                    with gr.Column(scale=1):
                        gr.Markdown("### Clip Navigation")
                        current_clip_display = gr.Markdown("Preview 1 of X")
                        prev_clip_btn = gr.Button("⬅️ Previous Preview")
                        next_clip_btn = gr.Button("Next Preview ➡️")
                        gr.Markdown("### Enter Label")
                        speaker_label_input = gr.Textbox(
                            label="Enter Speaker Name/Label",
                            placeholder="e.g., Alice (or leave blank)",
                        )
                        # --- Added Skip Button ---
                        skip_item_btn = gr.Button("Skip Rest of Item ⏭️")
                        submit_label_btn = gr.Button(
                            "Submit Label & Next Speaker ▶️", variant="primary"
                        )

            # --- Helper Functions for UI Logic ---
            def change_ui_mode(mode):
                log_info(f"Changing UI mode to: {mode}")
                is_labeling = mode == "labeling"
                is_idle_or_finished = (
                    mode == "idle" or mode == "finished" or mode == "error"
                )
                return {
                    labeling_ui_group: gr.update(visible=is_labeling),
                    batch_input_group: gr.update(visible=is_idle_or_finished),
                    status_output_group: gr.update(visible=True),
                    batch_process_btn: gr.update(interactive=is_idle_or_finished),
                }

            # --- Main Batch Processing Wrapper ---
            def process_batch_wrapper(
                xlsx_file_obj: Optional[str],
                include_audio: bool,
                include_json: bool,
                include_csv: bool,
                include_script: bool,
                include_plots: bool,
            ) -> Generator[Dict, Any, Any]:
                if xlsx_file_obj is None:
                    yield {
                        batch_status_output: "ERROR: Please upload an Excel file.",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }
                    return

                # Reset states
                current_batch_job_id = None
                yield {
                    batch_status_output: "Starting batch processing...",
                    batch_download_output: "",
                    ui_mode_state: "processing",
                    batch_job_id_state: None,
                    items_to_label_state: [],
                    current_item_index_state: 0,
                    current_youtube_url_state: "",
                    eligible_speakers_state: [],
                    current_speaker_index_state: 0,
                    current_start_times_state: [],
                    current_clip_index_state: 0,
                    collected_labels_state: {},
                    video_player_html: "",  # Clear HTML too
                    **change_ui_mode("processing"),
                }

                try:
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

                        items_requiring_labeling = self.pipeline.labeling_state.get(
                            current_batch_job_id, {}
                        ).get("items_requiring_labeling_order", [])

                        if not items_requiring_labeling:
                            log_error(
                                f"Batch [{current_batch_job_id}] needs labeling, but no items found in state."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{batch_status_output.value}\n\nERROR: Internal state error."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                            return

                        first_item_id = items_requiring_labeling[0]
                        log_info(f"Starting labeling with item: {first_item_id}")

                        # Get data: SpeakerID, YouTubeURL, List[StartTimes]
                        first_speaker_data = (
                            self.pipeline.start_interactive_labeling_for_item(
                                current_batch_job_id, first_item_id
                            )
                        )

                        if first_speaker_data:
                            first_speaker_id, yt_url, first_start_times = (
                                first_speaker_data
                            )
                            # Ensure eligible_speakers list is retrieved from state for progress display
                            eligible_speakers_list = self.pipeline.labeling_state[
                                current_batch_job_id
                            ][first_item_id].get("eligible_speakers", [])
                            initial_start_time = (
                                first_start_times[0] if first_start_times else 0
                            )
                            clip_count = len(first_start_times)
                            initial_html = get_youtube_embed_html(
                                yt_url, initial_start_time
                            )

                            yield {
                                batch_job_id_state: current_batch_job_id,
                                items_to_label_state: items_requiring_labeling,
                                current_item_index_state: 0,
                                current_youtube_url_state: yt_url,  # Store URL
                                eligible_speakers_state: eligible_speakers_list,
                                current_speaker_index_state: 0,
                                current_start_times_state: first_start_times,  # Store start times
                                current_clip_index_state: 0,
                                collected_labels_state: {},
                                ui_mode_state: "labeling",
                                labeling_progress_md: f"Labeling Speaker: **{first_speaker_id}** (Speaker 1 of {len(eligible_speakers_list)}, Item 1 of {len(items_requiring_labeling)})",
                                current_clip_display: f"Preview 1 of {clip_count}",
                                video_player_html: initial_html,  # Set initial embed
                                speaker_label_input: gr.update(value=""),
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                **change_ui_mode("labeling"),
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for item {first_item_id} in batch {current_batch_job_id}."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{batch_status_output.value}\n\nERROR: Failed to start labeling."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    else:
                        # --- No Labeling Required or Error Occurred ---
                        log_info(
                            f"No interactive labeling needed or initial processing failed."
                        )
                        final_mode = "finished" if "✅" in status_msg else "error"
                        yield {ui_mode_state: final_mode, **change_ui_mode(final_mode)}

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
                outputs=[  # Order matters for updates dictionary return
                    batch_status_output,
                    batch_download_output,
                    video_player_html,
                    speaker_label_input,
                    labeling_progress_md,
                    current_clip_display,
                    prev_clip_btn,
                    next_clip_btn,
                    labeling_ui_group,
                    batch_input_group,
                    status_output_group,
                    batch_process_btn,  # UI visibility/interactivity
                    # State variables must be outputs to receive updates
                    ui_mode_state,
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                    collected_labels_state,
                ],
            )

            # 2. Change Video Clip (Preview) Buttons
            def handle_change_clip(
                direction: int,
                current_clip_idx: int,
                start_times: list,
                youtube_url: str,
            ) -> Dict:
                """Handles changing the displayed YouTube embed start time."""
                if not start_times:
                    return {}
                num_clips = len(start_times)
                new_clip_idx = max(0, min(current_clip_idx + direction, num_clips - 1))
                if new_clip_idx == current_clip_idx:
                    return {}

                new_start_time = start_times[new_clip_idx]
                new_html = get_youtube_embed_html(youtube_url, new_start_time)
                return {
                    current_clip_index_state: new_clip_idx,
                    video_player_html: new_html,
                    current_clip_display: f"Preview {new_clip_idx + 1} of {num_clips}",
                }

            prev_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(-1),
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[
                    current_clip_index_state,
                    video_player_html,
                    current_clip_display,
                ],
            )
            next_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(1),
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[
                    current_clip_index_state,
                    video_player_html,
                    current_clip_display,
                ],
            )

            # --- Common Logic for Moving to Next State (Speaker/Item/Finish) ---
            def move_to_next_labeling_state(
                batch_id: str, items_to_label: list, item_idx: int, speaker_idx: int
            ) -> Generator[Dict, Any, Any]:
                """Handles transitions after submitting label or skipping item."""
                # Try to get the next speaker *within the current item* first
                next_speaker_data = self.pipeline.get_next_speaker_for_labeling(
                    batch_id, items_to_label[item_idx], speaker_idx
                )

                if next_speaker_data:
                    # --- Still more speakers in the CURRENT item ---
                    next_speaker_id, yt_url, next_start_times = next_speaker_data
                    next_speaker_idx = speaker_idx + 1
                    initial_start_time = next_start_times[0] if next_start_times else 0
                    clip_count = len(next_start_times)
                    new_html = get_youtube_embed_html(yt_url, initial_start_time)
                    eligible_speakers_list = self.pipeline.labeling_state[batch_id][
                        items_to_label[item_idx]
                    ].get("eligible_speakers", [])  # Re-fetch just in case

                    yield {
                        current_speaker_index_state: next_speaker_idx,
                        current_start_times_state: next_start_times,
                        current_clip_index_state: 0,
                        # collected_labels_state: collected_labs, # Keep existing labels for item
                        labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker {next_speaker_idx + 1} of {len(eligible_speakers_list)}, Item {item_idx + 1} of {len(items_to_label)})",
                        current_clip_display: f"Preview 1 of {clip_count}",
                        video_player_html: new_html,
                        speaker_label_input: gr.update(value=""),  # Clear input
                        prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                        next_clip_btn: gr.update(interactive=(clip_count > 1)),
                    }
                else:
                    # --- Finished speakers for the CURRENT item ---
                    log_info(
                        f"[{batch_id}-{items_to_label[item_idx]}] Finished labeling/skipping item."
                    )
                    # Finalize this item in the backend (happens whether submitted or skipped)
                    # Note: If skipped, finalize_labeled_item uses labels collected *so far*
                    self.pipeline.finalize_labeled_item(
                        batch_id, items_to_label[item_idx]
                    )

                    # Check if there are more items to label
                    next_item_idx = item_idx + 1
                    if next_item_idx < len(items_to_label):
                        # --- Move to the NEXT item ---
                        next_item_id = items_to_label[next_item_idx]
                        log_info(
                            f"[{batch_id}] Moving to label next item: {next_item_id}"
                        )
                        first_speaker_data_next = (
                            self.pipeline.start_interactive_labeling_for_item(
                                batch_id, next_item_id
                            )
                        )

                        if first_speaker_data_next:
                            next_item_speaker_id, next_yt_url, next_item_start_times = (
                                first_speaker_data_next
                            )
                            next_item_elig_spkrs = self.pipeline.labeling_state[
                                batch_id
                            ][next_item_id].get("eligible_speakers", [])
                            initial_start_time = (
                                next_item_start_times[0] if next_item_start_times else 0
                            )
                            clip_count = len(next_item_start_times)
                            new_html = get_youtube_embed_html(
                                next_yt_url, initial_start_time
                            )

                            yield {
                                current_item_index_state: next_item_idx,
                                current_youtube_url_state: next_yt_url,  # Update URL
                                eligible_speakers_state: next_item_elig_spkrs,
                                current_speaker_index_state: 0,  # Reset speaker index
                                current_start_times_state: next_item_start_times,
                                current_clip_index_state: 0,
                                collected_labels_state: {},  # Reset labels for new item
                                labeling_progress_md: f"Labeling Speaker: **{next_item_speaker_id}** (Speaker 1 of {len(next_item_elig_spkrs)}, Item {next_item_idx + 1} of {len(items_to_label)})",
                                current_clip_display: f"Preview 1 of {clip_count}",
                                video_player_html: new_html,
                                speaker_label_input: gr.update(value=""),
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for next item {next_item_id}."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{batch_status_output.value}\n\nERROR: Failed start labeling item {next_item_id}."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    else:
                        # --- Finished all items in the batch ---
                        log_info(f"[{batch_id}] Finished labeling/skipping all items.")
                        yield {
                            batch_status_output: gr.update(
                                value=f"{batch_status_output.value}\n\nLabeling complete. Finalizing batch..."
                            )
                        }
                        final_zip_path = self.pipeline.check_batch_completion_and_zip(
                            batch_id
                        )
                        final_status = (
                            f"✅ Batch complete. Output: {final_zip_path}"
                            if final_zip_path
                            else f"❗️ Batch complete, but ZIP failed."
                        )
                        final_mode = "finished" if final_zip_path else "error"
                        yield {
                            batch_status_output: gr.update(
                                value=f"{batch_status_output.value}\n\n{final_status}"
                            ),
                            batch_download_output: str(final_zip_path)
                            if final_zip_path
                            else "",
                            ui_mode_state: final_mode,
                            **change_ui_mode(final_mode),
                        }

            # 3. Submit Label Button Handler
            def handle_submit_label_wrapper(*args):  # Use *args to adapt easily
                # args order needs to match inputs list below
                # batch_id, items_to_label, item_idx, speakers_to_label, speaker_idx, collected_labs, current_label_input
                (
                    batch_id,
                    items_to_label,
                    item_idx,
                    speakers_to_label,
                    speaker_idx,
                    collected_labs,
                    current_label_input,
                ) = args

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

                success = self.pipeline.store_speaker_label(
                    batch_id, item_id, speaker_id, current_label_input
                )
                if not success:
                    log_warning(
                        f"Failed to store label for {speaker_id} in backend."
                    )  # Log warning, continue
                collected_labs[speaker_id] = (
                    current_label_input  # Update local state copy
                )

                # Now transition to the next state (speaker/item/finish)
                yield from move_to_next_labeling_state(
                    batch_id, items_to_label, item_idx, speaker_idx
                )  # Pass current speaker index

            submit_label_btn.click(
                fn=handle_submit_label_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    collected_labels_state,
                    speaker_label_input,  # Get current value from textbox
                ],
                outputs=[  # Must list all components potentially updated by the generator
                    # Direct updates
                    batch_status_output,
                    batch_download_output,
                    video_player_html,
                    speaker_label_input,
                    labeling_progress_md,
                    current_clip_display,
                    prev_clip_btn,
                    next_clip_btn,
                    # Visibility/Mode updates
                    labeling_ui_group,
                    batch_input_group,
                    status_output_group,
                    batch_process_btn,
                    # State updates
                    ui_mode_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                    collected_labels_state,
                ],
            )

            # 4. Skip Rest of Item Button Handler
            def handle_skip_item_wrapper(*args):
                # args order needs to match inputs list below
                # batch_id, items_to_label, item_idx
                batch_id, items_to_label, item_idx = args

                if not batch_id or item_idx >= len(items_to_label):
                    log_error("Skip item called with invalid state.")
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nERROR: Invalid state during skip."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                log_info(
                    f"[{batch_id}-{item_id}] User skipping rest of speakers for item."
                )

                # Call backend skip function (which just finalizes with current labels)
                success = self.pipeline.skip_labeling_for_item(batch_id, item_id)
                if not success:
                    log_warning(
                        f"[{batch_id}-{item_id}] Backend finalization during skip returned failure, but attempting to proceed."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nWARN: Finalization during skip failed for item {item_id}."
                        )
                    }
                else:
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nSkipped remaining speakers for item {item_id}."
                        )
                    }

                # Now transition state - pretend we finished the *last* speaker of the skipped item
                # We need the count of speakers for the skipped item to pass the correct index to move_to_next
                # This requires accessing state, which is tricky here. Let's assume skip means index = len(speakers)-1
                # This might be fragile. A cleaner way might be needed if the state isn't available easily.
                # Let's retrieve the eligible speakers count from the pipeline state again for the skipped item.
                item_state_for_skip = self.pipeline._get_item_state(
                    batch_id, item_id
                )  # Use internal getter
                # We call _remove_item_state inside finalize/skip, so state might be gone already!
                # Let's adjust: skip_labeling_for_item should return True/False,
                # and we call move_to_next based on that.
                # Let's simplify: assume the skip always finalizes the item. We just need to move UI state.

                # Transition to next state (next item or finish batch)
                # We effectively finished the "last" speaker of the current item_idx
                # We need speaker_idx here, which is complex to get reliably after skip.
                # Let's modify move_to_next_labeling_state to accept item_idx directly
                # and figure out the next step from there (next item or finish).
                # Okay, simpler: the skip handler will determine the next item_idx and call start_interactive_labeling_for_item
                # if needed, or check_batch_completion_and_zip.

                next_item_idx = item_idx + 1
                if next_item_idx < len(items_to_label):
                    # --- Move to the NEXT item ---
                    next_item_id = items_to_label[next_item_idx]
                    log_info(
                        f"[{batch_id}] Moving to label next item after skip: {next_item_id}"
                    )
                    first_speaker_data_next = (
                        self.pipeline.start_interactive_labeling_for_item(
                            batch_id, next_item_id
                        )
                    )
                    if first_speaker_data_next:
                        next_item_speaker_id, next_yt_url, next_item_start_times = (
                            first_speaker_data_next
                        )
                        next_item_elig_spkrs = self.pipeline.labeling_state[batch_id][
                            next_item_id
                        ].get("eligible_speakers", [])
                        initial_start_time = (
                            next_item_start_times[0] if next_item_start_times else 0
                        )
                        clip_count = len(next_item_start_times)
                        new_html = get_youtube_embed_html(
                            next_yt_url, initial_start_time
                        )
                        yield {
                            current_item_index_state: next_item_idx,
                            current_youtube_url_state: next_yt_url,
                            eligible_speakers_state: next_item_elig_spkrs,
                            current_speaker_index_state: 0,
                            current_start_times_state: next_item_start_times,
                            current_clip_index_state: 0,
                            collected_labels_state: {},  # Reset labels
                            labeling_progress_md: f"Labeling Speaker: **{next_item_speaker_id}** (Speaker 1 of {len(next_item_elig_spkrs)}, Item {next_item_idx + 1} of {len(items_to_label)})",
                            current_clip_display: f"Preview 1 of {clip_count}",
                            video_player_html: new_html,
                            speaker_label_input: gr.update(value=""),
                            prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                            next_clip_btn: gr.update(interactive=(clip_count > 1)),
                        }
                    else:
                        log_error(
                            f"Failed get data for next item {next_item_id} after skip."
                        )
                        yield {
                            batch_status_output: gr.update(
                                value=f"{batch_status_output.value}\n\nERROR: Failed start labeling item {next_item_id} after skip."
                            ),
                            ui_mode_state: "error",
                            **change_ui_mode("error"),
                        }
                else:
                    # --- Finished all items after skip ---
                    log_info(f"[{batch_id}] Finished all items after skip.")
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\nLabeling complete. Finalizing batch..."
                        )
                    }
                    final_zip_path = self.pipeline.check_batch_completion_and_zip(
                        batch_id
                    )
                    final_status = (
                        f"✅ Batch complete. Output: {final_zip_path}"
                        if final_zip_path
                        else f"❗️ Batch complete, but ZIP failed."
                    )
                    final_mode = "finished" if final_zip_path else "error"
                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\n{final_status}"
                        ),
                        batch_download_output: str(final_zip_path)
                        if final_zip_path
                        else "",
                        ui_mode_state: final_mode,
                        **change_ui_mode(final_mode),
                    }

            skip_item_btn.click(
                fn=handle_skip_item_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                ],
                outputs=[  # Must list all components potentially updated by the generator
                    # Direct updates
                    batch_status_output,
                    batch_download_output,
                    video_player_html,
                    speaker_label_input,
                    labeling_progress_md,
                    current_clip_display,
                    prev_clip_btn,
                    next_clip_btn,
                    # Visibility/Mode updates
                    labeling_ui_group,
                    batch_input_group,
                    status_output_group,
                    batch_process_btn,
                    # State updates
                    ui_mode_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                    collected_labels_state,
                ],
            )

        return demo

    def launch(self, **kwargs):
        """Launches the Gradio interface."""
        try:
            self.interface_ui().launch(**kwargs)
        except Exception as e:
            log_error(f"FATAL ERROR during Gradio launch: {e}")
            log_error(traceback.format_exc())
            print(
                f"\nFATAL ERROR: Could not launch Gradio UI. Check logs. Error: {e}\n"
            )
