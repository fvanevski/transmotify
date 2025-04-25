# ui/webapp.py
"""
Defines the Gradio web interface for the speech analysis application.
Interacts with the core.orchestrator module to run processing pipelines
and manage interactive speaker labeling.
Moved from legacy ui/main_gui.py
"""

import os
import traceback
import re
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator, Union

# Gradio and Pandas are core dependencies for the UI
try:
    import gradio as gr
    from gradio import themes
    import pandas as pd

    GRADIO_AVAILABLE = True
except ImportError:
    print("ERROR: Gradio or Pandas library not found. UI cannot be launched.")
    GRADIO_AVAILABLE = False
    # Define dummy types if import fails
    gr = Any
    themes = Any
    pd = Any

# Import the backend orchestrator and logging
try:
    from core.orchestrator import Orchestrator  # The new backend class
    from core.logging import log_error, log_warning, log_info
except ImportError:
    print(
        "ERROR: Failed to import core modules (Orchestrator, logging). Ensure PYTHONPATH is correct."
    )

    # Define dummy classes/functions if core components missing
    class Orchestrator:
        pass

    log_error = log_warning = log_info = print


# --- Helper to generate YouTube embed HTML ---
# Moved from legacy ui/main_gui.py
def get_youtube_embed_html(youtube_url: str, start_time_seconds: int = 0) -> str:
    """Creates HTML for embedding a YouTube video starting at a specific time."""
    if (
        not youtube_url
        or not isinstance(youtube_url, str)
        or not youtube_url.startswith("http")
    ):
        log_warning(f"Invalid YouTube URL for embed: {youtube_url}")
        return "<p>Invalid YouTube URL</p>"

    video_id = None
    # Handle common YouTube URL formats (including potential youtube.com/watch?v=... )
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",  # Standard watch URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})",  # Shortened URL
        r"embed/([a-zA-Z0-9_-]{11})",  # Embed URL
        r"googleusercontent\.com/youtube\.com/\d+/([a-zA-Z0-9_-]{11})",  # User content URL (less common)
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        log_warning(f"Could not extract Video ID from URL: {youtube_url}")
        return f"<p>Could not extract Video ID from URL</p>"

    start_param = max(
        0, int(math.floor(start_time_seconds))
    )  # Ensure non-negative integer
    # Use standard YouTube embed URL
    embed_url = (
        f"https://www.youtube.com/embed/{video_id}?start={start_param}&controls=1"
    )

    # Use standard iframe attributes
    return (
        f'<iframe width="560" height="315" src="{embed_url}" '
        f'title="YouTube video player" frameborder="0" '
        f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        f'referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
    )


class UI:
    """Manages the Gradio interface and its interactions with the Orchestrator."""

    # --- MODIFIED: Accepts Orchestrator instance ---
    def __init__(self, orchestrator: Orchestrator):
        """
        Initializes the UI.

        Args:
            orchestrator: An instance of the core.orchestrator.Orchestrator class.
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio or Pandas not found. UI cannot be initialized.")
        self.orchestrator = orchestrator
        self.config_data = orchestrator.config_data  # Get config dict from orchestrator
        log_info("UI Initialized with Orchestrator instance.")

    def create_ui(self) -> gr.Blocks:
        """Creates the Gradio Blocks interface."""
        default_theme = themes.Default()

        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Speech Analysis Pipeline")  # Updated title slightly

            # --- UI States (remain the same) ---
            ui_mode_state = gr.State(
                "idle"
            )  # idle, processing, labeling, finished, error
            batch_job_id_state = gr.State(None)
            items_to_label_state = gr.State(
                []
            )  # List of item_identifiers requiring labeling
            current_item_index_state = gr.State(0)  # Index into items_to_label_state
            current_youtube_url_state = gr.State("")  # URL for the item being labeled
            eligible_speakers_state = gr.State(
                []
            )  # List of SPEAKER_IDs for the current item
            current_speaker_index_state = gr.State(
                0
            )  # Index into eligible_speakers_state
            current_start_times_state = gr.State(
                []
            )  # List of preview start times (int seconds)
            current_clip_index_state = gr.State(
                0
            )  # Index into current_start_times_state
            # collected_labels_state = gr.State({}) # No longer needed - state managed by orchestrator

            # --- Layout (largely the same, minor adjustments maybe) ---
            with gr.Row():
                with gr.Column(scale=1):
                    # --- BATCH INPUT GROUP ---
                    with gr.Group() as batch_input_group:
                        gr.Markdown("## 1. Batch Processing Input")
                        gr.Markdown(
                            "Upload an Excel file (.xlsx). The first column should contain YouTube URLs or local file paths."
                        )
                        batch_input_file = gr.File(
                            label="Upload Batch File",
                            type="filepath",
                            file_types=[".xlsx"],
                        )
                        gr.Markdown("### Output Options")
                        # Checkbox values default to config, passed to orchestrator on run
                        include_source_audio_checkbox = gr.Checkbox(
                            label="Include Source Audio in ZIP",
                            value=self.config_data.get("include_source_audio", True),
                        )
                        include_json_summary_checkbox = gr.Checkbox(
                            label="Include Detailed JSON Summary",
                            value=self.config_data.get("include_json_summary", True),
                        )
                        include_csv_summary_checkbox = gr.Checkbox(
                            label="Include High-Level CSV Summary",
                            value=self.config_data.get("include_csv_summary", False),
                        )
                        include_script_transcript_checkbox = gr.Checkbox(
                            label="Include Simple Text Transcript",
                            value=self.config_data.get(
                                "include_script_transcript", False
                            ),
                        )
                        include_plots_checkbox = gr.Checkbox(
                            label="Include Emotion Plots",
                            value=self.config_data.get("include_plots", False),
                        )
                        batch_process_btn = gr.Button(
                            "Start Batch Processing ▶️", variant="primary"
                        )

                with gr.Column(scale=1):
                    # --- STATUS OUTPUT GROUP ---
                    with gr.Group() as status_output_group:
                        gr.Markdown("## 2. Processing Status & Results")
                        batch_status_output = gr.Textbox(
                            label="Status Log",
                            interactive=False,
                            lines=15,
                            max_lines=20,
                        )
                        batch_download_output = gr.Textbox(
                            label="Final Output ZIP Path",
                            interactive=False,
                            lines=1,
                            placeholder="Path to ZIP bundle will appear here...",
                        )

            # --- INTERACTIVE LABELING GROUP (Initially Hidden) ---
            with gr.Column(visible=False) as labeling_ui_group:
                gr.Markdown("## 3. Interactive Speaker Labeling")
                labeling_progress_md = gr.Markdown(
                    "Labeling Speaker: ---"
                )  # Updated dynamically
                with gr.Row():
                    with gr.Column(scale=2):
                        video_player_html = gr.HTML(label="Speaker Preview")
                    with gr.Column(scale=1):
                        gr.Markdown("### Clip Navigation")
                        current_clip_display = gr.Markdown(
                            "Preview 1 of X"
                        )  # Updated dynamically
                        with gr.Row():
                            prev_clip_btn = gr.Button("⬅️ Previous")
                            next_clip_btn = gr.Button("Next ➡️")
                        gr.Markdown("### Enter Label")
                        speaker_label_input = gr.Textbox(
                            label="Enter Speaker Name/Label",
                            placeholder="e.g., Alice (leave blank to keep original ID)",
                        )
                        submit_label_btn = gr.Button(
                            "Submit Label & Next Speaker ▶️", variant="primary"
                        )
                        skip_item_btn = gr.Button("Skip Rest of Item ⏭️", variant="stop")

            # --- Helper Functions for UI Logic ---

            def change_ui_mode(mode):
                """Updates visibility of UI groups based on the current mode."""
                log_info(f"Changing UI mode to: {mode}")
                is_labeling = mode == "labeling"
                is_idle_or_finished = mode in ["idle", "finished", "error"]
                return {
                    labeling_ui_group: gr.update(visible=is_labeling),
                    # Keep batch input visible but disable button during processing/labeling
                    batch_input_group: gr.update(visible=True),
                    batch_process_btn: gr.update(interactive=is_idle_or_finished),
                    # Keep status group always visible
                    status_output_group: gr.update(visible=True),
                }

            # --- UPDATED: Calls self.orchestrator.process_batch ---
            def process_batch_wrapper(
                xlsx_file_obj: Optional[
                    str
                ],  # Gradio File component gives filepath string
                include_audio: bool,
                include_json: bool,
                include_csv: bool,
                include_script: bool,
                include_plots: bool,
            ) -> Generator[Dict, Any, Any]:
                """Wrapper to handle batch processing initiation and UI updates."""
                if xlsx_file_obj is None:
                    yield {
                        batch_status_output: "ERROR: Please upload an Excel file.",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }
                    return

                # Reset states for new batch run
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
                    video_player_html: "",
                    **change_ui_mode("processing"),
                }

                try:
                    # --- CALL ORCHESTRATOR ---
                    status_msg, results_summary, returned_batch_id = (
                        self.orchestrator.process_batch(
                            input_source=xlsx_file_obj,
                            include_source_audio=include_audio,
                            include_json_summary=include_json,
                            include_csv_summary=include_csv,
                            include_script_transcript=include_script,
                            include_plots=include_plots,
                        )
                    )
                    current_status_text = f"{status_msg}\n\n{results_summary}"
                    yield {batch_status_output: current_status_text}

                    if returned_batch_id:
                        # Batch requires labeling, get initial state from orchestrator
                        current_batch_job_id = returned_batch_id
                        log_info(
                            f"Batch [{current_batch_job_id}] requires labeling. Initializing UI."
                        )
                        # Get the ordered list of items to label for this batch
                        items_requiring_labeling = self.orchestrator.labeling_state.get(
                            current_batch_job_id, {}
                        ).get("items_requiring_labeling_order", [])

                        if not items_requiring_labeling:
                            log_error(
                                f"Batch [{current_batch_job_id}] needs labeling, but no items found in state."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{current_status_text}\n\nERROR: Internal state error finding items."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                            return

                        first_item_id = items_requiring_labeling[0]
                        log_info(f"Starting labeling UI with item: {first_item_id}")
                        # --- CALL ORCHESTRATOR ---
                        first_speaker_data = self.orchestrator.start_labeling_item(
                            current_batch_job_id, first_item_id
                        )

                        if first_speaker_data:
                            first_speaker_id, yt_url, first_start_times = (
                                first_speaker_data
                            )
                            # Retrieve eligible speakers list from the state managed by orchestrator
                            eligible_speakers_list = self.orchestrator.labeling_state[
                                current_batch_job_id
                            ][first_item_id].get("eligible_speakers", [])
                            initial_start_time = (
                                first_start_times[0] if first_start_times else 0
                            )
                            clip_count = len(first_start_times)
                            initial_html = get_youtube_embed_html(
                                yt_url, initial_start_time
                            )
                            total_labeling_items = len(items_requiring_labeling)

                            yield {
                                batch_job_id_state: current_batch_job_id,
                                items_to_label_state: items_requiring_labeling,
                                current_item_index_state: 0,
                                current_youtube_url_state: yt_url,
                                eligible_speakers_state: eligible_speakers_list,
                                current_speaker_index_state: 0,
                                current_start_times_state: first_start_times,
                                current_clip_index_state: 0,
                                ui_mode_state: "labeling",
                                labeling_progress_md: f"Labeling Speaker: **{first_speaker_id}** (Speaker 1/{len(eligible_speakers_list)}, Item 1/{total_labeling_items})",
                                current_clip_display: f"Preview 1 of {clip_count}",
                                video_player_html: initial_html,
                                speaker_label_input: gr.update(
                                    value=""
                                ),  # Clear input field
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                **change_ui_mode("labeling"),
                            }
                        else:
                            log_error(
                                f"Failed to get initial speaker data for item {first_item_id} from orchestrator."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{current_status_text}\n\nERROR: Failed to start labeling for {first_item_id}."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    else:
                        # Batch finished without needing labeling
                        log_info(
                            f"Batch processing finished. No interactive labeling required."
                        )
                        final_mode = "finished" if "✅" in status_msg else "error"
                        final_zip_path_str = ""
                        if final_mode == "finished":
                            match = re.search(r"Download ready: (.+\.zip)", status_msg)
                            if match:
                                final_zip_path_str = match.group(1).strip()
                        yield {
                            batch_download_output: final_zip_path_str,
                            ui_mode_state: final_mode,
                            **change_ui_mode(final_mode),
                        }

                except Exception as e:
                    error_trace = traceback.format_exc()
                    log_error(f"Error in process_batch_wrapper: {e}\n{error_trace}")
                    yield {
                        batch_status_output: f"An unexpected error occurred: {e}\n\n{error_trace}",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }

            # Change Clip Buttons (remain the same logic)
            def handle_change_clip(
                direction: int,
                current_clip_idx: int,
                start_times: list,
                youtube_url: str,
            ) -> Dict:
                if not start_times:
                    return {}
                num_clips = len(start_times)
                new_clip_idx = max(0, min(current_clip_idx + direction, num_clips - 1))
                if new_clip_idx == current_clip_idx:
                    return {}  # No change
                new_start_time = start_times[new_clip_idx]
                new_html = get_youtube_embed_html(youtube_url, new_start_time)
                return {
                    current_clip_index_state: new_clip_idx,
                    video_player_html: new_html,
                    current_clip_display: f"Preview {new_clip_idx + 1} of {num_clips}",
                }

            # --- UPDATED: Logic to move to next state after submit/skip ---
            # This function now handles UI updates based on orchestrator calls
            def move_to_next_labeling_state(
                batch_id: str,
                items_to_label: list,
                item_idx: int,  # Index of the item JUST completed/skipped
                current_status: str,  # Current status text to append to
            ) -> Generator[Dict, Any, Any]:
                """Handles UI transitions AFTER an item is finalized (via submit/skip)."""
                log_info(
                    f"[{batch_id}] UI moving state after finalizing item index {item_idx}"
                )

                item_id_finalized = items_to_label[item_idx]
                # yield {batch_status_output: gr.update(value=f"{current_status}\n\nFinalized item {item_id_finalized}. Checking next...")}
                # current_status = batch_status_output.value # Update status text internally

                next_item_idx = item_idx + 1

                if next_item_idx < len(items_to_label):
                    # --- Move UI to the NEXT item ---
                    next_item_id = items_to_label[next_item_idx]
                    log_info(
                        f"[{batch_id}] Moving UI to label next item: {next_item_id} (index {next_item_idx})"
                    )
                    # --- CALL ORCHESTRATOR ---
                    next_speaker_data = self.orchestrator.start_labeling_item(
                        batch_id, next_item_id
                    )

                    if next_speaker_data:
                        next_speaker_id, next_yt_url, next_start_times = (
                            next_speaker_data
                        )
                        # Retrieve state info for the new item from orchestrator
                        next_item_state = self.orchestrator.labeling_state.get(
                            batch_id, {}
                        ).get(next_item_id, {})
                        next_eligible_spkrs = next_item_state.get(
                            "eligible_speakers", []
                        )
                        initial_start_time = (
                            next_start_times[0] if next_start_times else 0
                        )
                        clip_count = len(next_start_times)
                        total_items_count = len(items_to_label)
                        new_html = get_youtube_embed_html(
                            next_yt_url, initial_start_time
                        )

                        yield {
                            current_item_index_state: next_item_idx,
                            current_youtube_url_state: next_yt_url,
                            eligible_speakers_state: next_eligible_spkrs,
                            current_speaker_index_state: 0,
                            current_start_times_state: next_start_times,
                            current_clip_index_state: 0,
                            labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker 1/{len(next_eligible_spkrs)}, Item {next_item_idx + 1}/{total_items_count})",
                            current_clip_display: f"Preview 1 of {clip_count}",
                            video_player_html: new_html,
                            speaker_label_input: gr.update(value=""),
                            prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                            next_clip_btn: gr.update(interactive=(clip_count > 1)),
                            ui_mode_state: "labeling",
                            **change_ui_mode("labeling"),
                        }
                    else:  # Error starting the next item
                        log_error(
                            f"Failed to get initial speaker data for next item {next_item_id}."
                        )
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status}\n\nERROR: Failed start labeling next item {next_item_id}."
                            ),
                            ui_mode_state: "error",
                            **change_ui_mode("error"),
                        }
                else:
                    # --- Finished all items in the batch ---
                    log_info(
                        f"[{batch_id}] UI: Finished labeling/skipping all items. Checking completion..."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status}\n\nLabeling complete for all items. Finalizing batch and creating ZIP..."
                        )
                    }
                    # current_status = batch_status_output.value # Update status

                    # --- CALL ORCHESTRATOR ---
                    final_zip_path = self.orchestrator.check_completion_and_zip(
                        batch_id
                    )
                    final_status_msg = (
                        f"✅ Batch complete. Output: {final_zip_path}"
                        if final_zip_path
                        else f"❗️ Batch complete, but ZIP creation failed."
                    )
                    final_mode = "finished" if final_zip_path else "error"

                    yield {
                        batch_status_output: gr.update(
                            value=f"{batch_status_output.value}\n\n{final_status_msg}"
                        ),  # Use .value to get latest status
                        batch_download_output: str(final_zip_path)
                        if final_zip_path
                        else "",
                        ui_mode_state: final_mode,
                        **change_ui_mode(final_mode),
                    }

            # --- UPDATED: Submit Label Button Handler ---
            def handle_submit_label_wrapper(*args):
                """Handles storing the label and moving to the next speaker or item."""
                (
                    batch_id,
                    items_to_label,
                    item_idx,
                    speakers_to_label,
                    speaker_idx,
                    current_label_input,
                    current_status_text,
                ) = args  # Removed collected_labels_state input

                # Basic state validation
                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                    or not speakers_to_label
                    or speaker_idx >= len(speakers_to_label)
                ):
                    log_error(
                        f"Submit label called with invalid state: {batch_id=}, {item_idx=}, {len(items_to_label)=}, {speaker_idx=}, {len(speakers_to_label)=}"
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: Invalid state during label submission."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                speaker_id = speakers_to_label[speaker_idx]
                log_info(
                    f"[{batch_id}-{item_id}] UI submitting label for {speaker_id}: '{current_label_input}'"
                )

                # --- CALL ORCHESTRATOR to store label ---
                success = self.orchestrator.store_label(
                    batch_id, item_id, speaker_id, current_label_input
                )
                if not success:
                    log_warning(
                        f"Failed to store label for {speaker_id} in orchestrator state."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nWARN: Failed to store label for {speaker_id}."
                        )
                    }
                    # Continue processing even if store fails? Maybe UI should stop? For now, continue.
                    # current_status_text = batch_status_output.value # Update status

                # --- CALL ORCHESTRATOR to get next speaker in *same* item ---
                next_speaker_data = self.orchestrator.get_next_labeling_speaker(
                    batch_id, item_id, speaker_idx
                )

                if next_speaker_data:
                    # --- Still more speakers in the CURRENT item ---
                    next_speaker_id, yt_url, next_start_times = next_speaker_data
                    next_speaker_idx = speaker_idx + 1
                    initial_start_time = next_start_times[0] if next_start_times else 0
                    clip_count = len(next_start_times)
                    total_speakers = len(speakers_to_label)
                    total_items_count = len(items_to_label)
                    new_html = get_youtube_embed_html(yt_url, initial_start_time)

                    yield {  # Update UI for next speaker in same item
                        current_speaker_index_state: next_speaker_idx,
                        current_start_times_state: next_start_times,
                        current_clip_index_state: 0,
                        labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker {next_speaker_idx + 1}/{total_speakers}, Item {item_idx + 1}/{total_items_count})",
                        current_clip_display: f"Preview 1 of {clip_count}",
                        video_player_html: new_html,
                        speaker_label_input: gr.update(value=""),  # Clear input
                        prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                        next_clip_btn: gr.update(interactive=(clip_count > 1)),
                    }
                else:
                    # --- Finished speakers for CURRENT item -> Finalize item & Transition UI ---
                    log_info(
                        f"[{batch_id}-{item_id}] Finished labeling speakers for item {item_id}. Triggering finalization."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nFinalizing item {item_id}..."
                        )
                    }
                    current_status_text = batch_status_output.value  # Update status

                    # --- CALL ORCHESTRATOR to finalize item ---
                    finalization_success = self.orchestrator.finalize_item(
                        batch_id, item_id
                    )
                    if not finalization_success:
                        log_error(
                            f"[{batch_id}-{item_id}] Finalization failed after submitting last label."
                        )
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\nERROR: Failed to finalize item {item_id}. Attempting to proceed..."
                            )
                        }
                        current_status_text = batch_status_output.value  # Update status
                    else:
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\nItem {item_id} finalized successfully."
                            )
                        }
                        current_status_text = batch_status_output.value

                    # Now call the UI transition generator to move to next item or finish batch
                    yield from move_to_next_labeling_state(
                        batch_id, items_to_label, item_idx, current_status_text
                    )

            # --- UPDATED: Skip Item Button Handler ---
            def handle_skip_item_wrapper(*args):
                """Handles skipping remaining speakers and finalizing the item."""
                (batch_id, items_to_label, item_idx, current_status_text) = args

                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                ):
                    log_error(
                        f"Skip item called with invalid state: {batch_id=}, {item_idx=}, {len(items_to_label)=}"
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: Invalid state during skip."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                log_info(
                    f"[{batch_id}-{item_id}] UI skipping rest of speakers for item."
                )
                yield {
                    batch_status_output: gr.update(
                        value=f"{current_status_text}\n\nSkipping & finalizing item {item_id}..."
                    )
                }
                current_status_text = batch_status_output.value

                # --- CALL ORCHESTRATOR to handle skip (which includes finalize) ---
                success = self.orchestrator.skip_item_labeling(batch_id, item_id)

                if not success:
                    log_warning(
                        f"[{batch_id}-{item_id}] Orchestrator finalize/skip returned failure, but attempting UI transition."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nWARN: Finalization during skip failed for item {item_id}."
                        )
                    }
                    current_status_text = batch_status_output.value  # Update status
                else:
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nItem {item_id} skipped and finalized."
                        )
                    }
                    current_status_text = batch_status_output.value

                # Call the common UI transition logic
                yield from move_to_next_labeling_state(
                    batch_id, items_to_label, item_idx, current_status_text
                )

            # --- Event Listeners ---
            # Batch processing button
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
                outputs=[  # List all UI components potentially affected
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
                    batch_process_btn,
                    # State outputs
                    ui_mode_state,
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                ],
            )
            # Previous clip button
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
            # Next clip button
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
            # Submit label button
            submit_label_btn.click(
                fn=handle_submit_label_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    speaker_label_input,
                    batch_status_output,
                ],  # Removed collected_labels_state input
                outputs=[  # List all potentially updated components
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
                    batch_process_btn,
                    # State outputs
                    ui_mode_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                ],
            )
            # Skip item button
            skip_item_btn.click(
                fn=handle_skip_item_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    batch_status_output,
                ],
                outputs=[  # List all potentially updated components
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
                    batch_process_btn,
                    # State outputs
                    ui_mode_state,
                    current_item_index_state,
                    current_youtube_url_state,
                    eligible_speakers_state,
                    current_speaker_index_state,
                    current_start_times_state,
                    current_clip_index_state,
                ],
            )

            return demo

    # --- Launch Method ---
    def launch(self, **kwargs):
        """Creates and launches the Gradio interface."""
        if not GRADIO_AVAILABLE:
            log_error("Cannot launch UI: Gradio or Pandas not available.")
            return

        try:
            log_info("Creating Gradio UI...")
            gradio_app = self.create_ui()
            log_info("Launching Gradio UI...")
            # Pass launch kwargs (e.g., server_name, share)
            gradio_app.launch(**kwargs)
        except Exception as e:
            log_error(f"FATAL ERROR during Gradio UI creation or launch: {e}")
            log_error(traceback.format_exc())
            print(
                f"\nFATAL ERROR: Could not launch Gradio UI. Check logs. Error: {e}\n"
            )
