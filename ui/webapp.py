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
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Generator, Union, TYPE_CHECKING

# Import logging module for module-level logger
import logging
logger = logging.getLogger(__name__)

# Imports for type checking only (avoids circular imports)
if TYPE_CHECKING:
    import gradio as gr
    from gradio import themes
    from core.orchestrator import Orchestrator

# Import custom exceptions
from core.errors import TransmotifyError, ResourceAccessError, ProcessingError

# Gradio and Pandas are core dependencies for the UI
# Declare GRADIO_AVAILABLE and component placeholders outside the try block
GRADIO_AVAILABLE = False
gr = themes = pd = Any  # type: ignore # Use Any as placeholders initially

try:
    import gradio as gr  # type: ignore
    from gradio import themes  # type: ignore
    import pandas as pd  # type: ignore

    GRADIO_AVAILABLE = True
except ImportError as e:
    logger.error("Gradio or Pandas library not found. UI cannot be launched.", exc_info=True)
    # UI cannot function without Gradio, so exiting or raising might be appropriate
    # depending on how main.py handles this. For now, the flag is set.

# Import the backend orchestrator
try:
    # Make Orchestrator import essential. If it fails, the app likely can't run.
    from core.orchestrator import Orchestrator
except ImportError as e:
    logger.exception("Failed to import core modules (Orchestrator). Ensure PYTHONPATH is correct.")
    raise ResourceAccessError("Critical component Orchestrator failed to import.") from e


# --- Helper to generate YouTube embed HTML ---
# Moved from legacy ui/main_gui.py
def get_youtube_embed_html(youtube_url: str, start_time_seconds: int = 0) -> str:
    """Creates HTML for embedding a YouTube video starting at a specific time."""
    if (
        not youtube_url
        or not isinstance(youtube_url, str)
        or not youtube_url.startswith("http")
    ):
        logger.warning(f"Invalid YouTube URL for embed: {youtube_url}")
        return "<p>Invalid YouTube URL</p>"

    video_id = None
    # Handle common YouTube URL formats (including potential youtube.com/watch?v=... )
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",  # Standard watch URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})",  # Shortened URL
        r"embed/([a-zA-Z0-9_-]{11})",  # Embed URL
        # Added more robust handling for shorts URLs
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        # User content URL pattern might be less common or stable
        # r"googleusercontent\.com/youtube\.com/\d+/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        logger.warning(f"Could not extract Video ID from URL: {youtube_url}")
        return f"<p>Could not extract Video ID from URL: {youtube_url}</p>"  # Show URL in error

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


# --- Main UI Class ---
class UI:
    """Manages the Gradio interface and its interactions with the Orchestrator."""

    # --- MODIFIED: Accepts Orchestrator instance ---
    def __init__(self, orchestrator: "Orchestrator"):
        """
        Initializes the UI.

        Args:
           orchestrator: An instance of the core.orchestrator.Orchestrator class.
        """
        if not GRADIO_AVAILABLE:
            # This check prevents initialization if Gradio isn't installed.
            raise ResourceAccessError("Gradio or Pandas not found. UI cannot be initialized.")

        # Type check orchestrator if possible (will be Any if import failed, but useful if it succeeded)
        if not isinstance(orchestrator, Orchestrator) and Orchestrator is not Any:
            raise TypeError("Orchestrator instance is required.")

        self.orchestrator = orchestrator
        # Access config via the orchestrator's config object's get method for safety
        self.config_data = (
            orchestrator.config.config
        )  # Get the raw config dict if needed
        logger.info("UI Initialized with Orchestrator instance.")

    def create_ui(self) -> "gr.Blocks":
        """Creates the Gradio Blocks interface."""
        # Ensure Gradio components are available before using them
        if not GRADIO_AVAILABLE or not gr or not themes:
            raise ResourceAccessError("Gradio components unavailable for UI creation.")

        default_theme = themes.Default()

        with gr.Blocks(theme=default_theme) as demo:
            gr.Markdown("# Speech Analysis Pipeline")

            # --- UI States ---
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

            # --- Layout ---
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
                        # Use orchestrator.config.get() for robust access to config values
                        include_source_audio_checkbox = gr.Checkbox(
                            label="Include Source Audio in ZIP",
                            value=self.orchestrator.config.get(
                                "include_source_audio", True
                            ),
                        )
                        include_json_summary_checkbox = gr.Checkbox(
                            label="Include Detailed JSON Summary",
                            value=self.orchestrator.config.get(
                                "include_json_summary", True
                            ),
                        )
                        include_csv_summary_checkbox = gr.Checkbox(
                            label="Include High-Level CSV Summary",
                            value=self.orchestrator.config.get(
                                "include_csv_summary", False
                            ),
                        )
                        include_script_transcript_checkbox = gr.Checkbox(
                            label="Include Simple Text Transcript",
                            value=self.orchestrator.config.get(
                                "include_script_transcript", False
                            ),
                        )
                        include_plots_checkbox = gr.Checkbox(
                            label="Include Emotion Plots",
                            value=self.orchestrator.config.get("include_plots", False),
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
                labeling_progress_md = gr.Markdown("Labeling Speaker: ---")
                with gr.Row():
                    with gr.Column(scale=2):
                        video_player_html = gr.HTML(label="Speaker Preview")
                    with gr.Column(scale=1):
                        gr.Markdown("### Clip Navigation")
                        current_clip_display = gr.Markdown("Preview 1 of X")
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

            def change_ui_mode(mode: str) -> Dict["gr.UIComponent", Dict[Any, Any]]:
                """Updates visibility of UI groups based on the current mode."""
                logger.info(f"Changing UI mode to: {mode}")
                is_labeling = mode == "labeling"
                is_idle_or_finished = mode in ["idle", "finished", "error"]
                return {
                    labeling_ui_group: gr.update(visible=is_labeling),
                    batch_input_group: gr.update(visible=True),  # Always visible
                    batch_process_btn: gr.update(interactive=is_idle_or_finished),
                    status_output_group: gr.update(visible=True),  # Always visible
                }

            def process_batch_wrapper(
                xlsx_file_path: Optional[
                    str
                ],  # Gradio File component gives filepath string
                include_audio: bool,
                include_json: bool,
                include_csv: bool,
                include_script: bool,
                include_plots: bool,
            ) -> Generator[Dict, Any, Any]:
                """Wrapper to handle batch processing initiation and UI updates."""
                if not xlsx_file_path:  # Check if path is None or empty
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
                            input_source=xlsx_file_path,  # Pass the path string
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
                        # Batch requires labeling
                        current_batch_job_id = returned_batch_id
                        logger.info(
                            f"Batch [{current_batch_job_id}] requires labeling. Initializing UI."
                        )
                        # Get the ordered list of items directly from orchestrator state
                        batch_state = self.orchestrator.labeling_state.get(
                            current_batch_job_id, {}
                        )
                        items_requiring_labeling = batch_state.get(
                            "items_requiring_labeling_order", []
                        )

                        if not items_requiring_labeling:
                            logger.error(
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
                        logger.info(f"Starting labeling UI with item: {first_item_id}")

                        # --- CALL ORCHESTRATOR to get first speaker data ---
                        first_speaker_data = self.orchestrator.start_labeling_item(
                            current_batch_job_id, first_item_id
                        )

                        if first_speaker_data:
                            first_speaker_id, yt_url, first_start_times = (
                                first_speaker_data
                            )
                            # Get eligible speakers for this specific item
                            item_state = batch_state.get(first_item_id, {})
                            eligible_speakers_list = item_state.get(
                                "eligible_speakers", []
                            )

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
                                speaker_label_input: gr.update(value=""),  # Clear input
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                **change_ui_mode("labeling"),
                            }
                        else:
                            logger.error(
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
                        logger.info(
                            "Batch processing finished. No interactive labeling required."
                        )
                        # Determine final status based on status message content
                        final_mode = (
                            "finished"
                            if "✅" in status_msg or "Batch complete" in status_msg
                            else "error"
                        )
                        final_zip_path_str = ""
                        # Try to extract zip path if finished successfully
                        if final_mode == "finished":
                            # Make regex more general for different success messages
                            match = re.search(
                                r"(?:Output|Download):\s*(.+\.zip)", status_msg
                            )
                            if match:
                                final_zip_path_str = match.group(1).strip()
                            else:
                                logger.warning(
                                    f"Could not extract ZIP path from success message: {status_msg}"
                                )

                        yield {
                            batch_download_output: final_zip_path_str,
                            ui_mode_state: final_mode,
                            **change_ui_mode(final_mode),
                        }

                except TransmotifyError as e:
                    logger.exception(f"TransmotifyError in process_batch_wrapper: {e}")
                    yield {
                        batch_status_output: f"Pipeline error: {e}",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }
                except Exception as e:
                    logger.exception(f"Unexpected error in process_batch_wrapper: {e}")
                    yield {
                        batch_status_output: f"An unexpected error occurred during batch processing: {e}",
                        ui_mode_state: "error",
                        **change_ui_mode("error"),
                    }

            def handle_change_clip(
                direction: int,
                current_clip_idx: int,
                start_times: list,
                youtube_url: str,
            ) -> Dict[str, Any]:  # Return type hint for clarity
                """Handles changing the preview clip."""
                if not start_times:
                    return {}  # No clips to change
                num_clips = len(start_times)
                # Basic bounds check
                if num_clips <= 1:
                    return {}  # Only one clip, nothing to change

                new_clip_idx = (
                    current_clip_idx + direction
                ) % num_clips  # Use modulo for wrapping
                # Simple calculation without complex min/max:
                # new_clip_idx = max(0, min(current_clip_idx + direction, num_clips - 1))

                # Avoid update if index didn't actually change (e.g., at boundaries with min/max)
                # if new_clip_idx == current_clip_idx:
                #     return {}

                new_start_time = start_times[new_clip_idx]
                new_html = get_youtube_embed_html(youtube_url, new_start_time)
                return {
                    current_clip_index_state: new_clip_idx,
                    video_player_html: new_html,
                    current_clip_display: f"Preview {new_clip_idx + 1} of {num_clips}",
                }

            def move_to_next_labeling_state(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,  # Index of the item JUST completed/skipped
                current_status_text: str,  # Pass the current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles UI transitions AFTER an item is finalized."""
                logger.info(
                    f"[{batch_id}] UI moving state after finalizing item index {item_idx}"
                )

                next_item_idx = item_idx + 1

                if next_item_idx < len(items_to_label):
                    # --- Move UI to the NEXT item ---
                    next_item_id = items_to_label[next_item_idx]
                    logger.info(
                        f"[{batch_id}] Moving UI to label next item: {next_item_id} (index {next_item_idx})"
                    )

                    try:
                        # --- CALL ORCHESTRATOR to get data for the next item ---
                        next_speaker_data = self.orchestrator.start_labeling_item(
                            batch_id, next_item_id
                        )

                        if next_speaker_data:
                            next_speaker_id, next_yt_url, next_start_times = (
                                next_speaker_data
                            )
                            # Retrieve state info for the new item from orchestrator state
                            batch_state = self.orchestrator.labeling_state.get(batch_id, {})
                            next_item_state = batch_state.get(next_item_id, {})
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
                                current_speaker_index_state: 0,  # Reset speaker index for new item
                                current_start_times_state: next_start_times,
                                current_clip_index_state: 0,  # Reset clip index
                                labeling_progress_md: f"Labeling Speaker: **{next_speaker_id}** (Speaker 1/{len(next_eligible_spkrs)}, Item {next_item_idx + 1}/{total_items_count})",
                                current_clip_display: f"Preview 1 of {clip_count}",
                                video_player_html: new_html,
                                speaker_label_input: gr.update(value=""),  # Clear input
                                prev_clip_btn: gr.update(interactive=(clip_count > 1)),
                                next_clip_btn: gr.update(interactive=(clip_count > 1)),
                                ui_mode_state: "labeling",  # Ensure UI stays in labeling mode
                                batch_status_output: current_status_text,  # Pass status through
                                **change_ui_mode("labeling"),
                            }
                        else:
                            # Error starting the next item
                            logger.error(
                                f"Failed to get initial speaker data for next item {next_item_id}."
                            )
                            yield {
                                batch_status_output: gr.update(
                                    value=f"{current_status_text}\n\nERROR: Failed start labeling next item {next_item_id}."
                                ),
                                ui_mode_state: "error",
                                **change_ui_mode("error"),
                            }
                    except TransmotifyError as e:
                        logger.exception(f"Error preparing next item: {e}")
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\nERROR: {e}"
                            ),
                            ui_mode_state: "error",
                            **change_ui_mode("error"),
                        }
                else:
                    # --- Finished all items in the batch ---
                    logger.info(
                        f"[{batch_id}] UI: Finished labeling/skipping all items. Checking completion..."
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nLabeling complete for all items. Finalizing batch and creating ZIP..."
                        )
                    }
                    # Update internal status text (not strictly necessary if only yielding once more)
                    current_status_text = f"{current_status_text}\n\nLabeling complete for all items. Finalizing batch and creating ZIP..."

                    try:
                        # --- CALL ORCHESTRATOR to check completion and create ZIP ---
                        final_zip_path = self.orchestrator.check_completion_and_zip(
                            batch_id
                        )

                        # --- CORRECTED: Redundant assignment removed ---
                        final_status_msg = (
                            f"✅ Batch complete. Output: {final_zip_path}"
                            if final_zip_path
                            else "❗️ Batch complete, but ZIP creation failed."
                        )
                        final_mode = "finished" if final_zip_path else "error"

                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\n{final_status_msg}"
                            ),  # Append final message
                            batch_download_output: (
                                str(final_zip_path) if final_zip_path else ""
                            ),
                            ui_mode_state: final_mode,
                            **change_ui_mode(final_mode),
                        }
                    except TransmotifyError as e:
                        logger.exception(f"Error finalizing batch: {e}")
                        yield {
                            batch_status_output: gr.update(
                                value=f"{current_status_text}\n\nERROR finalizing batch: {e}"
                            ),
                            ui_mode_state: "error",
                            **change_ui_mode("error"),
                        }

            def handle_submit_label_wrapper(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,
                speakers_to_label: List[str],
                speaker_idx: int,
                current_label_input: str,
                current_status_text: str,  # Pass current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles storing the label and moving to the next speaker or item."""
                # Basic state validation
                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                    or not speakers_to_label
                    or speaker_idx >= len(speakers_to_label)
                ):
                    logger.error(
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
                logger.info(
                    f"[{batch_id}-{item_id}] UI submitting label for {speaker_id}: '{current_label_input}'"
                )

                try:
                    # --- CALL ORCHESTRATOR to store label ---
                    success = self.orchestrator.store_label(
                        batch_id, item_id, speaker_id, current_label_input
                    )
                    if not success:
                        logger.warning(
                            f"Failed to store label for {speaker_id} in orchestrator state."
                        )
                        # Update status but continue
                        current_status_text = f"{current_status_text}\n\nWARN: Failed to store label for {speaker_id}."
                        yield {batch_status_output: gr.update(value=current_status_text)}
                        # Proceed to next speaker anyway? Or halt? Current logic proceeds.

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
                            batch_status_output: current_status_text,  # Pass status through
                        }
                    else:
                        # --- Finished speakers for CURRENT item -> Finalize item & Transition UI ---
                        logger.info(
                            f"[{batch_id}-{item_id}] Finished labeling speakers for item {item_id}. Triggering finalization."
                        )
                        current_status_text = (
                            f"{current_status_text}\n\nFinalizing item {item_id}..."
                        )
                        yield {batch_status_output: gr.update(value=current_status_text)}

                        # --- CALL ORCHESTRATOR to finalize item ---
                        finalization_success = self.orchestrator.finalize_item(
                            batch_id, item_id
                        )

                        if not finalization_success:
                            logger.error(
                                f"[{batch_id}-{item_id}] Finalization failed after submitting last label."
                            )
                            current_status_text = f"{current_status_text}\n\nERROR: Failed to finalize item {item_id}. Attempting to proceed..."
                            yield {
                                batch_status_output: gr.update(value=current_status_text)
                            }
                        else:
                            current_status_text = f"{current_status_text}\n\nItem {item_id} finalized successfully."
                            yield {
                                batch_status_output: gr.update(value=current_status_text)
                            }

                        # Now call the UI transition generator to move to next item or finish batch
                        yield from move_to_next_labeling_state(
                            batch_id, items_to_label, item_idx, current_status_text
                        )
                except TransmotifyError as e:
                    logger.exception(f"Error during label submission: {e}")
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: {e}"
                        ),
                    }

            def handle_skip_item_wrapper(
                batch_id: str,
                items_to_label: List[str],
                item_idx: int,
                current_status_text: str,  # Pass current status text
            ) -> Generator[Dict, Any, Any]:
                """Handles skipping remaining speakers and finalizing the item."""
                if (
                    not batch_id
                    or not items_to_label
                    or item_idx >= len(items_to_label)
                ):
                    logger.error(
                        f"Skip item called with invalid state: {batch_id=}, {item_idx=}, {len(items_to_label)=}"
                    )
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: Invalid state during skip."
                        )
                    }
                    return

                item_id = items_to_label[item_idx]
                logger.info(
                    f"[{batch_id}-{item_id}] UI skipping rest of speakers for item."
                )
                current_status_text = (
                    f"{current_status_text}\n\nSkipping & finalizing item {item_id}..."
                )
                yield {batch_status_output: gr.update(value=current_status_text)}

                try:
                    # --- CALL ORCHESTRATOR to handle skip (which includes finalize) ---
                    success = self.orchestrator.skip_item_labeling(batch_id, item_id)

                    if not success:
                        logger.warning(
                            f"[{batch_id}-{item_id}] Orchestrator finalize/skip returned failure, but attempting UI transition."
                        )
                        current_status_text = f"{current_status_text}\n\nWARN: Finalization during skip failed for item {item_id}."
                        yield {batch_status_output: gr.update(value=current_status_text)}
                    else:
                        current_status_text = f"{current_status_text}\n\nItem {item_id} skipped and finalized."
                        yield {batch_status_output: gr.update(value=current_status_text)}

                    # Call the common UI transition logic
                    yield from move_to_next_labeling_state(
                        batch_id, items_to_label, item_idx, current_status_text
                    )
                except TransmotifyError as e:
                    logger.exception(f"Error skipping item: {e}")
                    yield {
                        batch_status_output: gr.update(
                            value=f"{current_status_text}\n\nERROR: {e}"
                        ),
                    }

            # --- Event Listeners ---
            # Define outputs list once for state components often updated together
            state_outputs_labeling = [
                ui_mode_state,
                batch_job_id_state,  # Only set initially
                items_to_label_state,
                current_item_index_state,
                current_youtube_url_state,
                eligible_speakers_state,
                current_speaker_index_state,
                current_start_times_state,
                current_clip_index_state,
            ]
            ui_outputs_labeling = [
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
            ]

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
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Combine UI and state outputs
            )

            # Previous clip button
            prev_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(-1),  # Pass direction implicitly
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[  # Only updates these specific UI elements
                    current_clip_index_state,
                    video_player_html,
                    current_clip_display,
                ],
            )

            # Next clip button
            next_clip_btn.click(
                fn=handle_change_clip,
                inputs=[
                    gr.State(1),  # Pass direction implicitly
                    current_clip_index_state,
                    current_start_times_state,
                    current_youtube_url_state,
                ],
                outputs=[  # Only updates these specific UI elements
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
                    batch_status_output,  # Pass current status text
                ],
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Updates UI and state
            )

            # Skip item button
            skip_item_btn.click(
                fn=handle_skip_item_wrapper,
                inputs=[
                    batch_job_id_state,
                    items_to_label_state,
                    current_item_index_state,
                    batch_status_output,  # Pass current status text
                ],
                outputs=ui_outputs_labeling
                + state_outputs_labeling,  # Updates UI and state
            )

            return demo

    # --- Launch Method ---
    def launch(self, **kwargs):
        """Creates and launches the Gradio interface."""
        if not GRADIO_AVAILABLE:
            logger.error("Cannot launch UI: Gradio or Pandas not available.")
            raise ResourceAccessError("Gradio or Pandas not found, UI cannot be launched.")

        try:
            logger.info("Creating Gradio UI...")
            gradio_app = self.create_ui()
            logger.info("Launching Gradio UI...")
            # Pass launch kwargs (e.g., server_name, share)
            gradio_app.launch(**kwargs)
        except Exception as e:
            # Catch errors specifically during UI creation or launch
            logger.exception(f"FATAL ERROR during Gradio UI creation or launch: {e}")
            print(
                f"\nFATAL ERROR: Could not launch Gradio UI. Check logs. Error: {e}\n"
            )
            # Re-raise the exception so the main script knows launch failed
            raise


# Entry point for direct execution
def main():
    try:
        # Example of a top-level "boundary" to handle uncaught exceptions gracefully
        from core.orchestrator import Orchestrator
        from core.config import Config
        
        config = Config()
        orchestrator = Orchestrator(config)
        ui = UI(orchestrator)
        ui.launch(server_name="127.0.0.1", server_port=7860)
    except TransmotifyError as err:
        logger.error("Application failed: %s", err, exc_info=True)
        sys.exit(1)
    except Exception as err:
        logger.exception("Unexpected error occurred: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
