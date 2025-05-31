# emotion/visual_model.py
"""
Handles loading and running visual-based emotion classification models (e.g., DeepFace).
Analyzes video frames corresponding to speech segments.
"""

import traceback
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any, Union

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

# Attempt imports needed for DeepFace and video handling
try:
    import cv2
    import torch  # DeepFace uses torch backend selection implicitly sometimes
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    log_error(
        "DeepFace or OpenCV (cv2) library not found. Visual emotion analysis will be unavailable."
    )
    DEEPFACE_AVAILABLE = False
    # Define dummy types if import fails
    cv2 = Any
    DeepFace = Any


class VisualEmotionModel:
    """
    Uses DeepFace to analyze dominant emotions in video frames corresponding to
    given time segments.
    """

    def __init__(
        self,
        detector_backend: str = "opencv",
        analysis_frame_rate: int = 1,  # Target FPS for analysis
        device: str = "cpu",  # Device hint ('cpu' or 'cuda')
    ):
        """
        Initializes the Visual Emotion Model wrapper.

        Args:
            detector_backend: Face detector backend for DeepFace
                              (e.g., 'opencv', 'ssd', 'mtcnn', 'retinaface').
            analysis_frame_rate: Target frames per second to analyze.
            device: Target device ('cpu' or 'cuda'). Note: DeepFace device handling
                    can be complex; this acts as a hint.
        """
        self.detector_backend = detector_backend
        self.frame_rate = max(1, analysis_frame_rate)  # Ensure at least 1 FPS
        # DeepFace uses device IDs (-1 for CPU, 0+ for GPU) or backend-specific settings
        self.device_hint = device  # Store the general device hint

        if not DEEPFACE_AVAILABLE:
            log_error(
                "Cannot initialize VisualEmotionModel: Required libraries not installed."
            )
            return

        log_info(
            f"VisualEmotionModel initialized. Backend: '{self.detector_backend}', Target FPS: {self.frame_rate}, Device Hint: '{self.device_hint}'."
        )
        # Note: DeepFace models are often loaded on first use, not necessarily here.

    def _get_deepface_device_param(self) -> Optional[Union[str, int]]:
        """Determine the device parameter format DeepFace might expect."""
        # This is heuristic. DeepFace backends might handle devices differently.
        # Some might expect 'cuda', some 'cuda:0', some just 0.
        # Returning None often lets DeepFace try its default/auto-detection.
        if self.device_hint == "cuda" and torch.cuda.is_available():
            # Returning None might be safer, let DeepFace/backend decide GPU ID?
            # Or try common formats:
            # return "cuda"
            return 0  # Often works for first GPU
        else:
            # return "cpu"
            return -1  # Often works for CPU

    def predict_video_segments(
        self, video_path: Path, segments: List[Dict[str, Any]]
    ) -> Dict[int, Optional[str]]:
        """
        Analyzes dominant visual emotion for time ranges corresponding to input segments.

        Args:
            video_path: Path to the video file.
            segments: A list of segment dictionaries, each requiring 'start' and 'end' keys
                      (in seconds) to define the time range for analysis.

        Returns:
            A dictionary mapping the original index of each input segment to the
            dominant visual emotion label (str) found within its time range, or None
            if no dominant emotion could be determined for that segment.
            Returns empty dict if analysis fails globally.
        """
        segment_visual_emotions: Dict[int, Optional[str]] = {
            i: None for i in range(len(segments))
        }

        if not DEEPFACE_AVAILABLE:
            log_error("DeepFace not available. Skipping visual emotion analysis.")
            return segment_visual_emotions  # Return dict with all None

        if not video_path.is_file():
            log_error(f"Video file not found for visual analysis: {video_path}")
            return segment_visual_emotions  # Return dict with all None

        log_info(
            f"Analyzing visual emotions for {len(segments)} segments in video: {video_path.name}"
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log_error(f"Failed to open video file: {video_path}")
            return segment_visual_emotions  # Return dict with all None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                log_warning(
                    f"Could not determine valid FPS for {video_path.name}. Assuming 30 FPS."
                )
                fps = 30.0  # Assume standard FPS if detection fails

            frame_interval = max(
                1, int(round(fps / self.frame_rate))
            )  # Analyze every Nth frame
            log_info(
                f"Video FPS: {fps:.2f}. Analyzing frames at interval: {frame_interval} (target ~{self.frame_rate} analysis FPS)"
            )

            # Store dominant emotion per timestamp {timestamp_sec: dominant_emotion_str}
            frame_emotions: Dict[float, str] = {}
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Check if it was the end or an error
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # Allow a small tolerance for frame count mismatch
                    if total_frames > 0 and current_pos < total_frames - 2:
                        log_warning(
                            f"Error reading frame {frame_idx} / {int(total_frames)} from video file."
                        )
                    break  # End of video or error

                # Process frame at the desired interval
                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / fps
                    try:
                        # Run DeepFace analysis on the frame
                        analysis_results = DeepFace.analyze(
                            img_path=frame,
                            actions=["emotion"],
                            detector_backend=self.detector_backend,
                            enforce_detection=False,  # Don't fail if no face found
                            prog_bar=False,  # Disable internal progress bar
                            # device = self._get_deepface_device_param() # Pass device hint - careful! might cause issues
                        )

                        # DeepFace returns list of dicts, one per face
                        if (
                            isinstance(analysis_results, list)
                            and len(analysis_results) > 0
                        ):
                            # Find the dominant emotion among all detected faces in this frame
                            emotions_in_frame = [
                                res.get("dominant_emotion")
                                for res in analysis_results
                                if res.get("dominant_emotion")
                            ]
                            if emotions_in_frame:
                                # Simple majority vote for dominant emotion in the frame
                                most_common_emotion = Counter(
                                    emotions_in_frame
                                ).most_common(1)[0][0]
                                frame_emotions[timestamp_sec] = most_common_emotion
                                # log_info(f"  Frame {frame_idx} ({timestamp_sec:.2f}s): Detected emotion - {most_common_emotion}") # Verbose

                    except ValueError as ve:
                        # Catch errors like "Face could not be detected." if enforce_detection=True
                        # log_warning(f"DeepFace analysis skipped for frame {frame_idx} ({timestamp_sec:.2f}s): {ve}")
                        pass  # Ignore frames where analysis fails or no face detected
                    except Exception as e:
                        log_warning(
                            f"DeepFace analysis error at frame {frame_idx} ({timestamp_sec:.2f}s): {e}"
                        )
                        # Avoid logging full traceback for every frame error unless debugging
                        # log_warning(traceback.format_exc())
                        pass  # Continue to next frame

                frame_idx += 1

            log_info(
                f"Finished analyzing {frame_idx} video frames. Found emotions at {len(frame_emotions)} timestamps."
            )

            # --- Assign frame emotions to segments ---
            log_info("Assigning detected visual emotions to segments...")
            for i, seg in enumerate(segments):
                seg_start = seg.get("start")
                seg_end = seg.get("end")

                if seg_start is None or seg_end is None:
                    log_warning(
                        f"Segment {i} missing start/end time. Cannot assign visual emotion."
                    )
                    continue

                # Collect emotions from frames within the segment's time range
                emotions_in_segment_range = [
                    emo
                    for ts, emo in frame_emotions.items()
                    if seg_start <= ts < seg_end
                ]

                if emotions_in_segment_range:
                    # Assign the most frequent dominant emotion found within the segment's time range
                    dominant_seg_emotion = Counter(
                        emotions_in_segment_range
                    ).most_common(1)[0][0]
                    segment_visual_emotions[i] = dominant_seg_emotion
                    # log_info(f"  Segment {i} ({seg_start:.2f}-{seg_end:.2f}s): Assigned visual emotion - {dominant_seg_emotion}") # Verbose
                # else: # Keep as None if no visual emotion found in range
                #    log_info(f"  Segment {i} ({seg_start:.2f}-{seg_end:.2f}s): No visual emotion detected in range.") # Verbose

        except Exception as e:
            log_error(f"Error during video processing loop: {e}")
            log_error(traceback.format_exc())
            # Return potentially partially filled dict in case of error mid-processing
            return segment_visual_emotions
        finally:
            if cap and cap.isOpened():
                cap.release()
                # log_info("Video capture released.") # Verbose

        log_info("Finished assigning visual emotions to segments.")
        return segment_visual_emotions
