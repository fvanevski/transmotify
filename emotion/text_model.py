# emotion/text_model.py
"""
Handles loading and running text-based emotion classification models.
"""

from typing import Optional, List, Dict, Any, Union

# Assuming core.logging is available
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    log_info = log_warning = log_error = print

try:
    from transformers.pipelines import pipeline
    from transformers.pipelines.base import Pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    log_error(
        "Transformers library not found. Text emotion analysis will be unavailable."
    )
    TRANSFORMERS_AVAILABLE = False
    # Define dummy Pipeline type hint if import fails
    Pipeline = Any


class TextEmotionModel:
    """
    Loads and runs a transformers pipeline for text emotion classification.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[Union[str, int]] = None,
    ):
        """
        Initializes the text emotion classifier.

        Args:
            model_name: The name of the text classification model on Hugging Face Hub.
            device: The device to run the model on (e.g., "cpu", "cuda", 0, 1).
                    If None, transformers pipeline will attempt auto-detection.
        """
        self.model_name = model_name
        self.device_arg = device  # Store device arg for pipeline
        self.classifier: Optional[Pipeline] = None

        if not TRANSFORMERS_AVAILABLE:
            log_error(
                "Cannot initialize TextEmotionModel: Transformers library not installed."
            )
            return

        log_info(f"Loading text emotion classifier model: {self.model_name}...")
        try:
            # Ensure all scores are returned for detailed analysis later
            # Pass device explicitly if provided, otherwise let pipeline handle it
            pipeline_args = {"model": self.model_name, "return_all_scores": True}
            if self.device_arg is not None:
                pipeline_args["device"] = self.device_arg

            if TRANSFORMERS_AVAILABLE:
                self.classifier = pipeline("text-classification", **pipeline_args)
            else:
                self.classifier = None

            # Check if classifier was successfully initialized before accessing device
            if self.classifier:
                resolved_device = (
                    self.classifier.device
                )  # Get actual device used by pipeline
                log_info(
                    f"Text emotion classifier loaded successfully on device: {resolved_device}."
                )
            # No else needed here as the except block handles initialization failure

        except Exception as e:
            log_error(
                f"Failed to load text emotion classifier '{self.model_name}': {e}"
            )
            self.classifier = None

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyzes the emotion of a given text snippet.

        Args:
            text: The input text string.

        Returns:
            A list of dictionaries, each containing 'label' and 'score' for an emotion,
            e.g., [{'label': 'joy', 'score': 0.9}, ...].
            Returns a default list indicating failure or no text if analysis cannot be performed.
        """
        if not self.classifier:
            log_warning(
                "Text emotion classifier not loaded. Returning 'analysis_failed'."
            )
            return [{"label": "analysis_failed", "score": 1.0}]

        if not text or not isinstance(text, str) or not text.strip():
            # Return a default structure indicating no text was provided
            return [{"label": "no_text", "score": 1.0}]

        try:
            # The pipeline returns a list containing one list of dictionaries
            # e.g., [[{'label': 'joy', 'score': 0.9}, ...]]
            result: List[List[Dict[str, float]]] = self.classifier(text)
            if (
                result
                and isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], list)
            ):
                # Return the inner list of score dictionaries
                return result[0]
            else:
                log_warning(
                    f"Unexpected output format from text classifier for text: '{text[:50]}...'. Result: {result}"
                )
                return [{"label": "analysis_failed", "score": 1.0}]
        except Exception as e:
            log_error(
                f"Error during text emotion analysis for text snippet: '{text[:50]}...': {e}"
            )
            # Return a structure indicating analysis failure
            return [{"label": "analysis_failed", "score": 1.0}]
