# core/emotion_analysis.py
from transformers import pipeline
# ADDED logging imports
from .logging import log_info, log_error

class EmotionAnalysis:
    def __init__(self, config):
        self.config = config
        log_info("Loading text emotion classifier...")
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True # MODIFIED: Ensure all scores are returned
            )
            log_info("Text emotion classifier loaded successfully.")
        except Exception as e:
            log_error(f"Failed to load text emotion classifier: {e}")
            self.emotion_classifier = None


    def analyze_emotion(self, text):
        # MODIFIED: Return the full result (list of score dictionaries)
        if not self.emotion_classifier or not text or not text.strip():
             # Return a default structure indicating no analysis or failure
             return [{"label": "no_text", "score": 1.0}] # Indicate no text was available or classifier not loaded

        try:
            result = self.emotion_classifier(text)
            # result is a list containing one list of dictionaries like [{'label': 'joy', 'score': 0.9}, ...]
            return result[0] # Return the list of score dictionaries
        except Exception as e:
            log_error(f"Error during text emotion analysis for text snippet: '{text[:50]}...': {e}")
            # Return a structure indicating analysis failure
            return [{"label": "analysis_failed", "score": 1.0}]