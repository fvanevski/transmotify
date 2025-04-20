# core/constants.py

from typing import Dict

# Emotion‑to‑value map
EMO_VAL: Dict[str, float] = {
    "joy": 1.0,
    "love": 0.8,
    "surprise": 0.5,
    "neutral": 0.0,
    "fear": -1.5,
    "sadness": -1.0,
    "disgust": -1.8,
    "anger": -2.0,
    "unknown": 0.0,
    "analysis_skipped": 0.0,
    "analysis_failed": 0.0,
    "no_text": 0.0,
}

# Pipeline filenames & suffixes
LOG_FILE_NAME: str = "process_log.txt"
# MODIFIED: Renamed for clarity
INTERMEDIATE_STRUCTURED_TRANSCRIPT_NAME: str = "structured_transcript_intermediate.json"
FINAL_STRUCTURED_TRANSCRIPT_NAME: str = "structured_transcript_final.json"
EMOTION_SUMMARY_CSV_NAME: str = "emotion_summary.csv"
EMOTION_SUMMARY_JSON_NAME: str = "emotion_summary.json"
SCRIPT_TRANSCRIPT_NAME: str = "script_transcript.txt" # ADDED: Constant for script transcript
FINAL_ZIP_SUFFIX: str = "_final_bundle.zip"

# Default fuzzy‑match threshold for snippet mapping
DEFAULT_SNIPPET_MATCH_THRESHOLD: float = 0.80