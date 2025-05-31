import unittest
from pathlib import Path # Not strictly needed for these tests but good practice for test files
import sys
import os

# Adjust path to import from the parent directory's 'asr' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asr.asr import _get_dominant_speaker, _transform_riva_to_whisperx_format

class TestSpeakerLogic(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(_get_dominant_speaker([]), "SPEAKER_00")

    def test_single_speaker(self):
        self.assertEqual(_get_dominant_speaker([1, 1, 1]), "SPEAKER_01")

    def test_multiple_speakers_clear_dominant(self):
        self.assertEqual(_get_dominant_speaker([1, 2, 1, 1, 2]), "SPEAKER_01")

    def test_tie_behavior(self):
        # Counter.most_common(1) picks one in case of a tie.
        # The specific one can depend on insertion order if counts are equal.
        # For [1, 2, 1, 2], it's usually the first one encountered that reached the max count.
        # Let's assume it picks SPEAKER_01 based on typical Counter behavior.
        # If this test is flaky, it means the tie-breaking is not deterministic in this simple form.
        # However, for this specific input, '1' appears first.
        self.assertIn(_get_dominant_speaker([1, 2, 1, 2]), ["SPEAKER_01", "SPEAKER_02"])


    def test_tie_behavior_favors_earlier_in_list_if_counts_equal(self):
        # Counter([1,2,1,2]).most_common(1) -> [(1,2)]
        # Counter([2,1,2,1]).most_common(1) -> [(2,2)]
        self.assertEqual(_get_dominant_speaker([1, 2, 1, 2]), "SPEAKER_01")
        self.assertEqual(_get_dominant_speaker([2, 1, 2, 1]), "SPEAKER_02")


    def test_different_default(self):
        self.assertEqual(_get_dominant_speaker([], default_speaker="UNKNOWN_SPEAKER"), "UNKNOWN_SPEAKER")

    def test_zero_tag_dominant(self):
        self.assertEqual(_get_dominant_speaker([0, 0, 1]), "SPEAKER_00")

    def test_zero_tag_not_dominant(self):
        self.assertEqual(_get_dominant_speaker([0, 1, 1]), "SPEAKER_01")

class TestRivaToWhisperXFormat(unittest.TestCase):
    def test_basic_transformation_single_segment_single_speaker(self):
        riva_response_dict = {
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello world",
                            "words": [
                                {"word": "Hello", "start_time": 100, "end_time": 500, "confidence": 0.95, "speaker_tag": 1},
                                {"word": "world", "start_time": 600, "end_time": 1000, "confidence": 0.92, "speaker_tag": 1}
                            ]
                        }
                    ],
                    "channel_tag": 0, # Example other field
                }
            ]
        }
        expected_output = {
            "segments": [
                {
                    "start": 0.1, "end": 1.0, "text": "Hello world", "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.5, "score": 0.9500, "speaker": "SPEAKER_01"},
                        {"word": "world", "start": 0.6, "end": 1.0, "score": 0.9200, "speaker": "SPEAKER_01"}
                    ]
                }
            ]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_multiple_segments_multiple_speakers_and_mixed_segment(self):
        riva_response_dict = {
            "results": [
                { # Segment 1: Speaker 2 dominant
                    "alternatives": [{
                        "transcript": "This is speaker two mostly.",
                        "words": [
                            {"word": "This", "start_time": 50, "end_time": 300, "confidence": 0.9, "speaker_tag": 2},
                            {"word": "is", "start_time": 350, "end_time": 500, "confidence": 0.9, "speaker_tag": 2},
                            {"word": "speaker", "start_time": 550, "end_time": 1000, "confidence": 0.8, "speaker_tag": 1}, # Minority speaker
                            {"word": "two", "start_time": 1050, "end_time": 1500, "confidence": 0.9, "speaker_tag": 2},
                            {"word": "mostly", "start_time": 1550, "end_time": 2000, "confidence": 0.9, "speaker_tag": 2},
                        ]
                    }]
                },
                { # Segment 2: Speaker 1 dominant
                    "alternatives": [{
                        "transcript": "Speaker one here.",
                        "words": [
                            {"word": "Speaker", "start_time": 2500, "end_time": 3000, "confidence": 0.95, "speaker_tag": 1},
                            {"word": "one", "start_time": 3050, "end_time": 3500, "confidence": 0.94, "speaker_tag": 1},
                            {"word": "here", "start_time": 3550, "end_time": 4000, "confidence": 0.93, "speaker_tag": 1},
                        ]
                    }]
                }
            ]
        }
        expected_output = {
            "segments": [
                {
                    "start": 0.05, "end": 2.0, "text": "This is speaker two mostly.", "speaker": "SPEAKER_02",
                    "words": [
                        {"word": "This", "start": 0.05, "end": 0.3, "score": 0.9000, "speaker": "SPEAKER_02"},
                        {"word": "is", "start": 0.35, "end": 0.5, "score": 0.9000, "speaker": "SPEAKER_02"},
                        {"word": "speaker", "start": 0.55, "end": 1.0, "score": 0.8000, "speaker": "SPEAKER_02"},
                        {"word": "two", "start": 1.05, "end": 1.5, "score": 0.9000, "speaker": "SPEAKER_02"},
                        {"word": "mostly", "start": 1.55, "end": 2.0, "score": 0.9000, "speaker": "SPEAKER_02"},
                    ]
                },
                {
                    "start": 2.5, "end": 4.0, "text": "Speaker one here.", "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Speaker", "start": 2.5, "end": 3.0, "score": 0.9500, "speaker": "SPEAKER_01"},
                        {"word": "one", "start": 3.05, "end": 3.5, "score": 0.9400, "speaker": "SPEAKER_01"},
                        {"word": "here", "start": 3.55, "end": 4.0, "score": 0.9300, "speaker": "SPEAKER_01"},
                    ]
                }
            ]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_empty_riva_results(self):
        riva_response_dict = {"results": []}
        expected_output = {"segments": []}
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_segment_with_no_words(self):
        riva_response_dict = {
            "results": [
                {
                    "alternatives": [{"transcript": "This segment has no words.", "words": []}]
                },
                { # A valid segment to ensure it's not just an empty list due to all segments being invalid
                    "alternatives": [{
                        "transcript": "Valid segment.",
                        "words": [
                            {"word": "Valid", "start_time": 100, "end_time": 200, "confidence": 0.9, "speaker_tag": 1}
                        ]
                    }]
                }
            ]
        }
        expected_output = {
            "segments": [
                {
                    "start": 0.1, "end": 0.2, "text": "Valid segment.", "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Valid", "start": 0.1, "end": 0.2, "score": 0.9000, "speaker": "SPEAKER_01"}
                    ]
                }
            ]
        }
        # The segment with no words should be skipped
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_word_timestamps_and_confidence_precision(self):
        riva_response_dict = {
            "results": [{
                "alternatives": [{
                    "transcript": "Precise timing.",
                    "words": [
                        {"word": "Precise", "start_time": 123, "end_time": 456, "confidence": 0.98765, "speaker_tag": 1},
                        {"word": "timing", "start_time": 500, "end_time": 987, "confidence": 0.12345, "speaker_tag": 1}
                    ]
                }]
            }]
        }
        expected_output = {
            "segments": [{
                "start": 0.123, "end": 0.987, "text": "Precise timing.", "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Precise", "start": 0.123, "end": 0.456, "score": 0.9877, "speaker": "SPEAKER_01"}, # 0.98765 rounds to 0.9877
                    {"word": "timing", "start": 0.500, "end": 0.987, "score": 0.1235, "speaker": "SPEAKER_01"}  # 0.12345 rounds to 0.1235
                ]
            }]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_speaker_tag_zero_handling(self):
        riva_response_dict = {
            "results": [{
                "alternatives": [{
                    "transcript": "Speaker zero test.",
                    "words": [
                        # Speaker 0 should be the dominant one
                        {"word": "Speaker", "start_time": 100, "end_time": 500, "confidence": 0.9, "speaker_tag": 0},
                        {"word": "zero", "start_time": 600, "end_time": 1000, "confidence": 0.9, "speaker_tag": 0},
                        {"word": "test", "start_time": 1100, "end_time": 1500, "confidence": 0.8, "speaker_tag": 1}, # Minority speaker 1
                    ]
                }]
            }]
        }
        expected_output = {
            "segments": [{
                "start": 0.1, "end": 1.5, "text": "Speaker zero test.", "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Speaker", "start": 0.1, "end": 0.5, "score": 0.9000, "speaker": "SPEAKER_00"},
                    {"word": "zero", "start": 0.6, "end": 1.0, "score": 0.9000, "speaker": "SPEAKER_00"},
                    {"word": "test", "start": 1.1, "end": 1.5, "score": 0.8000, "speaker": "SPEAKER_00"},
                ]
            }]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_segment_without_alternatives(self):
        riva_response_dict = {
            "results": [
                {"alternatives": []}, # No alternatives
                { # A valid segment to ensure it's not just an empty list due to all segments being invalid
                    "alternatives": [{
                        "transcript": "Valid segment.",
                        "words": [
                            {"word": "Valid", "start_time": 100, "end_time": 200, "confidence": 0.9, "speaker_tag": 1}
                        ]
                    }]
                }
            ]
        }
        expected_output = {
            "segments": [
                {
                    "start": 0.1, "end": 0.2, "text": "Valid segment.", "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Valid", "start": 0.1, "end": 0.2, "score": 0.9000, "speaker": "SPEAKER_01"}
                    ]
                }
            ]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

    def test_word_without_speaker_tag(self):
        # This case assumes that if diarization is on, speaker_tag should ideally be there.
        # If it can be missing, the get("speaker_tag", 0) handles it by defaulting to 0.
        riva_response_dict = {
            "results": [{
                "alternatives": [{
                    "transcript": "Word no tag",
                    "words": [
                        {"word": "Word", "start_time": 100, "end_time": 500, "confidence": 0.95}, # No speaker_tag
                        {"word": "no", "start_time": 600, "end_time": 800, "confidence": 0.92, "speaker_tag": 1},
                        {"word": "tag", "start_time": 900, "end_time": 1200, "confidence": 0.90}  # No speaker_tag
                    ]
                }]
            }]
        }
        # Expected: Words without speaker_tag will default to speaker_tag 0.
        # If speaker 1 is the only one with a non-zero tag, it becomes dominant.
        # If all default to 0, then SPEAKER_00 is dominant.
        # In this case, [0, 1, 0] -> SPEAKER_00 is dominant because 0 appears twice.
        expected_output = {
            "segments": [{
                "start": 0.1, "end": 1.2, "text": "Word no tag", "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Word", "start": 0.1, "end": 0.5, "score": 0.9500, "speaker": "SPEAKER_00"},
                    {"word": "no", "start": 0.6, "end": 0.8, "score": 0.9200, "speaker": "SPEAKER_00"},
                    {"word": "tag", "start": 0.9, "end": 1.2, "score": 0.9000, "speaker": "SPEAKER_00"}
                ]
            }]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict), expected_output)

        # Test case where a non-zero speaker is dominant even with missing tags
        riva_response_dict_spk1_dominant = {
            "results": [{
                "alternatives": [{
                    "transcript": "Word no tag spk1",
                    "words": [
                        {"word": "Word", "start_time": 100, "end_time": 500, "confidence": 0.95},
                        {"word": "no", "start_time": 600, "end_time": 800, "confidence": 0.92, "speaker_tag": 1},
                        {"word": "tag", "start_time": 900, "end_time": 1200, "confidence": 0.90, "speaker_tag": 1}
                    ]
                }]
            }]
        }
        # Expected: [0, 1, 1] -> SPEAKER_01 is dominant
        expected_output_spk1_dominant = {
            "segments": [{
                "start": 0.1, "end": 1.2, "text": "Word no tag spk1", "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Word", "start": 0.1, "end": 0.5, "score": 0.9500, "speaker": "SPEAKER_01"},
                    {"word": "no", "start": 0.6, "end": 0.8, "score": 0.9200, "speaker": "SPEAKER_01"},
                    {"word": "tag", "start": 0.9, "end": 1.2, "score": 0.9000, "speaker": "SPEAKER_01"}
                ]
            }]
        }
        self.assertEqual(_transform_riva_to_whisperx_format(riva_response_dict_spk1_dominant), expected_output_spk1_dominant)


if __name__ == '__main__':
    unittest.main()
