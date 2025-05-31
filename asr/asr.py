# asr/asr.py
"""
 asr/asr.py – Riva Integration with WhisperX-like Output
 ---------------------------------------------------------
 Replaces the previous WhisperX implementation with Riva ASR and
 transforms Riva's output to a WhisperX-compatible JSON structure.

 Key points ▸
 • Uses the Riva Python client API, now calling the gRPC stub directly for timeout control.
 • Transforms raw Riva ASR output to the specified JSON format (segments, words, speaker tags).
 • Parameters like riva_server_uri, language_code, max_speakers_diarization are used.
 • Detailed logging for Riva ASR processing.
 • Graceful error handling for gRPC and other exceptions.
 """

from __future__ import annotations

import gc
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import grpc  # type: ignore – external dependency
import riva.client  # type: ignore – external dependency
import riva.client.proto.riva_asr_pb2 as rasr  # type: ignore – external dependency
from google.protobuf import json_format # type: ignore – external dependency

logger = logging.getLogger(__name__)

__all__ = ["run_riva_asr"]


def _get_dominant_speaker(speaker_tags: List[int], default_speaker: str = "SPEAKER_00") -> str:
    """Determines the most frequent speaker tag from a list.

    Args:
        speaker_tags: A list of integer speaker tags from word alignments.
        default_speaker: The speaker label to return if no tags are available or in case of a tie with no clear majority.

    Returns:
        A string representing the dominant speaker (e.g., "SPEAKER_01").
    """
    if not speaker_tags:
        return default_speaker

    counts = Counter(speaker_tags)
    dominant_tag = counts.most_common(1)[0][0]
    return f"SPEAKER_{dominant_tag:02d}"


def _transform_riva_to_whisperx_format(riva_response_dict: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Transforms Riva ASR output dictionary to a WhisperX-like JSON structure.

    Args:
        riva_response_dict: The dictionary obtained from json_format.MessageToDict(riva_response).
                           Assumes 'start_time' and 'end_time' for words are in milliseconds.
                           Assumes 'speaker_tag' is present for words if diarization was enabled.

    Returns:
        A dictionary with a single key "segments" pointing to a list of transformed segment objects.
    """
    transformed_segments = []
    riva_results = riva_response_dict.get("results", [])

    for riva_segment_result in riva_results:
        if not riva_segment_result.get("alternatives"):
            continue
        alternative = riva_segment_result["alternatives"][0]
        if not alternative.get("words"):
            logger.debug(f"[ASR Riva Transform] Skipping segment with no words: {alternative.get('transcript', 'N/A')}")
            continue

        transformed_words_for_segment = []
        current_segment_speaker_tags = []
        segment_text = alternative.get("transcript", "")

        for riva_word_info in alternative["words"]:
            start_time_ms = riva_word_info.get("start_time", 0)
            end_time_ms = riva_word_info.get("end_time", 0)
            word_start_sec = start_time_ms / 1000.0
            word_end_sec = end_time_ms / 1000.0
            confidence = riva_word_info.get("confidence", 0.0)
            word_text = riva_word_info.get("word", "")
            speaker_tag = riva_word_info.get("speaker_tag", 0)
            current_segment_speaker_tags.append(speaker_tag)

            transformed_words_for_segment.append({
                "word": word_text,
                "start": round(word_start_sec, 3),
                "end": round(word_end_sec, 3),
                "score": round(confidence, 4),
            })

        if not transformed_words_for_segment:
            logger.debug(f"[ASR Riva Transform] Segment skipped due to no transformed words: {segment_text}")
            continue

        segment_start_sec = transformed_words_for_segment[0]["start"]
        segment_end_sec = transformed_words_for_segment[-1]["end"]
        segment_speaker_str = _get_dominant_speaker(current_segment_speaker_tags)

        for word_data in transformed_words_for_segment:
            word_data["speaker"] = segment_speaker_str

        transformed_segments.append({
            "start": segment_start_sec,
            "end": segment_end_sec,
            "text": segment_text,
            "speaker": segment_speaker_str,
            "words": transformed_words_for_segment,
        })
        logger.debug(f"[ASR Riva Transform] Transformed segment: Speaker {segment_speaker_str}, Text: {segment_text[:50]}...")

    return {"segments": transformed_segments}


def run_riva_asr(
    # Input / output
    audio_path: Path,
    output_dir: Path,
    *,
    # Riva specific params
    riva_server_uri: str = "localhost:50051",
    language_code: str = "en-US",
    max_speakers_diarization: Optional[int] = 5,
    enable_automatic_punctuation: bool = True,
    riva_request_timeout: Optional[float] = 240, # Timeout parameter in seconds
    # Output discovery compatibility
    output_filename_exclusions: Optional[List[str]] = None,
    # Logging
    log_file_handle: Optional[TextIO] = None,
    log_prefix: str = "[ASR Riva]",
) -> Path:
    """Run Riva ASR, transform output to WhisperX-like format, and return the path of the resulting JSON.
    Uses direct gRPC stub call to enable request timeout.
    """
    t0 = time.perf_counter()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{audio_path.stem}.json"

    if log_file_handle:
        file_handler = logging.StreamHandler(log_file_handle)
        logger.addHandler(file_handler)

    logger.info(f"{log_prefix} Connecting to Riva server at {riva_server_uri} …")
    try:
        auth = riva.client.Auth(uri=riva_server_uri)
        asr_service = riva.client.ASRService(auth)

        logger.info(f"{log_prefix} Loading audio from: {audio_path}")
        with audio_path.open('rb') as fh:
            data = fh.read()

        audio_file_size_bytes = len(data)
        audio_file_size_mb = audio_file_size_bytes / (1024 * 1024)
        size_threshold_mb = 100
        if audio_file_size_mb > size_threshold_mb:
            timeout_msg_part = (
                f"{riva_request_timeout}s"
                if (riva_request_timeout is not None and riva_request_timeout > 0)
                else "server default"
            )
            logger.warning(
                f"{log_prefix} Input audio file {audio_path.name} is large ({audio_file_size_mb:.2f} MB). "
                f"Processing may take a significant time. Client timeout is set to {timeout_msg_part}."
            )

        # RecognitionConfig is a protobuf message.
        # The riva.client.RecognitionConfig helper populates this protobuf message.
        config = riva.client.RecognitionConfig(
            language_code=language_code,
            max_alternatives=1,
            enable_automatic_punctuation=enable_automatic_punctuation,
            enable_word_time_offsets=True,
            # encoding and sample_rate_hertz are typically set by the Riva server or defaults,
            # but can be specified if known and fixed, e.g., AudioEncoding.LINEAR_PCM and 16000 Hz
        )

        if max_speakers_diarization is not None and max_speakers_diarization > 0:
            logger.info(f"{log_prefix} Enabling speaker diarization with max_speakers={max_speakers_diarization}")
            # add_speaker_diarization_to_config modifies the config protobuf message in place
            riva.client.add_speaker_diarization_to_config(
                config, diarization_enable=True, diarization_max_speakers=max_speakers_diarization
            )
        else:
            logger.info(f"{log_prefix} Speaker diarization not enabled or max_speakers is invalid.")

        logger.info(f"{log_prefix} Recognition config prepared. Language: {language_code}")
        logger.info(f"{log_prefix} Sending ASR request to Riva server for {audio_path.name}…")

        # Construct the RecognizeRequest protobuf message
        request = rasr.RecognizeRequest(audio=data, config=config)

        actual_timeout_for_call = None
        if riva_request_timeout is not None and riva_request_timeout > 0:
            logger.info(f"{log_prefix} Using request timeout of {riva_request_timeout} seconds for gRPC call.")
            actual_timeout_for_call = riva_request_timeout
        else:
            logger.info(f"{log_prefix} No positive request timeout specified, gRPC call will use its default timeout.")

        # Call the gRPC stub directly to use the timeout parameter
        response: rasr.RecognizeResponse = asr_service.stub.Recognize(
            request,
            metadata=auth.get_auth_metadata(), # Get authentication metadata
            timeout=actual_timeout_for_call    # Pass the timeout in seconds
        )

        if not response.results:
            logger.warning(f"{log_prefix} Received empty ASR results for {audio_path.name}.")
            transformed_data = {"segments": []}
        else:
            logger.info(f"{log_prefix} Transcription complete. Received {len(response.results)} result(s) for {audio_path.name}.")
            response_dict = json_format.MessageToDict(
                response,
                preserving_proto_field_name=True
            )
            logger.debug(f"{log_prefix} Raw Riva response (dict): {str(response_dict)[:500]}...")
            logger.info(f"{log_prefix} Transforming Riva output to WhisperX format…")
            transformed_data = _transform_riva_to_whisperx_format(response_dict)

        with output_json.open("w", encoding="utf‑8") as fp:
            json.dump(transformed_data, fp, ensure_ascii=False, indent=2)
        logger.info(f"{log_prefix} Saved WhisperX-formatted transcript → {output_json.relative_to(output_dir.parent)}")

    except grpc.RpcError as e:
        err_code = e.code()
        err_details = e.details()
        logger.error(f"{log_prefix} gRPC Error processing {audio_path.name}: {err_code} - {err_details}")
        if err_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            logger.error(
                f"{log_prefix} The ASR request for {audio_path.name} timed out. "
                f"Configured timeout was {actual_timeout_for_call if actual_timeout_for_call is not None else 'gRPC default'}s. "
                f"Consider increasing 'riva_request_timeout'."
            )
        raise RuntimeError(f"Riva ASR request failed for {audio_path.name} ({err_code}): {err_details}") from e
    except Exception as e:
        logger.error(f"{log_prefix} An unexpected error occurred processing {audio_path.name}: {type(e).__name__} - {e}")
        raise RuntimeError(f"An unexpected error occurred during Riva ASR processing of {audio_path.name}: {e}") from e
    finally:
        gc.collect()
        if log_file_handle and 'file_handler' in locals(): # Ensure file_handler was defined
            try:
                logger.removeHandler(file_handler)
            except Exception as ex_handler:
                logger.error(f"{log_prefix} Error removing log handler: {ex_handler}")


    elapsed = time.perf_counter() - t0
    logger.info(f"{log_prefix} Completed processing {audio_path.name} in {elapsed:,.1f}s")

    exclusions = output_filename_exclusions or []
    if output_json.name in exclusions:
        logger.debug(f"{log_prefix} JSON file {output_json.name} in exclusion list, caller will handle.")

    return output_json