"""
Runpod serverless handler for speech-to-text processing via URL.

This worker implements a single action: `speech_to_text_url`.
It downloads the audio from the provided URL, runs full processing
(transcription -> alignment -> diarization -> speaker assignment),
and returns the final JSON result without touching the database.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import runpod
import requests

from app.audio import process_audio_file
from app.files import ALLOWED_EXTENSIONS, validate_extension
from app.logger import logger
from app.schemas import (
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    SpeechToTextProcessingParams,
    VADOptions,
    WhisperModelParams,
    AlignedTranscription,
)
from app.transcript import filter_aligned_transcription
from app.whisperx_services import (
    align_whisper_output,
    diarize,
    transcribe_with_whisper,
)
from whisperx import assign_word_speakers


def _download_to_temp(url: str) -> str:
    """Download file from URL to a temporary path preserving extension."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()

        # Try to infer filename from headers; fallback to URL basename.
        filename = None
        cd = r.headers.get("Content-Disposition")
        if cd and "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"')
        if not filename:
            filename = os.path.basename(url.split("?")[0]) or "audio"

        _, ext = os.path.splitext(filename)
        # default to .mp3 if no extension can be found
        ext = ext if ext else ".mp3"

        # Validate before writing
        validate_extension(f"dummy{ext}", ALLOWED_EXTENSIONS)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            return tmp.name


def _split_params(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Split the incoming model_params into their respective schema payloads."""
    model_keys = {
        "language",
        "task",
        "model",
        "device",
        "device_index",
        "threads",
        "batch_size",
        "chunk_size",
        "compute_type",
    }
    align_keys = {"align_model", "interpolate_method", "return_char_alignments"}
    asr_keys = {
        "beam_size",
        "best_of",
        "patience",
        "length_penalty",
        "temperatures",
        "compression_ratio_threshold",
        "log_prob_threshold",
        "no_speech_threshold",
        "suppress_tokens",
        "suppress_numerals",
        "initial_prompt",
        "hotwords",
    }

    mp = {k: v for k, v in model_params.items() if k in model_keys}
    ap = {k: v for k, v in model_params.items() if k in align_keys}
    ao = {k: v for k, v in model_params.items() if k in asr_keys}

    # Normalize optional field types
    if "suppress_tokens" in ao and isinstance(ao["suppress_tokens"], int):
        # API examples sometimes supply -1 instead of [-1]
        ao["suppress_tokens"] = [ao["suppress_tokens"]]

    return {"whisper": mp, "align": ap, "asr": ao}


def _process_full_pipeline(
    audio,
    whisper_params: WhisperModelParams,
    align_params: AlignmentParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
):
    """Run transcription, alignment, diarization and speaker assignment."""
    # Transcribe
    tx = transcribe_with_whisper(
        audio=audio,
        task=whisper_params.task.value,
        asr_options=asr_options.model_dump(),
        vad_options=vad_options.model_dump(),
        language=whisper_params.language,
        batch_size=whisper_params.batch_size,
        chunk_size=whisper_params.chunk_size,
        model=whisper_params.model,
        device=whisper_params.device,
        device_index=whisper_params.device_index,
        compute_type=whisper_params.compute_type,
        threads=whisper_params.threads,
    )

    # Align
    aligned = align_whisper_output(
        transcript=tx["segments"],
        audio=audio,
        language_code=tx["language"],
        align_model=align_params.align_model,
        interpolate_method=align_params.interpolate_method,
        return_char_alignments=align_params.return_char_alignments,
    )

    filtered = filter_aligned_transcription(AlignedTranscription(**aligned))

    # Diarize
    diar = diarize(
        audio,
        device=whisper_params.device,
        min_speakers=None,
        max_speakers=None,
    )

    # Combine
    combined = assign_word_speakers(diar, filtered.model_dump())
    return combined


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Runpod handler entry point.

    Expected event schema:
    {
      "input": {
        "action": "speech_to_text_url",
        "url": "https://...",
        "model_params": {...},
        "vad_options": {...}
      }
    }
    """
    try:
        data = (event or {}).get("input", {})
        action = data.get("action")
        if action != "speech_to_text_url":
            raise ValueError("Unsupported action. Use 'speech_to_text_url'.")

        url = data.get("url")
        if not url:
            raise ValueError("Missing 'url' in input.")

        model_params_in = data.get("model_params", {})
        vad_opts_in = data.get("vad_options", {})

        split = _split_params(model_params_in)
        whisper_params = WhisperModelParams(**split["whisper"]) if split["whisper"] else WhisperModelParams()
        align_params = AlignmentParams(**split["align"]) if split["align"] else AlignmentParams()
        asr_options = ASROptions(**split["asr"]) if split["asr"] else ASROptions()
        vad_options = VADOptions(**vad_opts_in) if vad_opts_in else VADOptions()

        tmp_path = _download_to_temp(url)
        logger.info("Downloaded input audio to %s", tmp_path)

        # Will also convert video to audio when needed
        audio = process_audio_file(tmp_path)

        # NOTE: identifier is not used in serverless handler, so pass a dummy
        result = _process_full_pipeline(
            audio=audio,
            whisper_params=whisper_params,
            align_params=align_params,
            asr_options=asr_options,
            vad_options=vad_options,
        )

        return {"status": "succeeded", "result": result}
    except Exception as e:
        logger.exception("Runpod handler failed: %s", str(e))
        return {"status": "failed", "error": str(e)}


runpod.serverless.start({"handler": handler})
