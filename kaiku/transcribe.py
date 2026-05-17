"""Transcription API calls for kaiku."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, NoReturn

from kaiku._vendor.httpclient import httpclient

from .utils import error, print_key_value, success, warning

if TYPE_CHECKING:
    from .config_types import Config


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""

    pass


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_TIMEOUT = 60.0  # seconds


def _handle_transcription_failure(
    error_msg: str,
    raise_on_error: bool,
    cause: Exception | None = None,
) -> NoReturn:
    if raise_on_error:
        raise TranscriptionError(error_msg) from cause
    error(error_msg)
    sys.exit(1)


def _build_api_headers(api_key: str, org_id: str | None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    if org_id:
        headers["OpenAI-Organization"] = org_id
    return headers


def _attempt_transcription(
    audio_file_path: str,
    url: str,
    headers: dict,
    model_name: str,
    timeout: float,
    language: str | None = None,
) -> str:
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": (os.path.basename(audio_file_path), audio_file, "audio/wav")}
        data = {"model": model_name}
        if language:
            data["language"] = language

        with httpclient.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, files=files, data=data)
            assert isinstance(response, httpclient.Response)

    if response.status_code != 200:
        raise TranscriptionError(f"API error {response.status_code}: {response.text}")

    result = response.json()
    return result.get("text", "")


def transcribe_audio(
    audio_file_path: str,
    config: Config,
    *,
    raise_on_error: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Transcribe audio via OpenAI-compatible API using ``config.asr_backend`` and ``config.language``."""
    asr = config.asr_backend
    api_key = asr.api_key or ""
    api_base_url = asr.api_base_url or ""
    if not api_base_url.endswith("/"):
        api_base_url += "/"
    url = f"{api_base_url}audio/transcriptions"
    headers = _build_api_headers(api_key, asr.org_id)
    language = config.language

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return _attempt_transcription(
                audio_file_path,
                url,
                headers,
                asr.model_name or "",
                timeout,
                language=language,
            )
        except (httpclient.HttpTimeoutError, httpclient.HttpClientError) as e:
            last_error = e
            if attempt < max_retries:
                warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                continue
            error_msg = f"Request failed after {max_retries + 1} attempts: {e}"
            _handle_transcription_failure(error_msg, raise_on_error, last_error)
        except TranscriptionError as e:
            _handle_transcription_failure(str(e), raise_on_error, e)
        except Exception as e:
            _handle_transcription_failure(
                f"Transcription error: {e}", raise_on_error, e
            )

    _handle_transcription_failure(
        "Unexpected error in transcription retry loop", raise_on_error
    )


def transcribe(
    audio_file_path: str,
    config: Config,
    raise_on_error: bool = False,
    timeout: float | None = None,
) -> str:
    """Transcribe audio using the backend resolved in ``config.asr_backend``."""
    if config.asr_backend.type == "mock":
        from .backends.mock import MockConfig, transcribe as mock_transcribe

        cfg = MockConfig(
            response=config.asr_backend.response,
            latency_ms=config.asr_backend.latency_ms or 0,
        )
        return mock_transcribe(audio_file_path, cfg, timeout=timeout)

    if config.asr_backend.type in ("mock-fwd", "mock-bwd"):
        from .backends.mock import MockTranscriptConfig, transcribe_from_transcript

        direction = "forward" if config.asr_backend.type == "mock-fwd" else "backward"
        cfg = MockTranscriptConfig(
            transcript_path=config.asr_backend.transcript_path or "",
            direction=direction,
            latency_ms=config.asr_backend.latency_ms or 0,
        )
        return transcribe_from_transcript(audio_file_path, cfg, timeout=timeout)

    if config.asr_backend.type == "mock-diarize":
        from .backends.mock import MockDiarizeConfig, transcribe_mock_diarize

        cfg = MockDiarizeConfig(
            transcript_path=config.asr_backend.transcript_path or "",
            speaker_count=config.asr_backend.speaker_count or 2,
        )
        return transcribe_mock_diarize(
            audio_file_path, cfg, num_speakers=config.num_speakers
        )

    if config.asr_backend.type == "whisperx":
        from .backends.whisperx import WhisperXConfig, transcribe as wx_transcribe

        hf_token = config.asr_backend.hf_token
        if not hf_token:
            error(
                "WhisperX requires a HuggingFace token.\n"
                "Set HF_TOKEN env var or add 'hf_token: hf_...' to the whisperx\n"
                "entry in asr_backends."
            )
            sys.exit(1)
        ns = config.num_speakers
        cfg = WhisperXConfig(
            hf_token=hf_token,
            min_speakers=ns or config.asr_backend.min_speakers,
            max_speakers=ns or config.asr_backend.max_speakers,
        )
        try:
            return wx_transcribe(
                audio_file_path, cfg, language=config.language, timeout=timeout
            )
        except TranscriptionError as e:
            if raise_on_error:
                raise
            error(f"WhisperX diarization failed: {e}")
            sys.exit(1)

    if config.asr_backend.type == "whisper_cpp":
        from .backends.whisper_cpp import WhisperCppConfig, transcribe as wc_transcribe

        cfg = WhisperCppConfig(
            binary=config.asr_backend.binary,
            model=config.asr_backend.model,
            threads=config.asr_backend.threads or 4,
            vad_model=config.asr_backend.vad_model,
        )
        if config.language:
            cfg.language = config.language
        try:
            return wc_transcribe(audio_file_path, cfg, timeout=timeout)
        except TranscriptionError as e:
            if raise_on_error:
                raise
            error(f"whisper.cpp transcription failed: {e}")
            sys.exit(1)

    return transcribe_audio(
        audio_file_path,
        config,
        raise_on_error=raise_on_error,
        timeout=timeout or DEFAULT_TIMEOUT,
    )


def test_openai_compat_connection(
    api_key: str,
    api_base_url: str,
    model_name: str,
    org_id: str | None = None,
) -> bool:
    """Probe ``GET …/models`` on an OpenAI-compatible server (credentials from any source)."""
    if not api_base_url.endswith("/"):
        api_base_url += "/"
    url = f"{api_base_url}models"
    headers = _build_api_headers(api_key, org_id)

    try:
        with httpclient.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            assert isinstance(response, httpclient.Response)

        if response.status_code == 200:
            success("API connection successful")
            print_key_value("Base URL", api_base_url)
            print_key_value("Model", model_name)
            return True
        else:
            error(f"API returned status {response.status_code}")
            print_key_value("Response", response.text[:200])
            return False

    except httpclient.HttpTimeoutError:
        error("Connection timed out")
        return False

    except httpclient.HttpClientError as e:
        error(f"Connection failed: {e}")
        return False


def test_transcription(config: Config) -> bool:
    """Test ASR OpenAI-compatible API connectivity using ``config.asr_backend``."""
    asr = config.asr_backend
    return test_openai_compat_connection(
        asr.api_key or "",
        asr.api_base_url or "",
        asr.model_name or "",
        asr.org_id,
    )
