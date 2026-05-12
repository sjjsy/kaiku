"""Transcription API calls for asr2clip."""

from __future__ import annotations

import os
import sys
import time
from typing import NoReturn

from asr2clip._vendor.httpclient import httpclient

from .utils import error, print_error, print_key_value, print_success, warning


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
    """Handle a transcription failure by raising or exiting.

    Args:
        error_msg: The error message.
        raise_on_error: If True, raise TranscriptionError.
        cause: Optional original exception.

    Raises:
        TranscriptionError: If raise_on_error is True.
        SystemExit: If raise_on_error is False.
    """
    if raise_on_error:
        raise TranscriptionError(error_msg) from cause
    error(error_msg)
    sys.exit(1)


def _build_api_headers(api_key: str, org_id: str | None) -> tuple[str, dict]:
    """Normalize the API base URL (ensure trailing slash) and build auth headers."""
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
    """Make a single transcription API request.

    Args:
        audio_file_path: Path to the audio file.
        url: API endpoint URL.
        headers: Request headers.
        model_name: Model name.
        timeout: Request timeout in seconds.

    Returns:
        Transcribed text.

    Raises:
        httpclient.HttpTimeoutError: On request timeout.
        httpclient.HttpClientError: On request failure.
        TranscriptionError: On API error response.
    """
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
    api_key: str,
    api_base_url: str,
    model_name: str,
    org_id: str | None = None,
    raise_on_error: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
    language: str | None = None,
) -> str:
    """Transcribe audio using the ASR API with automatic retry on timeout.

    Args:
        audio_file_path: Path to the audio file to transcribe.
        api_key: API key for authentication.
        api_base_url: Base URL of the API.
        model_name: Name of the model to use.
        org_id: Optional organization ID.
        raise_on_error: If True, raise exception on error instead of sys.exit().
        max_retries: Maximum number of retry attempts for timeout errors.
        retry_delay: Delay between retries in seconds.
        timeout: Request timeout in seconds.

    Returns:
        Transcribed text.

    Raises:
        TranscriptionError: If transcription fails and raise_on_error is True.
        SystemExit: If transcription fails and raise_on_error is False.
    """
    if not api_base_url.endswith("/"):
        api_base_url += "/"
    url = f"{api_base_url}audio/transcriptions"
    headers = _build_api_headers(api_key, org_id)

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return _attempt_transcription(
                audio_file_path, url, headers, model_name, timeout, language=language
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


def transcribe_with_config(
    audio_file_path: str,
    config: dict,
    raise_on_error: bool = False,
    timeout: float | None = None,
    language: str | None = None,
    backend: str | None = None,
) -> str:
    """Transcribe audio using whichever backend is configured.

    Args:
        audio_file_path: Path to the WAV file.
        config: Full configuration dictionary.
        raise_on_error: If True, raise TranscriptionError on failure.
        timeout: Optional timeout override in seconds.
        backend: Optional backend override.

    Returns:
        Transcribed text.
    """
    from .config import resolve_backend_name, resolve_backend_config

    # Determine which backend to use (file transcription mode)
    backend_name = resolve_backend_name(config, backend, "file")
    backend_config = resolve_backend_config(config, backend, "file")
    if backend_config.get("backend") == "whisper_cpp":
        from .backends.whisper_cpp import WhisperCppConfig, transcribe as wc_transcribe
        # Create a wrapper config with whisper_cpp at top level for WhisperCppConfig.from_config()
        wcpp_wrapper = {"whisper_cpp": backend_config.get("whisper_cpp", {})}
        cfg = WhisperCppConfig.from_config(wcpp_wrapper)
        if language:
            cfg.language = language
        try:
            return wc_transcribe(audio_file_path, cfg, timeout=timeout)
        except TranscriptionError:
            if raise_on_error:
                raise
            error(f"whisper.cpp transcription failed")
            import sys
            sys.exit(1)

    # Extract API config from the resolved backend config (not top-level config)
    api_key = backend_config.get("api_key")
    api_base_url = backend_config.get("api_base_url")
    model_name = backend_config.get("model_name")
    org_id = backend_config.get("org_id")
    effective_language = language or config.get("language")
    return transcribe_audio(
        audio_file_path,
        api_key,
        api_base_url,
        model_name,
        org_id,
        raise_on_error=raise_on_error,
        timeout=timeout or DEFAULT_TIMEOUT,
        language=effective_language,
    )


def test_transcription(
    api_key: str,
    api_base_url: str,
    model_name: str,
    org_id: str | None = None,
) -> bool:
    """Test the transcription API connection.

    Args:
        api_key: API key for authentication.
        api_base_url: Base URL of the API.
        model_name: Name of the model to use.
        org_id: Optional organization ID.

    Returns:
        True if the API is accessible, False otherwise.
    """
    if not api_base_url.endswith("/"):
        api_base_url += "/"
    url = f"{api_base_url}models"
    headers = _build_api_headers(api_key, org_id)

    try:
        with httpclient.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            assert isinstance(response, httpclient.Response)

        if response.status_code == 200:
            print_success("API connection successful")
            print_key_value("Base URL", api_base_url)
            print_key_value("Model", model_name)
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            print_key_value("Response", response.text[:200])
            return False

    except httpclient.HttpTimeoutError:
        print_error("Connection timed out")
        return False

    except httpclient.HttpClientError as e:
        print_error(f"Connection failed: {e}")
        return False
