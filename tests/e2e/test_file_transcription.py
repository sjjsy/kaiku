"""E2E tests: file transcription via whisper.cpp backend."""

from __future__ import annotations

import os
import pytest

from asr2clip.transcribe import transcribe
from asr2clip.backends.whisper_cpp import WhisperCppConfig, transcribe as wc_transcribe

from .conftest import (
    JFK_EXPECTED_FRAGMENT,
    skip_no_whisper,
    skip_no_model,
)


@skip_no_whisper
@skip_no_model
class TestWhisperCppFileTranscription:
    def test_jfk_transcription_contains_expected_text(self, jfk_wav, wcpp_config):
        """Full pipeline: jfk.wav → whisper.cpp → non-empty transcript with known content."""
        result = transcribe(jfk_wav, wcpp_config, raise_on_error=True)
        assert result.strip(), "Transcript should not be empty"
        assert JFK_EXPECTED_FRAGMENT in result.lower(), (
            f"Expected '{JFK_EXPECTED_FRAGMENT}' in transcript, got: {result!r}"
        )

    def test_transcription_returns_string(self, jfk_wav, wcpp_config):
        result = transcribe(jfk_wav, wcpp_config, raise_on_error=True)
        assert isinstance(result, str)

    def test_language_hint_accepted(self, jfk_wav, wcpp_config):
        """Passing language='en' should work without error."""
        result = transcribe(jfk_wav, wcpp_config, raise_on_error=True, language="en")
        assert result.strip()

    def test_direct_wc_transcribe(self, jfk_wav, wcpp_config):
        """Call the whisper_cpp backend directly, bypassing transcribe()."""
        backend = wcpp_config.asr_backend
        cfg = WhisperCppConfig(
            binary=backend.binary,
            model=backend.model,
            threads=backend.threads or 4,
        )
        result = wc_transcribe(jfk_wav, cfg)
        assert JFK_EXPECTED_FRAGMENT in result.lower()
