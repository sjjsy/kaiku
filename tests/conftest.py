"""Shared fixtures for the asr2clip test suite."""

from __future__ import annotations

import struct
import wave
import io
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Minimal config dicts
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """Bare-minimum config with per-mode backend selection."""
    return {
        "asr_backend_urgent": "api",
        "asr_backend_casual": "api",
        "asr_backends": {
            "api": {
                "type": "api",
                "api_key": "sk-test",
                "api_base_url": "http://localhost:11434/v1/",
                "model_name": "whisper-1",
            },
        },
    }


@pytest.fixture
def recorder_config():
    """Minimal config for recorder tests."""
    return {"recorder": "auto"}


@pytest.fixture
def postprocessor_config():
    """Config with a full postprocessors section for unit tests."""
    return {
        "postprocessor_backends": {
            "local": {
                "type": "openai_compat",
                "api_base_url": "http://localhost:11434/v1/",
                "api_key": "ollama",
                "model": "qwen3:14b",
            },
            "cc": {
                "type": "claude_code",
                "model": "claude-sonnet-4-6",
            },
        },
        "postprocessors": {
            "solo-base": {
                "backend": "local",
                "prompt": "Clean up this transcript.",
            },
            "solo-enhance": {
                "extends": "solo-base",
                "extra": "Also extract key points.",
            },
            "group": {
                "backend": "cc",
                "prompt": "Summarize this meeting.",
                "template": "full",
            },
        },
        "output_templates": {
            "default": "{result}",
            "full": "# Transcript\n\n{transcript}\n\n# Result\n\n{result}",
            "bare": "{result}",
        },
        "postprocessor_urgent": "none",
        "postprocessor_casual": "solo-base",
    }


@pytest.fixture
def full_backend_config():
    """Full ASR backend config with multiple backend options."""
    return {
        "asr_backend_urgent": "groq",
        "asr_backend_casual": "wcpp",
        "asr_backends": {
            "groq": {
                "type": "api",
                "api_base_url": "https://api.groq.com/openai/v1/",
                "api_key": "gsk_test_key",
                "model_name": "whisper-large-v3-turbo",
            },
            "wcpp": {
                "type": "whisper_cpp",
                "binary": "/path/to/whisper.cpp/whisper-cli",
                "model": "/path/to/models/ggml-large-v3-turbo-q8_0.bin",
                "threads": 4,
            },
            "ollama": {
                "type": "api",
                "api_base_url": "http://localhost:11434/v1/",
                "api_key": "ollama",
                "model_name": "whisper",
            },
        },
    }


@pytest.fixture
def full_preprocessor_config():
    """Config with per-mode preprocessor selection."""
    return {
        "preprocessor_urgent": "noisereduce",
        "preprocessor_casual": "deepfilter",
    }


@pytest.fixture
def device_config():
    """Config with device preference order."""
    return {
        "audio_device": "Snowball,Webcam,auto",
    }


@pytest.fixture
def clipboard_config():
    """Config with clipboard settings."""
    return {
        "clipboard_max_chars": 100_000,
    }


# ---------------------------------------------------------------------------
# Synthetic WAV generator
# ---------------------------------------------------------------------------

def make_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return bytes of a minimal valid WAV file (silence)."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


@pytest.fixture
def tmp_wav(tmp_path):
    """Path to a 1-second silent WAV file in a temp directory."""
    p = tmp_path / "test.wav"
    p.write_bytes(make_wav(1.0))
    return str(p)


@pytest.fixture
def tmp_wav_factory(tmp_path):
    """Factory: tmp_wav_factory(duration_s) → path."""
    def _make(duration_s: float = 1.0) -> str:
        p = tmp_path / f"test_{duration_s}s.wav"
        p.write_bytes(make_wav(duration_s))
        return str(p)
    return _make
