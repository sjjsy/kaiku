"""Shared fixtures for the asr2clip test suite."""

from __future__ import annotations

import struct
import wave
import io
import os
import tempfile

import pytest

from asr2clip.config_types import Config, Preset


def _config_with_postprocessors(sections: dict) -> Config:
    """Build a minimal Config wrapping postprocessor-focused config sections.

    Used by postprocessor unit tests that don't care about ASR backends or presets.
    """
    full = {
        "asr_backends": {"default": {"type": "mock", "response": ""}},
        "presets": {"default": ["none", "default", "none", "test"]},
        **sections,
    }
    full.setdefault("postprocessor_backends", {})
    full.setdefault("postprocessors", {})
    preset = Preset(name="default", preprocessor="none", asr_backend="default", postprocessor="none")
    return Config.resolve(full, preset=preset)


# ---------------------------------------------------------------------------
# Minimal config dicts
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """Bare-minimum config for testing."""
    return {
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


_POSTPROCESSOR_SECTIONS = {
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
}


@pytest.fixture
def postprocessor_config():
    """Raw postprocessors config dict — for testing internal helpers directly."""
    import copy
    return copy.deepcopy(_POSTPROCESSOR_SECTIONS)


@pytest.fixture
def postprocessor_config_obj():
    """Config object wrapping postprocessors config — for testing public API."""
    return _config_with_postprocessors(_POSTPROCESSOR_SECTIONS)


@pytest.fixture
def full_backend_config():
    """Full ASR backend config with multiple backend options."""
    return {
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
