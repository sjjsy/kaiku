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
    """Bare-minimum config that satisfies config-reading code paths."""
    return {
        "backend": "api",
        "api_key": "sk-test",
        "api_base_url": "http://localhost:11434/v1/",
        "model": "whisper-1",
    }


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
        "postprocessor_live": "none",
        "postprocessor_file": "solo-base",
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
