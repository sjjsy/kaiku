"""E2E tests: post-processing pipeline with mocked LLM backends.

These tests run the full pipeline (real whisper.cpp transcription + postprocessor)
but mock the LLM call so no API key or running model server is needed.
"""

from __future__ import annotations

import json
import subprocess
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from asr2clip.postprocessors import make_postprocessor
from asr2clip.postprocessors.base import PostMetadata
from asr2clip.transcribe import transcribe_casual

from .conftest import skip_no_whisper, skip_no_model, JFK_EXPECTED_FRAGMENT


def _meta(**kw) -> PostMetadata:
    defaults = dict(date="2026-05-11", duration_s=11.0, language="en", prompt_name="test")
    defaults.update(kw)
    return PostMetadata(**defaults)


def _mock_openai_response(content: str) -> MagicMock:
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# openai_compat backend with mocked HTTP
# ---------------------------------------------------------------------------

@skip_no_whisper
@skip_no_model
class TestOpenAICompatE2E:
    """Transcribe with real whisper.cpp, post-process with mocked HTTP."""

    def _pp_config(self):
        return {
            "postprocessor_backends": {
                "local": {
                    "type": "openai_compat",
                    "api_base_url": "http://localhost:11434/v1/",
                    "api_key": "ollama",
                    "model": "qwen3:14b",
                }
            },
            "postprocessors": {
                "clean": {
                    "backend": "local",
                    "prompt": "Clean up this transcript. Output only the corrected text.",
                }
            },
        }

    def test_full_pipeline_transcript_in_user_message(self, jfk_wav, wcpp_config):
        """The LLM user message must contain the real transcript."""
        transcript = transcribe_casual(jfk_wav, wcpp_config, raise_on_error=True)
        assert JFK_EXPECTED_FRAGMENT in transcript.lower()

        captured = {}
        mock_resp = _mock_openai_response("Mocked clean output.")

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return mock_resp

        pp = make_postprocessor("clean", self._pp_config())
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = pp.process(transcript, metadata=_meta())

        assert result == "Mocked clean output."
        user_content = captured["body"]["messages"][1]["content"]
        assert JFK_EXPECTED_FRAGMENT in user_content.lower()

    def test_system_prompt_present_in_payload(self, jfk_wav, wcpp_config):
        """System prompt from config must appear in the messages payload."""
        transcript = transcribe_casual(jfk_wav, wcpp_config, raise_on_error=True)
        captured = {}
        mock_resp = _mock_openai_response("ok")

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return mock_resp

        pp = make_postprocessor("clean", self._pp_config())
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process(transcript, metadata=_meta())

        system_msg = captured["body"]["messages"][0]
        assert system_msg["role"] == "system"
        assert "Clean up" in system_msg["content"]


# ---------------------------------------------------------------------------
# claude_code backend with mocked subprocess
# ---------------------------------------------------------------------------

@skip_no_whisper
@skip_no_model
class TestClaudeCodeE2E:
    """Transcribe with real whisper.cpp, post-process with mocked claude CLI."""

    def _pp_config(self):
        return {
            "postprocessor_backends": {
                "cc": {
                    "type": "claude_code",
                    "model": "claude-haiku-4-5-20251001",
                }
            },
            "postprocessors": {
                "summarize": {
                    "backend": "cc",
                    "prompt": "Summarize the following transcript in one sentence.",
                }
            },
        }

    def test_transcript_passed_to_claude_stdin(self, jfk_wav, wcpp_config):
        """The real transcript must reach claude's stdin input."""
        transcript = transcribe_casual(jfk_wav, wcpp_config, raise_on_error=True)
        assert JFK_EXPECTED_FRAGMENT in transcript.lower()

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["input"] = kwargs.get("input", "")
            return MagicMock(returncode=0, stdout="One-sentence summary.", stderr="")

        pp = make_postprocessor("summarize", self._pp_config())
        with patch("subprocess.run", side_effect=fake_run):
            result = pp.process(transcript, metadata=_meta())

        assert result == "One-sentence summary."
        assert JFK_EXPECTED_FRAGMENT in captured["input"].lower()

    def test_model_flag_passed_to_claude(self, jfk_wav, wcpp_config):
        transcript = transcribe_casual(jfk_wav, wcpp_config, raise_on_error=True)
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return MagicMock(returncode=0, stdout="ok", stderr="")

        pp = make_postprocessor("summarize", self._pp_config())
        with patch("subprocess.run", side_effect=fake_run):
            pp.process(transcript, metadata=_meta())

        assert "--model" in captured["cmd"]
        assert "claude-haiku-4-5-20251001" in captured["cmd"]
