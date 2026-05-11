"""Unit tests for OpenAICompatPostProcessor and ClaudeCodePostProcessor."""

from __future__ import annotations

import json
import subprocess
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest

from asr2clip.postprocessors.base import PostMetadata
from asr2clip.postprocessors.openai_compat import OpenAICompatPostProcessor
from asr2clip.postprocessors.claude_code import ClaudeCodePostProcessor


def _meta(**kw) -> PostMetadata:
    defaults = dict(date="2026-05-11", duration_s=60.0, language="fi", prompt_name="test")
    defaults.update(kw)
    return PostMetadata(**defaults)


def _make_http_response(content: str, status: int = 200) -> MagicMock:
    """Return a mock that mimics urllib urlopen context manager."""
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# OpenAICompatPostProcessor
# ---------------------------------------------------------------------------

class TestOpenAICompatPostProcessor:
    def _make(self, **kw) -> OpenAICompatPostProcessor:
        defaults = dict(
            prompt_name="solo-base",
            api_base_url="http://localhost:11434/v1/",
            model="qwen3:14b",
            system_prompt="Clean up this transcript.",
            api_key="ollama",
        )
        defaults.update(kw)
        return OpenAICompatPostProcessor(**defaults)

    def test_properties(self):
        pp = self._make()
        assert pp.name == "solo-base"
        assert pp.model == "qwen3:14b"
        assert pp.backend_type == "openai_compat"

    def test_process_returns_llm_content(self):
        pp = self._make()
        mock_resp = _make_http_response("Cleaned text.")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = pp.process("raw transcript", metadata=_meta())
        assert result == "Cleaned text."

    def test_request_payload_structure(self):
        pp = self._make()
        mock_resp = _make_http_response("ok")
        captured_request = {}

        def fake_urlopen(req, timeout=None):
            captured_request["data"] = json.loads(req.data.decode())
            captured_request["headers"] = dict(req.headers)
            captured_request["url"] = req.full_url
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process("my transcript", metadata=_meta())

        data = captured_request["data"]
        assert data["model"] == "qwen3:14b"
        messages = data["messages"]
        assert messages[0]["role"] == "system"
        assert "Clean up" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "my transcript" in messages[1]["content"]
        assert "chat/completions" in captured_request["url"]

    def test_bearer_token_in_header(self):
        pp = self._make(api_key="sk-secret")
        mock_resp = _make_http_response("ok")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process("t", metadata=_meta())

        assert captured["auth"] == "Bearer sk-secret"

    def test_context_text_prepended(self):
        pp = self._make(context_text="## Context\n\nsome context here")
        mock_resp = _make_http_response("ok")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process("transcript", metadata=_meta())

        user_content = captured["body"]["messages"][1]["content"]
        assert "some context here" in user_content
        assert "transcript" in user_content

    def test_custom_user_template(self):
        pp = self._make(user_template="Date: {date}\n\n{transcript}")
        mock_resp = _make_http_response("ok")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process("hello", metadata=_meta(date="2026-01-01"))

        user_content = captured["body"]["messages"][1]["content"]
        assert user_content.startswith("Date: 2026-01-01")

    def test_http_error_exits(self):
        pp = self._make()
        err = urllib.error.HTTPError(
            url="http://x", code=401,
            msg="Unauthorized", hdrs=None, fp=BytesIO(b"unauthorized")
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_connection_error_exits(self):
        pp = self._make()
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_malformed_response_exits(self):
        pp = self._make()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"unexpected": "format"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_trailing_slash_normalised(self):
        pp = self._make(api_base_url="http://localhost:11434/v1")  # no trailing slash
        mock_resp = _make_http_response("ok")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            pp.process("t", metadata=_meta())

        assert captured["url"].endswith("/chat/completions")
        assert "/v1//chat" not in captured["url"]  # no double slash


# ---------------------------------------------------------------------------
# ClaudeCodePostProcessor
# ---------------------------------------------------------------------------

class TestClaudeCodePostProcessor:
    def _make(self, **kw) -> ClaudeCodePostProcessor:
        defaults = dict(
            prompt_name="group",
            system_prompt="Summarize this meeting.",
            model="claude-sonnet-4-6",
        )
        defaults.update(kw)
        return ClaudeCodePostProcessor(**defaults)

    def test_properties(self):
        pp = self._make()
        assert pp.name == "group"
        assert pp.model == "claude-sonnet-4-6"
        assert pp.backend_type == "claude_code"

    def test_process_returns_stdout(self):
        pp = self._make()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  Meeting summary.  "
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = pp.process("raw", metadata=_meta())
        assert result == "Meeting summary."  # stripped

    def test_cmd_includes_model_flag(self):
        pp = self._make(model="claude-opus-4-7")
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("t", metadata=_meta())
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "claude-opus-4-7" in cmd

    def test_cmd_omits_model_flag_when_empty(self):
        pp = self._make(model="")
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("t", metadata=_meta())
        cmd = mock_run.call_args[0][0]
        assert "--model" not in cmd

    def test_cmd_includes_system_prompt(self):
        pp = self._make(system_prompt="Be concise.")
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("t", metadata=_meta())
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "Be concise."

    def test_no_markdown_flag(self):
        pp = self._make()
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("t", metadata=_meta())
        cmd = mock_run.call_args[0][0]
        assert "--no-markdown" in cmd

    def test_transcript_passed_via_stdin(self):
        pp = self._make()
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("my transcript text", metadata=_meta())
        kwargs = mock_run.call_args[1]
        assert "my transcript text" in kwargs["input"]

    def test_context_prepended_to_input(self):
        pp = self._make(context_text="Context here")
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            pp.process("transcript", metadata=_meta())
        kwargs = mock_run.call_args[1]
        assert "Context here" in kwargs["input"]
        assert "transcript" in kwargs["input"]

    def test_claude_not_found_exits(self):
        pp = self._make()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_timeout_exits(self):
        pp = self._make()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=180)):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_nonzero_returncode_exits(self):
        pp = self._make()
        mock_result = MagicMock(returncode=1, stdout="", stderr="some error")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(SystemExit):
                pp.process("t", metadata=_meta())

    def test_no_model_produces_empty_model_property(self):
        pp = ClaudeCodePostProcessor(prompt_name="x", system_prompt="p")
        assert pp.model == ""
