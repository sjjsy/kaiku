"""Unit tests for transcribe.py and backends/whisper_cpp.py."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from asr2clip.transcribe import (
    TranscriptionError,
    _build_api_headers,
    transcribe_audio,
)
from asr2clip.backends.whisper_cpp import (
    WhisperCppConfig,
    _clean_output,
    transcribe as wc_transcribe,
)
from asr2clip._vendor.httpclient import httpclient


# ---------------------------------------------------------------------------
# _build_api_headers
# ---------------------------------------------------------------------------

class TestBuildApiHeaders:
    def test_basic_auth_header(self):
        headers = _build_api_headers("sk-test", None)
        assert headers["Authorization"] == "Bearer sk-test"
        assert "OpenAI-Organization" not in headers

    def test_org_id_included_when_provided(self):
        headers = _build_api_headers("sk-test", "org-abc")
        assert headers["OpenAI-Organization"] == "org-abc"


# ---------------------------------------------------------------------------
# transcribe_audio — retry logic
# ---------------------------------------------------------------------------

def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpclient.Response)
    resp.status_code = status
    resp.text = f'{{"text": "{text}"}}'
    resp.json.return_value = {"text": text}
    return resp


class TestTranscribeAudioRetry:
    def _call(self, **kw):
        defaults = dict(
            audio_file_path="/tmp/fake.wav",
            api_key="sk-test",
            api_base_url="http://localhost:11434/v1/",
            model_name="whisper-1",
            raise_on_error=True,
            max_retries=2,
            retry_delay=0,
        )
        defaults.update(kw)
        return transcribe_audio(**defaults)

    def test_success_on_first_attempt(self, tmp_wav):
        resp = _mock_response("hello world")
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = resp
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            result = self._call(audio_file_path=tmp_wav)
        assert result == "hello world"

    def test_timeout_retries_then_succeeds(self, tmp_wav):
        resp = _mock_response("after retry")
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = [
            httpclient.HttpTimeoutError("timed out"),
            resp,
        ]
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            result = self._call(audio_file_path=tmp_wav)
        assert result == "after retry"
        assert mock_client.post.call_count == 2

    def test_max_retries_exhausted_raises(self, tmp_wav):
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpclient.HttpTimeoutError("timed out")
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            with pytest.raises(TranscriptionError):
                self._call(audio_file_path=tmp_wav, max_retries=2)
        assert mock_client.post.call_count == 3  # 1 + 2 retries

    def test_http_client_error_raises_immediately(self, tmp_wav):
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpclient.HttpClientError("connection refused")
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            with pytest.raises(TranscriptionError):
                self._call(audio_file_path=tmp_wav)

    def test_api_non_200_raises(self, tmp_wav):
        resp = MagicMock(spec=httpclient.Response)
        resp.status_code = 401
        resp.text = "Unauthorized"
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = resp
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            with pytest.raises(TranscriptionError, match="401"):
                self._call(audio_file_path=tmp_wav)

    def test_raise_on_error_false_calls_sys_exit(self, tmp_wav):
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpclient.HttpTimeoutError("t/o")
        with patch("asr2clip.transcribe.httpclient.Client", return_value=mock_client):
            with pytest.raises(SystemExit):
                self._call(audio_file_path=tmp_wav, raise_on_error=False, max_retries=0)


# ---------------------------------------------------------------------------
# WhisperCppConfig.from_config
# ---------------------------------------------------------------------------

class TestWhisperCppConfigFromConfig:
    def test_defaults(self):
        cfg = WhisperCppConfig.from_config({})
        assert cfg.binary == "whisper-cli"
        assert cfg.language == "auto"
        assert cfg.threads == 4
        assert cfg.timestamps is False
        assert cfg.timeout_multiplier == 4.0
        assert cfg.extra_args == []

    def test_all_fields_overridden(self):
        config = {
            "whisper_cpp": {
                "binary": "/usr/local/bin/whisper-cli",
                "model": "/models/ggml-large-v3.bin",
                "language": "fi",
                "threads": 8,
                "timestamps": True,
                "timeout_multiplier": 6.0,
                "extra_args": ["--print-progress"],
            }
        }
        cfg = WhisperCppConfig.from_config(config)
        assert cfg.binary == "/usr/local/bin/whisper-cli"
        assert cfg.model == "/models/ggml-large-v3.bin"
        assert cfg.language == "fi"
        assert cfg.threads == 8
        assert cfg.timestamps is True
        assert cfg.timeout_multiplier == 6.0
        assert cfg.extra_args == ["--print-progress"]

    def test_home_expanded_in_paths(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/testuser")
        config = {
            "whisper_cpp": {
                "binary": "~/bin/whisper-cli",
                "model": "~/models/ggml.bin",
            }
        }
        cfg = WhisperCppConfig.from_config(config)
        assert cfg.binary == "/home/testuser/bin/whisper-cli"
        assert cfg.model == "/home/testuser/models/ggml.bin"


# ---------------------------------------------------------------------------
# _clean_output
# ---------------------------------------------------------------------------

class TestCleanOutput:
    def test_removes_blank_audio(self):
        raw = "[BLANK_AUDIO]\nHello world."
        result = _clean_output(raw, strip_timestamps=False)
        assert "[BLANK_AUDIO]" not in result
        assert "Hello world." in result

    def test_removes_noise_music_inaudible(self):
        raw = "[MUSIC]\n[NOISE]\n[INAUDIBLE]\nActual text."
        result = _clean_output(raw, strip_timestamps=False)
        assert "[MUSIC]" not in result
        assert "[NOISE]" not in result
        assert "[INAUDIBLE]" not in result
        assert "Actual text." in result

    def test_strips_timestamps_when_requested(self):
        raw = "[00:00:00.000 --> 00:00:03.000]  Hello there."
        result = _clean_output(raw, strip_timestamps=True)
        assert "-->" not in result
        assert "Hello there." in result

    def test_preserves_timestamps_when_not_stripping(self):
        raw = "[00:00:00.000 --> 00:00:03.000]  Hello there."
        result = _clean_output(raw, strip_timestamps=False)
        assert "-->" in result

    def test_empty_lines_collapsed(self):
        raw = "Line one.\n\n\nLine two."
        result = _clean_output(raw, strip_timestamps=False)
        lines = result.splitlines()
        assert len(lines) == 2

    def test_artifact_case_insensitive(self):
        raw = "[blank_audio]\n[Music]\nText."
        result = _clean_output(raw, strip_timestamps=False)
        assert "blank_audio" not in result.lower() or "Text." in result


# ---------------------------------------------------------------------------
# transcribe() (whisper_cpp) — error paths
# ---------------------------------------------------------------------------

class TestWhisperCppTranscribe:
    def _cfg(self, **kw) -> WhisperCppConfig:
        defaults = dict(binary="whisper-cli", model="/models/ggml.bin", language="auto")
        defaults.update(kw)
        return WhisperCppConfig(**defaults)

    def test_binary_not_found_raises(self, tmp_wav):
        cfg = self._cfg(binary="/nonexistent/whisper-cli")
        with patch("shutil.which", return_value=None):
            with pytest.raises(TranscriptionError, match="binary not found"):
                wc_transcribe(tmp_wav, cfg)

    def test_model_not_found_raises(self, tmp_wav):
        cfg = self._cfg()
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", side_effect=lambda p: "whisper" in p):
                # binary found, model not found
                cfg_no_model = self._cfg(model="/nonexistent/model.bin")
                with pytest.raises(TranscriptionError, match="model not found"):
                    wc_transcribe(tmp_wav, cfg_no_model)

    def test_nonzero_exit_raises(self, tmp_wav):
        cfg = self._cfg()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: something failed"
        mock_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", return_value=True):
                with patch("asr2clip.backends.whisper_cpp.run_subprocess", return_value=mock_result):
                    with pytest.raises(TranscriptionError, match="something failed"):
                        wc_transcribe(tmp_wav, cfg)

    def test_timeout_raises(self, tmp_wav):
        cfg = self._cfg()
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", return_value=True):
                with patch(
                    "asr2clip.backends.whisper_cpp.run_subprocess",
                    side_effect=subprocess.TimeoutExpired(cmd="whisper-cli", timeout=30),
                ):
                    with pytest.raises(TranscriptionError, match="timed out"):
                        wc_transcribe(tmp_wav, cfg, timeout=30.0)

    def test_success_returns_cleaned_text(self, tmp_wav):
        cfg = self._cfg()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "[BLANK_AUDIO]\nHello from whisper."
        mock_result.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", return_value=True):
                with patch("asr2clip.backends.whisper_cpp.run_subprocess", return_value=mock_result):
                    result = wc_transcribe(tmp_wav, cfg, timeout=30.0)
        assert "[BLANK_AUDIO]" not in result
        assert "Hello from whisper." in result

    def test_language_auto_not_passed_to_cmd(self, tmp_wav):
        cfg = self._cfg(language="auto")
        mock_result = MagicMock(returncode=0, stdout="text", stderr="")
        captured_cmd = {}
        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return mock_result
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", return_value=True):
                with patch("asr2clip.backends.whisper_cpp.run_subprocess", side_effect=fake_run):
                    wc_transcribe(tmp_wav, cfg, timeout=30.0)
        assert "--language" not in captured_cmd["cmd"]

    def test_language_fi_passed_to_cmd(self, tmp_wav):
        cfg = self._cfg(language="fi")
        mock_result = MagicMock(returncode=0, stdout="text", stderr="")
        captured_cmd = {}
        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return mock_result
        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("os.path.isfile", return_value=True):
                with patch("asr2clip.backends.whisper_cpp.run_subprocess", side_effect=fake_run):
                    wc_transcribe(tmp_wav, cfg, timeout=30.0)
        assert "--language" in captured_cmd["cmd"]
        idx = captured_cmd["cmd"].index("--language")
        assert captured_cmd["cmd"][idx + 1] == "fi"
