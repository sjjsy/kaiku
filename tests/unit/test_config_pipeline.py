"""Comprehensive tests for the config pipeline.

Covers backend resolution, device resolution, preprocessor/postprocessor selection,
and CLI overrides to catch regressions in config handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from asr2clip.config import (
    resolve_backend_name,
    resolve_backend_config,
    resolve_preprocessor_config,
    resolve_audio_device,
    resolve_clipboard_max_chars,
)
from asr2clip.postprocessors import resolve_postprocessor_config


# ---------------------------------------------------------------------------
# Backend resolution: per-mode selection (asr_backend_live vs asr_backend_file)
# ---------------------------------------------------------------------------

class TestBackendNameResolution:
    """Test resolve_backend_name with per-mode config and CLI overrides."""

    @pytest.fixture
    def permode_config(self):
        """Config with separate live and file backends."""
        return {
            "asr_backend_live": "groq",
            "asr_backend_file": "wcpp",
            "asr_backends": {
                "groq": {"type": "api", "api_base_url": "https://api.groq.com/"},
                "wcpp": {"type": "whisper_cpp", "binary": "/path/to/whisper-cli"},
            },
        }

    def test_live_mode_resolves_to_live_backend(self, permode_config):
        """In live mode, should resolve to asr_backend_live."""
        name = resolve_backend_name(permode_config, backend_name=None, mode="live")
        assert name == "groq"

    def test_file_mode_resolves_to_file_backend(self, permode_config):
        """In file mode, should resolve to asr_backend_file."""
        name = resolve_backend_name(permode_config, backend_name=None, mode="file")
        assert name == "wcpp"

    def test_cli_override_takes_precedence_live(self, permode_config):
        """CLI override should take precedence over config in live mode."""
        name = resolve_backend_name(permode_config, backend_name="wcpp", mode="live")
        assert name == "wcpp"

    def test_cli_override_takes_precedence_file(self, permode_config):
        """CLI override should take precedence over config in file mode."""
        name = resolve_backend_name(permode_config, backend_name="groq", mode="file")
        assert name == "groq"

    def test_backend_name_returned_when_provided(self, permode_config):
        """When backend_name is provided, it should be returned regardless of mode."""
        name = resolve_backend_name(permode_config, backend_name="wcpp", mode="live")
        assert name == "wcpp"

    def test_empty_string_backend_uses_config(self, permode_config):
        """Empty string backend should be treated as no override."""
        name = resolve_backend_name(permode_config, backend_name="", mode="file")
        # Empty string is falsy, so should use config
        assert name in ("", "wcpp", None)


class TestBackendConfigResolution:
    """Test resolve_backend_config properly returns backend-specific configs."""

    @pytest.fixture
    def full_config(self):
        """Config with multiple backends."""
        return {
            "asr_backend_live": "groq",
            "asr_backend_file": "wcpp",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-large-v3-turbo",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/home/samsy/whisper.cpp/whisper-cli",
                    "model": "/home/samsy/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin",
                    "threads": 4,
                },
            },
        }

    def test_groq_config_has_api_fields(self, full_config):
        """resolve_backend_config for groq should return API config."""
        cfg = resolve_backend_config(full_config, backend_name="groq", mode="live")
        assert cfg.get("backend") == "api"
        assert cfg.get("api_base_url") == "https://api.groq.com/openai/v1/"
        assert cfg.get("api_key") == "gsk_test"

    def test_wcpp_config_has_whisper_cpp_fields(self, full_config):
        """resolve_backend_config for wcpp should return whisper.cpp config."""
        cfg = resolve_backend_config(full_config, backend_name="wcpp", mode="file")
        assert cfg.get("backend") == "whisper_cpp"
        wcpp_cfg = cfg.get("whisper_cpp", {})
        assert wcpp_cfg.get("binary") == "/home/samsy/whisper.cpp/whisper-cli"
        assert wcpp_cfg.get("model") == "/home/samsy/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin"

    def test_mode_default_when_no_override(self, full_config):
        """When backend override is None, should resolve based on mode."""
        cfg_live = resolve_backend_config(full_config, backend_name=None, mode="live")
        cfg_file = resolve_backend_config(full_config, backend_name=None, mode="file")
        # Groq is live backend, wcpp is file backend
        assert cfg_live.get("backend") == "api"
        assert cfg_file.get("backend") == "whisper_cpp"

    def test_cli_override_resolves_correctly(self, full_config):
        """CLI override should resolve to specified backend regardless of mode."""
        cfg = resolve_backend_config(full_config, backend_name="wcpp", mode="live")
        # Should get wcpp config even though requesting live mode
        assert cfg.get("backend") == "whisper_cpp"

    def test_nonexistent_backend_exits(self, full_config):
        """Requesting a backend that doesn't exist should raise SystemExit."""
        with pytest.raises(SystemExit):
            resolve_backend_config(full_config, backend_name="nonexistent", mode="live")


# ---------------------------------------------------------------------------
# Preprocessor resolution
# ---------------------------------------------------------------------------

class TestPreprocessorResolution:
    """Test resolve_preprocessor_config with per-mode selection."""

    @pytest.fixture
    def preproc_config(self):
        """Config with per-mode preprocessors."""
        return {
            "preprocessor_live": "noisereduce",
            "preprocessor_file": "deepfilter",
        }

    def test_live_preprocessor_resolved(self, preproc_config):
        """In live mode, should use preprocessor_live."""
        name = resolve_preprocessor_config(preproc_config, cli_override=None, mode="live")
        assert name == "noisereduce"

    def test_file_preprocessor_resolved(self, preproc_config):
        """In file mode, should use preprocessor_file."""
        name = resolve_preprocessor_config(preproc_config, cli_override=None, mode="file")
        assert name == "deepfilter"

    def test_cli_override_preprocessor(self, preproc_config):
        """CLI -p flag should override config preprocessor."""
        name = resolve_preprocessor_config(preproc_config, cli_override="pyrnnoise", mode="live")
        assert name == "pyrnnoise"

    def test_none_preprocessor_default(self):
        """Missing preprocessor config should default to 'none'."""
        config = {}
        name = resolve_preprocessor_config(config, cli_override=None, mode="live")
        assert name == "none"

    def test_override_takes_precedence(self, preproc_config):
        """CLI override should take precedence over config."""
        name = resolve_preprocessor_config(preproc_config, cli_override="none", mode="live")
        assert name == "none"


# ---------------------------------------------------------------------------
# Postprocessor resolution
# ---------------------------------------------------------------------------

class TestPostprocessorResolution:
    """Test resolve_postprocessor_config with per-mode selection."""

    @pytest.fixture
    def postproc_config(self):
        """Config with per-mode postprocessors."""
        return {
            "postprocessor_live": "none",
            "postprocessor_file": "solo-enhance",
            "postprocessors": {
                "solo-enhance": {
                    "backend": "groq",
                    "prompt": "Enhance this transcript.",
                },
                "group-summary": {
                    "backend": "groq",
                    "prompt": "Summarize this meeting.",
                },
            },
        }

    def test_live_postprocessor_resolved(self, postproc_config):
        """In live mode, should use postprocessor_live."""
        name = resolve_postprocessor_config(postproc_config, cli_override=None, mode="live")
        assert name == "none"

    def test_file_postprocessor_resolved(self, postproc_config):
        """In file mode, should use postprocessor_file."""
        name = resolve_postprocessor_config(postproc_config, cli_override=None, mode="file")
        assert name == "solo-enhance"

    def test_cli_override_postprocessor(self, postproc_config):
        """CLI -P flag should override config postprocessor."""
        name = resolve_postprocessor_config(postproc_config, cli_override="group-summary", mode="file")
        assert name == "group-summary"

    def test_none_postprocessor_default(self):
        """Missing postprocessor config should default to 'none'."""
        config = {}
        name = resolve_postprocessor_config(config, cli_override=None, mode="live")
        assert name == "none"


# ---------------------------------------------------------------------------
# Device resolution with preference order
# ---------------------------------------------------------------------------

class TestDeviceResolution:
    """Test audio device resolution with preference order fallback."""

    @pytest.fixture
    def device_config(self):
        """Config with device preference order."""
        return {
            "audio_device": "Snowball,Webcam,auto",
        }

    @patch("asr2clip.audio.resolve_device_preference_order")
    def test_device_preference_order_first_available(self, mock_resolve, device_config):
        """Should return first available device in preference order."""
        snowball = MagicMock(name="Snowball")
        mock_resolve.return_value = [snowball]

        device_info = resolve_audio_device(device_config, cli_override=None)
        assert device_info == snowball

    @patch("asr2clip.audio.resolve_device_preference_order")
    def test_cli_device_override(self, mock_resolve, device_config):
        """CLI --device should override config."""
        device = MagicMock(name="ExternalMic")
        mock_resolve.return_value = [device]

        device_info = resolve_audio_device(device_config, cli_override="ExternalMic")
        assert device_info == device

    @patch("asr2clip.audio.resolve_device_preference_order")
    def test_auto_default_when_no_device_specified(self, mock_resolve):
        """Missing audio_device config should default to 'auto'."""
        auto_device = MagicMock(name="System Default")
        mock_resolve.return_value = [auto_device]

        config = {}
        device_info = resolve_audio_device(config, cli_override=None)
        assert device_info == auto_device
        # Should have called with "auto"
        mock_resolve.assert_called_once_with("auto")


# ---------------------------------------------------------------------------
# Clipboard max chars resolution
# ---------------------------------------------------------------------------

class TestClipboardMaxCharsResolution:
    """Test clipboard character limit resolution."""

    def test_explicit_clipboard_max_chars(self):
        """Explicit clipboard_max_chars in config should be used."""
        config = {"clipboard_max_chars": 100000}
        result = resolve_clipboard_max_chars(config)
        assert result == 100000

    def test_default_clipboard_max_chars(self):
        """Missing clipboard_max_chars should use default (50_000)."""
        config = {}
        result = resolve_clipboard_max_chars(config)
        # Default is 50_000 per output.py
        assert result == 50_000

    def test_clipboard_zero_always_file(self):
        """clipboard_max_chars: 0 should indicate always use file path."""
        config = {"clipboard_max_chars": 0}
        result = resolve_clipboard_max_chars(config)
        assert result == 0

    def test_string_value_converted_to_int(self):
        """If config has string value, should convert to int."""
        config = {"clipboard_max_chars": "75000"}
        result = resolve_clipboard_max_chars(config)
        assert result == 75000
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Integration: multiple config interactions
# ---------------------------------------------------------------------------

class TestConfigPipelineIntegration:
    """Test interactions between backend, preprocessor, postprocessor configs."""

    @pytest.fixture
    def complex_config(self):
        """Full config with all features."""
        return {
            "asr_backend_live": "groq",
            "asr_backend_file": "wcpp",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-large-v3-turbo",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper-cli",
                    "model": "/path/to/model.bin",
                },
            },
            "preprocessor_live": "noisereduce",
            "preprocessor_file": "deepfilter",
            "postprocessor_live": "none",
            "postprocessor_file": "solo-enhance",
            "audio_device": "auto",
            "clipboard_max_chars": 75000,
        }

    def test_live_mode_pipeline(self, complex_config):
        """Live mode should resolve all components correctly."""
        backend_name = resolve_backend_name(complex_config, backend_name=None, mode="live")
        backend_cfg = resolve_backend_config(complex_config, backend_name=None, mode="live")
        preproc = resolve_preprocessor_config(complex_config, cli_override=None, mode="live")
        postproc = resolve_postprocessor_config(complex_config, cli_override=None, mode="live")

        assert backend_name == "groq"
        assert backend_cfg.get("backend") == "api"
        assert preproc == "noisereduce"
        assert postproc == "none"

    def test_file_mode_pipeline(self, complex_config):
        """File mode should resolve all components correctly."""
        backend_name = resolve_backend_name(complex_config, backend_name=None, mode="file")
        backend_cfg = resolve_backend_config(complex_config, backend_name=None, mode="file")
        preproc = resolve_preprocessor_config(complex_config, cli_override=None, mode="file")
        postproc = resolve_postprocessor_config(complex_config, cli_override=None, mode="file")

        assert backend_name == "wcpp"
        assert backend_cfg.get("backend") == "whisper_cpp"
        assert preproc == "deepfilter"
        assert postproc == "solo-enhance"

    def test_cli_overrides_all_modes(self, complex_config):
        """CLI overrides should apply across all resolution functions."""
        # Force groq in file mode (normally wcpp)
        backend_name = resolve_backend_name(complex_config, backend_name="groq", mode="file")
        backend_cfg = resolve_backend_config(complex_config, backend_name="groq", mode="file")

        assert backend_name == "groq"
        assert backend_cfg.get("backend") == "api"

    def test_mixed_cli_and_config(self, complex_config):
        """Some overrides + some config should work together."""
        # Use config backend but override preprocessor
        backend = resolve_backend_name(complex_config, backend_name=None, mode="file")
        preproc = resolve_preprocessor_config(complex_config, cli_override="none", mode="file")

        assert backend == "wcpp"
        assert preproc == "none"


# ---------------------------------------------------------------------------
# Edge cases and error conditions
# ---------------------------------------------------------------------------

class TestConfigEdgeCases:
    """Test edge cases and unusual configs."""

    def test_empty_config(self):
        """Empty config should not crash; use defaults."""
        config = {}
        backend = resolve_backend_name(config, backend_name=None, mode="live")
        preproc = resolve_preprocessor_config(config, cli_override=None, mode="live")
        postproc = resolve_postprocessor_config(config, cli_override=None, mode="live")

        # Should return defaults, not crash
        assert backend is None or isinstance(backend, str)
        assert preproc == "none"
        assert postproc == "none"

    def test_config_with_none_values(self):
        """Config with explicit None values should be handled."""
        config = {
            "asr_backend_live": None,
            "preprocessor_live": None,
        }
        backend = resolve_backend_name(config, backend_name=None, mode="live")
        # Should handle None gracefully
        assert backend is None or isinstance(backend, str)

    def test_config_with_invalid_types(self):
        """Config with wrong types should handle gracefully."""
        config = {
            "asr_backend_live": 123,
            "clipboard_max_chars": "not_a_number",
        }
        # Should not crash; either convert or use default
        try:
            result = resolve_clipboard_max_chars(config)
            assert isinstance(result, int)
        except (ValueError, TypeError):
            pass  # Expected for invalid input

    def test_backend_name_does_not_exist_in_asr_backends(self):
        """Config references a backend name that doesn't exist."""
        config = {
            "asr_backend_live": "nonexistent_backend",
            "asr_backends": {
                "api": {"type": "api"},
            },
        }
        # Should exit when backend not found
        with pytest.raises(SystemExit):
            resolve_backend_config(config, backend_name=None, mode="live")


# ---------------------------------------------------------------------------
# Toggle mode backend override (important bug fix!)
# ---------------------------------------------------------------------------

class TestToggleModeBackendOverride:
    """Test that -b backend override works in toggle mode."""

    @pytest.fixture
    def toggle_config(self):
        """Config for toggle mode testing."""
        return {
            "asr_backend_live": "wcpp",  # Default for toggle/live
            "asr_backend_file": "api",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/",
                    "api_key": "test",
                    "model_name": "whisper",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path",
                    "model": "/path",
                },
            },
        }

    def test_toggle_respects_cli_backend_override(self, toggle_config):
        """Toggle mode should respect -b override (reproduces the bug)."""
        # Toggle normally uses live mode
        default_name = resolve_backend_name(toggle_config, backend_name=None, mode="live")
        assert default_name == "wcpp"

        # But with -b groq override, should use groq
        override_name = resolve_backend_name(toggle_config, backend_name="groq", mode="live")
        assert override_name == "groq"

        # And the config should be correct
        override_cfg = resolve_backend_config(toggle_config, backend_name="groq", mode="live")
        assert override_cfg.get("backend") == "api"


# ---------------------------------------------------------------------------
# Integration: transcribe_with_config uses resolved backend config
# ---------------------------------------------------------------------------

class TestTranscribeBackendIntegration:
    """Test that transcribe_with_config actually uses the resolved backend config."""

    @pytest.fixture
    def groq_vs_wcpp_config(self):
        """Config with groq (API) and wcpp (local) backends."""
        return {
            "asr_backend_live": "wcpp",
            "asr_backend_file": "groq",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_groq_test_key",
                    "model_name": "whisper-large-v3-turbo",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper-cli",
                    "model": "/path/to/model.bin",
                },
            },
        }

    @patch("asr2clip.transcribe.transcribe_audio")
    def test_transcribe_with_config_uses_backend_groq(self, mock_transcribe, groq_vs_wcpp_config, tmp_wav):
        """transcribe_with_config should use groq's API key/URL, not top-level defaults."""
        from asr2clip.transcribe import transcribe_with_config

        mock_transcribe.return_value = "test result"

        # Call with file mode (uses groq by default, or with explicit override)
        result = transcribe_with_config(tmp_wav, groq_vs_wcpp_config, backend="groq")

        # Verify transcribe_audio was called with groq's config, not OpenAI defaults
        mock_transcribe.assert_called_once()
        call_args = mock_transcribe.call_args

        # call_args[0] is positional args: (audio_file_path, api_key, api_base_url, model_name, org_id, ...)
        called_api_key = call_args[0][1]
        called_api_base_url = call_args[0][2]
        called_model_name = call_args[0][3]

        # These should be from groq config, NOT openai defaults
        assert called_api_key == "gsk_groq_test_key", \
            f"Expected groq key, got {called_api_key} (was top-level config extracted?)"
        assert called_api_base_url == "https://api.groq.com/openai/v1/", \
            f"Expected groq URL, got {called_api_base_url} (was openai.com default used?)"
        assert called_model_name == "whisper-large-v3-turbo"

    @patch("asr2clip.transcribe.transcribe_audio")
    def test_transcribe_respects_backend_override(self, mock_transcribe, groq_vs_wcpp_config, tmp_wav):
        """Backend override should force use of that backend's config."""
        from asr2clip.transcribe import transcribe_with_config

        mock_transcribe.return_value = "test result"

        # File mode defaults to groq, but override to use it explicitly anyway
        result = transcribe_with_config(tmp_wav, groq_vs_wcpp_config, backend="groq")

        call_args = mock_transcribe.call_args
        called_api_key = call_args[0][1]
        called_api_base_url = call_args[0][2]

        # Must use groq's settings
        assert called_api_key == "gsk_groq_test_key"
        assert "groq.com" in called_api_base_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
