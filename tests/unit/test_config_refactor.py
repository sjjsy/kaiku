"""Tests for the refactored hierarchical config system."""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from asr2clip.config_types import (
    CliOverrides,
    ASRBackendConfig,
    PreprocessorConfig,
    RecorderConfig,
    PostprocessorConfig,
    OutputConfig,
    DiarizationConfig,
    Config,
    Preset,
)


class TestASRBackendConfig:
    """Test ASR backend resolution."""

    @pytest.fixture
    def config_with_groq(self):
        return {
            "asr_backend_urgent": "groq",
            "asr_backend_casual": "groq",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-large-v3-turbo",
                },
            },
        }

    def test_groq_backend_resolves_correctly(self, config_with_groq):
        """ASR backend config should resolve groq with correct API settings."""
        asr = ASRBackendConfig.resolve(config_with_groq, backend_name="groq")
        assert asr.name == "groq"
        assert asr.type == "api"
        assert asr.api_key == "gsk_test"
        assert asr.api_base_url == "https://api.groq.com/openai/v1/"
        assert asr.model_name == "whisper-large-v3-turbo"

    def test_cli_override_takes_precedence(self, config_with_groq):
        """Backend override should work correctly."""
        config_with_groq["asr_backends"]["wcpp"] = {
            "type": "whisper_cpp",
            "binary": "/path/to/binary",
            "model": "/path/to/model",
        }
        asr = ASRBackendConfig.resolve(config_with_groq, backend_name="wcpp")
        assert asr.name == "wcpp"
        assert asr.type == "whisper_cpp"

    def test_missing_required_field_raises(self, config_with_groq):
        """Missing required field should raise ValueError."""
        del config_with_groq["asr_backends"]["groq"]["api_key"]
        del config_with_groq["asr_backends"]["groq"]["api_base_url"]
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="requires api_key"):
            ASRBackendConfig.resolve(config_with_groq, backend_name="groq")


class TestPreprocessorConfig:
    """Test preprocessor resolution."""

    def test_preprocessor_cli_override(self):
        """CLI override should be used when provided."""
        config = {}
        pp = PreprocessorConfig.resolve(config, cli_override="noisereduce")
        assert pp.name == "noisereduce"

    def test_preprocessor_defaults_to_none(self):
        """Should default to 'none' when no CLI override provided."""
        config = {}
        pp = PreprocessorConfig.resolve(config)
        assert pp.name == "none"


class TestOutputConfig:
    """Test output configuration resolution."""

    def test_clipboard_max_chars_from_config(self):
        config = {"clipboard_max_chars": 100000}
        out = OutputConfig.resolve(config)
        assert out.clipboard_max_chars == 100000

    def test_clipboard_max_chars_default(self):
        config = {}
        out = OutputConfig.resolve(config)
        assert out.clipboard_max_chars == 50_000


class TestDiarizationConfig:
    """Test diarization configuration resolution."""

    def test_diarization_from_env_var(self):
        config = {}
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            dia = DiarizationConfig.resolve(config)
            assert dia.hf_token == "hf_test"

    def test_diarization_from_config(self):
        config = {"diarize_hf_token": "hf_config"}
        dia = DiarizationConfig.resolve(config)
        assert dia.hf_token == "hf_config"


class TestMasterConfig:
    """Test master Config coordinator and lazy loading."""

    @pytest.fixture
    def full_config(self):
        return {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-large-v3-turbo",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
            "clipboard_max_chars": 75000,
        }

    def test_config_resolves_asr_backend(self, full_config):
        """Config should resolve ASR backend."""
        config = Config.resolve(full_config, preset=Preset(name="test", preprocessor="deepfilter", asr_backend="groq", postprocessor="none"))
        pp = config.preprocessor
        assert pp.name == "deepfilter"
        asr = config.asr_backend
        assert asr.name == "groq"
        assert asr.api_key == "gsk_test"

    def test_config_resolves_preprocessor(self, full_config):
        """Config should resolve preprocessor."""
        config = Config.resolve(full_config, preset=Preset(name="test", preprocessor="deepfilter", asr_backend="groq", postprocessor="none"))
        pp = config.preprocessor
        assert pp.name == "deepfilter"
        pp = config.preprocessor
        assert pp.name == "deepfilter"

    def test_config_resolves_output(self, full_config):
        """Config should resolve output config."""
        config = Config.resolve(full_config, preset=Preset(name="test", preprocessor="deepfilter", asr_backend="groq", postprocessor="none"))
        pp = config.preprocessor
        assert pp.name == "deepfilter"
        out = config.output
        assert out.clipboard_max_chars == 75000

    def test_cli_overrides_apply(self, full_config):
        """CLI overrides should apply."""
        cli = CliOverrides(backend="groq", preprocessor="pyrnnoise")
        config = Config.resolve(full_config, preset=Preset(name="test", preprocessor="noisereduce", asr_backend="groq", postprocessor="none"), cli_overrides=cli)
        assert config.asr_backend.name == "groq"
        assert config.preprocessor.name == "pyrnnoise"

    def test_lazy_loading_caches(self, full_config):
        """Config should cache resolved values."""
        config = Config.resolve(full_config, preset=Preset(name="test", preprocessor="deepfilter", asr_backend="groq", postprocessor="none"))
        pp = config.preprocessor
        assert pp.name == "deepfilter"
        asr1 = config.asr_backend
        asr2 = config.asr_backend
        assert asr1 is asr2  # Same object, cached


class TestConfigIntegration:
    """Integration tests: Config replaces all old resolver functions."""

    @pytest.fixture
    def integration_config(self):
        return {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_live",
                    "model_name": "whisper-turbo",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper-cli",
                    "model": "/path/to/model.bin",
                    "threads": 4,
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
            "clipboard_max_chars": 50_000,
        }

    def test_live_mode_resolves_correctly(self, integration_config):
        """Live mode should use live-specific configs."""
        config = Config.resolve(integration_config, preset=Preset(name="test", preprocessor="none", asr_backend="wcpp", postprocessor="none"))
        assert config.asr_backend.name == "wcpp"
        assert config.asr_backend.type == "whisper_cpp"
        assert config.preprocessor.name == "none"

    def test_file_mode_resolves_correctly(self, integration_config):
        """File mode should use file-specific configs."""
        config = Config.resolve(integration_config, preset=Preset(name="test", preprocessor="noisereduce", asr_backend="groq", postprocessor="none"))
        assert config.asr_backend.name == "groq"
        assert config.asr_backend.type == "api"
        assert config.preprocessor.name == "noisereduce"

    def test_cli_override_file_backend_to_groq(self, integration_config):
        """CLI override should force backend selection."""
        cli = CliOverrides(backend="groq")
        config = Config.resolve(integration_config, preset=Preset(name="test", preprocessor="noisereduce", asr_backend="wcpp", postprocessor="none"), cli_overrides=cli)
        assert config.asr_backend.name == "groq"
        assert config.asr_backend.api_key == "gsk_live"

    def test_toggle_mode_uses_live_backend(self, integration_config):
        """Preset-based system selects backends atomically per preset.

        With the preset system, the user selects one preset per run.
        This test verifies that a preset with groq backend resolves correctly.
        """
        toggle_config = integration_config.copy()
        toggle_config["asr_backends"]["groq"] = {
            "type": "api",
            "api_base_url": "https://api.groq.com/openai/v1/",
            "api_key": "gsk_live",
            "model_name": "whisper-turbo",
        }

        # Speed preset: uses groq
        speed_config = Config.resolve(toggle_config, preset=Preset(name="speed", preprocessor="none", asr_backend="groq", postprocessor="none"))
        assert speed_config.asr_backend.name == "groq"
        assert speed_config.asr_backend.type == "api"

        # Quality preset: uses wcpp for offline quality
        quality_config = Config.resolve(toggle_config, preset=Preset(name="quality", preprocessor="deepfilter", asr_backend="wcpp", postprocessor="none"))
        assert quality_config.asr_backend.name == "wcpp"
        assert quality_config.asr_backend.type == "whisper_cpp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
