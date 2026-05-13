"""Integration tests for main features and config pipelines.

Tests the end-to-end behavior of:
- Preset-based pipeline selection
- Backend selection based on CLI overrides
- Preprocessor/postprocessor integration
- Device resolution
- Language parameter propagation
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from asr2clip.config_types import Config, CliOverrides, Preset


class TestPresetsDefineBackendSelection:
    """Test that presets define which backends and preprocessors are used."""

    @pytest.fixture
    def preset_backend_config(self):
        """Config with multiple backends and presets to select them."""
        return {
            "asr_backends": {
                "fast_api": {
                    "type": "api",
                    "api_base_url": "https://api.fast.com/v1/",
                    "api_key": "key_fast",
                    "model_name": "fast-model",
                },
                "slow_offline": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper",
                    "model": "/path/to/model.bin",
                    "threads": 4,
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
            "presets": {
                "speed": ["none", "fast_api", "none", "Fast API with no preprocessing"],
                "quality": ["noisereduce", "slow_offline", "none", "Offline with noise reduction"],
            },
        }

    def test_speed_preset_uses_fast_api(self, preset_backend_config):
        """Speed preset should use fast_api backend."""
        config = Config.resolve(preset_backend_config, preset=Preset(name="speed", preprocessor="none", asr_backend="fast_api", postprocessor="none"))
        assert config.asr_backend.name == "fast_api"
        assert config.asr_backend.type == "api"
        assert config.asr_backend.api_key == "key_fast"

    def test_quality_preset_uses_offline_backend(self, preset_backend_config):
        """Quality preset should use slow_offline backend."""
        config = Config.resolve(preset_backend_config, preset=Preset(name="quality", preprocessor="noisereduce", asr_backend="slow_offline", postprocessor="none"))
        assert config.asr_backend.name == "slow_offline"
        assert config.asr_backend.type == "whisper_cpp"
        assert config.asr_backend.binary == "/path/to/whisper"

    def test_speed_preset_uses_none_preprocessor(self, preset_backend_config):
        """Speed preset specifies none preprocessor."""
        config = Config.resolve(preset_backend_config, preset=Preset(name="speed", preprocessor="none", asr_backend="fast_api", postprocessor="none"))
        assert config.preprocessor.name == "none"

    def test_quality_preset_uses_noisereduce_preprocessor(self, preset_backend_config):
        """Quality preset specifies noisereduce preprocessor."""
        config = Config.resolve(preset_backend_config, preset=Preset(name="quality", preprocessor="noisereduce", asr_backend="slow_offline", postprocessor="none"))
        assert config.preprocessor.name == "noisereduce"

    def test_cli_override_takes_precedence_over_preset(self, preset_backend_config):
        """CLI backend override should force backend regardless of preset."""
        cli = CliOverrides(backend="fast_api")
        config = Config.resolve(preset_backend_config, preset=Preset(name="quality", preprocessor="noisereduce", asr_backend="slow_offline", postprocessor="none"), cli_overrides=cli)
        assert config.asr_backend.name == "fast_api"
        assert config.asr_backend.type == "api"

    def test_preprocessor_cli_override(self, preset_backend_config):
        """CLI preprocessor override should take precedence over preset."""
        cli = CliOverrides(preprocessor="deepfilter")
        config = Config.resolve(preset_backend_config, preset=Preset(name="quality", preprocessor="noisereduce", asr_backend="slow_offline", postprocessor="none"), cli_overrides=cli)
        assert config.preprocessor.name == "deepfilter"


class TestToggleModeIntegration:
    """Test toggle mode with various preset configurations."""

    @pytest.fixture
    def toggle_config(self):
        """Config optimized for toggle mode."""
        return {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-turbo",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper-cli",
                    "model": "/path/to/model.bin",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
            "presets": {
                "speed": ["none", "groq", "none", "Fast toggle with Groq API"],
                "privacy": ["none", "wcpp", "none", "Offline toggle with whisper.cpp"],
            },
        }

    def test_toggle_speed_preset_uses_groq(self, toggle_config):
        """Speed preset should use groq backend for fast toggle response."""
        config = Config.resolve(toggle_config, preset=Preset(name="speed", preprocessor="none", asr_backend="groq", postprocessor="none"))
        assert config.asr_backend.name == "groq"
        assert "api.groq.com" in config.asr_backend.api_base_url

    def test_toggle_privacy_preset_uses_offline(self, toggle_config):
        """Privacy preset can override to use offline whisper.cpp backend."""
        config = Config.resolve(toggle_config, preset=Preset(name="privacy", preprocessor="none", asr_backend="wcpp", postprocessor="none"))
        assert config.asr_backend.name == "wcpp"
        assert config.asr_backend.type == "whisper_cpp"

    def test_toggle_preset_with_backend_override(self, toggle_config):
        """Preset backend can be overridden with CLI flag."""
        cli = CliOverrides(backend="wcpp")
        config = Config.resolve(toggle_config, preset=Preset(name="speed", preprocessor="none", asr_backend="groq", postprocessor="none"), cli_overrides=cli)
        assert config.asr_backend.name == "wcpp"
        assert config.asr_backend.type == "whisper_cpp"


class TestFileModeIntegration:
    """Test file input mode with various preset configurations."""

    @pytest.fixture
    def file_config(self):
        """Config optimized for file input."""
        return {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_file",
                    "model_name": "whisper-large",
                },
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper-cli",
                    "model": "/path/to/model.bin",
                    "threads": 2,
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {
                "solo-base": {
                    "backend": "groq",
                    "prompt": "base",
                }
            },
            "presets": {
                "quality": ["deepfilter", "wcpp", "none", "Offline with quality preprocessing"],
                "speed": ["none", "groq", "solo-base", "Fast API with post-processing"],
            },
        }

    def test_quality_preset_uses_offline_backend(self, file_config):
        """Quality preset should use wcpp backend for accuracy."""
        config = Config.resolve(file_config, preset=Preset(name="quality", preprocessor="deepfilter", asr_backend="wcpp", postprocessor="none"))
        assert config.asr_backend.name == "wcpp"
        assert config.asr_backend.type == "whisper_cpp"

    def test_quality_preset_uses_deepfilter(self, file_config):
        """Quality preset uses deepfilter for preprocessing."""
        config = Config.resolve(file_config, preset=Preset(name="quality", preprocessor="deepfilter", asr_backend="wcpp", postprocessor="none"))
        assert config.preprocessor.name == "deepfilter"

    def test_speed_preset_uses_postprocessor(self, file_config):
        """Speed preset can include post-processor if configured."""
        config = Config.resolve(file_config, preset=Preset(name="speed", preprocessor="none", asr_backend="groq", postprocessor="solo-base"))
        assert config.postprocessor.name == "solo-base"

    def test_file_mode_backend_override(self, file_config):
        """File mode preset backend can be overridden with CLI flag."""
        cli = CliOverrides(backend="groq")
        config = Config.resolve(file_config, preset=Preset(name="quality", preprocessor="deepfilter", asr_backend="wcpp", postprocessor="none"), cli_overrides=cli)
        assert config.asr_backend.name == "groq"
        assert config.asr_backend.api_key == "gsk_file"


class TestConfigResolutionPipeline:
    """Test the full config resolution pipeline from preset to usage."""

    @pytest.fixture
    def pipeline_config(self):
        """Config for testing resolution pipeline."""
        return {
            "asr_backends": {
                "fast_backend": {
                    "type": "api",
                    "api_base_url": "https://fast.api/v1/",
                    "api_key": "fast_key",
                    "model_name": "fast_model",
                },
                "slow_backend": {
                    "type": "api",
                    "api_base_url": "https://slow.api/v1/",
                    "api_key": "slow_key",
                    "model_name": "slow_model",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {
                "enhance": {
                    "backend": "groq",
                    "prompt": "enhance_prompt",
                },
                "restructure": {
                    "backend": "ollama",
                    "prompt": "restructure_prompt",
                },
            },
            "clipboard_max_chars": 75000,
            "presets": {
                "speed": ["none", "fast_backend", "enhance", "Fast with enhancement"],
                "quality": ["noisereduce", "slow_backend", "restructure", "Slow with restructure"],
            },
        }

    def test_speed_preset_full_pipeline(self, pipeline_config):
        """Full speed pipeline should resolve all components from preset."""
        config = Config.resolve(pipeline_config, preset=Preset(name="speed", preprocessor="none", asr_backend="fast_backend", postprocessor="enhance"))
        assert config.asr_backend.name == "fast_backend"
        assert config.preprocessor.name == "none"
        assert config.postprocessor.name == "enhance"
        assert config.output.clipboard_max_chars == 75000

    def test_quality_preset_full_pipeline(self, pipeline_config):
        """Full quality pipeline should resolve all components from preset."""
        config = Config.resolve(pipeline_config, preset=Preset(name="quality", preprocessor="noisereduce", asr_backend="slow_backend", postprocessor="restructure"))
        assert config.asr_backend.name == "slow_backend"
        assert config.preprocessor.name == "noisereduce"
        assert config.postprocessor.name == "restructure"
        assert config.output.clipboard_max_chars == 75000

    def test_cli_overrides_affect_full_pipeline(self, pipeline_config):
        """CLI overrides should override individual stages of the preset."""
        cli = CliOverrides(
            backend="slow_backend",
            preprocessor="none",
            post="restructure",
        )
        config = Config.resolve(pipeline_config, preset=Preset(name="speed", preprocessor="none", asr_backend="fast_backend", postprocessor="enhance"), cli_overrides=cli)
        assert config.asr_backend.name == "slow_backend"
        assert config.preprocessor.name == "none"
        assert config.postprocessor.name == "restructure"


class TestDeviceResolution:
    """Test device resolution in config."""

    @pytest.fixture
    def device_config(self):
        """Config with device settings."""
        return {
            "audio_device": "auto",
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-turbo",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
            "presets": {
                "test": ["none", "groq", "none", "Test preset"],
            },
        }

    def test_device_from_config(self, device_config):
        """Device resolution should read from config."""
        config = Config.resolve(device_config, preset=Preset(name="test", preprocessor="none", asr_backend="groq", postprocessor="none"))
        # Device resolution is deferred until RecorderConfig.resolve()
        recorder = config.recorder
        assert recorder.name in ["sounddevice", "arecord", "auto"]

    def test_device_cli_override(self, device_config):
        """CLI device override should take precedence."""
        cli = CliOverrides(device="auto")
        config = Config.resolve(device_config, preset=Preset(name="test", preprocessor="none", asr_backend="groq", postprocessor="none"), cli_overrides=cli)
        recorder = config.recorder
        assert recorder.name in ["sounddevice", "arecord"]


class TestErrorHandling:
    """Test error handling in preset-based config resolution."""

    def test_preset_references_nonexistent_backend(self):
        """Preset referencing a backend that doesn't exist should raise ValueError."""
        config_dict = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-turbo",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
        }
        with pytest.raises(ValueError, match="Backend 'wcpp' not found"):
            config = Config.resolve(config_dict, preset=Preset(name="test", preprocessor="none", asr_backend="wcpp", postprocessor="none"))
            config.asr_backend  # Lazy loading triggers error

    def test_missing_api_key_in_preset_backend(self):
        """Backend in preset missing API key should raise ValueError."""
        config_dict = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "model_name": "whisper-turbo",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
        }
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="requires api_key"):
                config = Config.resolve(config_dict, preset=Preset(name="test", preprocessor="none", asr_backend="groq", postprocessor="none"))
                config.asr_backend  # Lazy loading triggers error

    def test_preset_references_nonexistent_postprocessor(self):
        """Preset referencing a postprocessor that doesn't exist should raise ValueError."""
        config_dict = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "gsk_test",
                    "model_name": "whisper-turbo",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
        }
        with pytest.raises(ValueError, match="Postprocessor 'nonexistent' not found"):
            config = Config.resolve(config_dict, preset=Preset(name="test", preprocessor="none", asr_backend="groq", postprocessor="nonexistent"))
            config.postprocessor  # Lazy loading triggers error


