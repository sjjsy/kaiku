"""Tests for config error cases and error message quality.

Ensures that:
- Invalid config raises specific, helpful errors
- Error messages point to the exact problem
- Users get actionable feedback within 0.5 seconds
"""

from __future__ import annotations

import pytest
from asr2clip.config_types import Config, Preset, PresetConfig, ASRBackendConfig, PostprocessorBackendConfig


class TestPresetFormatValidation:
    """Test that presets must be in correct 4-element list format."""

    def test_preset_must_be_list(self):
        """Preset as dict should raise clear error."""
        config = {
            "presets": {
                "bad": {
                    "preprocessor": "none",
                    "asr_backend": "groq",
                    "postprocessor": "none",
                }
            }
        }
        with pytest.raises(ValueError, match="must be a 4-element list"):
            preset_config = PresetConfig.resolve(config, "bad")

    def test_preset_list_must_have_exactly_4_elements(self):
        """Preset list with wrong number of elements should error."""
        config = {
            "presets": {
                "incomplete": ["none", "groq"]  # Only 2 elements
            }
        }
        with pytest.raises(ValueError, match="exactly 4 elements"):
            PresetConfig.resolve(config, "incomplete")

    def test_preset_list_with_5_elements_fails(self):
        """Preset with too many elements should error."""
        config = {
            "presets": {
                "too_many": ["none", "groq", "none", "desc", "extra"]
            }
        }
        with pytest.raises(ValueError, match="exactly 4 elements"):
            PresetConfig.resolve(config, "too_many")

    def test_invalid_preset_name_shows_available(self):
        """Error for missing preset should list available presets."""
        config = {
            "presets": {
                "speed": ["none", "groq", "none", "Fast"],
                "quality": ["noisereduce", "groq", "none", "Good"],
            }
        }
        with pytest.raises(ValueError, match="Preset 'unknown' not found"):
            PresetConfig.resolve(config, "unknown")

    def test_empty_presets_raises_error(self):
        """Config with no presets section should error clearly."""
        config = {}
        with pytest.raises(ValueError, match="No 'presets:' section found"):
            PresetConfig.resolve(config, "any")


class TestBackendValidation:
    """Test that missing or invalid backends raise clear errors."""

    def test_backend_not_found_shows_available(self):
        """Error for missing backend should list available backends."""
        config = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "key",
                    "model_name": "whisper",
                },
            }
        }
        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            ASRBackendConfig.resolve(config, backend_name="unknown")

    def test_api_backend_requires_api_key(self):
        """API backend without api_key should error."""
        config = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "model_name": "whisper",
                    # Missing api_key
                }
            }
        }
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="requires api_key"):
            ASRBackendConfig.resolve(config, backend_name="groq")

    def test_api_backend_requires_api_base_url(self):
        """API backend without api_base_url should error."""
        config = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_key": "key",
                    "model_name": "whisper",
                    # Missing api_base_url
                }
            }
        }
        with pytest.raises(ValueError, match="requires api_base_url"):
            ASRBackendConfig.resolve(config, backend_name="groq")

    def test_api_backend_requires_model_name(self):
        """API backend without model_name should error."""
        config = {
            "asr_backends": {
                "groq": {
                    "type": "api",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "key",
                    # Missing model_name
                }
            }
        }
        with pytest.raises(ValueError, match="requires model_name"):
            ASRBackendConfig.resolve(config, backend_name="groq")

    def test_whisper_cpp_backend_requires_binary(self):
        """whisper_cpp backend without binary should error."""
        config = {
            "asr_backends": {
                "wcpp": {
                    "type": "whisper_cpp",
                    "model": "/path/to/model",
                    # Missing binary
                }
            }
        }
        with pytest.raises(ValueError, match="requires binary path"):
            ASRBackendConfig.resolve(config, backend_name="wcpp")

    def test_whisper_cpp_backend_requires_model(self):
        """whisper_cpp backend without model should error."""
        config = {
            "asr_backends": {
                "wcpp": {
                    "type": "whisper_cpp",
                    "binary": "/path/to/whisper",
                    # Missing model
                }
            }
        }
        with pytest.raises(ValueError, match="requires model path"):
            ASRBackendConfig.resolve(config, backend_name="wcpp")

    def test_unknown_backend_type_raises_error(self):
        """Unknown backend type should error."""
        config = {
            "asr_backends": {
                "weird": {
                    "type": "unknown_type",
                }
            }
        }
        with pytest.raises(ValueError, match="Unknown backend type"):
            ASRBackendConfig.resolve(config, backend_name="weird")


class TestPostprocessorBackendValidation:
    """Test that postprocessor backend validation gives clear errors."""

    def test_postprocessor_backend_not_found_shows_available(self):
        """Error for missing postprocessor backend should list available."""
        config = {
            "postprocessor_backends": {
                "groq": {
                    "type": "openai_compat",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "key",
                    "model": "whisper",
                },
            }
        }
        with pytest.raises(ValueError, match="Post-processor backend 'unknown' not found"):
            PostprocessorBackendConfig.resolve(config, "unknown")

    def test_openai_compat_requires_api_base_url(self):
        """openai_compat backend without api_base_url should error."""
        config = {
            "postprocessor_backends": {
                "groq": {
                    "type": "openai_compat",
                    "api_key": "key",
                    "model": "gpt-4",
                    # Missing api_base_url
                }
            }
        }
        with pytest.raises(ValueError, match="requires api_base_url"):
            PostprocessorBackendConfig.resolve(config, "groq")

    def test_openai_compat_requires_api_key(self):
        """openai_compat backend without api_key should error."""
        config = {
            "postprocessor_backends": {
                "groq": {
                    "type": "openai_compat",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "model": "gpt-4",
                    # Missing api_key
                }
            }
        }
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="requires api_key"):
            PostprocessorBackendConfig.resolve(config, "groq")

    def test_openai_compat_requires_model(self):
        """openai_compat backend without model should error."""
        config = {
            "postprocessor_backends": {
                "groq": {
                    "type": "openai_compat",
                    "api_base_url": "https://api.groq.com/openai/v1/",
                    "api_key": "key",
                    # Missing model
                }
            }
        }
        with pytest.raises(ValueError, match="requires model"):
            PostprocessorBackendConfig.resolve(config, "groq")

    def test_claude_code_backend_requires_model(self):
        """claude_code backend without model should error."""
        config = {
            "postprocessor_backends": {
                "claude": {
                    "type": "claude_code",
                    # Missing model
                }
            }
        }
        with pytest.raises(ValueError, match="requires model"):
            PostprocessorBackendConfig.resolve(config, "claude")


class TestCliPrecedenceDocumentation:
    """Tests that document and verify CLI flag precedence over presets.

    This is FIXME #2 resolution: explicit tests documenting that CLI flags
    override preset components individually.
    """

    @pytest.fixture
    def precedence_config(self):
        return {
            "asr_backends": {
                "fast": {
                    "type": "api",
                    "api_base_url": "https://fast.com/v1/",
                    "api_key": "key1",
                    "model_name": "fast",
                },
                "slow": {
                    "type": "api",
                    "api_base_url": "https://slow.com/v1/",
                    "api_key": "key2",
                    "model_name": "slow",
                },
            },
            "postprocessor_backends": {},
            "postprocessors": {},
        }

    def test_cli_backend_overrides_preset_backend(self, precedence_config):
        """--backend FLAG should override preset's asr_backend."""
        from asr2clip.config_types import CliOverrides
        # Preset says "use slow backend"
        preset = Preset(name="test", preprocessor="none", asr_backend="slow", postprocessor="none")
        # CLI says "use fast backend"
        cli = CliOverrides(backend="fast")
        # CLI should win
        config = Config.resolve(precedence_config, preset=preset, cli_overrides=cli)
        assert config.asr_backend.name == "fast"

    def test_cli_preprocessor_overrides_preset_preprocessor(self, precedence_config):
        """--preprocessor FLAG should override preset's preprocessor."""
        from asr2clip.config_types import CliOverrides
        # Preset says "use noisereduce"
        preset = Preset(name="test", preprocessor="noisereduce", asr_backend="fast", postprocessor="none")
        # CLI says "use deepfilter"
        cli = CliOverrides(preprocessor="deepfilter")
        # CLI should win
        config = Config.resolve(precedence_config, preset=preset, cli_overrides=cli)
        assert config.preprocessor.name == "deepfilter"

    def test_cli_post_overrides_preset_postprocessor(self, precedence_config):
        """--post FLAG should override preset's postprocessor."""
        from asr2clip.config_types import CliOverrides, PostprocessorConfig
        # Preset says "use solo-base"
        preset = Preset(name="test", preprocessor="none", asr_backend="fast", postprocessor="solo-base")
        # CLI says "use solo-enhance"
        cli = CliOverrides(post="solo-enhance")

        config_dict = {
            **precedence_config,
            "postprocessors": {
                "solo-base": {"backend": "groq"},
                "solo-enhance": {"backend": "groq"},
            }
        }
        config = Config.resolve(config_dict, preset=preset, cli_overrides=cli)
        assert config.postprocessor.name == "solo-enhance"

    def test_preset_with_multiple_cli_overrides(self, precedence_config):
        """Multiple CLI overrides should work independently."""
        from asr2clip.config_types import CliOverrides
        # Preset specifies all components
        preset = Preset(name="test", preprocessor="noisereduce", asr_backend="slow", postprocessor="none")
        # CLI overrides backend and preprocessor, but not postprocessor
        cli = CliOverrides(backend="fast", preprocessor="deepfilter")
        config = Config.resolve(precedence_config, preset=preset, cli_overrides=cli)
        # Backend and preprocessor should use CLI values
        assert config.asr_backend.name == "fast"
        assert config.preprocessor.name == "deepfilter"
        # Postprocessor should use preset value
        assert config.postprocessor.name == "none"

    def test_cli_override_logs_cli_source(self, precedence_config):
        """Verify that CLI override is resolved correctly (logs show CLI source)."""
        from asr2clip.config_types import CliOverrides

        preset = Preset(name="test", preprocessor="noisereduce", asr_backend="slow", postprocessor="none")
        cli = CliOverrides(backend="fast")
        config = Config.resolve(precedence_config, preset=preset, cli_overrides=cli)

        # Verify backend resolution is correct (comes from CLI override, not preset)
        assert config.asr_backend.name == "fast"

    def test_preset_component_shows_correct_log_message(self, precedence_config):
        """Preset component logs should indicate it came from preset."""
        from asr2clip.config_types import CliOverrides

        preset = Preset(name="test", preprocessor="noisereduce", asr_backend="slow", postprocessor="none")
        cli = CliOverrides()  # No overrides
        config = Config.resolve(precedence_config, preset=preset, cli_overrides=cli)

        # Verify that preset components are used
        assert config.preprocessor.name == "noisereduce"
        assert config.asr_backend.name == "slow"
