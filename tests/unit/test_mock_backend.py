"""Tests for the mock ASR backend."""

from __future__ import annotations

import pytest

from asr2clip.backends.mock import MockConfig, transcribe as mock_transcribe
from asr2clip.config_types import ASRBackendConfig


class TestMockConfig:
    """Test MockConfig creation and defaults."""

    def test_mock_config_default_response(self):
        """MockConfig should use default response when not specified."""
        cfg = MockConfig.from_config({})
        assert "quick brown fox" in cfg.response.lower()
        assert cfg.latency_ms == 0

    def test_mock_config_custom_response(self):
        """MockConfig should accept custom response."""
        custom = "This is a test transcript"
        cfg = MockConfig.from_config({"response": custom})
        assert cfg.response == custom

    def test_mock_config_latency(self):
        """MockConfig should accept latency setting."""
        cfg = MockConfig.from_config({"latency_ms": 500})
        assert cfg.latency_ms == 500

    def test_mock_config_all_fields(self):
        """MockConfig should accept all fields."""
        cfg = MockConfig.from_config({
            "response": "Custom response",
            "latency_ms": 250
        })
        assert cfg.response == "Custom response"
        assert cfg.latency_ms == 250


class TestMockTranscribe:
    """Test mock transcription."""

    def test_mock_transcribe_returns_response(self, tmp_path):
        """Mock transcribe should return configured response."""
        audio_file = tmp_path / "dummy.wav"
        audio_file.touch()

        cfg = MockConfig(response="Test transcript")
        result = mock_transcribe(str(audio_file), cfg)
        assert result == "Test transcript"

    def test_mock_transcribe_default_response(self, tmp_path):
        """Mock transcribe should use default response."""
        audio_file = tmp_path / "dummy.wav"
        audio_file.touch()

        cfg = MockConfig()
        result = mock_transcribe(str(audio_file), cfg)
        assert len(result) > 0
        assert "quick" in result.lower()

    def test_mock_transcribe_ignores_audio_file(self, tmp_path):
        """Mock transcribe should work with non-existent audio file."""
        cfg = MockConfig(response="Test")
        # File doesn't need to exist
        result = mock_transcribe("/nonexistent/audio.wav", cfg)
        assert result == "Test"


class TestMockBackendConfig:
    """Test ASRBackendConfig with mock type."""

    def test_resolve_mock_backend(self):
        """Config should resolve mock backend type."""
        config = {
            "asr_backends": {
                "test_mock": {
                    "type": "mock",
                    "response": "Mock test response"
                }
            }
        }

        backend = ASRBackendConfig.resolve(config, backend_name="test_mock")
        assert backend.name == "test_mock"
        assert backend.type == "mock"
        assert backend.response == "Mock test response"

    def test_resolve_mock_backend_with_latency(self):
        """Config should resolve mock backend with latency."""
        config = {
            "asr_backends": {
                "slow_mock": {
                    "type": "mock",
                    "latency_ms": 1000
                }
            }
        }

        backend = ASRBackendConfig.resolve(config, backend_name="slow_mock")
        assert backend.type == "mock"
        assert backend.latency_ms == 1000

    def test_resolve_mock_backend_default_latency(self):
        """Config should default latency to 0."""
        config = {
            "asr_backends": {
                "mock": {
                    "type": "mock"
                }
            }
        }

        backend = ASRBackendConfig.resolve(config, backend_name="mock")
        assert backend.latency_ms == 0


class TestMockBackendTest:
    """Test the mock backend's test() function."""

    def test_mock_backend_test_always_succeeds(self):
        """Mock backend test should always return True."""
        cfg = MockConfig()
        from asr2clip.backends.mock import test
        assert test(cfg) is True

    def test_mock_backend_test_with_custom_config(self):
        """Mock backend test should succeed with custom config."""
        cfg = MockConfig(response="Test", latency_ms=100)
        from asr2clip.backends.mock import test
        assert test(cfg) is True
