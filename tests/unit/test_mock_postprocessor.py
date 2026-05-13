"""Tests for the mock LLM post-processor."""

from __future__ import annotations

import pytest

from asr2clip.postprocessors.mock import MockPostProcessor, _analyze_text
from asr2clip.postprocessors.base import PostMetadata
from asr2clip.postprocessors import make_postprocessor
from conftest import _config_with_postprocessors


class TestTextAnalysis:
    """Test the text analysis function."""

    def test_analyze_simple_text(self):
        """Analysis should extract longest, shortest, most frequent words and counts."""
        text = "the quick brown fox jumps over the lazy dog"
        result = _analyze_text(text)
        # Longest word: "quick" or "brown" (5 chars)
        assert "longest=" in result
        assert "shortest=" in result
        assert "most_frequent=" in result
        assert "lines=1" in result
        assert "words=9" in result
        assert "chars=" in result

    def test_analyze_empty_text(self):
        """Analysis of empty text should return all zeros."""
        result = _analyze_text("")
        assert result == "longest=-, shortest=-, most_frequent=-, lines=0, words=0, chars=0"

    def test_analyze_multiline_text(self):
        """Analysis should count multiple lines."""
        text = "line one\nline two\nline three"
        result = _analyze_text(text)
        assert "lines=3" in result

    def test_analyze_most_frequent_word(self):
        """Most frequent word should be identified correctly."""
        text = "test test test hello world"
        result = _analyze_text(text)
        assert "most_frequent=test" in result

    def test_analyze_case_insensitive_frequency(self):
        """Word frequency should be case-insensitive."""
        text = "Test test TEST hello"
        result = _analyze_text(text)
        assert "most_frequent=test" in result


class TestMockPostProcessor:
    """Test MockPostProcessor implementation."""

    def test_mock_postprocessor_basic_properties(self):
        """Mock postprocessor should have correct properties."""
        processor = MockPostProcessor(
            prompt_name="test",
            model="claude-opus",
            system_prompt="Test prompt"
        )
        assert processor.name == "test"
        assert processor.model == "claude-opus"
        assert processor.backend_type == "mock"

    def test_mock_postprocessor_analyzes_prompt_and_transcript(self):
        """Mock postprocessor should analyze both prompt and transcript."""
        prompt = "analyze this prompt"
        processor = MockPostProcessor(
            system_prompt=prompt,
            model="test-model"
        )

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=1.0,
            language="en",
            prompt_name="mock",
        )

        transcript = "this is the transcript to analyze"
        result = processor.process(transcript, metadata=metadata)

        # Should contain analysis format
        assert "Prompt analyzed:" in result
        assert "Transcript analyzed:" in result
        assert "*Yours truly, Test-Model\n*" in result

    def test_mock_postprocessor_model_title_case(self):
        """Model name should be converted to Title Case in output."""
        processor = MockPostProcessor(
            model="claude-opus-4-7",
            system_prompt="test"
        )

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=1.0,
            language="en",
            prompt_name="mock",
        )

        result = processor.process("test", metadata=metadata)
        assert "*Yours truly, Claude-Opus-4-7\n*" in result

    def test_mock_postprocessor_with_empty_prompt(self):
        """Mock postprocessor should handle empty prompt."""
        processor = MockPostProcessor(
            model="test-model",
            system_prompt=""
        )

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=1.0,
            language="en",
            prompt_name="mock",
        )

        result = processor.process("test transcript", metadata=metadata)
        assert "Prompt analyzed:" in result
        assert "longest=-, shortest=-, most_frequent=-" in result

    def test_mock_postprocessor_consistent_analysis(self):
        """Same input should produce consistent analysis."""
        processor = MockPostProcessor(
            model="test",
            system_prompt="consistent test"
        )

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=1.0,
            language="en",
            prompt_name="mock",
        )

        result1 = processor.process("transcript", metadata=metadata)
        result2 = processor.process("transcript", metadata=metadata)
        assert result1 == result2


class TestMockPostProcessorIntegration:
    """Test mock postprocessor with config and make_postprocessor."""

    def test_make_postprocessor_with_mock_backend(self):
        """make_postprocessor should create mock postprocessor from config."""
        config = _config_with_postprocessors({
            "postprocessors": {
                "test_mock": {"backend": "mock", "prompt": "Test prompt"},
            },
            "postprocessor_backends": {
                "mock": {"type": "mock", "model": "gpt-4"},
            },
        })
        processor = make_postprocessor("test_mock", config)
        assert isinstance(processor, MockPostProcessor)
        assert processor.name == "test_mock"
        assert processor.model == "gpt-4"

    def test_make_postprocessor_mock_backend_analysis(self):
        """Mock postprocessor should analyze from config."""
        config = _config_with_postprocessors({
            "postprocessors": {
                "analyze": {"backend": "mock", "prompt": "Analyze this content"},
            },
            "postprocessor_backends": {
                "mock": {"type": "mock", "model": "test-model"},
            },
        })
        processor = make_postprocessor("analyze", config)

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=1.0,
            language="en",
            prompt_name="analyze",
        )
        result = processor.process("test input", metadata=metadata)

        assert "Prompt analyzed:" in result
        assert "Transcript analyzed:" in result
        assert "*Yours truly, Test-Model\n*" in result

    def test_make_postprocessor_mock_with_model_override(self):
        """Mock postprocessor should accept model override."""
        config = _config_with_postprocessors({
            "postprocessors": {
                "mock": {"backend": "mock", "prompt": "Test"},
            },
            "postprocessor_backends": {
                "mock": {"type": "mock", "model": "default-model"},
            },
        })
        processor = make_postprocessor("mock", config, model_override="custom-model")
        assert processor.model == "custom-model"


class TestMockPostProcessorE2E:
    """End-to-end tests with mock postprocessor in full pipeline."""

    def test_mock_postprocessor_in_full_config(self):
        """Mock postprocessor should work in full config."""
        config = _config_with_postprocessors({
            "postprocessor_backends": {
                "mock": {"type": "mock", "model": "mock-analyzer"},
            },
            "postprocessors": {
                "analyze": {"backend": "mock", "prompt": "Analyze the content"},
            },
            "output_templates": {"default": "{result}"},
        })
        processor = make_postprocessor("analyze", config)
        assert isinstance(processor, MockPostProcessor)

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=5.0,
            language="en",
            prompt_name="analyze",
        )
        result = processor.process("test transcript content", metadata=metadata)

        # Verify analysis format
        assert "Prompt analyzed:" in result
        assert "Transcript analyzed:" in result
        assert "*Yours truly, Mock-Analyzer\n*" in result
        # Verify it contains word counts
        assert "words=" in result
        assert "chars=" in result
        assert "lines=" in result

    def test_mock_postprocessor_analyzes_long_content(self):
        """Mock postprocessor should handle long content."""
        prompt = "Once upon a time there was a very long prompt that went on and on"
        config = _config_with_postprocessors({
            "postprocessor_backends": {
                "mock": {"type": "mock", "model": "analyzer"},
            },
            "postprocessors": {
                "test": {"backend": "mock", "prompt": prompt},
            },
        })
        processor = make_postprocessor("test", config)

        metadata = PostMetadata(
            date="2026-05-12",
            duration_s=5.0,
            language="en",
            prompt_name="test",
        )
        transcript = "a much longer transcript with many words that we will analyze"
        result = processor.process(transcript, metadata=metadata)

        assert "Prompt analyzed:" in result
        assert "Transcript analyzed:" in result
        # Check word counts are reasonable
        assert "words=" in result
        assert "*Yours truly, Analyzer\n*" in result
