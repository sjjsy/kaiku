"""Unit tests for robust.py (chunk logic, quality check) and config.py."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch, MagicMock

import pytest
from pydub import AudioSegment

from asr2clip.robust import _find_chunk_boundaries, _check_quality
from asr2clip.config import find_config_path


# ---------------------------------------------------------------------------
# _check_quality
# ---------------------------------------------------------------------------

class TestCheckQuality:
    def test_normal_prose_passes(self):
        text = "The quick brown fox jumps over the lazy dog near the river bank."
        assert _check_quality(text) is True

    def test_too_few_words_fails(self):
        assert _check_quality("one two three") is False
        assert _check_quality("") is False
        assert _check_quality("   ") is False

    def test_exactly_five_words_passes(self):
        assert _check_quality("one two three four five") is True

    def test_hallucination_loop_fails(self):
        # Repeated single word = low unique ratio
        assert _check_quality("the the the the the the the the") is False

    def test_repeated_phrase_fails(self):
        phrase = "thank you " * 10
        assert _check_quality(phrase) is False

    def test_borderline_unique_ratio(self):
        # 50% unique: "a b c d a b c d" → 4 unique / 8 total = 0.5, boundary is < 0.5
        text = "alpha beta gamma delta alpha beta gamma delta"
        assert _check_quality(text) is True  # exactly 0.5, not < 0.5

    def test_just_below_unique_ratio(self):
        # 4 unique / 9 total ≈ 0.44 < 0.5
        text = "alpha beta gamma delta alpha beta gamma delta alpha"
        assert _check_quality(text) is False


# ---------------------------------------------------------------------------
# _find_chunk_boundaries
# ---------------------------------------------------------------------------

def _make_audio(duration_ms: int, sample_rate: int = 16000) -> AudioSegment:
    """Produce a silent AudioSegment of given duration."""
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


class TestFindChunkBoundaries:
    def test_short_audio_single_chunk(self):
        audio = _make_audio(30_000)  # 30 s
        bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        assert bounds == [(0, 30_000)]

    def test_exact_fit_single_chunk(self):
        audio = _make_audio(180_000)
        bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        assert bounds == [(0, 180_000)]

    def test_two_equal_chunks(self):
        audio = _make_audio(360_000)  # 6 min
        bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        assert len(bounds) == 2
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 360_000

    def test_chunks_are_contiguous(self):
        audio = _make_audio(500_000)
        bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        for i in range(len(bounds) - 1):
            assert bounds[i][1] == bounds[i + 1][0]

    def test_last_chunk_ends_at_total(self):
        audio = _make_audio(400_000)
        bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        assert bounds[-1][1] == 400_000

    def test_silence_midpoint_used(self):
        """If a silence falls in the valid window, chunk should end at its midpoint."""
        audio = _make_audio(200_000)
        # Fake a silence at 95000-105000 ms (midpoint 100000), well within the window
        fake_silences = [(95_000, 105_000)]
        with patch("asr2clip.robust.detect_silence", return_value=fake_silences):
            bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        # With a 180s window starting at 0, half=90000, end=180000.
        # Silence midpoint 100000 is in [90000, 180000], so first chunk should end at 100000.
        assert bounds[0][1] == 100_000

    def test_no_silence_hard_cut(self):
        audio = _make_audio(400_000)
        with patch("asr2clip.robust.detect_silence", return_value=[]):
            bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        # Hard cut: first chunk ends at exactly max_chunk_ms
        assert bounds[0] == (0, 180_000)

    def test_detect_silence_exception_falls_back(self):
        audio = _make_audio(400_000)
        with patch("asr2clip.robust.detect_silence", side_effect=RuntimeError("boom")):
            bounds = _find_chunk_boundaries(audio, max_chunk_ms=180_000)
        # Should still return valid boundaries (hard cuts)
        assert len(bounds) > 0
        assert bounds[-1][1] == 400_000


# ---------------------------------------------------------------------------
# find_config_path
# ---------------------------------------------------------------------------

class TestFindConfigPath:
    def test_explicit_path_found(self, tmp_path):
        cfg = tmp_path / "my.conf"
        cfg.write_text("backend: api\n")
        result = find_config_path(str(cfg))
        assert result == str(cfg)

    def test_explicit_path_not_found_falls_through(self, tmp_path):
        missing = str(tmp_path / "nonexistent.conf")
        # Should NOT return the missing explicit path; falls through to search
        with patch("asr2clip.config.CONFIG_PATHS", []):
            result = find_config_path(missing)
        assert result is None

    def test_search_order_first_match_wins(self, tmp_path):
        first = tmp_path / "first.conf"
        second = tmp_path / "second.conf"
        first.write_text("a: 1")
        second.write_text("b: 2")
        paths = [str(first), str(second)]
        with patch("asr2clip.config.CONFIG_PATHS", paths):
            result = find_config_path(None)
        assert result == str(first)

    def test_all_missing_returns_none(self, tmp_path):
        paths = [str(tmp_path / "a.conf"), str(tmp_path / "b.conf")]
        with patch("asr2clip.config.CONFIG_PATHS", paths):
            result = find_config_path(None)
        assert result is None

    def test_no_explicit_path_searches_defaults(self, tmp_path):
        cfg = tmp_path / "found.conf"
        cfg.write_text("backend: api\n")
        with patch("asr2clip.config.CONFIG_PATHS", [str(cfg)]):
            result = find_config_path(None)
        assert result == str(cfg)
