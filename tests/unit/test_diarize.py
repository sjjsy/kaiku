"""Unit tests for asr2clip.diarize — pure logic functions."""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest

from asr2clip.diarize import _fmt_ts, _format_transcript, DiarizationError, run_diarization


# ---------------------------------------------------------------------------
# _fmt_ts
# ---------------------------------------------------------------------------

class TestFmtTs:
    def test_zero(self):
        assert _fmt_ts(0.0) == "00:00:00"

    def test_seconds_only(self):
        assert _fmt_ts(45.9) == "00:00:45"

    def test_one_minute(self):
        assert _fmt_ts(60.0) == "00:01:00"

    def test_59_minutes_59_seconds(self):
        assert _fmt_ts(3599.0) == "00:59:59"

    def test_one_hour(self):
        assert _fmt_ts(3600.0) == "01:00:00"

    def test_over_one_hour(self):
        assert _fmt_ts(3661.0) == "01:01:01"

    def test_fractional_seconds_truncated(self):
        assert _fmt_ts(90.999) == "00:01:30"


# ---------------------------------------------------------------------------
# _format_transcript
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def test_single_segment(self):
        segs = [{"speaker": "SPEAKER_00", "text": "Hello world.", "start": 0.0}]
        result = _format_transcript(segs)
        assert result == "[00:00:00] SPEAKER_00: Hello world."

    def test_consecutive_same_speaker_merged(self):
        segs = [
            {"speaker": "SPEAKER_00", "text": "First sentence.", "start": 0.0},
            {"speaker": "SPEAKER_00", "text": "Second sentence.", "start": 5.0},
        ]
        result = _format_transcript(segs)
        lines = result.splitlines()
        assert len(lines) == 1
        assert "First sentence. Second sentence." in lines[0]

    def test_speaker_change_creates_new_line(self):
        segs = [
            {"speaker": "SPEAKER_00", "text": "Hello.", "start": 0.0},
            {"speaker": "SPEAKER_01", "text": "Hi there.", "start": 3.0},
        ]
        result = _format_transcript(segs)
        lines = result.splitlines()
        assert len(lines) == 2
        assert "SPEAKER_00" in lines[0]
        assert "SPEAKER_01" in lines[1]

    def test_empty_text_segments_skipped(self):
        segs = [
            {"speaker": "SPEAKER_00", "text": "Real content.", "start": 0.0},
            {"speaker": "SPEAKER_00", "text": "   ", "start": 1.0},
            {"speaker": "SPEAKER_00", "text": "", "start": 2.0},
        ]
        result = _format_transcript(segs)
        lines = result.splitlines()
        assert len(lines) == 1
        assert "Real content." in lines[0]

    def test_missing_speaker_defaults_to_speaker_00(self):
        segs = [{"text": "No speaker key.", "start": 0.0}]
        result = _format_transcript(segs)
        assert "SPEAKER_00" in result

    def test_timestamp_uses_segment_start(self):
        segs = [{"speaker": "SPEAKER_01", "text": "Late remark.", "start": 3661.0}]
        result = _format_transcript(segs)
        assert "[01:01:01]" in result

    def test_empty_segments(self):
        assert _format_transcript([]) == ""

    def test_three_speakers_interleaved(self):
        segs = [
            {"speaker": "A", "text": "One.", "start": 0.0},
            {"speaker": "B", "text": "Two.", "start": 5.0},
            {"speaker": "A", "text": "Three.", "start": 10.0},
            {"speaker": "C", "text": "Four.", "start": 15.0},
        ]
        result = _format_transcript(segs)
        lines = result.splitlines()
        assert len(lines) == 4
        assert "A: One." in lines[0]
        assert "B: Two." in lines[1]
        assert "A: Three." in lines[2]
        assert "C: Four." in lines[3]


# ---------------------------------------------------------------------------
# run_diarization error paths (no real WhisperX needed)
# ---------------------------------------------------------------------------

class TestRunDiarizationErrors:
    def test_whisperx_not_installed_raises_diarization_error(self):
        config = {"diarize_hf_token": "hf_test"}
        with patch.dict("sys.modules", {"whisperx": None}):
            with pytest.raises(DiarizationError, match="whisperx is not installed"):
                run_diarization("/tmp/fake.wav", config)

    def test_no_hf_token_raises_diarization_error(self):
        mock_wx = MagicMock()
        config = {}  # no token in config, no env var
        with patch.dict("sys.modules", {"whisperx": mock_wx, "torch": MagicMock()}):
            import os
            env_backup = os.environ.pop("HF_TOKEN", None)
            try:
                with pytest.raises(DiarizationError, match="HuggingFace token"):
                    run_diarization("/tmp/fake.wav", config)
            finally:
                if env_backup is not None:
                    os.environ["HF_TOKEN"] = env_backup

    def test_num_speakers_overrides_config_hints(self):
        """num_speakers=2 should set both min and max to 2, ignoring config hints."""
        mock_wx = MagicMock()
        mock_wx.load_model.return_value = MagicMock()
        mock_wx.load_audio.return_value = MagicMock()

        transcribe_result = {"segments": [], "language": "fi"}
        mock_wx.load_model.return_value.transcribe.return_value = transcribe_result
        mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_wx.align.return_value = {"segments": []}

        mock_diarize_pipeline = MagicMock(return_value=MagicMock())
        mock_wx.DiarizationPipeline.return_value = mock_diarize_pipeline
        mock_wx.assign_word_speakers.return_value = {"segments": []}

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        config = {
            "diarize_hf_token": "hf_test",
            "diarize_min_speakers": 3,
            "diarize_max_speakers": 6,
        }

        with patch.dict("sys.modules", {"whisperx": mock_wx, "torch": mock_torch}):
            result = run_diarization("/tmp/fake.wav", config, num_speakers=2)

        # Called with min_speakers=2, max_speakers=2
        call_kwargs = mock_diarize_pipeline.call_args[1]
        assert call_kwargs.get("min_speakers") == 2
        assert call_kwargs.get("max_speakers") == 2
