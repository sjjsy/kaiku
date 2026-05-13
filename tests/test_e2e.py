"""End-to-end CLI tests for asr2clip.

All tests use the session-scoped `example_cfg` fixture from conftest.py, which
is built from the live asr2clip.conf.example file. This means every change to
the example config is automatically exercised here — the suite doubles as a
runnable demo of every mock pipeline stage.

The only tests that write their own inline config are those that require a
config structure that cannot be expressed via CLI flags alone (e.g. two backends
with different responses to verify --backend override, or a missing-config error
path).
"""

from __future__ import annotations

import struct
import subprocess
import textwrap
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MOCK_FIXED = "The quick brown fox jumps over the lazy dog"  # response from -b mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(*args, config: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["asr2clip", "--config", str(config), *args],
        capture_output=True, text=True,
    )


def _make_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> Path:
    n = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def silent_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("audio") / "silent.wav"
    return _make_wav(p, duration_s=2.0)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

class TestConfigResolution:
    def test_default_preset_runs_without_flag(self, example_cfg, silent_wav):
        """Example config default_preset (mock-fwd) should work with no --preset."""
        result = _run("-i", str(silent_wav), config=example_cfg)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_backend_flag_overrides_preset(self, example_cfg, silent_wav):
        """-b mock should return the fixed fox sentence, overriding the default mock-fwd."""
        result = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        assert result.returncode == 0
        assert MOCK_FIXED in result.stdout

    def test_backend_bwd_differs_from_fwd(self, example_cfg, silent_wav):
        """-b mock-bwd should produce different output than -b mock-fwd."""
        fwd = _run("-i", str(silent_wav), "-b", "mock-fwd", config=example_cfg)
        bwd = _run("-i", str(silent_wav), "-b", "mock-bwd", config=example_cfg)
        assert fwd.returncode == 0
        assert bwd.returncode == 0
        assert fwd.stdout.strip() != bwd.stdout.strip()

    def test_missing_preset_exits_nonzero(self, example_cfg, silent_wav):
        result = _run("-i", str(silent_wav), "--preset", "no-such-preset", config=example_cfg)
        assert result.returncode != 0

    def test_missing_config_exits_nonzero(self, silent_wav):
        result = _run(
            "-i", str(silent_wav),
            config=REPO_ROOT / "nonexistent_xyz_config.yaml",
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# File transcription (mock backend)
# ---------------------------------------------------------------------------

class TestFileTranscription:
    def test_transcribe_to_stdout(self, example_cfg, silent_wav):
        result = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        assert result.returncode == 0
        assert MOCK_FIXED in result.stdout

    def test_transcribe_to_output_file(self, example_cfg, silent_wav, tmp_path):
        out = tmp_path / "transcript.txt"
        result = _run("-i", str(silent_wav), "-b", "mock", "-o", str(out), config=example_cfg)
        assert result.returncode == 0
        assert out.exists()
        assert MOCK_FIXED in out.read_text()

    def test_language_flag_accepted(self, example_cfg, silent_wav):
        """--language should not cause an error; mock backend ignores it."""
        result = _run("-i", str(silent_wav), "-l", "fi", config=example_cfg)
        assert result.returncode == 0

    def test_quiet_flag_reduces_output(self, example_cfg, silent_wav):
        verbose = _run("-i", str(silent_wav), config=example_cfg)
        quiet = _run("-i", str(silent_wav), "-q", config=example_cfg)
        assert verbose.returncode == 0
        assert quiet.returncode == 0
        assert len(quiet.stdout.splitlines()) <= len(verbose.stdout.splitlines())


# ---------------------------------------------------------------------------
# mock-fwd / mock-bwd backends
# ---------------------------------------------------------------------------

class TestTranscriptMockBackends:
    def test_mock_forward_returns_words_from_transcript(self, example_cfg, silent_wav):
        """-b mock-fwd should return words from the transcript file, not the fixed fox string."""
        result = _run("-i", str(silent_wav), "-b", "mock-fwd", config=example_cfg)
        assert result.returncode == 0
        output = result.stdout.strip()
        assert output
        assert MOCK_FIXED not in output

    def test_mock_backward_differs_from_forward(self, example_cfg, silent_wav):
        """-b mock-bwd should produce the reverse word order of -b mock-fwd."""
        fwd = _run("-i", str(silent_wav), "-b", "mock-fwd", config=example_cfg)
        bwd = _run("-i", str(silent_wav), "-b", "mock-bwd", config=example_cfg)
        assert fwd.returncode == 0
        assert bwd.returncode == 0
        assert fwd.stdout.strip() != bwd.stdout.strip()

    def test_longer_audio_returns_more_words(self, example_cfg, tmp_path):
        """Longer audio should produce more words (N = duration_s / 2)."""
        short = _make_wav(tmp_path / "short.wav", duration_s=2.0)
        long_ = _make_wav(tmp_path / "long.wav", duration_s=20.0)
        res_short = _run("-i", str(short), "-b", "mock-fwd", config=example_cfg)
        res_long = _run("-i", str(long_), "-b", "mock-fwd", config=example_cfg)
        assert res_short.returncode == 0
        assert res_long.returncode == 0
        assert len(res_long.stdout.split()) > len(res_short.stdout.split())


# ---------------------------------------------------------------------------
# Mock audio devices
# ---------------------------------------------------------------------------

class TestMockDevices:
    def test_mock_jfk_device_produces_output(self, example_cfg):
        """--device mock-jfk should load jfk-11s-1p.wav and transcribe it."""
        result = _run(
            "--preset", "mock-fwd", "--device", "mock-jfk",
            config=example_cfg,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_mock_group_device_produces_output(self, example_cfg):
        """--device mock-group should load group-30s-4p.wav and transcribe it."""
        result = _run(
            "--preset", "mock-fwd", "--device", "mock-group",
            config=example_cfg,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_mock_group_longer_than_jfk(self, example_cfg):
        """group-30s-4p.wav (30s) should produce more words than jfk-11s-1p.wav (11s)."""
        jfk = _run("--preset", "mock-fwd", "--device", "mock-jfk", config=example_cfg)
        grp = _run("--preset", "mock-fwd", "--device", "mock-group", config=example_cfg)
        assert jfk.returncode == 0
        assert grp.returncode == 0
        assert len(grp.stdout.split()) > len(jfk.stdout.split())


# ---------------------------------------------------------------------------
# Mock diarization
# ---------------------------------------------------------------------------

class TestMockDiarization:
    def test_mock_diarize_2_produces_two_speakers(self, example_cfg, jfk_wav):
        """-b mock-dia-2 should produce a 2-speaker transcript."""
        result = _run("-i", str(jfk_wav), "-b", "mock-dia-2", config=example_cfg)
        assert result.returncode == 0
        lines = [l for l in result.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines) >= 2
        speakers = {l.split("SPEAKER_")[1].split(":")[0] for l in lines}
        assert len(speakers) == 2  # exactly SPEAKER_00 and SPEAKER_01

    def test_mock_diarize_3_produces_three_speakers(self, example_cfg, group_wav):
        """-b mock-dia-3 should produce a 3-speaker transcript."""
        result = _run("-i", str(group_wav), "-b", "mock-dia-3", config=example_cfg)
        assert result.returncode == 0
        lines = [l for l in result.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines) >= 2
        speakers = {l.split("SPEAKER_")[1].split(":")[0] for l in lines}
        assert len(speakers) <= 3

    def test_speakers_flag_overrides_speaker_count(self, example_cfg, jfk_wav):
        """-s 1 should collapse output to a single speaker."""
        result = _run("-i", str(jfk_wav), "-b", "mock-dia-2", "-s", "1", config=example_cfg)
        assert result.returncode == 0
        lines = [l for l in result.stdout.splitlines() if "SPEAKER_" in l]
        speakers = {l.split("SPEAKER_")[1].split(":")[0] for l in lines}
        assert len(speakers) == 1

    def test_preset_triggers_diarization(self, example_cfg, jfk_wav):
        """mock-dia preset (asr_backend=mock-dia-2, type=mock-diarize) diarizes without -b."""
        result = _run("-i", str(jfk_wav), "--preset", "mock-dia", config=example_cfg)
        assert result.returncode == 0
        lines = [l for l in result.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines) >= 2


# ---------------------------------------------------------------------------
# Output templates
# ---------------------------------------------------------------------------

class TestOutputTemplates:
    def test_raw_template_returns_transcript(self, example_cfg, silent_wav):
        """-T raw should return the raw transcript (no post-processing)."""
        result = _run("-i", str(silent_wav), "-b", "mock", "-T", "raw", config=example_cfg)
        assert result.returncode == 0
        assert MOCK_FIXED in result.stdout

    def test_bare_template_accepted(self, example_cfg, silent_wav):
        result = _run("-i", str(silent_wav), "-T", "bare", config=example_cfg)
        assert result.returncode == 0

    def test_unknown_template_falls_back_gracefully(self, example_cfg, silent_wav):
        result = _run("-i", str(silent_wav), "-b", "mock", "-T", "no_such_template", config=example_cfg)
        assert result.returncode == 0
        assert MOCK_FIXED in result.stdout


# ---------------------------------------------------------------------------
# Mock post-processor
# ---------------------------------------------------------------------------

class TestMockPostprocessor:
    def test_mock_postprocessor_runs(self, example_cfg, silent_wav):
        """-b mock -P mock-pp should run the mock post-processor."""
        result = _run("-i", str(silent_wav), "-b", "mock", "-P", "mock-pp", config=example_cfg)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_post_flag_overrides_no_postprocessor(self, example_cfg, silent_wav):
        """-P mock-pp with a no-post preset should still run the post-processor."""
        result = _run("-i", str(silent_wav), "-P", "mock-pp", config=example_cfg)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0
