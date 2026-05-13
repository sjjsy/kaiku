"""End-to-end CLI tests for asr2clip.

# Testing strategy

All tests treat asr2clip as a black box: they run the CLI as a subprocess and
assert on exit code, stdout, stderr, and written files.  No internal module is
imported; no mocks are injected.

This is intentional.  The tool's contract lives entirely at the CLI boundary:
given a config file and a set of flags, the output (stdout, written files) and
the log lines (stderr) must be exactly what the documentation says.  A test
that passes but allows the wrong backend to be used is worthless.

## How to test config resolution

The logging contract from CLAUDE.md requires that every config decision states
*why* a value was chosen.  Those log lines appear in stderr in verbose mode.
We use them as assertions:

    result = _run("-i", wav, "-b", "wcpp", config=cfg)
    assert "Using backend: wcpp (CLI -b)" in result.stderr

This verifies end-to-end that the CLI flag reached the right code path and
that the correct value was selected — without importing a single module.

## Fixture source

The session-scoped `example_cfg` fixture is derived directly from
`asr2clip.conf.example`.  Every change to the example config is automatically
exercised — the suite doubles as a runnable demo of every mock pipeline stage.

Tests that require a config structure not expressible via flags alone (e.g. two
backends with conflicting settings to verify override order) write their own
inline YAML config.
"""

from __future__ import annotations

import os
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).parent.parent
MOCK_FIXED = "The quick brown fox jumps over the lazy dog"  # response from -b mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    *args,
    config: Path,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["asr2clip", "--config", str(config), *args],
        capture_output=True, text=True,
        env=env,
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


@pytest.fixture
def toggle_runtime(tmp_path: Path, example_cfg: Path):
    """Per-test XDG_RUNTIME_DIR for --toggle E2E tests.

    The mock recorder forks a child that sleeps until the *second* --toggle
    sends SIGTERM. Tests that only assert on the *first* toggle would otherwise
    leave that child running (visible as extra ``asr2clip`` rows in ``ps``).
    If the lock file still exists after a test, we invoke --toggle once more
    to stop the recorder and reap the mock child.
    """
    env = {**os.environ, "XDG_RUNTIME_DIR": str(tmp_path)}
    yield tmp_path, env
    if (tmp_path / "asr2clip.lock").exists():
        _run(
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            config=example_cfg, env=env,
        )


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

class TestConfigResolution:
    def test_default_preset_runs_without_flag(self, example_cfg, silent_wav):
        """Example config default_preset (mock-fwd) should work with no --preset."""
        result = _run("-i", str(silent_wav), config=example_cfg)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_short_preset_flag(self, example_cfg, silent_wav):
        """-x NAME should work identically to --preset NAME."""
        result = _run("-i", str(silent_wav), "-x", "mock-fwd", config=example_cfg)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_backend_flag_overrides_preset(self, example_cfg, silent_wav):
        """-b mock should return the fixed fox sentence, overriding the default mock-fwd."""
        result = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        assert result.returncode == 0
        assert MOCK_FIXED in result.stdout

    def test_backend_override_logged_with_source(self, example_cfg, silent_wav):
        """-b mock should log 'Using backend: mock (CLI -b)' in stderr."""
        result = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        assert result.returncode == 0
        assert "Using backend: mock (CLI -b)" in result.stderr

    def test_preset_backend_logged_with_source(self, example_cfg, silent_wav):
        """When no -b flag is given, stderr should state the preset as the source."""
        result = _run("-i", str(silent_wav), "-x", "mock-fwd", config=example_cfg)
        assert result.returncode == 0
        assert "preset 'mock-fwd'" in result.stderr

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
        assert len(quiet.stderr.splitlines()) <= len(verbose.stderr.splitlines())


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
        assert len(speakers) == 2

    def test_mock_diarize_3_produces_three_speakers(self, example_cfg, group_wav):
        """-b mock-dia-3 should produce exactly 3 distinct speaker labels."""
        result = _run("-i", str(group_wav), "-b", "mock-dia-3", config=example_cfg)
        assert result.returncode == 0
        lines = [l for l in result.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines) >= 3
        speakers = {l.split("SPEAKER_")[1].split(":")[0] for l in lines}
        assert len(speakers) == 3

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


# ---------------------------------------------------------------------------
# Device resolution failures — must abort, not silently fall back
# ---------------------------------------------------------------------------

class TestDeviceAbortOnFailure:
    def _cfg_with_broken_mock_device(self, tmp_path: Path) -> Path:
        """Config with a mock device whose source_file does not exist."""
        cfg = textwrap.dedent("""\
            default_preset: mock-fwd

            mock_devices:
              broken-mock:
                source_file: /nonexistent/path/to/audio.wav

            presets:
              mock-fwd: [none, mock-fwd, none, "mock forward backend"]

            asr_backends:
              mock-fwd:
                type: mock-fwd
                transcript_path: /dev/null

            postprocessors: {}
        """)
        p = tmp_path / "config.yaml"
        p.write_text(cfg)
        return p

    def test_unknown_device_name_exits_nonzero(self, example_cfg):
        """Requesting a device that does not exist must abort before recording starts.

        Device resolution is lazy and only fires in the recording path (no -i flag).
        The bad device name triggers sys.exit(1) before any audio capture begins.
        """
        result = _run("--device", "no_such_device_xyz_abc", config=example_cfg)
        assert result.returncode != 0

    def test_broken_mock_device_source_exits_nonzero(self, silent_wav, tmp_path):
        """A mock device with a missing source_file must abort, not silently fall through."""
        cfg = self._cfg_with_broken_mock_device(tmp_path)
        result = _run("--device", "broken-mock", config=cfg)
        assert result.returncode != 0
        assert "broken-mock" in result.stderr or "source file" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Robust mode (-r)
# ---------------------------------------------------------------------------

class TestRobustMode:
    """Tests for -r/--robust chunked transcription of long audio files.

    Uses the long_speech fixture (~3.5 min OGA file) with -b mock so chunks
    always succeed.  With -C 20 (20-second chunks) a 210-second file produces
    ~10 chunks; we assert that multiple independent transcript segments appear
    in the output.
    """

    def test_robust_exits_zero(self, example_cfg, long_speech):
        """-r on a long audio file with -b mock must exit 0."""
        result = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            config=example_cfg,
        )
        assert result.returncode == 0

    def test_robust_produces_multiple_chunks(self, example_cfg, long_speech):
        """-r -C 20 on ~3.5 min audio should split into multiple chunks."""
        result = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            config=example_cfg,
        )
        assert result.returncode == 0
        # Each chunk prints the mock transcript followed by a blank line; count paragraphs
        paragraphs = [p for p in result.stdout.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 5, (
            f"Expected at least 5 chunks from a ~3.5 min file with 20 s chunks, "
            f"got {len(paragraphs)}.  stdout:\n{result.stdout[:500]}"
        )

    def test_robust_chunk_progress_logged(self, example_cfg, long_speech):
        """Chunk progress lines ('Chunk N/M') must appear in stderr."""
        result = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            config=example_cfg,
        )
        assert result.returncode == 0
        assert "Chunk 1/" in result.stderr
        assert "Chunk 2/" in result.stderr

    def test_robust_output_file_appended(self, example_cfg, long_speech, tmp_path):
        """-r -o FILE should append each chunk to the file as it completes."""
        out = tmp_path / "transcript.txt"
        result = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            "-o", str(out),
            config=example_cfg,
        )
        assert result.returncode == 0
        assert out.exists()
        content = out.read_text()
        # Each chunk should appear separated by blank lines
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 5

    def test_robust_short_chunk_duration_flag(self, example_cfg, long_speech):
        """-C 10 should produce roughly twice as many chunks as -C 20."""
        r20 = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            config=example_cfg,
        )
        r10 = _run(
            "-i", str(long_speech), "-r", "-C", "10", "-b", "mock",
            config=example_cfg,
        )
        assert r20.returncode == 0
        assert r10.returncode == 0
        chunks20 = [p for p in r20.stdout.split("\n\n") if p.strip()]
        chunks10 = [p for p in r10.stdout.split("\n\n") if p.strip()]
        assert len(chunks10) > len(chunks20)


# ---------------------------------------------------------------------------
# Toggle mode (--toggle)
# ---------------------------------------------------------------------------

class TestToggleMode:
    """Tests for --toggle lock-file start/stop protocol.

    The mock recorder forks a child that copies the source WAV to a temp path
    and sleeps until killed.  We isolate the lock file by setting XDG_RUNTIME_DIR
    to a per-test tmp directory so tests never interfere with each other or with
    a live asr2clip toggle session.

    The ``toggle_runtime`` fixture tears down any still-active toggle (stops the
    mock recorder child) so ``pytest`` never leaves stray ``asr2clip`` processes.
    """

    def test_first_toggle_creates_lock(self, example_cfg, toggle_runtime):
        """First --toggle invocation must create the lock file and exit 0."""
        runtime_dir, env = toggle_runtime
        result = _run(
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            config=example_cfg, env=env,
        )
        assert result.returncode == 0
        assert (runtime_dir / "asr2clip.lock").exists(), (
            "Lock file was not created by --toggle start.\n"
            f"stderr: {result.stderr}"
        )

    def test_second_toggle_removes_lock_and_transcribes(self, example_cfg, toggle_runtime):
        """Second --toggle invocation must remove the lock and write a transcript to stdout."""
        runtime_dir, env = toggle_runtime
        args = ("--toggle", "--preset", "mock-fwd", "--device", "mock-jfk")

        r1 = _run(*args, config=example_cfg, env=env)
        assert r1.returncode == 0
        assert (runtime_dir / "asr2clip.lock").exists()

        r2 = _run(*args, config=example_cfg, env=env)
        assert r2.returncode == 0
        assert not (runtime_dir / "asr2clip.lock").exists(), (
            "Lock file was not removed by --toggle stop."
        )
        assert len(r2.stdout.strip()) > 0, (
            "Second --toggle should output a transcript, got empty stdout.\n"
            f"stderr: {r2.stderr}"
        )

    def test_toggle_stdout_is_empty_on_start(self, example_cfg, toggle_runtime):
        """First --toggle (start) must not write a transcript to stdout."""
        _runtime_dir, env = toggle_runtime
        result = _run(
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            config=example_cfg, env=env,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "", (
            "Start invocation should produce no transcript output."
        )

    def test_toggle_with_output_file(self, example_cfg, toggle_runtime):
        """--toggle -o FILE should append the transcript to the file on stop."""
        runtime_dir, env = toggle_runtime
        out = runtime_dir / "out.txt"
        args = (
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            "-o", str(out),
        )

        r1 = _run(*args, config=example_cfg, env=env)
        assert r1.returncode == 0

        r2 = _run(*args, config=example_cfg, env=env)
        assert r2.returncode == 0
        assert out.exists()
        assert len(out.read_text().strip()) > 0
