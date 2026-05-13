"""End-to-end CLI tests for asr2clip.

# Testing strategy

Tests treat asr2clip as a black box: each scenario runs the CLI as a subprocess
and asserts a *checklist* of independent predicates (exit code, stdout shape,
multiple stderr substrings, files on disk).  No internal imports, no mocks.

**Invocation budget:** expensive paths (`long_speech`, toggle pairs) run only
where the contract requires them.  Several former micro-tests are folded into
one scenario so a single ``CompletedProcess`` catches multiple failure classes
(wrong backend, broken ``-o``, missing log *why*, post-processor not wired, …).

By default ``_run`` injects ``--no-clipboard`` so the suite does not spawn
clipboard helpers.  Pass ``clipboard=True`` only where clipboard behaviour is
the subject.

Config resolution *why* lines live in stderr (see CLAUDE.md); we assert those
substrings end-to-end without importing ``Config``.
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
    clipboard: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``asr2clip`` as a subprocess (black-box E2E).

    Unless ``clipboard=True``, ``--no-clipboard`` is inserted after
    ``--config`` so the suite does not spawn wl-copy / copykitten helpers.
    """
    cmd = ["asr2clip", "--config", str(config)]
    if not clipboard:
        cmd.append("--no-clipboard")
    cmd.extend(args)
    return subprocess.run(
        cmd,
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


def _chunk_paragraphs(stdout: str) -> list[str]:
    return [p for p in stdout.split("\n\n") if p.strip()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def silent_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("audio") / "silent.wav"
    return _make_wav(p, duration_s=2.0)


@pytest.fixture
def toggle_runtime(tmp_path: Path, example_cfg: Path):
    """Per-test XDG_RUNTIME_DIR for --toggle E2E tests."""
    env = {**os.environ, "XDG_RUNTIME_DIR": str(tmp_path)}
    yield tmp_path, env
    if (tmp_path / "asr2clip.lock").exists():
        _run(
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            config=example_cfg, env=env,
        )


# ---------------------------------------------------------------------------
# Preset + file pipeline (few invocations, dense assertions)
# ---------------------------------------------------------------------------

class TestPresetAndFilePipeline:
    def test_default_preset_matches_short_flag(self, example_cfg, silent_wav):
        """YAML ``default_preset`` and ``-x mock-fwd`` must yield identical transcripts."""
        a = _run("-i", str(silent_wav), config=example_cfg)
        b = _run("-i", str(silent_wav), "-x", "mock-fwd", config=example_cfg)
        assert a.returncode == 0
        assert b.returncode == 0
        assert a.stdout == b.stdout
        assert len(a.stdout.strip()) > 0
        assert MOCK_FIXED not in a.stdout
        why = "Using backend: mock-fwd (preset 'mock-fwd')"
        assert why in a.stderr
        assert why in b.stderr

    def test_file_input_dense_primary_contract(self, example_cfg, silent_wav, tmp_path):
        """One stacked file run exercises backend, template, post, language, ``-o``, logs, clipboard skip."""
        out = tmp_path / "mega.txt"
        r = _run(
            "-i", str(silent_wav),
            "-b", "mock",
            "-T", "raw",
            "-P", "mock-pp",
            "-l", "fi",
            "-o", str(out),
            config=example_cfg,
        )
        assert r.returncode == 0
        err = r.stderr
        assert "Using backend: mock (CLI -b)" in err
        assert "Using post-processor: mock-pp (CLI -P)" in err
        assert "Post-processing with 'mock-pp'" in err
        assert "Mock backend: returning canned transcript" in err
        assert "nothing was placed on the system clipboard" in err.lower()
        assert "Transcript analyzed:" in r.stdout
        assert "words=9" in r.stdout
        assert MOCK_FIXED not in r.stdout
        assert out.exists()
        assert "Transcript analyzed:" in out.read_text()
        assert "words=9" in out.read_text()

        r2 = _run(
            "-i", str(silent_wav), "-b", "mock", "-T", "no_such_template",
            config=example_cfg,
        )
        assert r2.returncode == 0
        assert MOCK_FIXED in r2.stdout

        r3 = _run("-i", str(silent_wav), "-P", "mock-pp", config=example_cfg)
        assert r3.returncode == 0
        assert "Using backend: mock-fwd (preset 'mock-fwd')" in r3.stderr
        assert "Using post-processor: mock-pp (CLI -P)" in r3.stderr
        assert "Transcript analyzed:" in r3.stdout
        assert MOCK_FIXED not in r3.stdout

    def test_quiet_matches_verbose_transcript_with_fewer_logs(
        self, example_cfg, silent_wav,
    ):
        """``-q`` must not change the transcript for a fixed backend; stderr should shrink."""
        verbose = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        quiet = _run("-i", str(silent_wav), "-b", "mock", "-q", config=example_cfg)
        assert verbose.returncode == 0
        assert quiet.returncode == 0
        assert verbose.stdout == quiet.stdout
        assert len(quiet.stderr.splitlines()) <= len(verbose.stderr.splitlines())


class TestConfigErrors:
    def test_missing_preset_exits_nonzero(self, example_cfg, silent_wav):
        r = _run("-i", str(silent_wav), "--preset", "no-such-preset", config=example_cfg)
        assert r.returncode != 0
        assert "no-such-preset" in r.stderr.lower() or "preset" in r.stderr.lower()

    def test_missing_config_exits_nonzero(self, silent_wav):
        r = _run(
            "-i", str(silent_wav),
            config=REPO_ROOT / "nonexistent_xyz_config.yaml",
        )
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# Mock transcript backends
# ---------------------------------------------------------------------------

class TestMockTranscriptBackends:
    def test_mock_fwd_bwd_and_duration_scaling(self, example_cfg, tmp_path):
        """``mock-fwd`` vs ``mock-bwd`` differ; longer audio yields more words than short."""
        short = _make_wav(tmp_path / "short.wav", duration_s=2.0)
        long_ = _make_wav(tmp_path / "long.wav", duration_s=20.0)
        fwd = _run("-i", str(short), "-b", "mock-fwd", config=example_cfg)
        bwd = _run("-i", str(short), "-b", "mock-bwd", config=example_cfg)
        res_long = _run("-i", str(long_), "-b", "mock-fwd", config=example_cfg)
        assert fwd.returncode == 0
        assert bwd.returncode == 0
        assert res_long.returncode == 0
        assert fwd.stdout.strip() != bwd.stdout.strip()
        assert MOCK_FIXED not in fwd.stdout
        assert len(res_long.stdout.split()) > len(fwd.stdout.split())


# ---------------------------------------------------------------------------
# Recording + mock devices
# ---------------------------------------------------------------------------

class TestMockRecordingDevices:
    def test_devices_transcribe_with_resolution_logs_and_duration_ordering(self, example_cfg):
        jfk = _run("--preset", "mock-fwd", "--device", "mock-jfk", config=example_cfg)
        grp = _run("--preset", "mock-fwd", "--device", "mock-group", config=example_cfg)
        assert jfk.returncode == 0
        assert grp.returncode == 0
        assert len(jfk.stdout.strip()) > 0
        assert len(grp.stdout.strip()) > 0
        assert "Using mock device: mock-jfk" in jfk.stderr
        assert "Using mock device: mock-group" in grp.stderr
        assert len(grp.stdout.split()) > len(jfk.stdout.split())


# ---------------------------------------------------------------------------
# Mock diarization
# ---------------------------------------------------------------------------

class TestMockDiarization:
    def test_speaker_counts_and_cli_override(self, example_cfg, jfk_wav, group_wav):
        r2 = _run("-i", str(jfk_wav), "-b", "mock-dia-2", config=example_cfg)
        assert r2.returncode == 0
        lines2 = [l for l in r2.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines2) >= 2
        speakers2 = {l.split("SPEAKER_")[1].split(":")[0] for l in lines2}
        assert len(speakers2) == 2

        r3 = _run("-i", str(group_wav), "-b", "mock-dia-3", config=example_cfg)
        assert r3.returncode == 0
        lines3 = [l for l in r3.stdout.splitlines() if "SPEAKER_" in l]
        assert len(lines3) >= 3
        speakers3 = {l.split("SPEAKER_")[1].split(":")[0] for l in lines3}
        assert len(speakers3) == 3

        r1 = _run("-i", str(jfk_wav), "-b", "mock-dia-2", "-s", "1", config=example_cfg)
        assert r1.returncode == 0
        lines1 = [l for l in r1.stdout.splitlines() if "SPEAKER_" in l]
        speakers1 = {l.split("SPEAKER_")[1].split(":")[0] for l in lines1}
        assert len(speakers1) == 1


# ---------------------------------------------------------------------------
# Device resolution failures
# ---------------------------------------------------------------------------

class TestDeviceAbortOnFailure:
    def _cfg_with_broken_mock_device(self, tmp_path: Path) -> Path:
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
        r = _run("--device", "no_such_device_xyz_abc", config=example_cfg)
        assert r.returncode != 0

    def test_broken_mock_device_source_exits_nonzero(self, tmp_path):
        cfg = self._cfg_with_broken_mock_device(tmp_path)
        r = _run("--device", "broken-mock", config=cfg)
        assert r.returncode != 0
        assert "broken-mock" in r.stderr or "source file" in r.stderr.lower()


# ---------------------------------------------------------------------------
# Robust mode (-r)
# ---------------------------------------------------------------------------

class TestRobustMode:
    def test_chunking_stderr_file_and_chunk_duration_flag(
        self, example_cfg, long_speech, tmp_path,
    ):
        """With ``-o``, per-chunk text is appended to the file (stdout stays minimal); ``-C`` changes chunk count."""
        out20 = tmp_path / "robust20.txt"
        r20 = _run(
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            "-o", str(out20),
            config=example_cfg,
        )
        assert r20.returncode == 0
        assert "Chunk 1/" in r20.stderr
        assert "Chunk 2/" in r20.stderr
        assert "Full transcript written to" in r20.stderr or "appended to" in r20.stderr.lower()
        assert out20.exists()
        paras20 = _chunk_paragraphs(out20.read_text())
        assert len(paras20) >= 5, out20.read_text()[:800]

        out10 = tmp_path / "robust10.txt"
        r10 = _run(
            "-i", str(long_speech), "-r", "-C", "10", "-b", "mock",
            "-o", str(out10),
            config=example_cfg,
        )
        assert r10.returncode == 0
        paras10 = _chunk_paragraphs(out10.read_text())
        assert len(paras10) > len(paras20)


# ---------------------------------------------------------------------------
# Toggle mode (--toggle)
# ---------------------------------------------------------------------------

class TestToggleMode:
    def test_lifecycle_lock_stdout_stderr_and_output_file(self, example_cfg, toggle_runtime):
        runtime_dir, env = toggle_runtime
        out = runtime_dir / "toggle_out.txt"
        args = (
            "--toggle", "--preset", "mock-fwd", "--device", "mock-jfk",
            "-o", str(out),
        )
        r1 = _run(*args, config=example_cfg, env=env)
        assert r1.returncode == 0
        assert (runtime_dir / "asr2clip.lock").exists()
        assert r1.stdout.strip() == ""
        assert "Recording started" in r1.stderr

        r2 = _run(*args, config=example_cfg, env=env)
        assert r2.returncode == 0
        assert not (runtime_dir / "asr2clip.lock").exists()
        assert len(r2.stdout.strip()) > 0
        assert "Stopping recorder" in r2.stderr or "Transcribing" in r2.stderr
        assert out.exists()
        assert len(out.read_text().strip()) > 0


# ---------------------------------------------------------------------------
# Clipboard (opt-in only)
# ---------------------------------------------------------------------------

class TestClipboardE2E:
    def test_clipboard_opt_in_reports_transcript_copied(self, example_cfg, silent_wav):
        r = _run(
            "-i", str(silent_wav), "-b", "mock",
            config=example_cfg,
            clipboard=True,
        )
        assert r.returncode == 0
        assert "Transcript text copied to clipboard" in r.stderr
