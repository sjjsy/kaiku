"""End-to-end CLI tests for kaiku.

# Testing strategy

Tests treat kaiku as a black box: each scenario runs the CLI as a subprocess
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

Log lines use the coloured formatter (``│  INFO │``, ``│   OK │``, …).  Tests
match stable message text; level tags may vary in width.
"""

from __future__ import annotations

import importlib.util
import os
import re
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).parent.parent
TRANSCRIPTS_DIR = REPO_ROOT / "tests" / "fixtures" / "transcripts"
MOCK_FIXED = (
    "The quick brown fox jumps over the lazy dog under sunshine and birds singing"
)  # response from -b mock
GB0_TXT = TRANSCRIPTS_DIR / "demo-1p-127s-en-gb0.txt"  # mock-fwd / mock-bwd transcript
_SPEAKER_LINE = re.compile(r"^SPEAKER_\d+:\s*(.*)$", re.I)
_WHISPERX_READY = (
    importlib.util.find_spec("whisperx") is not None and bool(os.environ.get("HF_TOKEN"))
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    *args,
    config: Path,
    env: Optional[dict] = None,
    clipboard: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``kaiku`` as a subprocess (black-box E2E).

    Unless ``clipboard=True``, ``--no-clipboard`` is inserted after
    ``--config`` so the suite does not spawn wl-copy / copykitten helpers.
    """
    cmd = ["kaiku", "--config", str(config)]
    if not clipboard:
        cmd.append("--no-clipboard")
    cmd.extend(args)
    return subprocess.run(
        cmd,
        capture_output=True, text=True,
        env=env,
    )


def _transcript_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SPEAKER_LINE.match(line)
        words.extend((m.group(1) if m else line).split())
    return words


def _word_recall(haystack: str, words: list[str]) -> float:
    if not words:
        return 1.0
    low = haystack.lower()
    return sum(1 for w in words if w.lower() in low) / len(words)


def _assert_word_recall(fixture: Path, stdout: str, *, min_ratio: float = 0.55) -> None:
    ratio = _word_recall(stdout, _transcript_words(fixture))
    assert ratio >= min_ratio, f"word recall {ratio:.2f} below {min_ratio} for {fixture.name}"


def _assert_stdout_prefix_of_transcript(fixture: Path, stdout: str) -> None:
    """mock-fwd emits the first N words of the transcript (N ≈ duration_s / 2)."""
    expected = _transcript_words(fixture)
    actual = stdout.split()
    assert len(actual) >= 3
    assert actual == expected[: len(actual)]


def _speaker_ids(text: str) -> set[str]:
    return {m.group(0).upper() for m in re.finditer(r"SPEAKER_\d+", text, re.I)}


def _fixture_speakers(path: Path) -> set[str]:
    return {
        line.split(":", 1)[0].strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().upper().startswith("SPEAKER_")
    }


def _assert_diarized_matches_fixture(
    stdout: str, fixture: Path, *, min_speakers: int, min_word_ratio: float = 0.45,
) -> None:
    expected_sp = _fixture_speakers(fixture)
    out_sp = _speaker_ids(stdout)
    assert len(out_sp) >= min_speakers
    assert len(out_sp & expected_sp) >= min(2, len(expected_sp))
    _assert_word_recall(fixture, stdout, min_ratio=min_word_ratio)


def _make_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> Path:
    n = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


def _read_wav(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


def _silence_pcm(duration_s: float, channels: int, sampwidth: int, rate: int) -> bytes:
    n_samples = int(rate * duration_s) * channels
    if sampwidth != 2:
        raise ValueError(f"unsupported sample width {sampwidth}")
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _write_wav(path: Path, channels: int, sampwidth: int, rate: int, *parts: bytes) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        for part in parts:
            wf.writeframes(part)
    return path


def _wav_with_silence_padding(
    speech: Path,
    out: Path,
    *,
    lead_s: float = 0,
    trail_s: float = 0,
    mid_s: float = 0,
) -> Path:
    """Concatenate digital silence around (and optionally inside) a speech WAV."""
    ch, sw, rate, pcm = _read_wav(speech)
    parts: list[bytes] = []
    if lead_s:
        parts.append(_silence_pcm(lead_s, ch, sw, rate))
    if mid_s:
        mid = len(pcm) // 2
        parts.extend([pcm[:mid], _silence_pcm(mid_s, ch, sw, rate), pcm[mid:]])
    else:
        parts.append(pcm)
    if trail_s:
        parts.append(_silence_pcm(trail_s, ch, sw, rate))
    return _write_wav(out, ch, sw, rate, *parts)


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
    if (tmp_path / "kaiku.lock").exists():
        _run(
            "--toggle", "--preset", "mock-fwd", "--device", "demo-1p-011s-en-jfk",
            config=example_cfg, env=env,
        )


# ---------------------------------------------------------------------------
# Demo audio (early): mock device + file input vs mock-fwd transcript (gb0.txt)
# ---------------------------------------------------------------------------

class TestDemoAudioEarly:
    def test_mock_1p_device_records_jfk_clip(self, example_cfg, demo_1p_wav):
        """``demo-1p-011s-en-jfk`` device audio; ``mock-fwd`` scales words from ``demo-1p-127s-en-gb0.txt``."""
        r = _run("--preset", "mock-fwd", "--device", "demo-1p-011s-en-jfk", config=example_cfg)
        assert r.returncode == 0
        assert "Using mock device: demo-1p-011s-en-jfk" in r.stderr
        assert len(r.stdout.strip()) > 0
        _assert_stdout_prefix_of_transcript(GB0_TXT, r.stdout)

    def test_jfk_file_input_mock_fwd_matches_transcript(self, example_cfg, demo_1p_wav):
        r = _run("-i", str(demo_1p_wav), "-b", "mock-fwd", config=example_cfg)
        assert r.returncode == 0
        assert "Using backend: mock-fwd (CLI -b)" in r.stderr
        _assert_stdout_prefix_of_transcript(GB0_TXT, r.stdout)

    def test_german_demo_device_with_language_flag(self, example_cfg):
        r = _run(
            "--preset", "mock-fwd", "--device", "demo-3p-096s-de-eoc", "-l", "de",
            config=example_cfg,
        )
        assert r.returncode == 0
        assert "Using language: de (CLI -l)" in r.stderr
        assert "Using mock device: demo-3p-096s-de-eoc" in r.stderr


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
        """Stacked file run: backend, ``-p none``, template, post, language, ``-o``, clipboard skip; template fallback; ``mock-pp2`` extends ``mock-pp``."""
        out = tmp_path / "mega.txt"
        r = _run(
            "-i", str(silent_wav),
            "-p", "none",
            "-b", "mock",
            "-T", "bare",
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
        assert "preprocessing audio with" not in err.lower()
        assert "clipboard: skipped (--no-clipboard)" in err.lower()
        assert "Transcript analyzed:" in r.stdout
        assert "words=14" in r.stdout
        assert MOCK_FIXED not in r.stdout
        assert out.exists()
        assert "Transcript analyzed:" in out.read_text()
        assert "words=14" in out.read_text()
        assert "Prompt analyzed:" in r.stdout
        assert ", lines=1, words=8" in r.stdout

        r2 = _run(
            "-i", str(silent_wav), "-b", "mock", "-T", "no_such_template",
            config=example_cfg,
        )
        assert r2.returncode == 0
        assert MOCK_FIXED in r2.stdout

        r3 = _run(
            "-i", str(silent_wav), "-b", "mock", "-P", "mock-pp2",
            config=example_cfg,
        )
        assert r3.returncode == 0
        assert "Using post-processor: mock-pp2 (CLI -P)" in r3.stderr
        assert "Post-processing with 'mock-pp2'" in r3.stderr
        assert "Transcript analyzed:" in r3.stdout
        assert MOCK_FIXED not in r3.stdout
        assert ", lines=3, words=22" in r3.stdout
        assert "Prompt analyzed:" in r3.stdout

    def test_quiet_matches_verbose_transcript_with_fewer_logs(
        self, example_cfg, silent_wav,
    ):
        """``-q`` must not change the transcript for a fixed backend; stderr should shrink."""
        verbose = _run("-i", str(silent_wav), "-b", "mock", config=example_cfg)
        quiet = _run("-i", str(silent_wav), "-b", "mock", "-q", config=example_cfg)
        assert verbose.returncode == 0
        assert quiet.returncode == 0
        assert verbose.stdout == quiet.stdout
        lines_when_verbose = len(verbose.stderr.splitlines())
        lines_when_quiet = len(quiet.stderr.splitlines())
        assert lines_when_quiet < lines_when_verbose
        assert lines_when_verbose > 5
        assert lines_when_quiet == 0

    def test_context_flag_loads_and_appears_in_mock_output(self, example_cfg, silent_wav):
        """``-X FILE`` injects context into post-processor; MockPostProcessor lists files."""
        context_file = Path("/etc/hostname")
        if not context_file.exists():
            pytest.skip("No /etc/hostname; skipping context flag test")
        r = _run(
            "-i", str(silent_wav), "-b", "mock", "-P", "mock-pp",
            "-X", str(context_file), "-o", "/dev/null",
            config=example_cfg,
        )
        assert r.returncode == 0
        assert "Context files:" in r.stdout
        assert "hostname" in r.stdout


class TestPlainTextInput:
    def test_txt_input_skips_asr_and_postprocesses(self, example_cfg, tmp_path):
        """``.txt`` input bypasses ASR; transcript is read from file then post-processed."""
        note = tmp_path / "note.txt"
        note.write_text("Delta echo foxtrot.\n")
        r = _run("-i", str(note), "-P", "mock-pp2", config=example_cfg)
        assert r.returncode == 0
        assert f"Processing file: {note}" in r.stderr
        assert "Mock backend" not in r.stderr
        assert "Transcription completed" not in r.stderr
        assert "Post-processing with 'mock-pp2'" in r.stderr
        assert "Transcript analyzed:" in r.stdout
        assert "most_frequent=delta" in r.stdout.lower() or "echo" in r.stdout.lower()


class TestSelfCheck:
    def test_cli_test_mode_passes_with_mock_preset(self, example_cfg):
        r = _run("--test", config=example_cfg)
        assert r.returncode == 0
        e = r.stderr
        assert "All checks passed" in e
        assert "no API connectivity check" in e
        assert "Preprocessor: none" in e
        assert "No post-processors configured to check" in e or "post-processors" in e.lower()


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
        p1 = _run("--preset", "mock-fwd", "--device", "demo-1p-011s-en-jfk", config=example_cfg)
        p2 = _run("--preset", "mock-fwd", "--device", "demo-4p-030s-en-ami", config=example_cfg)
        p3 = _run("--preset", "mock-fwd", "--device", "demo-3p-096s-de-eoc", config=example_cfg)
        assert p1.returncode == 0 and p2.returncode == 0 and p3.returncode == 0
        assert len(p1.stdout.strip()) > 0 and len(p2.stdout.strip()) > 0 and len(p3.stdout.strip()) > 0
        assert "Using mock device: demo-1p-011s-en-jfk" in p1.stderr
        assert "Using mock device: demo-4p-030s-en-ami" in p2.stderr
        assert "Using mock device: demo-3p-096s-de-eoc" in p3.stderr
        w1, w2, w3 = len(p1.stdout.split()), len(p2.stdout.split()), len(p3.stdout.split())
        assert w1 < w2 < w3


# ---------------------------------------------------------------------------
# Diarization (optional): mock-diarize in diarization_cfg; whisperx if installed
# ---------------------------------------------------------------------------

class TestDiarizationOptional:
    """Not part of the default mock-only contract; uses ``diarization_cfg`` fixture."""

    def test_mock_dia_4p_matches_fixture(self, diarization_cfg, demo_4p_wav):
        r = _run("-i", str(demo_4p_wav), "-b", "mock-dia-4", config=diarization_cfg)
        assert r.returncode == 0
        _assert_diarized_matches_fixture(r.stdout, TRANSCRIPTS_DIR / "demo-4p-030s-en-ami.txt", min_speakers=4)

    def test_mock_dia_3p_matches_fixture(self, diarization_cfg, demo_3p_wav):
        r = _run("-i", str(demo_3p_wav), "-b", "mock-dia-3", config=diarization_cfg)
        assert r.returncode == 0
        _assert_diarized_matches_fixture(r.stdout, TRANSCRIPTS_DIR / "demo-3p-096s-de-eoc.txt", min_speakers=3)

    @pytest.mark.skipif(not _WHISPERX_READY, reason="requires kaiku[diarize] and HF_TOKEN")
    def test_whisperx_4p_matches_fixture(self, diarization_cfg, demo_4p_wav):
        r = _run("-i", str(demo_4p_wav), "-b", "whisperx", "-s", "4", config=diarization_cfg)
        assert r.returncode == 0
        _assert_diarized_matches_fixture(
            r.stdout, TRANSCRIPTS_DIR / "demo-4p-030s-en-ami.txt", min_speakers=4, min_word_ratio=0.25,
        )

    @pytest.mark.skipif(not _WHISPERX_READY, reason="requires kaiku[diarize] and HF_TOKEN")
    def test_whisperx_3p_matches_fixture(self, diarization_cfg, demo_3p_wav):
        r = _run("-i", str(demo_3p_wav), "-b", "whisperx", "-s", "3", config=diarization_cfg)
        assert r.returncode == 0
        _assert_diarized_matches_fixture(
            r.stdout, TRANSCRIPTS_DIR / "demo-3p-096s-de-eoc.txt", min_speakers=3, min_word_ratio=0.2,
        )


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
        """With ``-o``, chunks go to file; ``-C`` changes chunk count.  Optionally ``-p noisereduce``."""
        out20 = tmp_path / "robust20.txt"
        args20: list[str | Path] = [
            "-i", str(long_speech), "-r", "-C", "20", "-b", "mock",
            "-o", str(out20),
        ]
        if importlib.util.find_spec("noisereduce") is not None:
            args20.extend(["-p", "noisereduce"])
        r20 = _run(*args20, config=example_cfg)
        assert r20.returncode == 0
        assert "Chunk 1/" in r20.stderr
        assert "Chunk 2/" in r20.stderr
        assert out20.exists()
        assert len(out20.read_text()) > 0
        assert r20.stderr.count("Chunk ") >= 5
        if importlib.util.find_spec("noisereduce") is not None:
            assert "using preprocessor: noisereduce" in r20.stderr.lower()

        out10 = tmp_path / "robust10.txt"
        r10 = _run(
            "-i", str(long_speech), "-r", "-C", "10", "-b", "mock",
            "-o", str(out10),
            config=example_cfg,
        )
        assert r10.returncode == 0
        assert r10.stderr.count("Chunk ") > r20.stderr.count("Chunk ")

    def test_robust_omits_synthetic_silence(
        self, example_cfg, medium_speech, tmp_path,
    ):
        """Padded silence at start/end/middle is omitted; speech still chunks with modest ``-C``."""
        padded = _wav_with_silence_padding(
            medium_speech,
            tmp_path / "padded.wav",
            lead_s=10.0,
            trail_s=15.0,
            mid_s=12.0,
        )
        out = tmp_path / "robust_silence.txt"
        r = _run(
            "-i", str(padded), "-r", "-C", "12", "-b", "mock",
            "-o", str(out),
            config=example_cfg,
        )
        assert r.returncode == 0
        assert out.exists() and MOCK_FIXED in out.read_text()
        assert r.stderr.count("Omitting chunk") >= 2
        omitted = re.search(r"omitted (\d+) segments of silence", r.stderr)
        assert omitted is not None and int(omitted.group(1)) >= 2
        split = re.search(r"Splitting into (\d+) chunk\(s\)", r.stderr)
        assert split is not None and int(split.group(1)) >= 4
        assert r.stderr.count("Chunk to transcribe") >= 4


# ---------------------------------------------------------------------------
# Toggle mode (--toggle)
# ---------------------------------------------------------------------------

class TestToggleMode:
    def test_lifecycle_lock_stdout_stderr_and_output_file(self, example_cfg, toggle_runtime):
        runtime_dir, env = toggle_runtime
        out = runtime_dir / "toggle_out.txt"
        args = (
            "--toggle", "--preset", "mock-fwd", "--device", "demo-1p-011s-en-jfk",
            "-o", str(out),
        )
        r1 = _run(*args, config=example_cfg, env=env)
        assert r1.returncode == 0
        assert (runtime_dir / "kaiku.lock").exists()
        assert r1.stdout.strip() == ""
        assert "Recording started" in r1.stderr

        r2 = _run(*args, config=example_cfg, env=env)
        assert r2.returncode == 0
        assert not (runtime_dir / "kaiku.lock").exists()
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
