"""Mock ASR backends for testing and demos without API credentials.

Three backend types — all return text without network calls or external binaries:

  mock          — fixed canned response string (configurable)
  mock-fwd  — N words from a transcript file (N = audio_duration_s / 2),
                  repeated cyclically if audio is longer than the transcript
  mock-bwd — same as mock-fwd but words are reversed

All types accept an optional latency_ms for realistic latency simulation.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import Optional

from ..transcribe import TranscriptionError
from ..utils import info


_DEFAULT_RESPONSE = (
    "The quick brown fox jumps over the lazy dog. This is a mock transcription "
    "provided by asr2clip's mock backend for testing and demonstrations."
)


@dataclass
class MockConfig:
    """Configuration for mock transcription backend.

    Args:
        response: Text to return as the transcription. If not provided, uses default.
        latency_ms: Simulated processing latency in milliseconds (optional, for realism).
    """
    response: str = _DEFAULT_RESPONSE
    latency_ms: int = 0

    @classmethod
    def from_config(cls, config: dict) -> MockConfig:
        """Create MockConfig from backend configuration dict."""
        return cls(
            response=config.get("response", _DEFAULT_RESPONSE),
            latency_ms=int(config.get("latency_ms", 0)),
        )


def transcribe(
    audio_path: str,
    cfg: MockConfig,
    timeout: float | None = None,
) -> str:
    """Transcribe using mock backend (returns canned response).

    Args:
        audio_path: Path to the audio file (unused in mock).
        cfg: Mock backend configuration.
        timeout: Timeout (ignored in mock).

    Returns:
        The configured mock response text.
    """
    if cfg.latency_ms > 0:
        import time
        latency_s = cfg.latency_ms / 1000.0
        info(f"Mock backend: simulating {cfg.latency_ms}ms latency...")
        time.sleep(latency_s)

    info("Mock backend: returning canned transcript")
    return cfg.response


def _wav_duration(audio_path: str) -> float:
    """Return duration of a WAV file in seconds; fallback to 10s on error."""
    try:
        with wave.open(audio_path) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 10.0


# ---------------------------------------------------------------------------
# Transcript-based mock (mock-fwd / mock-bwd)
# ---------------------------------------------------------------------------

@dataclass
class MockTranscriptConfig:
    """Configuration for transcript-based mock backends.

    Args:
        transcript_path: Path to the source transcript text file.
        direction: 'forward' (natural order) or 'backward' (words reversed).
        latency_ms: Optional simulated latency in milliseconds.
    """
    transcript_path: str
    direction: str = "forward"
    latency_ms: int = 0

    @classmethod
    def from_config(cls, config: dict, direction: str) -> "MockTranscriptConfig":
        return cls(
            transcript_path=config.get("transcript_path", ""),
            direction=direction,
            latency_ms=int(config.get("latency_ms", 0)),
        )


def transcribe_from_transcript(
    audio_path: str,
    cfg: MockTranscriptConfig,
    timeout: float | None = None,
) -> str:
    """Return N words from a transcript file (N = audio_duration_s / 2).

    Words are taken cyclically (transcript repeats if audio is longer).
    Direction 'backward' reverses the word order before taking N words.

    Args:
        audio_path: WAV file whose duration determines how many words to return.
        cfg: Mock transcript backend configuration.
        timeout: Ignored; present for API compatibility.

    Returns:
        A string of N words from the transcript.
    """
    import os

    if cfg.latency_ms > 0:
        import time
        time.sleep(cfg.latency_ms / 1000.0)
        info(f"Mock-{cfg.direction} backend: simulated {cfg.latency_ms}ms latency")

    duration_s = _wav_duration(audio_path)
    n_words = max(1, int(duration_s / 2))
    info(f"Mock-{cfg.direction} backend: {duration_s:.1f}s audio → {n_words} words")

    transcript_path = os.path.expanduser(cfg.transcript_path)
    if not os.path.exists(transcript_path):
        raise TranscriptionError(
            f"Mock transcript file not found: {transcript_path}"
        )

    with open(transcript_path, encoding="utf-8") as fh:
        text = fh.read()

    words = text.split()
    if not words:
        return ""

    if cfg.direction == "backward":
        words = list(reversed(words))

    # Repeat cyclically to cover the requested word count
    if n_words > len(words):
        repeats = (n_words // len(words)) + 1
        words = words * repeats

    return " ".join(words[:n_words])


# ---------------------------------------------------------------------------
# Mock diarization backend (mock-diarize)
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class MockDiarizeConfig:
    """Configuration for the mock-diarize backend.

    Assigns all transcript file lines to speakers in round-robin order;
    timestamps are distributed evenly across the audio duration.

    Args:
        transcript_path: Path to the source transcript text file.
        speaker_count: Number of speakers to cycle through.
    """
    transcript_path: str
    speaker_count: int = 2


def transcribe_mock_diarize(
    audio_path: str,
    cfg: MockDiarizeConfig,
    num_speakers: int | None = None,
) -> str:
    """Return a mock speaker-attributed transcript from a text file.

    Output format: "[HH:MM:SS] SPEAKER_NN: line text"

    Args:
        audio_path: WAV file whose duration determines timestamp spacing.
        cfg: Mock diarize configuration.
        num_speakers: Overrides cfg.speaker_count when provided.
    """
    import os
    import wave

    transcript_path = os.path.expanduser(cfg.transcript_path)
    if not os.path.exists(transcript_path):
        raise TranscriptionError(
            f"Mock diarize transcript file not found: {transcript_path!r}"
        )

    n_speakers = num_speakers or cfg.speaker_count

    try:
        with wave.open(audio_path) as wf:
            duration_s = wf.getnframes() / wf.getframerate()
    except Exception:
        duration_s = 10.0

    with open(transcript_path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if not lines:
        return ""

    time_per_line = duration_s / len(lines)
    output: list[str] = []
    for i, line in enumerate(lines):
        speaker = f"SPEAKER_{i % n_speakers:02d}"
        ts = _fmt_ts(i * time_per_line)
        output.append(f"[{ts}] {speaker}: {line}")

    return "\n".join(output)


def test(cfg: MockConfig) -> bool:
    """Verify the mock configuration.

    Returns:
        Always True (mock backend has no external dependencies).
    """
    from ..utils import print_key_value, print_success

    print_success("Mock backend is always available")
    if cfg.response != _DEFAULT_RESPONSE:
        print_key_value("Custom response", f"{len(cfg.response)} characters")
    if cfg.latency_ms > 0:
        print_key_value("Simulated latency", f"{cfg.latency_ms}ms")

    return True
