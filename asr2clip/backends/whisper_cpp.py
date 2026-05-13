"""whisper.cpp subprocess backend for asr2clip."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from ..transcribe import TranscriptionError
from ..utils import info, run_subprocess, warning


_ARTIFACT_RE = re.compile(
    r"^\s*\[(?:BLANK_AUDIO|MUSIC|NOISE|INAUDIBLE)\]\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(r"^\[[\d:.,\s>-]+\]\s*", re.MULTILINE)


@dataclass
class WhisperCppConfig:
    binary: str
    model: str
    language: str = "auto"
    threads: int = 4
    timestamps: bool = False
    timeout_multiplier: float = 4.0
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict) -> "WhisperCppConfig":
        wc = config.get("whisper_cpp", {})
        return cls(
            binary=os.path.expanduser(wc.get("binary", "whisper-cli")),
            model=os.path.expanduser(wc.get("model", "")),
            language=wc.get("language", "auto"),
            threads=int(wc.get("threads", 4)),
            timestamps=bool(wc.get("timestamps", False)),
            timeout_multiplier=float(wc.get("timeout_multiplier", 4.0)),
            extra_args=list(wc.get("extra_args", [])),
        )


def _clean_output(raw: str, strip_timestamps: bool) -> str:
    text = _ARTIFACT_RE.sub("", raw)
    if strip_timestamps:
        text = _TIMESTAMP_RE.sub("", text)
    return "\n".join(line for line in text.splitlines() if line.strip())


def transcribe(
    audio_path: str,
    cfg: WhisperCppConfig,
    timeout: float | None = None,
) -> str:
    """Transcribe audio_path using whisper-cli subprocess.

    Args:
        audio_path: Path to the WAV file.
        cfg: whisper.cpp backend configuration.
        timeout: Override timeout in seconds (None = auto from cfg.timeout_multiplier).

    Returns:
        Transcribed text.

    Raises:
        TranscriptionError: On non-zero exit or timeout.
    """
    if not shutil.which(cfg.binary) and not os.path.isfile(cfg.binary):
        raise TranscriptionError(f"whisper-cli binary not found: {cfg.binary}")
    if cfg.model and not os.path.isfile(cfg.model):
        raise TranscriptionError(f"whisper.cpp model not found: {cfg.model}")

    cmd = [cfg.binary, "-m", cfg.model, "-f", audio_path, "-t", str(cfg.threads)]
    if not cfg.timestamps:
        cmd.append("-nt")
    if cfg.language and cfg.language != "auto":
        cmd += ["--language", cfg.language]
    cmd += cfg.extra_args

    if timeout is None:
        # Rough estimate: audio duration × multiplier, minimum 30 s
        try:
            import wave
            with wave.open(audio_path) as wf:
                dur = wf.getnframes() / wf.getframerate()
            timeout = max(30.0, dur * cfg.timeout_multiplier)
        except Exception:
            timeout = 300.0

    t0 = time.time()
    try:
        result = run_subprocess(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TranscriptionError(
            f"whisper-cli timed out after {timeout:.0f}s"
        )
    elapsed = time.time() - t0
    info(f"whisper-cli completed in {elapsed:.1f}s")

    if result.returncode != 0:
        msg = result.stderr.strip() or f"whisper-cli exited {result.returncode}"
        raise TranscriptionError(msg)

    return _clean_output(result.stdout, strip_timestamps=not cfg.timestamps)


def test(cfg: WhisperCppConfig) -> bool:
    """Verify the whisper.cpp configuration.

    Returns:
        True if binary and model are usable.
    """
    from ..utils import error, print_key_value, success

    ok = True

    binary_path = shutil.which(cfg.binary) or cfg.binary
    if os.path.isfile(binary_path):
        success(f"Binary found: {binary_path}")
    else:
        error(f"Binary not found: {cfg.binary}")
        ok = False

    if cfg.model:
        if os.path.isfile(cfg.model):
            success(f"Model found: {cfg.model}")
        else:
            error(f"Model not found: {cfg.model}")
            ok = False
    else:
        error("No model path configured (whisper_cpp.model)")
        ok = False

    if ok:
        try:
            result = run_subprocess(
                [cfg.binary, "--help"], capture_output=True, text=True, timeout=10
            )
            # whisper-cli prints version info in its help header
            output = result.stdout + result.stderr
            for line in output.splitlines():
                if "whisper" in line.lower() or "version" in line.lower():
                    print_key_value("Info", line.strip())
                    break
        except Exception as e:
            warning(f"Could not retrieve version info: {e}")

    return ok
