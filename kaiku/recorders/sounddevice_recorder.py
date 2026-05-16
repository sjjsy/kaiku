"""sounddevice-based recorder for toggle mode (cross-platform).

The recorder class spawns this module as a subprocess via
  python -m kaiku.recorders.sounddevice_recorder <audio_path> <rate> [device]

The subprocess records until it receives SIGTERM, then writes a WAV file and exits.
_run_subprocess() is a named function (not an inline __main__ block) so tests can
call it directly with a mocked sounddevice without spawning a child process.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time

from ..audio import _device_native_rate
from ..utils import popen_subprocess, warning
from .arecord import _ALSA_PREFIXES
from .base import AudioRecorder


class SounddeviceRecorder(AudioRecorder):
    """Records audio using the sounddevice Python library (cross-platform)."""

    @property
    def name(self) -> str:
        return "sounddevice"

    def is_available(self) -> bool:
        return importlib.util.find_spec("sounddevice") is not None

    def start(self, audio_path: str, device_info) -> int | None:
        device = None
        if device_info:
            device = device_info.get_spec("sounddevice")
        rate = _device_native_rate(device) or 44100
        device_arg = "" if device is None else str(device)
        cmd = [
            sys.executable,
            "-m", "kaiku.recorders.sounddevice_recorder",
            audio_path,
            str(rate),
            device_arg,
        ]
        proc = popen_subprocess(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(0.5)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace").strip()
            warning(f"sounddevice recorder exited immediately: {stderr}")
            return None
        return proc.pid


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def _run_subprocess(audio_path: str, rate: int, device: str | int | None) -> None:
    """Record until SIGTERM, then write audio to audio_path as WAV."""
    import signal
    import wave

    import numpy as np
    import sounddevice as sd

    chunks: list[np.ndarray] = []

    def _cb(indata, frames, t, status):
        chunks.append(indata.copy())

    def _on_sigterm(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=_cb,
        ):
            signal.pause()
    except SystemExit:
        pass
    except Exception as e:
        sys.stderr.write(f"sounddevice error: {e}\n")
        sys.exit(1)

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="float32")
    audio_i16 = (audio.flatten() * 32767).clip(-32768, 32767).astype("int16")
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio_i16.tobytes())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: python -m kaiku.recorders.sounddevice_recorder"
            " <audio_path> <rate> [device]\n"
        )
        sys.exit(1)

    _audio_path = sys.argv[1]
    _rate = int(sys.argv[2])
    _device_arg = sys.argv[3] if len(sys.argv) > 3 else ""

    _device: str | int | None
    if not _device_arg:
        _device = None
    elif _device_arg.isdigit():
        _device = int(_device_arg)
    else:
        _device = _device_arg

    _run_subprocess(_audio_path, _rate, _device)
