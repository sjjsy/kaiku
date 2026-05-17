"""ALSA arecord-based recorder for toggle mode (Linux)."""

from __future__ import annotations

import re
import shutil
import subprocess
import time

from ..audio import _device_native_rate
from ..utils import popen_subprocess, warning
from .base import AudioRecorder

_ALSA_PREFIXES = ("hw:", "plughw:", "pulse", "pipewire", "default", "sysdefault")


def _is_alsa_device(device: str | int | None) -> bool:
    if device is None or isinstance(device, int):
        return False
    return any(device.startswith(p) for p in _ALSA_PREFIXES)


def _friendly_to_alsa(device: str) -> str | None:
    """Resolve a PortAudio friendly name to an ALSA plughw: string.

    sounddevice names often embed the ALSA index, e.g.
    'Blue Snowball: USB Audio (hw:2,0)' → 'plughw:2,0'.
    Returns None when the index cannot be extracted.
    """
    try:
        import sounddevice as sd

        info = sd.query_devices(device, "input")
        m = re.search(r"\(hw:(\d+),(\d+)\)", info["name"])
        if m:
            return f"plughw:{m.group(1)},{m.group(2)}"
    except Exception:
        pass
    return None


class ArecordRecorder(AudioRecorder):
    """Records audio using the system arecord binary (ALSA, Linux only)."""

    @property
    def name(self) -> str:
        return "arecord"

    def is_available(self) -> bool:
        return bool(shutil.which("arecord"))

    def start(self, audio_path: str, device_info) -> int | None:
        if not self.is_available():
            return None

        alsa_device: str | int | None = None
        if device_info:
            alsa_device = device_info.get_spec("arecord")

        rate = (
            _device_native_rate(
                alsa_device
                if isinstance(alsa_device, str) and _is_alsa_device(alsa_device)
                else None
            )
            or 44100
        )
        cmd = ["arecord", "-f", "S16_LE", "-r", str(rate), "-c", "1"]

        if alsa_device is not None and _is_alsa_device(alsa_device):
            cmd += ["-D", str(alsa_device)]

        cmd.append(audio_path)
        proc = popen_subprocess(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(0.3)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace").strip()
            warning(f"arecord exited immediately: {stderr}")
            return None
        return proc.pid
