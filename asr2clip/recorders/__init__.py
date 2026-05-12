"""Toggle-mode audio recorder package for asr2clip."""

from __future__ import annotations

import sys

from .arecord import ArecordRecorder, _ALSA_PREFIXES
from .base import AudioRecorder, _kill_process, _pid_alive
from .sounddevice_recorder import SounddeviceRecorder

__all__ = [
    "AudioRecorder",
    "ArecordRecorder",
    "SounddeviceRecorder",
    "PREFERENCE_ORDER",
    "VALID_NAMES",
    "_kill_process",
    "_pid_alive",
    "probe_available",
    "make_recorder",
]

# sounddevice is first: it is a required Python dependency and works on all platforms.
# arecord is an explicit opt-in for users who want direct ALSA access on Linux.
PREFERENCE_ORDER = ["sounddevice", "arecord"]

VALID_NAMES = ["auto"] + PREFERENCE_ORDER

_CLASS_MAP: dict[str, type[AudioRecorder]] = {
    "sounddevice": SounddeviceRecorder,
    "arecord": ArecordRecorder,
}


def probe_available() -> list[str]:
    """Return names of available recorders in preference order."""
    return [name for name in PREFERENCE_ORDER if _CLASS_MAP[name]().is_available()]


def make_recorder(name: str, device_info=None) -> AudioRecorder:
    """Instantiate a recorder by name.

    'auto' selects sounddevice by default, prefers arecord for ALSA-only devices.
    Exits with an error message for unknown names or when nothing is available.

    Args:
        name: Recorder name ('auto', 'sounddevice', 'arecord').
        device_info: Optional DeviceInfo object. If device is ALSA-only, arecord is preferred.
    """
    if name in (None, "auto"):
        order = PREFERENCE_ORDER
        if device_info and device_info.alsa_name and device_info.alsa_name and not device_info.portaudio_name:
            order = ["arecord", "sounddevice"]
        for candidate in order:
            recorder = _CLASS_MAP[candidate]()
            if recorder.is_available():
                return recorder
        print(
            "Error: no recorder available. "
            "Install sounddevice (pip install sounddevice) or alsa-utils (arecord).",
            file=sys.stderr,
        )
        sys.exit(1)

    cls = _CLASS_MAP.get(name)
    if cls is None:
        print(
            f"Error: unknown recorder '{name}'.\n"
            f"Valid options: {', '.join(VALID_NAMES)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return cls()
