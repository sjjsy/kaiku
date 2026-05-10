"""Audio pre-processor package for asr2clip."""

from __future__ import annotations

import importlib.util
import sys

from .base import AudioPreprocessor
from .none import NonePreprocessor

__all__ = [
    "AudioPreprocessor",
    "NonePreprocessor",
    "QUALITY_ORDER",
    "LATENCY_ORDER",
    "VALID_NAMES",
    "probe_available",
    "check_preprocessor_available",
    "make_preprocessor",
]

# Quality ranking (best first) — used for file transcription default
QUALITY_ORDER = ["deepfilter", "noisereduce", "pyrnnoise"]

# Latency ranking (lowest overhead first) — used for live recording default.
# noisereduce operates natively at 16 kHz with no resampling.
# pyrnnoise requires 16→48→16 kHz resampling for every live recording.
LATENCY_ORDER = ["noisereduce", "pyrnnoise", "deepfilter"]

VALID_NAMES = ["none"] + QUALITY_ORDER

_MODULE_MAP = {
    "deepfilter": "df",
    "noisereduce": "noisereduce",
    "pyrnnoise": "pyrnnoise",
}

_INSTALL_HINT = {
    "deepfilter": "pip install asr2clip[deepfilter]",
    "noisereduce": "pip install asr2clip[noisereduce]",
    "pyrnnoise": "pip install asr2clip[pyrnnoise]",
}


def probe_available() -> list[str]:
    """Return names of installed preprocessor libraries, in quality order."""
    return [
        name
        for name in QUALITY_ORDER
        if importlib.util.find_spec(_MODULE_MAP[name]) is not None
    ]


def check_preprocessor_available(name: str) -> tuple[bool, str]:
    """Return (is_available, install_hint_or_empty_string).

    Always returns True for 'none'.
    """
    if name in (None, "none"):
        return True, ""
    module = _MODULE_MAP.get(name)
    if module is None:
        return False, f"Unknown preprocessor: '{name}'"
    available = importlib.util.find_spec(module) is not None
    hint = "" if available else _INSTALL_HINT.get(name, f"pip install {name}")
    return available, hint


def make_preprocessor(name: str) -> AudioPreprocessor:
    """Instantiate a preprocessor by name.

    Raises SystemExit with a helpful message for unknown names.
    """
    if name in (None, "none"):
        return NonePreprocessor()
    if name == "noisereduce":
        from .noisereduce import NoiseReducePreprocessor
        return NoiseReducePreprocessor()
    if name == "pyrnnoise":
        from .pyrnnoise import PyRNNoisePreprocessor
        return PyRNNoisePreprocessor()
    if name == "deepfilter":
        from .deepfilter import DeepFilterNetPreprocessor
        return DeepFilterNetPreprocessor()
    print(
        f"Error: unknown preprocessor '{name}'.\n"
        f"Valid options: {', '.join(VALID_NAMES)}",
        file=sys.stderr,
    )
    sys.exit(1)
