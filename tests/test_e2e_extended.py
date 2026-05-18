"""Optional extended E2E: real ASR backends vs reference transcripts.

Run only when requested::

    KAIKU_EXTENDED_BACKEND=wcpp pytest tests/test_e2e_extended.py -v

For a diarizing backend (e.g. whisperx), multi-speaker demo devices are used;
otherwise all solo (``demo-1p-*``) mock devices are exercised with ``-l`` where
the clip language is not English.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kaiku.fixtures import DEMO_GROUP_DEVICE_NAMES, DEMO_SOLO_DEVICE_NAMES

from test_e2e import TRANSCRIPTS_DIR, _assert_diarized_matches_fixture, _assert_word_recall, _run

pytestmark = pytest.mark.extended

_EXTENDED_BACKEND = os.environ.get("KAIKU_EXTENDED_BACKEND", "").strip()
_DIARIZE_BACKENDS = frozenset({"whisperx", "mock-diarize"})

_DEVICE_LANG: dict[str, str | None] = {
    "demo-1p-011s-en-jfk": "en",
    "demo-1p-127s-en-gb0": "en",
    "demo-3p-096s-de-eoc": "de",
    "demo-3p-051s-fi-metro": "fi",
}

_DIARIZE_SPEAKERS: dict[str, int] = {
    "demo-2p-023s-en-courtney": 2,
    "demo-4p-030s-en-ami": 4,
    "demo-3p-096s-de-eoc": 3,
    "demo-4p-082s-en-agni": 4,
    "demo-3p-051s-fi-metro": 3,
}


def _devices_for_backend(backend: str) -> tuple[str, ...]:
    if backend in _DIARIZE_BACKENDS:
        return DEMO_GROUP_DEVICE_NAMES
    return DEMO_SOLO_DEVICE_NAMES


def _transcript_for_device(device: str) -> Path:
    return TRANSCRIPTS_DIR / f"{device}.txt"


@pytest.fixture(scope="session")
def extended_backend() -> str:
    if not _EXTENDED_BACKEND:
        pytest.skip("set KAIKU_EXTENDED_BACKEND (e.g. wcpp, groq, whisperx)")
    return _EXTENDED_BACKEND


@pytest.fixture(scope="session")
def extended_cfg(example_cfg: Path) -> Path:
    return example_cfg


class TestExtendedAsrBattery:
    def test_backend_on_demo_devices(self, extended_cfg, extended_backend: str):
        """One subprocess per selected device; compare stdout to reference transcript."""
        for device in _devices_for_backend(extended_backend):
            args = ["-d", device, "-b", extended_backend, "-p", "none", "-P", "none"]
            if lang := _DEVICE_LANG.get(device):
                args += ["-l", lang]
            if extended_backend in _DIARIZE_BACKENDS:
                args += ["-s", str(_DIARIZE_SPEAKERS.get(device, 2))]
            r = _run(*args, config=extended_cfg)
            assert r.returncode == 0, (device, r.stderr)
            ref = _transcript_for_device(device)
            if extended_backend in _DIARIZE_BACKENDS:
                _assert_diarized_matches_fixture(
                    r.stdout, ref, min_speakers=2, min_word_ratio=0.2,
                )
            else:
                _assert_word_recall(ref, r.stdout, min_ratio=0.35)
