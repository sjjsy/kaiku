"""E2E test fixtures: audio download, test config, skip markers."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

# The whisper.cpp JFK sample — stable, public domain, ~11 s, known transcript
JFK_WAV_URL = (
    "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
)
JFK_WAV_PATH = DATA_DIR / "jfk.wav"

# Expected transcript fragment (case-insensitive substring check)
JFK_EXPECTED_FRAGMENT = "ask not what your country"

# ---------------------------------------------------------------------------
# Whisper backend discovery
# Note: override via env vars for CI or non-standard installs.
#   WHISPER_CLI_BINARY   — path to whisper-cli binary
#   WHISPER_MODEL_PATH   — path to ggml model file
# ---------------------------------------------------------------------------

WHISPER_BINARY = os.environ.get(
    "WHISPER_CLI_BINARY",
    os.path.expanduser("~/snc/git/whisper.cpp/build/bin/whisper-cli"),
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL_PATH",
    os.path.expanduser("~/snc/git/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin"),
)

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

skip_no_whisper = pytest.mark.skipif(
    not (shutil.which(WHISPER_BINARY) or os.path.isfile(WHISPER_BINARY)),
    reason=f"whisper-cli not found at {WHISPER_BINARY} (set WHISPER_CLI_BINARY to override)",
)

skip_no_model = pytest.mark.skipif(
    not os.path.isfile(WHISPER_MODEL),
    reason=f"whisper model not found at {WHISPER_MODEL} (set WHISPER_MODEL_PATH to override)",
)


# ---------------------------------------------------------------------------
# Audio download fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def jfk_wav() -> str:
    """Return path to the JFK WAV file, downloading it if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not JFK_WAV_PATH.exists():
        urllib.request.urlretrieve(JFK_WAV_URL, JFK_WAV_PATH)
    return str(JFK_WAV_PATH)


# ---------------------------------------------------------------------------
# Minimal test config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wcpp_config() -> dict:
    """Minimal config dict for whisper.cpp backend E2E tests."""
    return {
        "asr_backends": {
            "wcpp": {
                "type": "whisper_cpp",
                "binary": WHISPER_BINARY,
                "model": WHISPER_MODEL,
                "threads": 4,
            },
        },
        "postprocessor_backends": {},
        "postprocessors": {},
        "presets": {
            "test": ["none", "wcpp", "none", "Test preset"],
        },
        "_preset_for_testing": "test",
    }
