"""Shared pytest fixtures for kaiku E2E tests.

The primary fixture `example_cfg` reads the repo's kaiku.conf.example,
injects `default_preset: mock-fwd`, resolves test_data/ paths to absolute, and
writes the result to a session-scoped temp file. Every change to the example
config is therefore automatically exercised by the test suite.

Audio fixtures (`jfk_wav`, `group_wav`, `long_speech`) are auto-downloaded on
first run and cached in test_data/. They are gitignored. Tests fail with a
clear message if the download fails (no network, broken URL).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "kaiku.conf.example"
TEST_DATA = REPO_ROOT / "test_data"

# Remote sources for test audio fixtures.
# Keys are the local filenames; values are stable public URLs.
_AUDIO_URLS: dict[str, str] = {
    "jfk-11s-1p.wav": (
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
    ),
    "group-30s-4p.wav": (
        "https://raw.githubusercontent.com/nikhilraghav29/diarizen-tutorial"
        "/main/example/EN2002a_30s.wav"
    ),
    # George W. Bush weekly radio address, Nov 1 2008 (~3.5 min, 1 speaker).
    # Same source used by whisper.cpp's own test suite (Makefile: gb0.ogg).
    # Wikimedia Commons URLs are permanent — files are never moved or deleted.
    "gb0-3min.oga": (
        "https://upload.wikimedia.org/wikipedia/commons/2/22/"
        "George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga"
    ),
}


def _ensure_audio(filename: str) -> Path:
    """Return path to a test audio file, downloading it from the known URL if absent.

    Calls pytest.fail() with a clear message if the download fails, so the
    error surfaces as a collection-time failure rather than a confusing
    FileNotFoundError mid-test.

    A User-Agent header is sent; Wikimedia Commons (and many other hosts)
    return 403 for requests that omit it.
    """
    path = TEST_DATA / filename
    if path.exists():
        return path
    url = _AUDIO_URLS[filename]
    TEST_DATA.mkdir(exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "kaiku-tests/1.0 (https://github.com/sjjsy/kaiku)"},
        )
        with urllib.request.urlopen(req) as resp:
            path.write_bytes(resp.read())
    except Exception as exc:
        pytest.fail(
            f"Cannot download test fixture '{filename}' from:\n  {url}\n"
            f"Error: {exc}\n"
            "Check network connectivity or place the file manually in test_data/."
        )
    return path


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def jfk_wav() -> Path:
    """11-second JFK speech clip (auto-downloaded from whisper.cpp samples)."""
    return _ensure_audio("jfk-11s-1p.wav")


@pytest.fixture(scope="session")
def group_wav() -> Path:
    """30-second group conversation clip (auto-downloaded from diarizen-tutorial)."""
    return _ensure_audio("group-30s-4p.wav")


@pytest.fixture(scope="session")
def long_speech() -> Path:
    """~3.5-minute George W. Bush radio address (auto-downloaded from Wikimedia Commons).

    Used for robust-mode (-r) tests that require multi-chunk splitting.
    The file is an OGG/Vorbis audio file (.oga); kaiku's process_file_robust
    loads it via pydub which uses ffmpeg to decode it.
    """
    return _ensure_audio("gb0-3min.oga")


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def example_cfg(
    jfk_wav: Path,    # noqa: ARG001 — ensures WAVs exist before config is written
    group_wav: Path,  # noqa: ARG001
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Session-scoped config derived from kaiku.conf.example.

    Patches applied:
      - Prepends `default_preset: mock-fwd` so tests run without --preset
      - Replaces relative `test_data/` references with absolute paths so the
        config works regardless of current working directory
    """
    text = EXAMPLE_CONFIG.read_text()

    abs_test_data = str(REPO_ROOT / "test_data")
    text = text.replace("test_data/", f"{abs_test_data}/")

    text = "default_preset: mock-fwd\n\n" + text

    p = tmp_path_factory.mktemp("cfg") / "config.yaml"
    p.write_text(text)
    return p
