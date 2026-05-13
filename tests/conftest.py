"""Shared pytest fixtures for asr2clip E2E tests.

The primary fixture `example_cfg` reads the repo's asr2clip.conf.example,
injects `default_preset: mock-fwd`, resolves test_data/ paths to absolute, and
writes the result to a session-scoped temp file. Every change to the example
config is therefore automatically exercised by the test suite.

WAV fixtures (`jfk_wav`, `group_wav`) are auto-downloaded on first run and
cached in test_data/. They are gitignored. Tests fail with a clear message if
the download fails (no network, broken URL).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "asr2clip.conf.example"
TEST_DATA = REPO_ROOT / "test_data"

# Remote sources for test WAV fixtures.
_WAV_URLS: dict[str, str] = {
    "jfk-11s-1p.wav": (
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
    ),
    "group-30s-4p.wav": (
        "https://raw.githubusercontent.com/nikhilraghav29/diarizen-tutorial"
        "/main/example/EN2002a_30s.wav"
    ),
}


def _ensure_wav(filename: str) -> Path:
    """Return path to a test WAV, downloading it from the known URL if absent.

    Calls pytest.fail() with a clear message if the download fails, so the
    error surfaces as a collection-time failure rather than a confusing
    FileNotFoundError mid-test.
    """
    path = TEST_DATA / filename
    if path.exists():
        return path
    url = _WAV_URLS[filename]
    TEST_DATA.mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        pytest.fail(
            f"Cannot download test fixture '{filename}' from:\n  {url}\n"
            f"Error: {exc}\n"
            "Check network connectivity or place the file manually in test_data/."
        )
    return path


# ---------------------------------------------------------------------------
# WAV fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def jfk_wav() -> Path:
    """11-second JFK speech clip (auto-downloaded from whisper.cpp samples)."""
    return _ensure_wav("jfk-11s-1p.wav")


@pytest.fixture(scope="session")
def group_wav() -> Path:
    """30-second group conversation clip (auto-downloaded from diarizen-tutorial)."""
    return _ensure_wav("group-30s-4p.wav")


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def example_cfg(
    jfk_wav: Path,  # noqa: ARG001 — ensures WAVs exist before config is written
    group_wav: Path,  # noqa: ARG001
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Session-scoped config derived from asr2clip.conf.example.

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
