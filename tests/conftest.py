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

from pathlib import Path

import pytest

from kaiku.fixtures import FixtureDownloadError, ensure_fixture

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "kaiku" / "kaiku.conf.example"
TEST_DATA = REPO_ROOT / "test_data"


def _ensure_audio(filename: str) -> Path:
    try:
        return ensure_fixture(filename, TEST_DATA)
    except FixtureDownloadError as exc:
        pytest.fail(str(exc))


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
