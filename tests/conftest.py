"""Shared pytest fixtures for kaiku E2E tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaiku.fixtures import (
    FIXTURE_DIR_CONFIG,
    FixtureDownloadError,
    download_fixtures,
)

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "kaiku" / "kaiku.conf.example"
TRANSCRIPTS_DIR = REPO_ROOT / "tests" / "fixtures" / "transcripts"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download demo clips once (same paths as ``kaiku --download-fixtures``)."""
    dest = tmp_path_factory.mktemp("kaiku_fixtures")
    try:
        download_fixtures(dest_dir=dest)
    except FixtureDownloadError as exc:
        pytest.fail(str(exc))
    os.environ["KAIKU_FIXTURE_DIR"] = str(dest)
    return dest


def _example_config_text(*, fixture_dir: Path, with_diarization_backends: bool) -> str:
    text = EXAMPLE_CONFIG.read_text()
    text = text.replace(f"{FIXTURE_DIR_CONFIG}/", f"{fixture_dir.resolve()}/")
    abs_tx = str(TRANSCRIPTS_DIR)
    if with_diarization_backends:
        _post = "## ── Post-processing (with AI models)"
        _dia = (
            f"  mock-dia-4:\n    type: mock-diarize\n    speaker_count: 4\n"
            f'    transcript_path: "{abs_tx}/demo-4p-030s-en-ami.txt"\n'
            f"  mock-dia-3:\n    type: mock-diarize\n    speaker_count: 3\n"
            f'    transcript_path: "{abs_tx}/demo-3p-096s-de-eoc.txt"\n'
        )
        token = os.environ.get("HF_TOKEN", "")
        if token:
            _dia += (
                f"  whisperx:\n    type: whisperx\n    hf_token: \"{token}\"\n"
                f"    speakers_min: 2\n    speakers_max: 6\n"
            )
        text = text.replace(_post, _dia + _post, 1)
    return "default_preset: mock-fwd\n\n" + text


@pytest.fixture(scope="session")
def example_cfg(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session config: example YAML + downloaded fixtures (devices via KAIKU_FIXTURE_DIR)."""
    p = tmp_path_factory.mktemp("cfg") / "config.yaml"
    p.write_text(_example_config_text(fixture_dir=fixture_dir, with_diarization_backends=False))
    return p


@pytest.fixture(scope="session")
def diarization_cfg(fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Like example_cfg plus mock-diarize backends; whisperx block if HF_TOKEN is set."""
    p = tmp_path_factory.mktemp("cfg-dia") / "config.yaml"
    p.write_text(_example_config_text(fixture_dir=fixture_dir, with_diarization_backends=True))
    return p


@pytest.fixture(scope="session")
def demo_1p_wav(fixture_dir: Path) -> Path:
    return fixture_dir / "demo-1p-011s-en-jfk.wav"


@pytest.fixture(scope="session")
def demo_4p_wav(fixture_dir: Path) -> Path:
    return fixture_dir / "demo-4p-030s-en-ami.wav"


@pytest.fixture(scope="session")
def demo_3p_wav(fixture_dir: Path) -> Path:
    return fixture_dir / "demo-3p-096s-de-eoc.wav"


@pytest.fixture(scope="session")
def long_speech(fixture_dir: Path) -> Path:
    return fixture_dir / "demo-1p-127s-en-gb0.wav"
