"""Download mock/demo fixture files (audio + transcripts) for config examples and tests."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from .utils import info, success

_USER_AGENT = "kaiku/1.0 (https://github.com/sjjsy/kaiku)"
_GITHUB_RAW = "https://raw.githubusercontent.com/sjjsy/kaiku/master/test_data"

# Remote audio (same sources as tests/conftest.py).
_AUDIO_URLS: dict[str, str] = {
    "jfk-11s-1p.wav": (
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
    ),
    "group-30s-4p.wav": (
        "https://raw.githubusercontent.com/nikhilraghav29/diarizen-tutorial"
        "/main/example/EN2002a_30s.wav"
    ),
    "gb0-3min.oga": (
        "https://upload.wikimedia.org/wikipedia/commons/2/22/"
        "George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga"
    ),
}

# Default bundle for `kaiku --download-fixtures` (mock presets in kaiku.conf.example).
DEFAULT_FIXTURES: tuple[str, ...] = (
    "jfk-11s-1p.wav",
    "group-30s-4p.wav",
    "group-2p-2.txt",
    "group-2p-1.txt",
    "group-3p-1.txt",
    "solo-1.txt",
    "solo-2.txt",
    "solo-jfk.txt",
)


class FixtureDownloadError(Exception):
    """Raised when a fixture file cannot be downloaded."""


def default_fixture_dir() -> Path:
    """XDG-style cache dir: ``$XDG_DATA_HOME/kaiku/fixtures`` or ``~/.local/share/kaiku/fixtures``."""
    if base := os.environ.get("XDG_DATA_HOME"):
        return Path(base).expanduser() / "kaiku" / "fixtures"
    return Path.home() / ".local" / "share" / "kaiku" / "fixtures"


def fixture_url(name: str) -> str:
    if name in _AUDIO_URLS:
        return _AUDIO_URLS[name]
    return f"{_GITHUB_RAW}/{name}"


def ensure_fixture(name: str, dest_dir: Path) -> Path:
    """Return path to fixture, downloading into *dest_dir* if missing."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    if path.exists():
        info(f"Fixture present: {path}")
        return path
    url = fixture_url(name)
    info(f"Downloading {name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req) as resp:
            path.write_bytes(resp.read())
    except Exception as exc:
        raise FixtureDownloadError(
            f"Cannot download fixture '{name}' from:\n  {url}\nError: {exc}"
        ) from exc
    success(f"Downloaded: {path}")
    return path


def download_fixtures(
    dest_dir: Path | None = None, names: tuple[str, ...] | None = None
) -> Path:
    """Download the default mock fixture set (or *names*) into *dest_dir*."""
    dest = dest_dir or default_fixture_dir()
    info(f"Fixture directory: {dest}")
    for name in names or DEFAULT_FIXTURES:
        ensure_fixture(name, dest)
    return dest


def print_fixture_config_help(dest: Path) -> None:
    """Print paths and example config snippets for mock presets."""
    d = dest.resolve()
    jfk = d / "jfk-11s-1p.wav"
    group = d / "group-30s-4p.wav"
    transcript = d / "group-2p-2.txt"
    print()
    success("Fixtures ready. Update your config (e.g. ~/.config/kaiku/config.yaml):")
    print()
    print("mock_devices:")
    print(f"  mock-jfk:")
    print(f'    source_file: "{jfk}"')
    print(f"  mock-group:")
    print(f'    source_file: "{group}"')
    print()
    print("asr_backends:  # mock-fwd, mock-bwd")
    print(f'  mock-fwd:')
    print(f'    transcript_path: "{transcript}"')
    print("  # use the same transcript_path for mock-bwd")
    print()
    print("Then try:")
    print("  kaiku --generate-config   # if you have not already")
    print("  kaiku --test -x mock-fwd -d mock-jfk")
    print(f"  kaiku -x mock-fwd -d mock-jfk              # record from mock device (JFK clip)")
    print(f'  kaiku -x mock-fwd -i "{jfk}"   # or transcribe that file directly (no -d)')
