"""Download mock/demo fixture files (audio + transcripts) for config examples and tests."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from .utils import info, success

_USER_AGENT = "kaiku/1.0 (https://github.com/sjjsy/kaiku)"
_GITHUB_RAW = (
    "https://raw.githubusercontent.com/sjjsy/kaiku/master/tests/fixtures/transcripts"
)

# Remote sources: key = local filename under dest_dir.
_AUDIO_URLS: dict[str, str] = {
    "demo-1p-011s-en-jfk.wav": (
        "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
    ),
    "demo-4p-030s-en-ami.wav": (
        "https://raw.githubusercontent.com/nikhilraghav29/diarizen-tutorial"
        "/main/example/EN2002a_30s.wav"
    ),
    "demo-3p-096s-de-eoc.wav": (
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/"
        "Element_of_Crime_Interview_1987.ogg"
    ),
    "demo-1p-127s-en-gb0.oga": (
        "https://upload.wikimedia.org/wikipedia/commons/2/22/"
        "George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga"
    ),
    "demo-2p-023s-en-courtney.wav": (
        "https://upload.wikimedia.org/wikipedia/commons/5/57/"
        "Courtney_Love_BBC_Radio_4_-_Woman%27s_Hour_4_April_2014_%28with_reference_to_her_own_Wikipedia%29.ogg"
    ),
    "demo-4p-082s-en-agni.wav": (
        "https://upload.wikimedia.org/wikipedia/commons/7/78/"
        "India_postpones_test-firing_of_Agni-III_ballistic_missile.ogg"
    ),
    "demo-3p-051s-fi-metro.wav": (
        "https://upload.wikimedia.org/wikipedia/commons/b/bd/"
        "Finnish_dialogue_-_Tapaaminen_metroasemalla.ogg"
    ),
}

# Tilde path shown in kaiku.conf.example (default when XDG_DATA_HOME is unset).
FIXTURE_DIR_CONFIG = "~/.local/share/kaiku/fixtures"

# Mock device name = audio basename without extension; value = audio filename in fixture dir.
DEMO_MOCK_DEVICES: tuple[tuple[str, str], ...] = (
    ("demo-1p-011s-en-jfk", "demo-1p-011s-en-jfk.wav"),
    ("demo-4p-030s-en-ami", "demo-4p-030s-en-ami.wav"),
    ("demo-3p-096s-de-eoc", "demo-3p-096s-de-eoc.wav"),
    ("demo-1p-127s-en-gb0", "demo-1p-127s-en-gb0.oga"),
    ("demo-2p-023s-en-courtney", "demo-2p-023s-en-courtney.wav"),
    ("demo-4p-082s-en-agni", "demo-4p-082s-en-agni.wav"),
    ("demo-3p-051s-fi-metro", "demo-3p-051s-fi-metro.wav"),
)

# Short tier for device duration-ordering tests (11 s → 30 s → 96 s).
DEMO_WAV_NAMES: tuple[str, ...] = (
    "demo-1p-011s-en-jfk.wav",
    "demo-4p-030s-en-ami.wav",
    "demo-3p-096s-de-eoc.wav",
)

DEMO_TXT_NAMES: tuple[str, ...] = tuple(f"{stem}.txt" for stem, _ in DEMO_MOCK_DEVICES)

EXTENDED_DEMO_WAV_NAMES: tuple[str, ...] = tuple(
    fn for _, fn in DEMO_MOCK_DEVICES if fn not in DEMO_WAV_NAMES and fn.endswith(".wav")
)

LONG_DEMO_OGA_NAMES: tuple[str, ...] = ("demo-1p-127s-en-gb0.oga",)

# Solo clips (extended ASR): basename without extension.
DEMO_SOLO_DEVICE_NAMES: tuple[str, ...] = tuple(
    name for name, _ in DEMO_MOCK_DEVICES if name.startswith("demo-1p-")
)

# Multi-speaker clips (extended diarization ASR).
DEMO_GROUP_DEVICE_NAMES: tuple[str, ...] = tuple(
    name for name, _ in DEMO_MOCK_DEVICES if not name.startswith("demo-1p-")
)

_DEFAULT_DOWNLOAD = tuple(dict.fromkeys(
    [fn for _, fn in DEMO_MOCK_DEVICES] + list(DEMO_TXT_NAMES)
))

_WAV_FROM_REMOTE_OGG = frozenset({
    "demo-3p-096s-de-eoc.wav",
    "demo-2p-023s-en-courtney.wav",
    "demo-4p-082s-en-agni.wav",
    "demo-3p-051s-fi-metro.wav",
})

_FIXTURE_AUDIO_SUFFIXES = frozenset({".wav", ".oga", ".ogg", ".mp3", ".m4a", ".flac", ".webm"})


class FixtureDownloadError(Exception):
    """Raised when a fixture file cannot be downloaded."""


def bundled_transcripts_dir() -> Path | None:
    """In-repo transcript fixtures (``tests/fixtures/transcripts``) when developing from source."""
    root = Path(__file__).resolve().parent.parent
    d = root / "tests" / "fixtures" / "transcripts"
    return d if d.is_dir() else None


def default_fixture_dir() -> Path:
    """Fixture dir: ``$KAIKU_FIXTURE_DIR``, else ``$XDG_DATA_HOME/kaiku/fixtures``, else ``~/.local/share/kaiku/fixtures``."""
    if override := os.environ.get("KAIKU_FIXTURE_DIR"):
        return Path(override).expanduser()
    if base := os.environ.get("XDG_DATA_HOME"):
        return Path(base).expanduser() / "kaiku" / "fixtures"
    return Path.home() / ".local" / "share" / "kaiku" / "fixtures"


def list_fixture_audio_devices(fixture_dir: Path | None = None) -> dict[str, Path]:
    """Map device name (audio stem) → absolute path for each audio file in the fixture dir."""
    root = fixture_dir or default_fixture_dir()
    if not root.is_dir():
        return {}
    devices: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in _FIXTURE_AUDIO_SUFFIXES:
            devices[path.stem] = path.resolve()
    return devices


def resolve_fixture_audio_device(spec: str) -> Path | None:
    """Return fixture audio path when *spec* matches a filename stem in the fixture dir."""
    return list_fixture_audio_devices().get(spec.strip())


def fixture_url(name: str) -> str:
    if name in _AUDIO_URLS:
        return _AUDIO_URLS[name]
    return f"{_GITHUB_RAW}/{name}"


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        path.write_bytes(resp.read())


def ensure_fixture(name: str, dest_dir: Path) -> Path:
    """Return path to fixture, downloading into *dest_dir* if missing."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    if path.exists():
        info(f"Fixture present: {path}")
        return path
    if name.endswith(".txt") and (src := bundled_transcripts_dir()):
        bundled = src / name
        if bundled.is_file():
            shutil.copy2(bundled, path)
            success(f"Installed from repo: {path}")
            return path
    url = fixture_url(name)
    info(f"Downloading {name} ...")
    try:
        if name in _WAV_FROM_REMOTE_OGG:
            ogg = dest_dir / f".{name}.src.ogg"
            _download(url, ogg)
            from .audio import convert_audio_to_wav

            convert_audio_to_wav(str(ogg), str(path))
            ogg.unlink(missing_ok=True)
        else:
            _download(url, path)
    except Exception as exc:
        raise FixtureDownloadError(
            f"Cannot download fixture '{name}' from:\n  {url}\nError: {exc}"
        ) from exc
    success(f"Downloaded: {path}")
    return path


def ensure_demo_wavs(dest_dir: Path) -> dict[str, Path]:
    """Ensure the three tiered demo WAV fixtures exist; return name → path."""
    return {name: ensure_fixture(name, dest_dir) for name in DEMO_WAV_NAMES}


def download_fixtures(
    dest_dir: Path | None = None, names: tuple[str, ...] | None = None
) -> Path:
    """Download the default mock fixture set (or *names*) into *dest_dir*."""
    dest = dest_dir or default_fixture_dir()
    info(f"Fixture directory: {dest}")
    for name in names or _DEFAULT_DOWNLOAD:
        ensure_fixture(name, dest)
    return dest


def report_fixture_setup(fixture_dir: Path) -> None:
    """Print fixture dir location and ``-d`` examples for auto-discovered mock devices."""
    d = fixture_dir.resolve()
    devices = list_fixture_audio_devices(d)
    print()
    success(f"Fixtures ready: {d}")
    info("Any audio file here is a mock device named by its basename (no mock_devices config needed).")
    print()
    success("Try a clip (mock recording + mock-fwd preset):")
    for name in sorted(devices):
        print(f"  kaiku -d {name} -x mock-fwd")
    print()
    info("Reference transcripts in repo: tests/fixtures/transcripts/")
    info("List devices: kaiku --list-devices")
