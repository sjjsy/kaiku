"""Unit tests for clipboard path vs text selection in ``output.copy_transcript_to_clipboard``."""

from __future__ import annotations

import argparse
import os

from asr2clip import output
from asr2clip.config_types import Config, Preset

copy_transcript_to_clipboard = output.copy_transcript_to_clipboard


def _preset() -> Preset:
    return Preset.from_dict("p", ["none", "mock", "none", "test preset"])


def _config(
    *,
    clipboard_max_chars: int = 50_000,
    output: str | None = None,
    no_clipboard: bool = False,
) -> Config:
    return Config(
        {"clipboard_max_chars": clipboard_max_chars},
        _preset(),
        argparse.Namespace(output=output, no_clipboard=no_clipboard),
    )


def test_short_transcript_copies_plain_text(monkeypatch):
    copied: list[str] = []

    def record_copy(text: str) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr(output, "copy_to_clipboard", record_copy)
    assert copy_transcript_to_clipboard("hello", _config()) is True
    assert copied == ["hello"]


def test_long_transcript_without_output_file_writes_temp_and_copies_path(monkeypatch):
    copied: list[str] = []

    def record_copy(text: str) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr(output, "copy_to_clipboard", record_copy)

    text = "x" * 80
    cfg = _config(clipboard_max_chars=50)
    assert copy_transcript_to_clipboard(text, cfg) is True
    assert len(copied) == 1
    path = copied[0]
    assert os.path.isabs(path)
    assert path.endswith(".txt")
    try:
        with open(path, encoding="utf-8") as f:
            assert f.read() == text
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_long_transcript_with_output_file_copies_that_path(tmp_path, monkeypatch):
    copied: list[str] = []

    def record_copy(text: str) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr(output, "copy_to_clipboard", record_copy)

    out = tmp_path / "saved.txt"
    out.write_text("existing", encoding="utf-8")
    text = "y" * 100
    cfg = _config(clipboard_max_chars=10, output=str(out))
    assert copy_transcript_to_clipboard(text, cfg) is True
    assert len(copied) == 1
    assert copied[0] == os.path.abspath(str(out))


def test_no_clipboard_skips_copy(monkeypatch):
    called: list[str] = []

    def record_copy(text: str) -> bool:
        called.append(text)
        return True

    monkeypatch.setattr(output, "copy_to_clipboard", record_copy)
    assert copy_transcript_to_clipboard("hello", _config(no_clipboard=True)) is False
    assert called == []
