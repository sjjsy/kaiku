"""Unit tests for ``--no-clipboard`` resolution on ``Config``."""

from __future__ import annotations

import argparse

from asr2clip.config_types import Config, Preset


def _preset() -> Preset:
    return Preset.from_dict("p", ["none", "mock", "none", "test preset"])


def test_no_clipboard_true_when_cli_flag_set() -> None:
    args = argparse.Namespace(no_clipboard=True)
    cfg = Config({}, _preset(), args)
    assert cfg.no_clipboard is True


def test_no_clipboard_false_when_cli_flag_absent() -> None:
    args = argparse.Namespace()
    cfg = Config({}, _preset(), args)
    assert cfg.no_clipboard is False


def test_no_clipboard_false_when_cli_flag_false() -> None:
    args = argparse.Namespace(no_clipboard=False)
    cfg = Config({}, _preset(), args)
    assert cfg.no_clipboard is False
