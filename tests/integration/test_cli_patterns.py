"""Integration tests for common CLI use case patterns.

These tests load config from a YAML fixture (the same path as Config.from_file()),
apply CLI overrides, and verify that the correct backend/preprocessor/postprocessor
is selected end-to-end. They also verify that transcription and process_file()
honor the same overrides.

A test here failing means a real user command is broken.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from asr2clip.config_types import Config, CliOverrides
from asr2clip.transcribe import transcribe

# ---------------------------------------------------------------------------
# YAML fixture: two mock ASR backends, two presets
# ---------------------------------------------------------------------------

_YAML = textwrap.dedent("""\
    default_preset: default

    presets:
      default: [none,        mock1, none, "Default preset"]
      alt:     [noisereduce, mock2, none, "Alt preset"]

    asr_backends:
      mock1:
        type: mock
        response: "mock1 response"
        latency_ms: 0
      mock2:
        type: mock
        response: "mock2 response"
        latency_ms: 0

    postprocessor_backends: {}
    postprocessors: {}
""")


@pytest.fixture
def cfg_file(tmp_path) -> str:
    """YAML config file with two mock backends and two presets."""
    p = tmp_path / "config.yaml"
    p.write_text(_YAML)
    return str(p)


# ---------------------------------------------------------------------------
# Config.from_file() resolution
# ---------------------------------------------------------------------------

class TestConfigFromFile:
    """Config.from_file() correctly resolves preset and CLI overrides from YAML."""

    def test_default_preset_backend(self, cfg_file):
        config = Config.from_file(cfg_file)
        assert config.asr_backend.name == "mock1"

    def test_default_preset_preprocessor(self, cfg_file):
        config = Config.from_file(cfg_file)
        assert config.preprocessor.name == "none"

    def test_preset_flag_selects_alt(self, cfg_file):
        config = Config.from_file(cfg_file, preset_name="alt")
        assert config.asr_backend.name == "mock2"
        assert config.preprocessor.name == "noisereduce"

    def test_backend_override_beats_default_preset(self, cfg_file):
        """-b mock2 with no --preset must override the default preset's mock1."""
        config = Config.from_file(cfg_file, cli_overrides=CliOverrides(backend="mock2"))
        assert config.asr_backend.name == "mock2"

    def test_backend_override_beats_named_preset(self, cfg_file):
        """-b mock1 with --preset alt must override alt's mock2."""
        config = Config.from_file(cfg_file, preset_name="alt",
                                  cli_overrides=CliOverrides(backend="mock1"))
        assert config.asr_backend.name == "mock1"

    def test_preprocessor_override_beats_preset(self, cfg_file):
        """-p deepfilter with --preset alt must override alt's noisereduce."""
        config = Config.from_file(cfg_file, preset_name="alt",
                                  cli_overrides=CliOverrides(preprocessor="deepfilter"))
        assert config.preprocessor.name == "deepfilter"

    def test_preset_preprocessor_used_without_cli_override(self, cfg_file):
        """Without -p, the preset's preprocessor must be used."""
        config = Config.from_file(cfg_file, preset_name="alt")
        assert config.preprocessor.name == "noisereduce"


# ---------------------------------------------------------------------------
# transcribe() backend selection
# ---------------------------------------------------------------------------

class TestTranscribeBackendSelection:
    """transcribe() routes audio to the correct backend."""

    def test_transcribe_uses_preset_backend(self, cfg_file, tmp_wav):
        config = Config.from_file(cfg_file)
        result = transcribe(tmp_wav, config)
        assert result == "mock1 response"

    def test_transcribe_backend_override(self, cfg_file, tmp_wav):
        """-b mock2 must reach transcription, returning mock2's response, not mock1's."""
        config = Config.from_file(cfg_file, cli_overrides=CliOverrides(backend="mock2"))
        result = transcribe(tmp_wav, config)
        assert result == "mock2 response"

    def test_transcribe_alt_preset_backend(self, cfg_file, tmp_wav):
        config = Config.from_file(cfg_file, preset_name="alt")
        result = transcribe(tmp_wav, config)
        assert result == "mock2 response"


# ---------------------------------------------------------------------------
# process_file() end-to-end with backend override
# ---------------------------------------------------------------------------

class TestProcessFileEndToEnd:
    """process_file() honors the backend CLI override end-to-end."""

    def test_process_file_uses_preset_backend(self, cfg_file, tmp_wav, tmp_path):
        """Without -b, process_file() uses the preset's backend."""
        config = Config.from_file(cfg_file)
        output = str(tmp_path / "out.txt")
        from asr2clip.asr2clip import process_file
        process_file(config, tmp_wav, output_file=output)
        assert "mock1 response" in Path(output).read_text()

    def test_process_file_backend_override(self, cfg_file, tmp_wav, tmp_path):
        """-b mock2 must override the preset's mock1 in process_file()."""
        config = Config.from_file(cfg_file, cli_overrides=CliOverrides(backend="mock2"))
        output = str(tmp_path / "out.txt")
        from asr2clip.asr2clip import process_file
        process_file(config, tmp_wav, output_file=output)
        assert "mock2 response" in Path(output).read_text()

    def test_process_file_alt_preset(self, cfg_file, tmp_wav, tmp_path):
        """--preset alt uses mock2 in process_file()."""
        config = Config.from_file(cfg_file, preset_name="alt")
        output = str(tmp_path / "out.txt")
        from asr2clip.asr2clip import process_file
        process_file(config, tmp_wav, output_file=output)
        assert "mock2 response" in Path(output).read_text()
