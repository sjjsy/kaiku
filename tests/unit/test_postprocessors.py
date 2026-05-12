"""Unit tests for asr2clip.postprocessors package internals."""

from __future__ import annotations

import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from asr2clip.postprocessors import (
    format_output,
    make_postprocessor,
    resolve_output_template,
)
from asr2clip.postprocessors.base import PostMetadata
from asr2clip.postprocessors.none import NonePostProcessor
from asr2clip.postprocessors import _resolve_prompt, _resolve_backend, _load_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(**kw) -> PostMetadata:
    defaults = dict(
        date="2026-05-11",
        duration_s=60.0,
        language="fi",
        prompt_name="solo-base",
    )
    defaults.update(kw)
    return PostMetadata(**defaults)


# ---------------------------------------------------------------------------
# resolve_output_template
# ---------------------------------------------------------------------------

class TestResolveOutputTemplate:
    def test_cli_override_takes_precedence(self, postprocessor_config):
        """CLI override should take priority over postprocessor template."""
        t = resolve_output_template(
            postprocessor_config,
            postprocessor_template="{result}",
            cli_override="full"
        )
        assert "{result}" in t and "{transcript}" in t

    def test_cli_override_missing_warns_and_falls_through(self, postprocessor_config, capsys):
        """Missing CLI template should warn and fall back to default."""
        t = resolve_output_template(
            postprocessor_config,
            postprocessor_template="{result}",
            cli_override="nonexistent"
        )
        captured = capsys.readouterr()
        assert "not found" in captured.err
        assert t == postprocessor_config["output_templates"]["default"]

    def test_uses_postprocessor_template_when_provided(self, postprocessor_config):
        """Should use postprocessor_template if it's not the default."""
        t = resolve_output_template(
            postprocessor_config,
            postprocessor_template="{transcript}",
            cli_override=None
        )
        assert t == "{transcript}"

    def test_default_template_when_empty(self):
        """Should use fallback when config has no templates."""
        t = resolve_output_template({}, postprocessor_template="{result}", cli_override=None)
        assert t == "{result}"

    def test_default_from_config_when_no_postprocessor_template(self, postprocessor_config):
        """Should use config default when postprocessor template is default."""
        t = resolve_output_template(
            postprocessor_config,
            postprocessor_template="{result}",  # default
            cli_override=None
        )
        assert t == postprocessor_config["output_templates"]["default"]


# ---------------------------------------------------------------------------
# format_output
# ---------------------------------------------------------------------------

class TestFormatOutput:
    def test_basic_substitution(self):
        meta = _meta(date="2026-01-01", duration_s=120.0, prompt_name="solo-base")
        out = format_output(
            "{result} | {date} | {duration_s:.0f}s | {prompt_name}",
            result="RESULT",
            transcript="RAW",
            metadata=meta,
        )
        assert out == "RESULT | 2026-01-01 | 120s | solo-base"

    def test_transcript_placeholder(self):
        meta = _meta()
        out = format_output(
            "{transcript}",
            result="R",
            transcript="original text",
            metadata=meta,
        )
        assert out == "original text"

    def test_model_and_backend(self):
        meta = _meta()
        out = format_output(
            "{model}/{backend}",
            result="",
            transcript="",
            metadata=meta,
            model="qwen3:14b",
            backend="openai_compat",
        )
        assert out == "qwen3:14b/openai_compat"

    def test_bad_key_falls_back_to_result(self, capsys):
        meta = _meta()
        out = format_output(
            "{unknown_key}",
            result="fallback",
            transcript="",
            metadata=meta,
        )
        captured = capsys.readouterr()
        assert "template error" in captured.err
        assert out == "fallback"


# ---------------------------------------------------------------------------
# NonePostProcessor
# ---------------------------------------------------------------------------

class TestNonePostProcessor:
    def test_returns_transcript_unchanged(self):
        pp = NonePostProcessor()
        meta = _meta()
        result = pp.process("hello world", metadata=meta)
        assert result == "hello world"

    def test_name(self):
        assert NonePostProcessor().name == "none"

    def test_model_empty(self):
        assert NonePostProcessor().model == ""


# ---------------------------------------------------------------------------
# _resolve_prompt
# ---------------------------------------------------------------------------

class TestResolvePrompt:
    def test_simple_prompt(self, postprocessor_config):
        r = _resolve_prompt("solo-base", postprocessor_config)
        assert r["system_prompt"] == "Clean up this transcript."
        assert r["backend_name"] == "local"

    def test_extends_chain(self, postprocessor_config):
        r = _resolve_prompt("solo-enhance", postprocessor_config)
        assert "Clean up this transcript." in r["system_prompt"]
        assert "Also extract key points." in r["system_prompt"]

    def test_child_extra_appended_to_base(self, postprocessor_config):
        r = _resolve_prompt("solo-enhance", postprocessor_config)
        lines = r["system_prompt"].splitlines()
        assert any("Clean up" in l for l in lines)
        assert any("key points" in l for l in lines)

    def test_unknown_name_exits(self, postprocessor_config):
        with pytest.raises(SystemExit):
            _resolve_prompt("nonexistent", postprocessor_config)

    def test_circular_extends_exits(self):
        config = {
            "postprocessors": {
                "a": {"extends": "b", "prompt": ""},
                "b": {"extends": "a", "prompt": ""},
            }
        }
        with pytest.raises(SystemExit):
            _resolve_prompt("a", config)

    def test_empty_prompt_exits(self):
        config = {"postprocessors": {"bad": {"prompt": ""}}}
        with pytest.raises(SystemExit):
            _resolve_prompt("bad", config)

    def test_child_overrides_base_prompt(self, postprocessor_config):
        postprocessor_config["postprocessors"]["override"] = {
            "extends": "solo-base",
            "prompt": "Completely different prompt.",
        }
        r = _resolve_prompt("override", postprocessor_config)
        assert r["system_prompt"] == "Completely different prompt."

    def test_context_path_inherited_and_extended(self, postprocessor_config):
        postprocessor_config["postprocessors"]["base-ctx"] = {
            "prompt": "Base.",
            "context_path": ["/tmp/a.md"],
        }
        postprocessor_config["postprocessors"]["child-ctx"] = {
            "extends": "base-ctx",
            "extra": "Extra.",
            "context_path": ["/tmp/b.md"],
        }
        r = _resolve_prompt("child-ctx", postprocessor_config)
        assert "/tmp/a.md" in r["context_paths"]
        assert "/tmp/b.md" in r["context_paths"]


# ---------------------------------------------------------------------------
# _resolve_backend
# ---------------------------------------------------------------------------

class TestResolveBackend:
    def test_named_backend(self, postprocessor_config):
        b = _resolve_backend(postprocessor_config, "local", None)
        assert b["type"] == "openai_compat"
        assert b["model"] == "qwen3:14b"

    def test_first_backend_as_default(self, postprocessor_config):
        b = _resolve_backend(postprocessor_config, None, None)
        assert b["type"] == "openai_compat"

    def test_model_override(self, postprocessor_config):
        b = _resolve_backend(postprocessor_config, "local", "llama3:8b")
        assert b["model"] == "llama3:8b"

    def test_unknown_backend_exits(self, postprocessor_config):
        with pytest.raises(SystemExit):
            _resolve_backend(postprocessor_config, "nonexistent", None)

    def test_no_backends_exits(self):
        with pytest.raises(SystemExit):
            _resolve_backend({}, None, None)

    def test_api_key_from_env(self, postprocessor_config, monkeypatch):
        postprocessor_config["postprocessor_backends"]["env-key"] = {
            "type": "openai_compat",
            "api_base_url": "http://x/v1/",
            "api_key_env": "MY_KEY",
            "model": "m",
        }
        monkeypatch.setenv("MY_KEY", "secret-from-env")
        b = _resolve_backend(postprocessor_config, "env-key", None)
        assert b["api_key"] == "secret-from-env"


# ---------------------------------------------------------------------------
# _load_context
# ---------------------------------------------------------------------------

class TestLoadContext:
    def test_empty_paths_returns_none(self):
        assert _load_context([]) is None

    def test_single_file(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("Hello context")
        result = _load_context([str(f)])
        assert result is not None
        assert "Hello context" in result
        assert "note.md" in result

    def test_multiple_files_indexed(self, tmp_path):
        (tmp_path / "a.md").write_text("Content A")
        (tmp_path / "b.md").write_text("Content B")
        result = _load_context([str(tmp_path / "*.md")])
        assert result is not None
        assert "Content A" in result
        assert "Content B" in result
        assert "Context files:" in result

    def test_nonmatching_glob_returns_none_silently(self):
        # glob.glob returns [] for non-matching pattern; no warning needed
        result = _load_context(["/nonexistent/path_that_will_never_exist_*.md"])
        assert result is None

    def test_unreadable_file_warns(self, tmp_path, capsys):
        f = tmp_path / "locked.md"
        f.write_text("content")
        f.chmod(0o000)
        try:
            result = _load_context([str(f)])
            captured = capsys.readouterr()
            # If we can't read it (non-root), should warn
            if result is None:
                assert "could not read" in captured.err
        finally:
            f.chmod(0o644)

    def test_common_prefix_relative_paths(self, tmp_path):
        sub = tmp_path / "context"
        sub.mkdir()
        (sub / "personal.md").write_text("personal")
        (sub / "private.md").write_text("private")
        result = _load_context([str(sub / "*.md")])
        assert "personal.md" in result
        assert "private.md" in result
        assert str(sub) not in result


# ---------------------------------------------------------------------------
# make_postprocessor — factory dispatch
# ---------------------------------------------------------------------------

class TestMakePostprocessor:
    def test_none_name_returns_none_processor(self, postprocessor_config):
        pp = make_postprocessor("none", postprocessor_config)
        assert isinstance(pp, NonePostProcessor)

    def test_null_name_returns_none_processor(self, postprocessor_config):
        pp = make_postprocessor(None, postprocessor_config)
        assert isinstance(pp, NonePostProcessor)

    def test_named_openai_compat(self, postprocessor_config):
        from asr2clip.postprocessors.openai_compat import OpenAICompatPostProcessor
        pp = make_postprocessor("solo-base", postprocessor_config)
        assert isinstance(pp, OpenAICompatPostProcessor)

    def test_named_claude_code(self, postprocessor_config):
        from asr2clip.postprocessors.claude_code import ClaudeCodePostProcessor
        pp = make_postprocessor("group", postprocessor_config)
        assert isinstance(pp, ClaudeCodePostProcessor)

    def test_raw_prompt_string_dispatches_to_first_backend(self, postprocessor_config):
        from asr2clip.postprocessors.openai_compat import OpenAICompatPostProcessor
        pp = make_postprocessor("Summarize this text as bullet points.", postprocessor_config)
        assert isinstance(pp, OpenAICompatPostProcessor)
        assert pp.name == "custom"

    def test_model_override_applied(self, postprocessor_config):
        pp = make_postprocessor("solo-base", postprocessor_config, model_override="llama3:8b")
        assert pp.model == "llama3:8b"
