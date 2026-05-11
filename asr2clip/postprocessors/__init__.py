"""LLM post-processor package for asr2clip.

All prompts are user-defined in config (no hardcoded built-ins).
The config template ships solo-memo as the one uncommented example.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime as _datetime

from .base import PostMetadata, PostProcessor
from .none import NonePostProcessor

__all__ = [
    "PostMetadata",
    "PostProcessor",
    "NonePostProcessor",
    "resolve_postprocessor_config",
    "make_postprocessor",
    "resolve_output_template",
    "format_output",
]

_FALLBACK_TEMPLATE = "{result}"


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------

def resolve_postprocessor_config(
    config: dict,
    cli_override: str | None = None,
    mode: str = "live",
) -> str:
    """Return the effective post-processor name for the given mode.

    Args:
        config: Full configuration dictionary.
        cli_override: Name supplied via -P/--post flag (takes priority).
        mode: 'live' for microphone/toggle recording, 'file' for -i file input.
    """
    if cli_override:
        return cli_override
    key = "postprocessor_live" if mode == "live" else "postprocessor_file"
    return config.get(key, config.get("postprocessor", "none"))


def resolve_output_template(
    config: dict,
    prompt_name: str = "",
    cli_override: str | None = None,
) -> str:
    """Return the output template string to use for this run.

    Lookup order:
    1. -T/--template CLI override
    2. The template named in the prompt's 'template:' field
    3. The 'default' entry in output_templates
    4. Built-in fallback: '{result}'
    """
    templates = config.get("output_templates", {})

    if cli_override:
        t = templates.get(cli_override)
        if t is None:
            print(
                f"Warning: output template '{cli_override}' not found in config; "
                "using default.",
                file=sys.stderr,
            )
        else:
            return t

    if prompt_name and prompt_name != "none":
        postprocessors = config.get("postprocessors", {})
        if prompt_name in postprocessors:
            tmpl_name = postprocessors[prompt_name].get("template", "default")
            t = templates.get(tmpl_name)
            if t is not None:
                return t

    return templates.get("default", _FALLBACK_TEMPLATE)


def format_output(
    template: str,
    *,
    result: str,
    transcript: str,
    metadata: PostMetadata,
    model: str = "",
    backend: str = "",
) -> str:
    """Apply an output template, substituting all known placeholders.

    Placeholders: {result} {transcript} {date} {datetime}
                  {prompt_name} {model} {backend} {duration_s}
    """
    now = _datetime.now()
    try:
        return template.format(
            result=result,
            transcript=transcript,
            date=metadata.date,
            datetime=now.strftime("%Y-%m-%d %H:%M"),
            prompt_name=metadata.prompt_name,
            model=model,
            backend=backend,
            duration_s=metadata.duration_s,
        )
    except (KeyError, ValueError) as e:
        print(
            f"Warning: output template error ({e}); using raw result.",
            file=sys.stderr,
        )
        return result


# ---------------------------------------------------------------------------
# Prompt resolution (extends + extra, context_path)
# ---------------------------------------------------------------------------

def _load_context(paths: list) -> str | None:
    """Expand glob patterns, read matching files, return organized context block.

    Output format includes:
    - Index: list of loaded files with relative paths
    - Separator: clear delimiters between files
    - Content: each file's content separated by visual breaks

    File paths are shown relative to their common prefix, e.g.
    ~/.asr2clip/context/personal.md and ~/.asr2clip/context/private.md
    are shown as context/personal.md and context/private.md.
    """
    if not paths:
        return None
    files: list[tuple[str, str]] = []  # (full_path, content)
    for pattern in paths:
        for path in sorted(glob.glob(os.path.expanduser(str(pattern)))):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                files.append((path, content))
            except OSError as e:
                print(f"Warning: context_path could not read '{path}': {e}", file=sys.stderr)

    if not files:
        return None

    # Find common directory: paths relative to this will show distinguishing parts
    full_paths = [path for path, _ in files]
    if len(files) > 1:
        common_dir = os.path.commonpath(full_paths)
    else:
        common_dir = os.path.dirname(files[0][0])

    # If commonpath is a file (shouldn't happen with files), use its parent
    if not os.path.isdir(common_dir):
        common_dir = os.path.dirname(common_dir)

    # Build index + content
    lines = ["Context files:", ""]
    for path, _ in files:
        rel_path = os.path.relpath(path, common_dir)
        lines.append(f"  • {rel_path}")
    lines.append("")

    # Add each file with clear delimiters
    for path, content in files:
        rel_path = os.path.relpath(path, common_dir)
        lines.append(f"\n{'=' * 70}")
        lines.append(f"Context File: {rel_path}")
        lines.append(f"{'=' * 70}\n")
        lines.append(content)

    return "\n".join(lines)


def _resolve_prompt(name: str, config: dict, _seen: set | None = None) -> dict:
    """Resolve a prompt definition, following extends chains.

    Returns a dict with keys:
      system_prompt, backend_name, model, context_paths, template_name
    """
    if _seen is None:
        _seen = set()
    if name in _seen:
        print(
            f"Error: circular 'extends' chain involving post-processor '{name}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    _seen = _seen | {name}

    postprocessors = config.get("postprocessors", {})
    if name not in postprocessors:
        available = ", ".join(postprocessors) if postprocessors else "(none defined)"
        print(
            f"Error: post-processor '{name}' not found in config.\n"
            f"Defined processors: {available}\n\n"
            "Add it under 'postprocessors:' in your config, or run:\n"
            "    asr2clip --edit",
            file=sys.stderr,
        )
        sys.exit(1)

    defn = postprocessors[name]

    base_prompt = ""
    base_backend: str | None = None
    base_model: str | None = None
    base_context_paths: list = []

    if "extends" in defn:
        base = _resolve_prompt(defn["extends"], config, _seen)
        base_prompt = base["system_prompt"]
        base_backend = base["backend_name"]
        base_model = base["model"]
        base_context_paths = base["context_paths"]

    own_prompt: str = defn.get("prompt", "")
    extra: str = defn.get("extra", "").strip()

    if own_prompt and base_prompt:
        # Explicit 'prompt:' in a child overrides the base entirely
        system_prompt = own_prompt.rstrip()
    elif base_prompt:
        system_prompt = base_prompt.rstrip()
        if extra:
            system_prompt += "\n\n" + extra
    else:
        system_prompt = own_prompt.rstrip()
        if extra:
            system_prompt += "\n\n" + extra

    if not system_prompt:
        print(
            f"Error: post-processor '{name}' has no 'prompt:' and its 'extends' "
            "chain produced an empty system prompt.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Child fields override base; fall back to base if child does not specify
    backend_name: str | None = defn.get("backend") or base_backend
    model: str | None = defn.get("model") or base_model
    context_paths: list = base_context_paths + (defn.get("context_path") or [])
    template_name: str = defn.get("template", "default")

    return {
        "system_prompt": system_prompt,
        "backend_name": backend_name,
        "model": model,
        "context_paths": context_paths,
        "template_name": template_name,
    }


# ---------------------------------------------------------------------------
# Backend config resolution
# ---------------------------------------------------------------------------

def _resolve_backend(
    config: dict,
    backend_name: str | None,
    model_override: str | None,
) -> dict:
    """Return a flat backend dict from postprocessor_backends.

    Keys returned: type, api_base_url (openai_compat only), api_key, model
    """
    backends = config.get("postprocessor_backends", {})

    if not backends:
        print(
            "Error: no 'postprocessor_backends:' section found in config.\n"
            "Add one, for example:\n\n"
            "  postprocessor_backends:\n"
            "    local:\n"
            "      type: openai_compat\n"
            "      api_base_url: 'http://localhost:11434/v1/'\n"
            "      api_key: 'ollama'\n"
            "      model: 'qwen3:14b'\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if backend_name and backend_name in backends:
        section = backends[backend_name]
    elif backend_name:
        available = ", ".join(backends)
        print(
            f"Error: postprocessor backend '{backend_name}' not found.\n"
            f"Defined: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        # Default: first defined backend
        section = next(iter(backends.values()))

    btype = section.get("type", "openai_compat")

    # api_key: direct value or env-var reference
    api_key = section.get("api_key")
    api_key_env = section.get("api_key_env")
    if api_key is None and api_key_env:
        api_key = os.environ.get(api_key_env)
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "sk-none")

    model = model_override or section.get("model", "")

    return {
        "type": btype,
        "api_base_url": section.get("api_base_url", ""),
        "api_key": api_key,
        "model": model,
        "user_template": section.get("user_template"),
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_postprocessor(
    name: str,
    config: dict,
    model_override: str | None = None,
) -> PostProcessor:
    """Instantiate a post-processor by name from config.

    Args:
        name: Post-processor name (key in config['postprocessors']) or 'none'.
              May also be a raw prompt string (contains spaces / length > 40).
        config: Full asr2clip configuration dict.
        model_override: Model name from -M/--post-model CLI flag (highest priority).

    Returns:
        A PostProcessor instance ready to call .process().
    """
    if name in (None, "none"):
        return NonePostProcessor()

    # Raw system-prompt string passed directly via --post "Summarize as bullets."
    if " " in name or len(name) > 60:
        system_prompt = name
        resolved = {
            "system_prompt": system_prompt,
            "backend_name": None,
            "model": model_override,
            "context_paths": [],
            "template_name": "default",
        }
    else:
        resolved = _resolve_prompt(name, config)
        if model_override:
            resolved["model"] = model_override

    backend_cfg = _resolve_backend(config, resolved["backend_name"], resolved["model"])
    context_text = _load_context(resolved["context_paths"])

    btype = backend_cfg["type"]

    if btype == "openai_compat":
        from .openai_compat import OpenAICompatPostProcessor
        return OpenAICompatPostProcessor(
            prompt_name=name if " " not in name else "custom",
            api_base_url=backend_cfg["api_base_url"],
            model=backend_cfg["model"],
            system_prompt=resolved["system_prompt"],
            api_key=backend_cfg["api_key"],
            user_template=backend_cfg.get("user_template"),
            context_text=context_text,
        )
    elif btype == "claude_code":
        from .claude_code import ClaudeCodePostProcessor
        return ClaudeCodePostProcessor(
            prompt_name=name if " " not in name else "custom",
            system_prompt=resolved["system_prompt"],
            model=backend_cfg["model"],
            user_template=backend_cfg.get("user_template"),
            context_text=context_text,
        )
    else:
        print(
            f"Error: unknown postprocessor backend type '{btype}'.\n"
            "Valid types: openai_compat, claude_code",
            file=sys.stderr,
        )
        sys.exit(1)
