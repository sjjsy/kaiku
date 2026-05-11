"""Claude Code CLI post-processor.

Invokes the `claude` CLI (Claude Code) as a subprocess, piping the transcript
via stdin. This uses whatever model and subscription the user's Claude Code
session is authenticated to — including Claude Code MAX plans where API usage
is included in the subscription fee.

Requires: `claude` on PATH and an active Claude Code session.
Install: https://claude.ai/code
"""

from __future__ import annotations

import subprocess
import sys

from .base import PostMetadata, PostProcessor

_DEFAULT_USER_TEMPLATE = "Transcript (recorded {date}, {duration_s:.0f}s):\n\n{transcript}"


class ClaudeCodePostProcessor(PostProcessor):
    """Post-processor that delegates to the `claude` CLI subprocess."""

    def __init__(
        self,
        prompt_name: str,
        system_prompt: str,
        model: str | None = None,
        user_template: str | None = None,
        context_text: str | None = None,
    ):
        self._prompt_name = prompt_name
        self._system_prompt = system_prompt
        self._model = model or ""
        self._user_template = user_template or _DEFAULT_USER_TEMPLATE
        self._context_text = context_text

    @property
    def name(self) -> str:
        return self._prompt_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend_type(self) -> str:
        return "claude_code"

    def process(self, transcript: str, *, metadata: PostMetadata) -> str:
        user_content = self._user_template.format(
            transcript=transcript,
            date=metadata.date,
            duration_s=metadata.duration_s,
            language=metadata.language,
            speakers=", ".join(metadata.speakers) if metadata.speakers else "",
        )
        if self._context_text:
            user_content = f"## Context\n\n{self._context_text}\n\n---\n\n{user_content}"

        cmd = ["claude"]
        if self._model:
            cmd += ["-m", self._model]
        cmd += ["-p", self._system_prompt, "--no-markdown"]

        try:
            result = subprocess.run(
                cmd,
                input=user_content,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except FileNotFoundError:
            print(
                "Error: 'claude' CLI not found.\n"
                "Install Claude Code (https://claude.ai/code) to use the claude_code backend.",
                file=sys.stderr,
            )
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("Error: claude CLI timed out after 180 s.", file=sys.stderr)
            sys.exit(1)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(
                f"Error: claude CLI exited {result.returncode}"
                + (f": {stderr}" if stderr else ""),
                file=sys.stderr,
            )
            sys.exit(1)

        return result.stdout.strip()
