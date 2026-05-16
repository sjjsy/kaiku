"""OpenAI-compatible LLM post-processor.

Covers any endpoint that speaks the OpenAI chat completions API:
  - Ollama  (http://localhost:11434/v1/)
  - Groq    (https://api.groq.com/openai/v1/)
  - Anthropic (https://api.anthropic.com/v1/)
  - OpenAI  (https://api.openai.com/v1/)
  - Any self-hosted OpenAI-compatible server

No extra SDK required — uses only the stdlib urllib.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .base import PostMetadata, PostProcessor

_DEFAULT_USER_TEMPLATE = "Transcript (recorded {date}, {duration_s:.0f}s):\n\n{transcript}"


class OpenAICompatPostProcessor(PostProcessor):
    """Send the transcript to any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        prompt_name: str,
        api_base_url: str,
        model: str,
        system_prompt: str,
        api_key: str | None = None,
        user_template: str | None = None,
        context_text: str | None = None,
    ):
        self._prompt_name = prompt_name
        self._api_base_url = api_base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-none")
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
        return "openai_compat"

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

        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
        }).encode()

        url = f"{self._api_base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            print(f"Post-processor HTTP error {e.code}: {err}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Post-processor request failed: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            print(f"Unexpected post-processor response: {body}", file=sys.stderr)
            sys.exit(1)
