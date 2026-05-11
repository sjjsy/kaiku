"""Pass-through post-processor — returns the transcript unchanged."""

from __future__ import annotations

from .base import PostMetadata, PostProcessor


class NonePostProcessor(PostProcessor):
    """No-op — preserves current behaviour when no --post flag is given."""

    @property
    def name(self) -> str:
        return "none"

    @property
    def model(self) -> str:
        return ""

    @property
    def backend_type(self) -> str:
        return "none"

    def process(self, transcript: str, *, metadata: PostMetadata) -> str:
        return transcript
