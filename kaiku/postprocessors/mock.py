"""Mock LLM post-processor for testing and demos.

Analyzes prompts and transcripts without making API calls or requiring credentials.
Returns linguistic statistics (word frequencies, lengths, counts) for testing.
"""

from __future__ import annotations

from collections import Counter

from .base import PostMetadata, PostProcessor


def _analyze_text(text: str) -> str:
    """Analyze text and return statistics in a single line.

    Returns: "longest=word, shortest=word, most_frequent=word, lines=N, words=N, chars=N"
    """
    if not text or not text.strip():
        return "longest=-, shortest=-, most_frequent=-, lines=0, words=0, chars=0"

    words = text.split()
    lines = len(text.strip().split("\n"))
    char_count = len(text)

    if not words:
        return f"longest=-, shortest=-, most_frequent=-, lines={lines}, words=0, chars={char_count}"

    longest = max(words, key=len)
    shortest = min(words, key=len)

    # Find most frequent word (case-insensitive)
    word_counts = Counter(w.lower() for w in words)
    most_frequent = word_counts.most_common(1)[0][0]

    return (
        f"longest={longest}, shortest={shortest}, "
        f"most_frequent={most_frequent}, lines={lines}, "
        f"words={len(words)}, chars={char_count}"
    )


class MockPostProcessor(PostProcessor):
    """Mock post-processor that analyzes prompts and transcripts."""

    def __init__(
        self,
        prompt_name: str = "mock",
        model: str = "mock-model",
        system_prompt: str | None = None,
        context_text: str | None = None,
    ):
        """Initialize mock post-processor.

        Args:
            prompt_name: Name of this post-processor (for output metadata).
            model: Model name (for output metadata, used in response).
            system_prompt: System prompt (will be analyzed in response).
            context_text: Pre-formatted context text (will list files in response).
        """
        self._prompt_name = prompt_name
        self._model = model
        self._system_prompt = system_prompt
        self._context_text = context_text

    @property
    def name(self) -> str:
        return self._prompt_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend_type(self) -> str:
        return "mock"

    def process(
        self,
        transcript: str,
        *,
        metadata: PostMetadata,
    ) -> str:
        """Return mock response with linguistic analysis.

        Args:
            transcript: Input transcript (will be analyzed).
            metadata: Recording metadata (ignored in mock).

        Returns:
            Analysis of the system prompt and transcript with statistics, plus context file listing.
        """
        import re
        prompt_text = self._system_prompt or ""
        prompt_analysis = _analyze_text(prompt_text)
        transcript_analysis = _analyze_text(transcript)
        model_title = self._model.title()

        result = ""
        if self._context_text:
            files = re.findall(r"^\s+•\s+(.+)$", self._context_text, re.MULTILINE)
            if files:
                result += f"Context files: {', '.join(files)}\n"

        result += (
            f"Prompt analyzed: {prompt_analysis}\n"
            f"Transcript analyzed: {transcript_analysis}\n"
            f"*Yours truly, {model_title}*\n"
        )
        return result
