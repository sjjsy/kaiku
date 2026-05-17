"""Base class and metadata for LLM post-processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PostMetadata:
    date: str  # YYYY-MM-DD
    duration_s: float  # recording duration in seconds
    language: str  # ISO-639-1 or "auto"
    prompt_name: str  # which post-processor was used
    speakers: list[str] = field(default_factory=list)  # known speaker names
    diarized: bool = False  # whether transcript has speaker labels
    source: str = "file"  # "toggle", "file", "vad", "interval"


class PostProcessor(ABC):
    """Abstract base for LLM post-processors applied after transcription."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this post-processor (prompt name)."""

    @property
    def model(self) -> str:
        """Model name used for this run, or empty string if not applicable."""
        return ""

    @property
    def backend_type(self) -> str:
        """Backend type string for output template formatting."""
        return self.name

    @abstractmethod
    def process(
        self,
        transcript: str,
        *,
        metadata: PostMetadata,
    ) -> str:
        """Send transcript to an LLM and return the result.

        Args:
            transcript: Full transcript text (plain or speaker-attributed).
            metadata: Recording context (date, duration, speakers, etc.).

        Returns:
            Post-processed text to replace the transcript in output.
        """
