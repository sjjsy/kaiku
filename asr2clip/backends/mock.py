"""Mock ASR backend for testing and demos without API credentials.

Returns a canned transcript without making actual API calls or running external binaries.
Useful for:
- Testing asr2clip without credentials
- Demos without network calls
- CI/CD pipelines
- Development iteration
"""

from __future__ import annotations

from dataclasses import dataclass

from ..transcribe import TranscriptionError
from ..utils import info


_DEFAULT_RESPONSE = (
    "The quick brown fox jumps over the lazy dog. This is a mock transcription "
    "provided by asr2clip's mock backend for testing and demonstrations."
)


@dataclass
class MockConfig:
    """Configuration for mock transcription backend.

    Args:
        response: Text to return as the transcription. If not provided, uses default.
        latency_ms: Simulated processing latency in milliseconds (optional, for realism).
    """
    response: str = _DEFAULT_RESPONSE
    latency_ms: int = 0

    @classmethod
    def from_config(cls, config: dict) -> MockConfig:
        """Create MockConfig from backend configuration dict."""
        return cls(
            response=config.get("response", _DEFAULT_RESPONSE),
            latency_ms=int(config.get("latency_ms", 0)),
        )


def transcribe(
    audio_path: str,
    cfg: MockConfig,
    timeout: float | None = None,
) -> str:
    """Transcribe using mock backend (returns canned response).

    Args:
        audio_path: Path to the audio file (unused in mock).
        cfg: Mock backend configuration.
        timeout: Timeout (ignored in mock).

    Returns:
        The configured mock response text.
    """
    if cfg.latency_ms > 0:
        import time
        latency_s = cfg.latency_ms / 1000.0
        info(f"Mock backend: simulating {cfg.latency_ms}ms latency...")
        time.sleep(latency_s)

    info("Mock backend: returning canned transcript")
    return cfg.response


def test(cfg: MockConfig) -> bool:
    """Verify the mock configuration.

    Returns:
        Always True (mock backend has no external dependencies).
    """
    from ..utils import print_key_value, print_success

    print_success("Mock backend is always available")
    if cfg.response != _DEFAULT_RESPONSE:
        print_key_value("Custom response", f"{len(cfg.response)} characters")
    if cfg.latency_ms > 0:
        print_key_value("Simulated latency", f"{cfg.latency_ms}ms")

    return True
