"""Base class for audio preprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def loudnorm(
    audio: np.ndarray,
    target_rms_db: float = -20.0,
    peak_db: float = -0.1,
) -> np.ndarray:
    """Normalise audio loudness after noise reduction.

    After any noise suppression step the speech signal may be quieter than
    before (the noise floor was raising the apparent level). This two-step
    normalisation restores a consistent volume:

    1. RMS normalisation toward target_rms_db (-20 dBFS by default).
       This brings quiet, de-noised speech up to a level where Whisper's
       attention mechanism works well.
    2. Peak limiting to peak_db (-0.1 dBFS by default) to prevent clipping.

    Args:
        audio: Float32 mono array, values nominally in [-1.0, 1.0].
        target_rms_db: Target RMS level in dBFS (default -20 dBFS ≈ 0.1 linear).
        peak_db: Peak ceiling in dBFS (default -0.1 dBFS ≈ 0.989 linear).

    Returns:
        Level-normalised float32 array.
    """
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 1e-8:
        target_rms = 10 ** (target_rms_db / 20.0)
        audio = audio * (target_rms / rms)
    ceiling = 10 ** (peak_db / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio.astype(np.float32)


class AudioPreprocessor(ABC):
    """Abstract base for audio preprocessing applied before transcription."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this preprocessor."""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Clean audio and return at the same sample rate.

        Args:
            audio: Float32 mono array, values in [-1.0, 1.0].
            sample_rate: Sample rate of the input audio in Hz.

        Returns:
            Cleaned float32 mono array at sample_rate Hz.
        """
