"""noisereduce-based audio preprocessor (spectral noise reduction, scipy-backed)."""

from __future__ import annotations

import sys

import numpy as np

from .base import AudioPreprocessor, loudnorm


class NoiseReducePreprocessor(AudioPreprocessor):
    """Spectral noise reduction via the noisereduce library.

    Uses non-stationary noise estimation — no reference noise clip needed.
    Requires: pip install asr2clip[noisereduce]
    """

    @property
    def name(self) -> str:
        return "noisereduce"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import noisereduce as nr
        except ImportError:
            print(
                "Error: noisereduce is not installed.\n"
                "Install with: pip install asr2clip[noisereduce]",
                file=sys.stderr,
            )
            sys.exit(1)

        if audio.ndim > 1:
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        cleaned = nr.reduce_noise(y=audio, sr=sample_rate, stationary=False)
        return loudnorm(np.clip(cleaned, -1.0, 1.0).astype(np.float32))
