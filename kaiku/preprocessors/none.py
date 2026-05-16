"""No-op preprocessor."""

from __future__ import annotations

import numpy as np

from .base import AudioPreprocessor


class NonePreprocessor(AudioPreprocessor):
    """Pass-through — returns audio unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio
