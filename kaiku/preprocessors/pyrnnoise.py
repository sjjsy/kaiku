"""pyrnnoise-based audio preprocessor (Mozilla RNNoise GRU, near-zero CPU)."""

from __future__ import annotations

import sys
from math import gcd

import numpy as np

from .base import AudioPreprocessor, loudnorm

_RNNOISE_RATE = 48000
_FRAME_SIZE = 480  # 10 ms at 48 kHz


class PyRNNoisePreprocessor(AudioPreprocessor):
    """Mozilla RNNoise GRU noise reduction via pyrnnoise.

    Very low CPU overhead. Internally operates at 48 kHz; input is resampled
    automatically via scipy. May sound slightly robotic on very heavy noise.
    Requires: pip install kaiku[pyrnnoise]
    """

    @property
    def name(self) -> str:
        return "pyrnnoise"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import pyrnnoise
        except ImportError:
            print(
                "Error: pyrnnoise is not installed.\n"
                "Install with: pip install kaiku[pyrnnoise]",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            from scipy.signal import resample_poly
        except ImportError:
            print(
                "Error: scipy is required for pyrnnoise resampling.\n"
                "Install with: pip install scipy",
                file=sys.stderr,
            )
            sys.exit(1)

        if audio.ndim > 1:
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

        # Resample to 48 kHz for RNNoise
        if sample_rate != _RNNOISE_RATE:
            g = gcd(sample_rate, _RNNOISE_RATE)
            up, down = _RNNOISE_RATE // g, sample_rate // g
            audio_48k = resample_poly(audio, up, down).astype(np.float32)
        else:
            up, down = 1, 1
            audio_48k = audio.astype(np.float32)

        # RNNoise expects float values in int16 range [-32768, 32767]
        audio_scaled = (audio_48k * 32767.0).clip(-32768.0, 32767.0)

        denoiser = pyrnnoise.RNNoise()
        n_frames = len(audio_scaled) // _FRAME_SIZE
        out_frames: list[np.ndarray] = []
        for i in range(n_frames):
            frame = audio_scaled[i * _FRAME_SIZE : (i + 1) * _FRAME_SIZE].tolist()
            out_frame = denoiser.process_frame(frame)
            out_frames.append(np.array(out_frame, dtype=np.float32))

        if not out_frames:
            return audio

        cleaned_48k = np.concatenate(out_frames) / 32767.0

        # Resample back to original sample rate
        if sample_rate != _RNNOISE_RATE:
            cleaned = resample_poly(cleaned_48k, down, up).astype(np.float32)
        else:
            cleaned = cleaned_48k

        # Trim/pad to match original length
        target_len = len(audio)
        if len(cleaned) > target_len:
            cleaned = cleaned[:target_len]
        elif len(cleaned) < target_len:
            cleaned = np.pad(cleaned, (0, target_len - len(cleaned)))

        return loudnorm(np.clip(cleaned, -1.0, 1.0).astype(np.float32))
