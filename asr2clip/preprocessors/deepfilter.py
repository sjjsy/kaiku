"""DeepFilterNet3 audio preprocessor (best quality, medium CPU, Rust/torch wheel)."""

from __future__ import annotations

import sys

import numpy as np

from .base import AudioPreprocessor, loudnorm


class DeepFilterNetPreprocessor(AudioPreprocessor):
    """DeepFilterNet3 neural noise reduction.

    Best quality of the available options. Handles speakers at variable distances
    and suppresses background hum, music, and crowd noise. Requires a one-time
    model download (~70 MB) on first use. Resampling uses torchaudio (already
    a deepfilternet dependency) so no scipy is needed.
    Requires: pip install asr2clip[deepfilter]
    """

    @property
    def name(self) -> str:
        return "deepfilter"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            from df import init_df, enhance
        except ImportError:
            print(
                "Error: deepfilternet is not installed.\n"
                "Install with: pip install asr2clip[deepfilter]",
                file=sys.stderr,
            )
            sys.exit(1)

        import torch
        import torchaudio.functional as F_audio

        model, df_state, _ = init_df()
        model_sr = df_state.sr()

        # Resample to model sample rate via torchaudio (no scipy needed)
        audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        if sample_rate != model_sr:
            audio_t = F_audio.resample(audio_t, sample_rate, model_sr)

        # enhance() expects (C, T) numpy array
        audio_2d = audio_t.numpy()
        enhanced_2d = enhance(model, df_state, audio_2d)
        enhanced_t = torch.from_numpy(np.asarray(enhanced_2d))  # (1, T)

        # Resample back to original sample rate
        if sample_rate != model_sr:
            enhanced_t = F_audio.resample(enhanced_t, model_sr, sample_rate)

        enhanced = enhanced_t.squeeze(0).numpy()

        # Trim/pad to match original length
        target_len = len(audio)
        if len(enhanced) > target_len:
            enhanced = enhanced[:target_len]
        elif len(enhanced) < target_len:
            enhanced = np.pad(enhanced, (0, target_len - len(enhanced)))

        return loudnorm(np.clip(enhanced, -1.0, 1.0).astype(np.float32))
