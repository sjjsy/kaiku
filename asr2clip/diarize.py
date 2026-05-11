"""Speaker diarization via WhisperX.

When --diarize / -D is set, this module replaces the standard ASR backend entirely.
WhisperX runs its own Whisper inference + word-level alignment + pyannote
diarization in one pass — no double transcription.

Output format: "[HH:MM:SS] SPEAKER_NN: text"
Speaker label substitution (SPEAKER_00 → real names) is left to the calling
assistant or post-processor, not done here.

Requires: pip install asr2clip[diarize]
"""

from __future__ import annotations

import os
import sys


class DiarizationError(Exception):
    pass


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_transcript(segments: list[dict]) -> str:
    """Merge consecutive same-speaker segments and format as timestamped turns."""
    lines: list[str] = []
    current_speaker: str | None = None
    current_text: list[str] = []
    current_start: float = 0.0

    def flush() -> None:
        if current_text and current_speaker is not None:
            lines.append(f"[{_fmt_ts(current_start)}] {current_speaker}: {' '.join(current_text).strip()}")

    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        if not text:
            continue
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            current_text = [text]
            current_start = start
        else:
            current_text.append(text)

    flush()
    return "\n".join(lines)


def run_diarization(
    audio_path: str,
    config: dict,
    language: str | None = None,
    num_speakers: int | None = None,
) -> str:
    """Run WhisperX on audio_path; return a speaker-attributed transcript.

    Format per turn: "[HH:MM:SS] SPEAKER_NN: text"
    Speaker label substitution is left to the caller (assistant / post-processor).

    Args:
        audio_path: Path to a WAV or supported audio file.
        config: Full asr2clip config dict (for HF token and speaker count hints).
        language: ISO-639-1 language code, or None for auto-detect.
        num_speakers: Expected speaker count (overrides config hints).

    Returns:
        Formatted speaker-attributed transcript string.

    Raises:
        DiarizationError: If whisperx is not installed or diarization fails.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        raise DiarizationError(
            "whisperx is not installed.\n"
            "Install with: pip install asr2clip[diarize]\n"
            "Then accept the pyannote model licence at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "and set HF_TOKEN in your environment or 'diarize_hf_token' in config."
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    hf_token = config.get("diarize_hf_token") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise DiarizationError(
            "Diarization requires a HuggingFace token to download the pyannote model.\n"
            "Set HF_TOKEN in your environment or add 'diarize_hf_token: hf_...' to config.\n"
            "Accept the licence at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    if num_speakers is not None:
        min_speakers = max_speakers = num_speakers
    else:
        min_speakers = config.get("diarize_min_speakers")
        max_speakers = config.get("diarize_max_speakers")

    try:
        model = whisperx.load_model(
            "large-v3", device, compute_type=compute_type, language=language
        )
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16, language=language)

        align_model, align_meta = whisperx.load_align_model(
            language_code=result.get("language", language or "en"),
            device=device,
        )
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio, device,
            return_char_alignments=False,
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_kwargs: dict = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)

        result = whisperx.assign_word_speakers(diarize_segments, result)

    except DiarizationError:
        raise
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    return _format_transcript(result.get("segments", []))
