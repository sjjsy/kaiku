"""WhisperX diarization backend for kaiku.

Performs speaker diarization: runs Whisper + word-level alignment + pyannote
speaker assignment in one pass and returns a timestamped, speaker-attributed
transcript.

Output format: "[HH:MM:SS] SPEAKER_NN: text"
Speaker label substitution (SPEAKER_00 → real names) is left to the caller.

Requires: pip install kaiku[diarize]
"""

from __future__ import annotations

from dataclasses import dataclass

from ..transcribe import TranscriptionError


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_transcript(segments: list[dict]) -> str:
    """Merge consecutive same-speaker segments into timestamped turns."""
    lines: list[str] = []
    current_speaker: str | None = None
    current_text: list[str] = []
    current_start: float = 0.0

    def flush() -> None:
        if current_text and current_speaker is not None:
            lines.append(
                f"[{_fmt_ts(current_start)}] {current_speaker}: {' '.join(current_text).strip()}"
            )

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


@dataclass
class WhisperXConfig:
    """Parameters for a WhisperX diarization run.

    Args:
        hf_token: HuggingFace token for the pyannote speaker-diarization model.
        min_speakers: Lower bound on speaker count passed to pyannote (optional).
        max_speakers: Upper bound on speaker count passed to pyannote (optional).
    """

    hf_token: str
    min_speakers: int | None = None
    max_speakers: int | None = None


def transcribe(
    audio_path: str,
    cfg: WhisperXConfig,
    language: str | None = None,
    timeout: float | None = None,
) -> str:
    """Run WhisperX on audio_path and return a speaker-attributed transcript.

    Args:
        audio_path: Path to a WAV or supported audio file.
        cfg: WhisperX backend configuration (token + optional speaker hints).
        language: ISO-639-1 language code, or None for auto-detect.
        timeout: Ignored; present for API compatibility.

    Returns:
        Formatted transcript string: "[HH:MM:SS] SPEAKER_NN: text" per turn.

    Raises:
        TranscriptionError: If whisperx is not installed or the run fails.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        raise TranscriptionError(
            "whisperx is not installed.\n"
            "Install with: pip install kaiku[diarize]\n"
            "Then accept the pyannote model licence at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "and set HF_TOKEN in your environment or 'hf_token:' in the whisperx\n"
            "asr_backends entry."
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    diarize_kwargs: dict = {}
    if cfg.min_speakers is not None:
        diarize_kwargs["min_speakers"] = cfg.min_speakers
    if cfg.max_speakers is not None:
        diarize_kwargs["max_speakers"] = cfg.max_speakers

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
            result["segments"],
            align_model,
            align_meta,
            audio,
            device,
            return_char_alignments=False,
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=cfg.hf_token, device=device
        )
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"WhisperX diarization failed: {e}") from e

    return _format_transcript(result.get("segments", []))
