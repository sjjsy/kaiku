"""Robust chunked transcription for long audio files."""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import TYPE_CHECKING

from pydub import AudioSegment
from pydub.silence import detect_silence

from .audio import audiosegment_to_float32, float32_to_audiosegment
from .output import copy_transcript_to_clipboard
from .transcribe import TranscriptionError, transcribe
from .utils import info, safe_unlink, warning

if TYPE_CHECKING:
    from .config_types import Config


def _find_chunk_boundaries(
    audio: AudioSegment, max_chunk_ms: int
) -> list[tuple[int, int]]:
    """Split audio at silence boundaries up to max_chunk_ms each.

    Tries to end each chunk at a silence midpoint; hard-cuts if none found.
    """
    total_ms = len(audio)
    silence_thresh = audio.dBFS - 16  # adaptive threshold
    try:
        silences = detect_silence(
            audio, min_silence_len=500, silence_thresh=silence_thresh
        )
    except Exception:
        silences = []

    boundaries: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + max_chunk_ms, total_ms)
        if end < total_ms:
            half = pos + max_chunk_ms // 2
            for s_start, s_end in silences:
                mid = (s_start + s_end) // 2
                if half <= mid <= end:
                    end = mid
                    break
        boundaries.append((pos, end))
        pos = end
    return boundaries


def _check_quality(text: str) -> bool:
    """Return False if text looks like a hallucination loop or is nearly empty."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return False
    if len(set(words)) / len(words) < 0.5:
        return False
    return True


def _estimate_timeout(chunk_duration_s: float) -> float:
    """Estimate timeout for chunk transcription. Uses conservative 4x multiplier for safety."""
    return max(30.0, chunk_duration_s * 4.0)


def process_file_robust(config: "Config"):
    """Transcribe a long audio file in silence-bounded chunks with quality checks.

    Chunk ASR text is appended iteratively to ``-o`` FILE or to a temp file (when
    there is no ``-o``, chunks still print to stdout). The assembly file is then
    read as ``transcript``, optionally post-processed, passed through
    ``format_output`` as ``text_output``, and FILE is overwritten with
    ``text_output`` only (no second timestamped append).

    Args:
        config: Resolved run config (``input_file``, ``output_file``, ``chunk_duration``, …).
    """
    import sys
    from .postprocessors import NonePostProcessor, PostMetadata, format_output, make_postprocessor, resolve_output_template
    from .preprocessors import NonePreprocessor, make_preprocessor

    if not config.input_file or not os.path.exists(config.input_file):
        print(f"File not found: {config.input_file}")
        sys.exit(1)

    info(f"Loading audio: {config.input_file}")
    t0 = time.time()
    audio = AudioSegment.from_file(config.input_file)
    # Normalise to 16 kHz mono for consistent transcription
    audio = audio.set_frame_rate(16000).set_channels(1)
    total_s = len(audio) / 1000.0
    info(f"Audio loaded: {total_s:.1f}s total ({time.time() - t0:.1f}s to load)")

    preprocessor = make_preprocessor(config)
    if not isinstance(preprocessor, NonePreprocessor):
        t_pre = time.time()
        try:
            audio_np = audiosegment_to_float32(audio)
            audio_np = preprocessor.process(audio_np, 16000)
            audio = float32_to_audiosegment(audio_np, 16000)
            info(f"Preprocessing completed in {time.time() - t_pre:.2f}s")
        except Exception as e:
            warning(f"Preprocessing failed ({e}), using audio as-is.")

    max_chunk_ms = config.chunk_duration * 1000
    boundaries = _find_chunk_boundaries(audio, max_chunk_ms)
    n_chunks = len(boundaries)
    info(f"Splitting into {n_chunks} chunk(s)")

    scratch: str | None = None
    output_file = config.output_file
    if output_file:
        d = os.path.dirname(os.path.abspath(output_file))
        if d:
            os.makedirs(d, exist_ok=True)
        chunk_target = output_file
    else:
        fd, scratch = tempfile.mkstemp(suffix=".txt", prefix="kaiku_robust_")
        os.close(fd)
        chunk_target = scratch

    info(f"Target file for chunks: {chunk_target}")

    had_transcribed = False

    for idx, (start_ms, end_ms) in enumerate(boundaries, 1):
        chunk_s = (end_ms - start_ms) / 1000.0
        timeout = _estimate_timeout(chunk_s)

        chunk_audio = audio[start_ms:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="kaiku_chunk_", delete=False)
        tmp_path = tmp.name
        tmp.close()
        chunk_audio.export(tmp_path, format="wav")

        text: str | None = None
        quality_ok = False
        retries = 3
        t_chunk = time.time()

        for attempt in range(retries):
            try:
                candidate = transcribe(tmp_path, config, raise_on_error=True, timeout=timeout)
                if _check_quality(candidate):
                    text = candidate
                    quality_ok = True
                    break
                else:
                    if attempt < retries - 1:
                        warning(
                            f"Chunk {idx}/{n_chunks}: quality check failed "
                            f"(attempt {attempt + 1}/{retries}), retrying…"
                        )
            except TranscriptionError as e:
                if attempt < retries - 1:
                    warning(f"Chunk {idx}/{n_chunks}: error (attempt {attempt + 1}/{retries}): {e}")
                else:
                    warning(f"Chunk {idx}/{n_chunks}: failed after {retries} attempts: {e}")
            except Exception as e:
                warning(f"Chunk {idx}/{n_chunks}: unexpected error: {e}")
                break

        safe_unlink(tmp_path)

        elapsed = time.time() - t_chunk
        preview = (text or "")[:60].replace("\n", " ")
        info(f"Chunk {idx}/{n_chunks} ({chunk_s:.1f}s audio → {elapsed:.1f}s): {preview}…")

        if text is None:
            warning(f"Chunk {idx}/{n_chunks} could not be transcribed; skipping.")
            with open(chunk_target, "a", encoding="utf-8") as f:
                f.write(f"\n# [ERROR: chunk {idx}/{n_chunks} — transcription failed]\n")
            continue

        if not quality_ok:
            # Last attempt produced text but quality is uncertain
            warning_line = f"\n# [WARNING: chunk {idx}/{n_chunks} — quality uncertain, may contain errors]\n"
        else:
            warning_line = ""

        had_transcribed = True

        with open(chunk_target, "a", encoding="utf-8") as f:
            if warning_line:
                f.write(warning_line)
            f.write(text)
            f.write("\n\n")
        if output_file:
            info(f"Chunk {idx}/{n_chunks} of size {len(text)} appended to {output_file}")
        else:
            print(text)
            print()

    if not had_transcribed:
        info("No speech detected in any chunk.")
        if scratch:
            safe_unlink(scratch)
        return

    with open(chunk_target, encoding="utf-8") as f:
        transcript = f.read()
    if scratch:
        safe_unlink(scratch)

    if not transcript.strip():
        info("No speech detected in any chunk.")
        return

    from datetime import date
    postprocessor = make_postprocessor(config)
    template = resolve_output_template(config)
    metadata = PostMetadata(
        date=date.today().isoformat(),
        duration_s=total_s,
        language=config.language or "auto",
        prompt_name=postprocessor.name,
        diarized=config.asr_backend.type in ("whisperx", "mock-diarize"),
        source="file",
    )
    if not isinstance(postprocessor, NonePostProcessor):
        info(f"Post-processing full transcript with '{postprocessor.name}'…")
        t_post = time.time()
        result = postprocessor.process(transcript, metadata=metadata)
        info(f"Post-processing completed in {time.time() - t_post:.1f}s")
    else:
        result = transcript

    text_output = format_output(
        template, result=result, transcript=transcript,
        metadata=metadata, model=postprocessor.model,
        backend=postprocessor.backend_type,
    )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text_output)

    copy_transcript_to_clipboard(text_output, config)

    info(f"Done. {n_chunks} chunk(s), {total_s:.0f}s audio transcribed.")
