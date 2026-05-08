"""Robust chunked transcription for long audio files."""

import os
import re
import tempfile
import time

from pydub import AudioSegment
from pydub.silence import detect_silence

from .output import copy_to_clipboard, append_transcript_to_file
from .transcribe import TranscriptionError, transcribe_with_config
from .utils import info, log, warning


_MAX_CLIPBOARD_CHARS = 4000


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


def _estimate_timeout(chunk_duration_s: float, backend: str) -> float:
    multiplier = 4.0 if backend == "whisper_cpp" else 2.0
    return max(30.0, chunk_duration_s * multiplier)


def process_file_robust(
    config: dict,
    input_file: str,
    output_file: str | None = None,
    chunk_duration: int = 180,
):
    """Transcribe a long audio file in silence-bounded chunks with quality checks.

    Each chunk is written to output_file immediately so the user can
    `tail -f` it while processing continues.

    Args:
        config: Full configuration dictionary.
        input_file: Path to the audio file.
        output_file: Optional file to append chunks to (tail-f friendly).
        chunk_duration: Maximum chunk length in seconds (default 180).
    """
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        import sys
        sys.exit(1)

    log(f"Loading audio: {input_file}")
    t0 = time.time()
    audio = AudioSegment.from_file(input_file)
    # Normalise to 16 kHz mono for consistent transcription
    audio = audio.set_frame_rate(16000).set_channels(1)
    total_s = len(audio) / 1000.0
    info(f"Audio loaded: {total_s:.1f}s total ({time.time() - t0:.1f}s to load)")

    max_chunk_ms = chunk_duration * 1000
    boundaries = _find_chunk_boundaries(audio, max_chunk_ms)
    n_chunks = len(boundaries)
    backend = config.get("backend", "api")
    log(f"Splitting into {n_chunks} chunk(s), backend: {backend}")

    all_text_parts: list[str] = []

    for idx, (start_ms, end_ms) in enumerate(boundaries, 1):
        chunk_s = (end_ms - start_ms) / 1000.0
        timeout = _estimate_timeout(chunk_s, backend)

        chunk_audio = audio[start_ms:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="asr2clip_chunk_", delete=False)
        tmp_path = tmp.name
        tmp.close()
        chunk_audio.export(tmp_path, format="wav")

        text: str | None = None
        quality_ok = False
        retries = 3
        t_chunk = time.time()

        for attempt in range(retries):
            try:
                candidate = transcribe_with_config(
                    tmp_path, config, raise_on_error=True, timeout=timeout
                )
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

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        elapsed = time.time() - t_chunk
        preview = (text or "")[:60].replace("\n", " ")
        info(f"Chunk {idx}/{n_chunks} ({chunk_s:.1f}s audio → {elapsed:.1f}s): {preview}…")

        if text is None:
            warning(f"Chunk {idx}/{n_chunks} could not be transcribed; skipping.")
            if output_file:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n# [ERROR: chunk {idx}/{n_chunks} — transcription failed]\n")
            continue

        if not quality_ok:
            # Last attempt produced text but quality is uncertain
            warning_line = f"\n# [WARNING: chunk {idx}/{n_chunks} — quality uncertain, may contain errors]\n"
        else:
            warning_line = ""

        all_text_parts.append(text)

        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                if warning_line:
                    f.write(warning_line)
                f.write(text)
                f.write("\n\n")
            log(f"Chunk {idx}/{n_chunks} appended to {output_file}")
        else:
            print(text)
            print()

    if not all_text_parts:
        log("No speech detected in any chunk.")
        return

    full_text = "\n\n".join(all_text_parts)

    if len(full_text) <= _MAX_CLIPBOARD_CHARS:
        if copy_to_clipboard(full_text):
            log("Full transcript copied to clipboard.")
        else:
            warning("Clipboard copy failed.")
    else:
        last_chunk = all_text_parts[-1]
        if copy_to_clipboard(last_chunk):
            log(
                f"Transcript too long for clipboard ({len(full_text)} chars); "
                "last chunk copied instead."
            )
        if output_file:
            log(f"Full transcript written to {output_file}")

    log(f"Done. {n_chunks} chunk(s), {total_s:.0f}s audio transcribed.")
