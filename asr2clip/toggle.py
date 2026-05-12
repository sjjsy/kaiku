"""Toggle-mode recording for asr2clip.

First invocation: start a background recorder, write lock file, exit.
Second invocation: stop the recorder, transcribe, copy to clipboard.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from .audio import load_wav, save_audio
from .config import resolve_audio_device, resolve_clipboard_max_chars, resolve_recorder_config
from .output import output_transcript
from .postprocessors import NonePostProcessor, PostMetadata, PostProcessor, format_output
from .preprocessors import AudioPreprocessor, NonePreprocessor
from .recorders import _kill_process, _pid_alive, make_recorder
from .transcribe import transcribe_with_config
from .utils import info, log, run_subprocess, safe_unlink, warning


def _lock_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return os.path.join(runtime, "asr2clip.lock")


def _notify(title: str, body: str):
    if shutil.which("notify-send"):
        try:
            run_subprocess(
                ["notify-send", "-t", "8000", title, body],
                check=False, capture_output=True,
            )
        except Exception:
            pass


def toggle_recording(
    config: dict,
    device: str | int | None = None,
    output_file: str | None = None,
    language: str | None = None,
    preprocessor: AudioPreprocessor | None = None,
    postprocessor: PostProcessor | None = None,
    template_str: str = "{result}",
    diarize: bool = False,
    num_speakers: int | None = None,
    backend: str | None = None,
):
    """Start or stop toggle-mode recording."""
    lock_path = _lock_path()

    if os.path.exists(lock_path):
        _stop_and_transcribe(
            lock_path, config, output_file, language, preprocessor,
            postprocessor, template_str, diarize, num_speakers, backend,
        )
    else:
        _start_recording(lock_path, device_cli=device, config=config)


def _start_recording(lock_path: str, device_cli: str | None, config: dict):
    audio_path = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="asr2clip_", delete=False
    ).name

    device_info = resolve_audio_device(config, cli_override=device_cli)
    recorder = make_recorder(resolve_recorder_config(config), device_info=device_info)
    pid = recorder.start(audio_path, device_info)
    if pid is None:
        safe_unlink(audio_path)
        warning("Could not start recorder. Check device availability.")
        _notify("asr2clip", "Failed to start recording.")
        return

    lock_data = {"pid": pid, "audio": audio_path, "recorder": recorder.name}
    with open(lock_path, "w") as f:
        json.dump(lock_data, f)

    device_desc = f" with {device_info.name}" if device_info else " (default device)"
    info(f"Recording started ({recorder.name}, pid {pid}){device_desc}")
    _notify("asr2clip", f"Recording{device_desc.replace(' with ', ' with ')}… (run asr2clip --toggle to stop)")


def _stop_and_transcribe(
    lock_path: str,
    config: dict,
    output_file: str | None,
    language: str | None = None,
    preprocessor: AudioPreprocessor | None = None,
    postprocessor: PostProcessor | None = None,
    template_str: str = "{result}",
    diarize: bool = False,
    num_speakers: int | None = None,
    backend: str | None = None,
):
    with open(lock_path) as f:
        lock_data = json.load(f)

    pid = lock_data.get("pid", 0)
    audio_path = lock_data.get("audio", "")

    if not _pid_alive(pid):
        warning(f"Recorder PID {pid} is no longer running (stale lock). Cleaning up.")
        os.unlink(lock_path)
        if audio_path and os.path.exists(audio_path):
            _transcribe_and_output(
                audio_path, config, output_file, language, preprocessor,
                postprocessor, template_str, diarize, num_speakers, backend,
            )
        return

    log(f"Stopping recorder (pid {pid})…")
    _kill_process(pid)
    os.unlink(lock_path)

    time.sleep(0.3)

    _transcribe_and_output(
        audio_path, config, output_file, language, preprocessor,
        postprocessor, template_str, diarize, num_speakers, backend,
    )


def _transcribe_and_output(
    audio_path: str,
    config: dict,
    output_file: str | None,
    language: str | None = None,
    preprocessor: AudioPreprocessor | None = None,
    postprocessor: PostProcessor | None = None,
    template_str: str = "{result}",
    diarize: bool = False,
    num_speakers: int | None = None,
    backend: str | None = None,
):
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        log("Audio file is empty or missing — nothing to transcribe.")
        return

    duration = 0.0
    try:
        import wave as _wave
        with _wave.open(audio_path) as wf:
            duration = wf.getnframes() / wf.getframerate()
        info(f"Recorded {duration:.1f}s of audio, transcribing…")
    except Exception:
        info("Transcribing recorded audio…")

    preprocessed_path: str | None = None
    if preprocessor is not None and not isinstance(preprocessor, NonePreprocessor):
        try:
            audio_data, sr = load_wav(audio_path)
            log(f"Preprocessing audio with {preprocessor.name}…")
            t_pre = time.time()
            audio_data = preprocessor.process(audio_data, sr)
            info(f"Preprocessing completed in {time.time() - t_pre:.2f}s")
            preprocessed_path = save_audio(audio_data, sr)
            transcribe_path = preprocessed_path
        except Exception as e:
            warning(f"Preprocessing failed ({e}), using original audio.")
            transcribe_path = audio_path
    else:
        transcribe_path = audio_path

    t0 = time.time()
    transcript: str = ""
    is_diarized = False
    try:
        if diarize:
            from .diarize import DiarizationError, run_diarization
            try:
                log("Running diarization (WhisperX)…")
                transcript = run_diarization(
                    transcribe_path, config,
                    language=language, num_speakers=num_speakers,
                )
                is_diarized = True
            except DiarizationError as e:
                warning(f"Diarization failed: {e}")
                _notify("asr2clip", f"Diarization failed: {e}")
                return
        else:
            transcript = transcribe_with_config(
                transcribe_path, config, raise_on_error=True, language=language, backend=backend
            )
    except Exception as e:
        warning(f"Transcription failed: {e}")
        _notify("asr2clip", f"Transcription failed: {e}")
        return
    finally:
        safe_unlink(audio_path)
        if preprocessed_path and preprocessed_path != audio_path:
            safe_unlink(preprocessed_path)

    info(f"Transcription completed in {time.time() - t0:.1f}s")

    if not transcript.strip():
        log("No speech detected.")
        _notify("asr2clip", "No speech detected.")
        return

    final = transcript
    if postprocessor is not None and not isinstance(postprocessor, NonePostProcessor):
        from datetime import date
        metadata = PostMetadata(
            date=date.today().isoformat(),
            duration_s=duration,
            language=language or "auto",
            prompt_name=postprocessor.name,
            diarized=is_diarized,
            source="toggle",
        )
        log(f"Post-processing with '{postprocessor.name}'…")
        t_post = time.time()
        result = postprocessor.process(transcript, metadata=metadata)
        info(f"Post-processing completed in {time.time() - t_post:.1f}s")
        final = format_output(
            template_str, result=result, transcript=transcript,
            metadata=metadata, model=postprocessor.model,
            backend=postprocessor.backend_type,
        )

    output_transcript(
        final, to_clipboard=True, to_stdout=True, to_file=output_file,
        max_clipboard_chars=resolve_clipboard_max_chars(config),
    )
    _notify("asr2clip", final[:100])
