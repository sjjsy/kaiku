"""Toggle-mode recording for kaiku.

First invocation: start a background recorder, write lock file, exit.
Second invocation: stop the recorder, transcribe, copy to clipboard.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

from .audio import load_wav, save_audio
from .output import output_transcript
from .postprocessors import NonePostProcessor, PostMetadata, format_output, make_postprocessor
from .preprocessors import NonePreprocessor, make_preprocessor
from .recorders import _kill_process, _pid_alive, make_recorder
from .transcribe import transcribe
from .utils import info, run_subprocess, safe_unlink, warning

if TYPE_CHECKING:
    from .config_types import Config


def _lock_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return os.path.join(runtime, "kaiku.lock")


def _notify(title: str, body: str):
    if shutil.which("notify-send"):
        try:
            run_subprocess(
                ["notify-send", "-t", "8000", title, body],
                check=False, capture_output=True,
            )
        except Exception:
            pass


def toggle_recording(config: "Config"):
    """Start or stop toggle-mode recording."""
    lock_path = _lock_path()

    if os.path.exists(lock_path):
        _stop_and_transcribe(lock_path, config)
    else:
        _start_recording(lock_path, config)


def _start_recording(lock_path: str, config: "Config"):
    audio_path = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="kaiku_", delete=False
    ).name

    recorder = make_recorder(config.recorder.name, device_info=config.recorder.device)
    pid = recorder.start(audio_path, config.recorder.device)
    if pid is None:
        safe_unlink(audio_path)
        warning("Could not start recorder. Check device availability.")
        _notify("kaiku", "Failed to start recording.")
        return

    lock_data = {"pid": pid, "audio": audio_path, "recorder": recorder.name}
    with open(lock_path, "w") as f:
        json.dump(lock_data, f)

    device_desc = f" with {config.recorder.device.name}" if config.recorder.device else " (default device)"
    info(f"Recording started ({recorder.name}, pid {pid}){device_desc}")
    _notify("kaiku", f"Recording{device_desc}… (run kaiku --toggle to stop)")


def _stop_and_transcribe(lock_path: str, config: "Config"):
    with open(lock_path) as f:
        lock_data = json.load(f)

    pid = lock_data.get("pid", 0)
    audio_path = lock_data.get("audio", "")

    if not _pid_alive(pid):
        warning(f"Recorder PID {pid} is no longer running (stale lock). Cleaning up.")
        os.unlink(lock_path)
        if audio_path and os.path.exists(audio_path):
            _transcribe_and_output(audio_path, config)
        return

    info(f"Stopping recorder (pid {pid})…")
    _kill_process(pid)
    os.unlink(lock_path)

    time.sleep(0.3)

    _transcribe_and_output(audio_path, config)


def _transcribe_and_output(audio_path: str, config: "Config"):
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        info("Audio file is empty or missing — nothing to transcribe.")
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
    preprocessor = make_preprocessor(config)
    if not isinstance(preprocessor, NonePreprocessor):
        try:
            audio_data, sr = load_wav(audio_path)
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
    try:
        transcript = transcribe(transcribe_path, config, raise_on_error=True)
    except Exception as e:
        warning(f"Transcription failed: {e}")
        _notify("kaiku", f"Transcription failed: {e}")
        return
    finally:
        safe_unlink(audio_path)
        if preprocessed_path and preprocessed_path != audio_path:
            safe_unlink(preprocessed_path)

    info(f"Transcription completed in {time.time() - t0:.1f}s")

    if not transcript.strip():
        info("No speech detected.")
        _notify("kaiku", "No speech detected.")
        return

    from datetime import date
    from .postprocessors import resolve_output_template
    postprocessor = make_postprocessor(config)
    template = resolve_output_template(config)
    metadata = PostMetadata(
        date=date.today().isoformat(),
        duration_s=duration,
        language=config.language or "auto",
        prompt_name=postprocessor.name,
        diarized=config.asr_backend.type in ("whisperx", "mock-diarize"),
        source="toggle",
    )
    if not isinstance(postprocessor, NonePostProcessor):
        info(f"Post-processing with '{postprocessor.name}'…")
        t_post = time.time()
        result = postprocessor.process(transcript, metadata=metadata)
        info(f"Post-processing completed in {time.time() - t_post:.1f}s")
    else:
        result = transcript
    final = format_output(
        template, result=result, transcript=transcript,
        metadata=metadata, model=postprocessor.model,
        backend=postprocessor.backend_type,
    )

    output_transcript(final, config)
    _notify("kaiku", final[:100])
