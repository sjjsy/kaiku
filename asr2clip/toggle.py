"""Toggle-mode recording for asr2clip.

First invocation: start arecord in background, write lock file, exit.
Second invocation: stop arecord, transcribe, copy to clipboard.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .output import copy_to_clipboard, output_transcript
from .transcribe import transcribe_with_config
from .utils import info, log, warning


def _lock_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return os.path.join(runtime, "asr2clip.lock")


def _notify(title: str, body: str):
    if shutil.which("notify-send"):
        try:
            subprocess.run(
                ["notify-send", "-t", "4000", title, body],
                check=False, capture_output=True,
            )
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


_ALSA_PREFIXES = ("hw:", "plughw:", "pulse", "pipewire", "default", "sysdefault")


def _is_alsa_device(device: str | int | None) -> bool:
    """Return True if device looks like a valid ALSA device string for arecord."""
    if device is None or isinstance(device, int):
        return False
    return any(device.startswith(p) for p in _ALSA_PREFIXES)


def _native_rate(device: str | int | None) -> int:
    """Query the sounddevice native sample rate for a device (default 44100)."""
    try:
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])
    except Exception:
        return 44100


def _start_arecord(audio_path: str, device: str | int | None) -> int | None:
    """Start arecord in background. Returns PID, or None if arecord is unusable."""
    if not shutil.which("arecord"):
        return None

    rate = _native_rate(device if _is_alsa_device(device) else None)
    cmd = ["arecord", "-f", "S16_LE", "-r", str(rate), "-c", "1"]

    if isinstance(device, int):
        warning("Toggle mode uses arecord; integer device indices not supported. Using default device.")
    elif device is not None and _is_alsa_device(device):
        cmd += ["-D", device]
    elif device is not None:
        # Friendly name (e.g. "Blue Snowball") — not valid for arecord; use pulse
        if shutil.which("arecord"):
            warning(
                f"Device '{device}' is not an ALSA name; arecord will use the "
                "default device. Set audio_device to an ALSA name (e.g. 'pulse') "
                "in your config to route to a specific device."
            )

    cmd.append(audio_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # Give arecord ~0.3 s to fail on a bad device before declaring success
    time.sleep(0.3)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace").strip()
        warning(f"arecord exited immediately: {stderr}")
        return None
    return proc.pid


def _start_sounddevice(audio_path: str, device: str | int | None) -> int:
    """Fallback recorder using a detached Python subprocess."""
    rate = _native_rate(device)
    script = (
        "import sounddevice as sd, wave, numpy as np, signal, sys\n"
        "chunks=[]\n"
        "def cb(d,f,t,s): chunks.append(d.copy())\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        f"dev={repr(device)}\n"
        f"rate={rate}\n"
        "with sd.InputStream(samplerate=rate,channels=1,dtype='float32',device=dev,callback=cb):\n"
        "    signal.pause()\n"
        "audio=np.concatenate(chunks) if chunks else np.zeros((0,1),dtype='float32')\n"
        "audio_i16=(audio.flatten()*32767).clip(-32768,32767).astype('int16')\n"
        f"wf=wave.open({repr(audio_path)},'wb')\n"
        "wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)\n"
        "wf.writeframes(audio_i16.tobytes()); wf.close()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def _kill_process(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(20):
        time.sleep(0.1)
        if not _pid_alive(pid):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def toggle_recording(
    config: dict,
    device: str | int | None = None,
    output_file: str | None = None,
):
    """Start or stop toggle-mode recording.

    Args:
        config: Full configuration dictionary.
        device: Audio input device (name string or index).
        output_file: Optional file to append transcript to.
    """
    lock_path = _lock_path()

    if os.path.exists(lock_path):
        _stop_and_transcribe(lock_path, config, output_file)
    else:
        _start_recording(lock_path, device)


def _start_recording(lock_path: str, device: str | int | None):
    audio_path = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="asr2clip_", delete=False
    ).name

    pid = None
    recorder = "sounddevice"
    if shutil.which("arecord"):
        pid = _start_arecord(audio_path, device)
        if pid is not None:
            recorder = "arecord"
        else:
            warning("arecord failed; falling back to sounddevice recorder.")
    else:
        warning("arecord not found; falling back to sounddevice recorder.")

    if pid is None:
        pid = _start_sounddevice(audio_path, device)

    lock_data = {"pid": pid, "audio": audio_path, "recorder": recorder}
    with open(lock_path, "w") as f:
        json.dump(lock_data, f)

    info(f"Recording started (pid {pid}, device: {device or 'default'})")
    _notify("asr2clip", "Recording… (run asr2clip --toggle to stop)")


def _stop_and_transcribe(
    lock_path: str,
    config: dict,
    output_file: str | None,
):
    with open(lock_path) as f:
        lock_data = json.load(f)

    pid = lock_data.get("pid", 0)
    audio_path = lock_data.get("audio", "")

    if not _pid_alive(pid):
        warning(f"Recorder PID {pid} is no longer running (stale lock). Cleaning up.")
        os.unlink(lock_path)
        if audio_path and os.path.exists(audio_path):
            _transcribe_and_output(audio_path, config, output_file)
        return

    log(f"Stopping recorder (pid {pid})…")
    _kill_process(pid)
    os.unlink(lock_path)

    # Give arecord a moment to flush its WAV header
    time.sleep(0.3)

    _transcribe_and_output(audio_path, config, output_file)


def _transcribe_and_output(audio_path: str, config: dict, output_file: str | None):
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        log("Audio file is empty or missing — nothing to transcribe.")
        return

    try:
        import wave
        with wave.open(audio_path) as wf:
            duration = wf.getnframes() / wf.getframerate()
        info(f"Recorded {duration:.1f}s of audio, transcribing…")
    except Exception:
        info("Transcribing recorded audio…")

    t0 = time.time()
    try:
        from .transcribe import transcribe_with_config
        text = transcribe_with_config(audio_path, config, raise_on_error=True)
    except Exception as e:
        warning(f"Transcription failed: {e}")
        _notify("asr2clip", f"Transcription failed: {e}")
        return
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    elapsed = time.time() - t0
    info(f"Transcription completed in {elapsed:.1f}s")

    if not text.strip():
        log("No speech detected.")
        _notify("asr2clip", "No speech detected.")
        return

    output_transcript(text, to_clipboard=True, to_stdout=True, to_file=output_file)
    _notify("asr2clip", text[:100])
