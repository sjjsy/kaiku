"""Audio recording and processing for kaiku."""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .utils import info, is_stop_requested, run_subprocess, warning


@dataclass
class DeviceInfo:
    """Resolved audio device with specs for different backends."""

    index: int
    name: str  # human-readable name from PortAudio
    portaudio_name: str  # full PortAudio device name
    alsa_name: str | None  # ALSA name extracted from device info (e.g., "hw:2,0")
    channels: int
    sample_rate: int
    is_default: bool = False
    mock_source: str | None = None  # path to source WAV for mock_devices entries

    def get_spec(self, recorder_name: str) -> str | int | None:
        """Return device spec that the given recorder understands."""
        if recorder_name == "sounddevice":
            return self.index
        elif recorder_name == "arecord":
            if self.alsa_name:
                return f"plughw:{self.alsa_name}"
            return None
        return None

    def __str__(self) -> str:
        """User-friendly device description."""
        desc = self.name
        if self.is_default:
            desc += " [DEFAULT]"
        if self.alsa_name:
            desc += f" (ALSA: {self.alsa_name})"
        return desc


def query_devices() -> list[DeviceInfo]:
    """Query all available input devices from sounddevice and ALSA."""
    devices: list[DeviceInfo] = []
    seen_names: set[str] = set()

    try:
        default_input = sd.query_devices(kind="input")
        default_name = default_input.get("name", "")
    except Exception:
        default_name = ""

    try:
        all_devices = sd.query_devices()
    except Exception:
        all_devices = []

    # Add sounddevice devices
    for i, device in enumerate(all_devices):
        if device.get("max_input_channels", 0) <= 0:
            continue

        # Extract ALSA name from device name (e.g., "Snowball: USB Audio (hw:2,0)" → "2,0")
        alsa_name = None
        m = re.search(r"\(hw:(\d+,\d+)\)", device.get("name", ""))
        if m:
            alsa_name = m.group(1)

        name = device.get("name", f"Device {i}")
        is_default = name == default_name
        dev_info = DeviceInfo(
            index=i,
            name=name,
            portaudio_name=name,
            alsa_name=alsa_name,
            channels=device.get("max_input_channels", 0),
            sample_rate=int(device.get("default_samplerate", 44100)),
            is_default=is_default,
        )
        devices.append(dev_info)
        seen_names.add(name.lower())

    # Also try to get ALSA devices directly (arecord -L) for hardware not visible to sounddevice
    try:
        result = run_subprocess(["arecord", "-L"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                i += 1

                if not line or line.startswith("!"):
                    continue

                # Look for hw: or plughw: entries (including named format like hw:CARD=Snowball,DEV=0)
                if line.startswith("hw:") or line.startswith("plughw:"):
                    # Skip numeric hw:X,Y format (already covered by sounddevice)
                    if re.match(r"hw:\d+,\d+$", line) or re.match(r"plughw:\d+,\d+$", line):
                        continue

                    alsa_name = line.replace("plughw:", "").replace("hw:", "")

                    # Get the description from the next line(s)
                    description = ""
                    if i < len(lines):
                        desc_line = lines[i].strip()
                        if desc_line and not desc_line.startswith(("hw:", "plughw:")):
                            description = desc_line
                            i += 1

                    # Avoid duplicates
                    if alsa_name.lower() not in seen_names:
                        # Use description if available, else construct from device name
                        if description:
                            name = f"{description} ({line})"
                        else:
                            name = line

                        dev_info = DeviceInfo(
                            index=-1,  # No sounddevice index
                            name=name,
                            portaudio_name=None,
                            alsa_name=alsa_name,
                            channels=2,  # Assume stereo; arecord -L doesn't provide this
                            sample_rate=44100,
                            is_default=False,
                        )
                        devices.append(dev_info)
                        seen_names.add(alsa_name.lower())
    except Exception:
        pass

    return devices


def find_device(name: str) -> DeviceInfo | None:
    """Find a device by name with relevance scoring.

    Supports:
    - Special names: 'auto', 'default' → system default
    - Integer index: '0', '12'
    - Substring match (case-insensitive): 'Snowball' matches 'Snowball: USB Audio'
    - ALSA name: 'hw:2,0' or 'plughw:2,0'

    Returns first matching device (prioritizing exact/higher-relevance matches).
    """
    if not name or name.lower() in ("auto", "default"):
        # Return system default
        available = query_devices()
        for dev in available:
            if dev.is_default:
                return dev
        return available[0] if available else None

    # Try numeric index
    try:
        idx = int(name)
        available = query_devices()
        if 0 <= idx < len(available):
            return available[idx]
    except ValueError:
        pass

    # Try ALSA name match (hw:X,Y or plughw:X,Y)
    available = query_devices()
    for dev in available:
        if dev.alsa_name and dev.alsa_name == name.replace("hw:", "").replace("plughw:", ""):
            return dev

    # Try case-insensitive substring match (exact > partial)
    name_lower = name.lower()
    exact_matches = [
        dev for dev in available if dev.name.lower() == name_lower
    ]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [
        dev for dev in available if name_lower in dev.name.lower()
    ]
    if partial_matches:
        info(f"Device '{name}' not exact; using first match: {partial_matches[0].name}")
        return partial_matches[0]

    return None


def resolve_device_preference_order(spec: str | list | None) -> list[DeviceInfo]:
    """Parse device preference order and return available devices.

    Args:
        spec: Comma-separated string ("Snowball,Webcam,auto") or list ["Snowball", "Webcam"].
              Empty/None defaults to "auto".

    Returns:
        List of available devices in preference order.
    """
    if spec is None or spec == "":
        spec = "auto"

    # Parse into list of names
    if isinstance(spec, str):
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = [str(s).strip() for s in spec if str(s).strip()]

    if not names:
        names = ["auto"]

    # Find each device in preference order
    resolved = []
    tried = []
    for name in names:
        dev = find_device(name)
        if dev:
            resolved.append(dev)
            tried.append(f"{name} (→ {dev.name})")
        else:
            tried.append(f"{name} (not available)")

    if tried:
        info(f"Device preference order: {', '.join(tried)}")

    return resolved


def list_audio_devices():
    """List all available audio input devices with human-readable info."""
    available = query_devices()
    if not available:
        print("No audio input devices found.")
        print("Check microphone permissions or system audio configuration.")
        return

    print("Available audio input devices:")
    print("-" * 80)
    for dev in available:
        print(f"  {dev.index}: {dev}")
        print(
            f"      Channels: {dev.channels}, Sample Rate: {dev.sample_rate} Hz"
        )
    print("-" * 80)
    print("\nUse --device to select by:")
    print("  Index:        kaiku --device 0")
    print("  Name:         kaiku --device Snowball")
    print("  Preference:   kaiku --device Snowball,Webcam,auto")
    print("  Special:      kaiku --device auto  (system default)")
    print("  ALSA:         kaiku --device hw:2,0")


def write_wav(audio_data: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Write audio data to WAV format bytes using stdlib wave module.

    Args:
        audio_data: Audio data as numpy array (float32, range -1.0 to 1.0).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        WAV file content as bytes.
    """
    # Flatten if multi-dimensional (e.g., from stereo recording)
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()

    # Convert float32 to int16
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

    # Convert to bytes using numpy's tobytes (more efficient than struct.pack)
    audio_bytes = audio_int16.tobytes()

    # Write WAV using stdlib wave module
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return buffer.getvalue()


def _device_native_rate(device: str | int | None) -> int | None:
    """Return the device's default sample rate, or None on failure."""
    try:
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])
    except Exception:
        return None


def _sounddevice_device(device: str | int | None) -> str | int | None:
    """Translate ALSA hw:/plughw: strings to a PortAudio-compatible form.

    arecord accepts 'plughw:2,0' but sounddevice/PortAudio does not.
    Strip the prefix so PortAudio can match by card name substring.
    """
    if not isinstance(device, str):
        return device
    if device.startswith("hw:") or device.startswith("plughw:"):
        return device.split(":", 1)[1].split(",")[0]
    return device


def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a (N, channels) float32 array from from_rate to to_rate via pydub."""
    from pydub import AudioSegment  # noqa: PLC0415 — intentional lazy import

    channels = audio.shape[1] if audio.ndim > 1 else 1
    flat = audio.flatten()
    pcm = (flat * 32767).clip(-32768, 32767).astype(np.int16)
    seg = AudioSegment(
        pcm.tobytes(), frame_rate=from_rate, sample_width=2, channels=channels
    )
    seg = seg.set_frame_rate(to_rate)
    resampled = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767
    if channels > 1:
        resampled = resampled.reshape(-1, channels)
    else:
        resampled = resampled.flatten()
    return resampled


def record_audio(
    sample_rate: int = 16000,
    channels: int = 1,
    device: str | int | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Record audio from the microphone until stop is requested.

    If the device does not support sample_rate, falls back to the device's
    native rate and resamples the result to sample_rate before returning.

    Args:
        sample_rate: Desired sample rate in Hz (default 16000).
        channels: Number of audio channels.
        device: Audio device name or index, or None for default.
        callback: Optional callback called with each raw audio chunk.

    Returns:
        Recorded audio as numpy array at sample_rate Hz.
    """
    device = _sounddevice_device(device)
    audio_chunks: list = []
    actual_rate = sample_rate

    def audio_callback(indata, frames, time, status):
        if status:
            warning(f"Audio status: {status}")
        audio_chunks.append(indata.copy())
        if callback:
            callback(indata)

    rates_to_try = [sample_rate]
    native = _device_native_rate(device)
    if native and native != sample_rate:
        rates_to_try.append(native)

    last_exc = None
    for rate in rates_to_try:
        audio_chunks.clear()
        try:
            with sd.InputStream(
                samplerate=rate,
                channels=channels,
                dtype="float32",
                device=device,
                callback=audio_callback,
            ):
                actual_rate = rate
                if rate != sample_rate:
                    warning(
                        f"Device does not support {sample_rate} Hz; "
                        f"recording at {rate} Hz and resampling."
                    )
                while not is_stop_requested():
                    sd.sleep(100)
            last_exc = None
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            last_exc = e
            if rate == sample_rate:
                info(f"Recording error at {rate} Hz: {e}. Trying device native rate...")
                continue
            info(f"Recording error: {e}")
            raise

    if last_exc:
        raise last_exc

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(audio_chunks, axis=0)

    if actual_rate != sample_rate:
        audio = _resample_audio(audio, actual_rate, sample_rate)

    # sounddevice gives (n_frames, channels); squeeze mono to 1D
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    return audio


def save_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Save audio data to a temporary WAV file.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    wav_bytes = write_wav(audio_data, sample_rate)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    info(f"Saving the audio to a temp file: {temp_file.name}")
    temp_file.write(wav_bytes)
    temp_file.close()

    return temp_file.name


def convert_audio_to_wav(input_path: str, output_path: str | None = None) -> str:
    """Convert an audio or video file to a 16 kHz mono WAV file.

    Supports all formats accepted by ffmpeg (mp3, m4a, ogg, flac, aac, opus,
    wma, mp4, mov, mkv, webm, avi, flv, mvi, ...). Video streams are discarded.
    Basic spectral cleaning (highpass 200 Hz, lowpass 3 kHz, loudnorm) is applied
    during conversion to improve transcription quality.

    Falls back to pydub (no cleaning, no video support) when ffmpeg is not on PATH.

    Args:
        input_path: Path to the source audio or video file.
        output_path: Destination WAV path; a temp file is created when omitted.

    Returns:
        Path to the converted WAV file.
    """
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",                              # drop video stream
            "-ar", "16000",                     # 16 kHz
            "-ac", "1",                         # mono
            "-af", "highpass=f=200,lowpass=f=3000,loudnorm",
            output_path,
        ]
        run_subprocess(cmd, check=True, capture_output=True)
    else:
        warning(
            "ffmpeg not found on PATH — falling back to pydub (no video support, "
            "no spectral cleaning). Install ffmpeg for full format support."
        )
        from pydub import AudioSegment  # noqa: PLC0415
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")

    return output_path


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file into a float32 numpy array.

    Args:
        path: Path to a 16-bit PCM WAV file.

    Returns:
        Tuple of (audio_float32, sample_rate). Audio is mono float32 in [-1, 1].
    """
    with wave.open(path) as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sample_rate


def get_audio_duration(audio_data: np.ndarray, sample_rate: int = 16000) -> float:
    """Calculate the duration of audio data in seconds.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    if len(audio_data) == 0:
        return 0.0
    return len(audio_data) / sample_rate


def calculate_rms(audio_data: np.ndarray) -> float:
    """Calculate the RMS (root mean square) of audio data.

    Args:
        audio_data: Audio data as numpy array.

    Returns:
        RMS value.
    """
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_data**2)))


def audiosegment_to_float32(audio: "AudioSegment") -> np.ndarray:  # type: ignore[name-defined]
    """Convert a mono AudioSegment to a float32 numpy array in [-1, 1]."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_audiosegment(audio: np.ndarray, sample_rate: int = 16000) -> "AudioSegment":  # type: ignore[name-defined]
    """Convert a float32 numpy array in [-1, 1] to a mono AudioSegment."""
    from pydub import AudioSegment
    samples_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return AudioSegment(samples_int16.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1)
