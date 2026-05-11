# asr2clip -- Speech-to-Text Clipboard Tool

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[中文](README_zh.md)

This ASR tool records speech, transcribes it, and copies the result to your clipboard, or a local file; and supports VAD:
- **ASR** (Automatic Speech Recognition) converts spoken audio into text.
- **VAD** (Voice Activity Detection) classifies audio frames as speech or silence, enabling hands-free continuous transcription.

The tool can provide ASR through multiple backends:
cloud APIs (OpenAI-compatible, such as [Groq](https://api.groq.com/openai/v1/)) for maximum model choice and accuracy with no local setup, and/or
offline fully local backends ([whisper.cpp](https://github.com/ggerganov/whisper.cpp), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)).

VAD requires [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and it uses the [Silero VAD](https://github.com/snakers4/silero-vad) model.

Audio preprocessing, chunked transcription, language specification, and other options to enhance transcription quality and the user experience for various use cases are provided but not all features are available for all backends and/or the VAD mode.

## TL;DR

**Cloud (API) path:**
```bash
pip install asr2clip
asr2clip --generate_config   # create config with all backend examples
asr2clip --edit              # fill in your API key
asr2clip --test              # verify
asr2clip                     # record and transcribe
```

**Local offline path — whisper.cpp:**
```bash
pip install asr2clip
# build whisper.cpp and download a model, then configure it in config
asr2clip --generate_config   # shows a wcpp backend example
asr2clip --test -b wcpp
asr2clip -b wcpp
```

**Local offline path — sherpa-onnx (model auto-downloads):**
```bash
pip install asr2clip[vad]
asr2clip --download-model    # download SenseVoice model on first use
asr2clip --serve &           # start local ASR API server
# configure a backend pointing to http://127.0.0.1:8000/v1/ — see Local ASR server below
asr2clip --test -b sonnx
asr2clip -b sonnx
```

## Prerequisites

- **Python 3.8 or higher**
- **ASR backend** — one of:
  - A cloud API key (OpenAI Whisper, Groq, SiliconFlow, xinference, or any OpenAI-compatible endpoint)
  - A local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) binary and model file (fully offline, no key needed)
  - A local [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ASR server (`pip install asr2clip[vad]`, model auto-downloads)

### System Dependencies

Required:

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| **ffmpeg** | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| **PortAudio** | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |
| **Clipboard** | Copy to clipboard | Built-in (copykitten) | Built-in | Built-in |

Optional Python packages (install only what you need):

| Extra | Install command | Purpose | Section |
|-------|----------------|---------|---------|
| `vad` | `pip install asr2clip[vad]` | VAD continuous recording + local sherpa-onnx ASR server | [Continuous recording](#continuous-recording), [Local ASR server](#local-asr-server-sherpa-onnx) |
| `noisereduce` | `pip install asr2clip[noisereduce]` | Spectral noise reduction | [Audio pre-processing](#audio-pre-processing-noise-reduction) |
| `pyrnnoise` | `pip install asr2clip[pyrnnoise]` | Neural GRU noise reduction | [Audio pre-processing](#audio-pre-processing-noise-reduction) |
| `deepfilter` | `pip install asr2clip[deepfilter]` | DeepFilterNet3 best-quality denoising | [Audio pre-processing](#audio-pre-processing-noise-reduction) |
| `enhance` | `pip install asr2clip[enhance]` | All three noise reduction options above | [Audio pre-processing](#audio-pre-processing-noise-reduction) |

## Installation

```bash
pip install asr2clip

# or in an isolated environment
pipx install asr2clip

# upgrade
pip install --upgrade asr2clip
```

Everything — all noise reduction options + VAD and the local sherpa-onnx ASR server:
```bash
pip install asr2clip[enhance,vad]
```

Only the deepfilter package for high quality single-shot/toggle/file recordings:
```bash
pip install asr2clip[deepfilter]
```

Note: The audio pre-processing options are not yet applied in VAD/interval continuous mode, but of course they are still useful if you do not use VAD exclusively.

**From source:**
```bash
git clone https://github.com/Oaklight/asr2clip.git
cd asr2clip
pip install -e .
```

## Configuration

```bash
asr2clip --generate_config   # write a fully annotated config with all backend examples
asr2clip --edit              # create/open config in your default editor
asr2clip --print_config      # print the annotated template to stdout
```

Config file is created at `~/.config/asr2clip/config.yaml`. Locations searched in order:
1. `./asr2clip.conf`
2. `~/.config/asr2clip/config.yaml`
3. `~/.config/asr2clip.conf`
4. `~/.asr2clip.conf`

### Backends

All backends are defined under a `backends:` section. Name them whatever you like and switch at runtime with `-b NAME`. `default_backend_live` and `default_backend_file` name which backend to use for each mode when `-b` is not given — the same idea as `preprocessor_live` / `preprocessor_file`. `-b` overrides both.

`asr2clip --generate_config` writes a fully annotated config with every supported backend type listed as commented-out examples — uncomment and fill in the one(s) you want.

A minimal working config (cloud API):

```yaml
backends:
  openai:
    type: api
    api_base_url: "https://api.openai.com/v1/"
    api_key: "YOUR_API_KEY"
    model_name: "whisper-1"
default_backend_live: openai
default_backend_file: openai
```

A multi-backend example with per-mode defaults:

```yaml
backends:
  groq:
    type: api
    api_base_url: "https://api.groq.com/openai/v1/"
    api_key: "YOUR_GROQ_KEY"
    model_name: "whisper-large-v3-turbo"
  sonnx:
    type: api
    api_base_url: "http://127.0.0.1:8000/v1/"
    model_name: "SenseVoiceSmall"
  wcpp:
    type: whisper_cpp
    binary: ~/path/to/whisper.cpp/build/bin/whisper-cli
    model:  ~/path/to/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin
default_backend_live: groq    # live/toggle/VAD recording
default_backend_file: wcpp    # -i file transcription
```

```bash
asr2clip                       # live recording → groq (default_backend_live)
asr2clip -i audio.wav          # file transcription → wcpp (default_backend_file)
asr2clip -b sonnx              # override for this run (both modes)
asr2clip -b wcpp -i audio.wav  # transcribe a file with a specific backend
asr2clip --test                # tests both live and file backends if they differ
asr2clip --test -b groq        # test a specific backend
```

**Supported `type` values:**

| `type` | Description | Requires |
|--------|-------------|---------|
| `api` | Any OpenAI-compatible HTTP endpoint | API key or local server |
| `whisper_cpp` | whisper.cpp binary via subprocess | whisper.cpp build + `.bin` model file |

Compatible cloud services for `type: api`: OpenAI, [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), and others.

### Local ASR: whisper.cpp vs. sherpa-onnx

Both provide fully offline, no-API-key ASR running on your machine. Here is how to choose between them:

| | whisper.cpp | sherpa-onnx (via `--serve`) |
|---|---|---|
| **What it is** | C++ reimplementation of OpenAI Whisper | ONNX-runtime inference toolkit with Python bindings |
| **ASR models** | Whisper family (GGML quantised) | Whisper, SenseVoice, paraformer, zipformer, and more |
| **Integration** | Subprocess call to external C++ binary | Python-native; exposes a local HTTP API |
| **Setup** | Build C++ from source; download `.bin` model manually | `pip install asr2clip[vad]` + `asr2clip --download-model` |
| **Model auto-download** | No | Yes |
| **Streaming ASR** | Separate `whisper-stream` binary | Built-in (model-dependent) |
| **Dev activity** | Mature, stable | Very active (k2-fsa / Next-gen Kaldi team) |
| **Already a dependency?** | No | Yes — `pip install asr2clip[vad]` already pulls in sherpa-onnx |

**When to choose whisper.cpp:** You already have it built, or you prefer GGML-quantised Whisper models and want the C++ inference stack without extra Python ML packages.

**When to choose sherpa-onnx:** You want zero C++ build steps, you already installed `asr2clip[vad]` (sherpa-onnx is already there), you want to try models beyond the Whisper family, or you want an actively developed upstream.

See [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for build instructions and model downloads, and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for its model zoo and documentation.

### Local ASR server (sherpa-onnx)

`asr2clip` can run a local OpenAI-compatible ASR API server backed by sherpa-onnx. Once running, configure it as a regular `type: api` backend:

```bash
pip install asr2clip[vad]
asr2clip --download-model          # download SenseVoice model (~1 GB, once)
asr2clip --serve                   # start server at 127.0.0.1:8000
```

Corresponding config backend:
```yaml
backends:
  sonnx:
    type: api
    api_base_url: "http://127.0.0.1:8000/v1/"
    model_name: "SenseVoiceSmall"
```

Server options: `--host HOST`, `--port PORT`, `--model-dir DIR`, `--num-threads N`.

### Audio device

```bash
asr2clip --list_devices            # list available input devices with names and indices
asr2clip -d "plughw:Snowball"      # use a specific ALSA device for this run
```

**Recommended — use system audio routing:**

On Linux, set `audio_device` to `"pulse"` or `"pipewire"` and select the active microphone in `pavucontrol` or your desktop sound settings. The system mixer handles format conversion and device matching.

```yaml
audio_device: "pulse"              # PulseAudio routes to whichever mic is set as default input
```

**Targeting a specific device directly:**

For dedicated hardware (USB mic, audio interface) you can bypass the system mixer using an ALSA `plughw:` device name. `plughw:` handles sample-rate conversion; `hw:` is raw (may fail if rates mismatch).

```yaml
audio_device: "plughw:Snowball"    # ALSA plughw — format conversion included
audio_device: 3                    # device index from --list_devices
```

| Value | System | Notes |
|-------|--------|-------|
| `"pulse"` | PulseAudio (Linux) | Recommended; configure mic via `pavucontrol` |
| `"pipewire"` | PipeWire (Linux) | Recommended on modern Linux |
| `"plughw:Snowball"` | ALSA (Linux) | Direct USB mic access with format conversion |
| `"hw:2,0"` | ALSA (Linux) | Raw direct access, card 2 device 0 |
| `3` | Any | Device index from `--list_devices` |
| `"BlackHole 2ch"` | macOS | Virtual routing device |

### Audio pre-processing (noise reduction)

asr2clip can denoise audio before sending it to the transcription backend. This
is especially useful in noisy environments (café, open-plan office, outdoor) or
when recording with variable speaker distances. Whisper and similar models
are sensitive to background noise and tend to hallucinate when the signal is poor.

**Available pre-processors:**

| Name | Technology | Extra dependencies | Con | Best for |
|------|-----------|-------------------|-----|----------|
| `none` | — | none (default) | — | clean recordings |
| `noisereduce` | Spectral subtraction | scipy | Works best on stationary noise | steady hum, fan noise |
| `pyrnnoise` | Mozilla RNNoise GRU | scipy | Requires 16 kHz→48 kHz resampling; may sound slightly robotic | babble, non-stationary noise |
| `deepfilter` | DeepFilterNet3 neural | torch + Rust wheel | Medium CPU; ~70 MB model download on first use | noisy files, variable speaker distance, best quality |

**A note on `pyrnnoise` and sample rates:** RNNoise internally operates at 48 kHz. Because
asr2clip records and converts all audio to 16 kHz before transcription, pyrnnoise always
performs a 16 kHz→48 kHz→16 kHz resampling round-trip. The inference itself is very fast,
but the resampling adds overhead. For live recordings with tight latency requirements,
`noisereduce` avoids this round-trip and is often the more predictable choice.

**`noisereduce` vs `pyrnnoise`:** They differ in kind, not just speed. `noisereduce` (spectral
subtraction) is better at removing *stationary* noise — constant hum from ventilation, a fan,
or electrical interference. `pyrnnoise` (neural GRU) handles *non-stationary* noise better —
babble, footsteps, intermittent sounds. Both require scipy; neither needs a GPU.

**Installation:**
```bash
pip install asr2clip[noisereduce]   # spectral subtraction (scipy-based)
pip install asr2clip[pyrnnoise]     # RNNoise GRU (scipy for resampling)
pip install asr2clip[deepfilter]    # DeepFilterNet3 (torch-based, no scipy)
pip install asr2clip[enhance]       # all three
```

**Usage:**
```bash
asr2clip -P deepfilter              # denoise live recording with DeepFilterNet
asr2clip -P noisereduce             # spectral denoising, no resampling overhead
asr2clip -P pyrnnoise               # RNNoise GRU denoising
asr2clip -P deepfilter -i talk.mp4  # denoise video file before transcription
asr2clip --test                     # also checks that configured preprocessors are available
```

**Config** (set different preprocessors for live vs. file transcription):
```yaml
preprocessor_live: noisereduce      # spectral, no resampling overhead for live 16 kHz audio
preprocessor_file: deepfilter       # best quality for longer file transcription
```

`asr2clip --generate_config` probes your system and writes these keys automatically
with the best available option for each context. Override for a single run with `-P`.

**Loudness normalisation:** After any noise reduction step the speech signal may be quieter
than before (the noise floor was raising the apparent level). All three preprocessors
apply a loudnorm pass after cleaning: RMS → −20 dBFS, peak ceiling −0.1 dBFS. This
ensures Whisper's attention mechanism receives a consistently strong, unclipped signal.

## Usage

### Basic

```bash
asr2clip                           # record until Ctrl+C, transcribe, copy to clipboard
asr2clip -l fi -b wcpp             # same but for Finnish, using local Whisper.cpp backend
asr2clip -i audio.mp3              # transcribe an existing audio file
asr2clip -i meeting.mp4            # transcribe from a video file (audio extracted automatically)
asr2clip -i audio.mp3 -b wcpp      # transcribe a file offline with Whisper.cpp
asr2clip -o transcript.txt         # also append transcript to a file
asr2clip -P deepfilter             # record with DeepFilterNet noise reduction
asr2clip -P deepfilter -i talk.mkv # denoise + transcribe a video file
```

**Supported input formats:**
- Audio: `wav`, `mp3`, `m4a`, `ogg`, `flac`, `aac`, `opus`, `wma`
- Video: `mp4`, `mov`, `mkv`, `webm`, `avi`, `flv`, `mvi`

Requires `ffmpeg` on PATH for non-WAV input. Video streams are discarded automatically;
basic spectral cleaning (highpass 200 Hz + lowpass 3 kHz + loudnorm) is applied during
conversion. Install ffmpeg: `apt install ffmpeg` / `brew install ffmpeg`.

Note:
- **Language support:** Many backends support multiple languages but some tend to favor a specific language, most frequently English.
  Sometimes they understand foreign languages but translate the output transcript into English.
  With the `-l LL` flag you can "force" the backend to try to interpret the audio and produce the transcript in a specific language (Use ISO-639-1 two letter codes such as `fi`, `fr` and `de`).
- **Clipboard size limit:** When a transcript exceeds ~4000 characters, the full text is too large to paste conveniently.
  If you also specified `-o FILE`, the absolute path to that file is copied to the clipboard instead — paste it wherever you need to open or attach the transcript.
  Without `-o`, the full text is copied regardless and clipboard behaviour depends on your system.

### Toggle mode

Toggle mode lets you bind a single keyboard shortcut to start and stop recording. The recording runs as a background process; the second invocation stops it, transcribes, and copies to clipboard. A desktop notification is shown on start and finish (requires `notify-send` on Linux).

```bash
asr2clip --toggle              # first press: start recording in background
asr2clip --toggle              # second press: stop, transcribe, copy to clipboard
asr2clip -b wcpp --toggle      # same, using local whisper.cpp
```

Toggle mode requires a POSIX system (Linux, macOS). Example awesome WM keybinding:

```lua
awful.key({ modkey }, "r", function()
    awful.spawn("asr2clip --toggle")
end)
```

### Continuous recording

Two modes are available:

| Mode | Flag | Trigger | Use case |
|------|------|---------|----------|
| Voice Activity Detection | `--vad` | Silence after speech | Meetings, dictation, any unscripted speech |
| Fixed interval | `--interval SEC` | Every N seconds | Lectures, podcasts with predictable pauses |

```bash
asr2clip --vad -o ~/meeting.txt          # auto-transcribe when silence is detected
asr2clip --interval 60 -o ~/meeting.txt  # transcribe every 60 seconds
```

**VAD requires [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):**

```bash
pip install asr2clip[vad]
```

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) model run locally via sherpa-onnx — a fast, lightweight speech/silence classifier. The model (~629 KB) is downloaded automatically on first use. No internet connection is required after the first run.

VAD options:
- `--silence_threshold PROB`: speech probability threshold, 0.0–1.0 (default: 0.5); lower = more sensitive
- `--silence_duration SEC`: how long silence must last to trigger a transcription (default: 1.5 s)

> **Note:** Audio pre-processing (`-P`) is not yet applied in VAD/interval continuous mode — only in single recording, toggle, and file transcription modes.

### Robust long-file transcription

For long recordings, `--robust` splits at silence boundaries, quality-checks each chunk, retries bad chunks, and writes output incrementally:

```bash
asr2clip -i meeting.mp3 -R                    # chunked, quality-checked
asr2clip -i meeting.mp3 -R -C 60              # 60 s chunks instead of default 180
asr2clip -i meeting.mp3 -R -o transcript.txt  # write chunks to file as they complete
asr2clip -i m.mp3 -Rb wcpp -l fi -o o.txt     # fully offline, Finnish language
```

Long transcripts often exceed the clipboard size limit; using `-o FILE` is recommended. When the limit is exceeded, the file path is copied to clipboard automatically (see note above).

### CLI reference

```
usage: asr2clip [-h] [-v] [-c FILE] [-q] [-b NAME] [-i FILE] [-P NAME]
                [-o FILE] [--test] [--list_devices] [--device DEV]
                [-l LANG] [-e] [--generate_config] [--print_config]
                [--vad] [--interval SEC] [--silence_threshold PROB]
                [--silence_duration SEC] [-R] [-C SEC] [--toggle]
                [--serve] [--host HOST] [--port PORT]
                [--model-dir MODEL_DIR] [--num-threads NUM_THREADS]
                [--download-model]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c FILE, --config FILE
                        Path to configuration file
  -q, --quiet           Quiet mode - only output transcription and errors
  -b NAME, --backend NAME
                        Named backend to use (defined under 'backends:' in config).
                        Overrides default_backend_live / default_backend_file in config.
  -i FILE, --input FILE
                        Transcribe an existing audio or video file instead of
                        recording. Supported: wav, mp3, m4a, ogg, flac, aac,
                        opus, wma, mp4, mov, mkv, webm, avi, flv, mvi
  -P NAME, --preprocessor NAME
                        Audio pre-processor to apply before transcription.
                        Choices: none, noisereduce, pyrnnoise, deepfilter.
                        Overrides preprocessor_live / preprocessor_file in config.
  -o FILE, --output FILE
                        Append transcripts to file
  --test                Test backend connectivity and configured preprocessors,
                        then exit
  --list_devices        List available audio input devices
  -d DEV, --device DEV  Audio input device (name, ALSA name, or index).
                        Overrides config.
  -l LANG, --language LANG
                        Language hint for transcription (ISO-639-1, e.g. fi, en).
                        Overrides config. Omit to auto-detect.
  -e, --edit            Open configuration file in editor
  --generate_config     Create config file at ~/.config/asr2clip/config.yaml
                        (probes installed enhancement libraries automatically)
  --print_config        Print template configuration to stdout
  --vad                 Continuous recording with voice activity detection (VAD).
                        Transcribes automatically when silence is detected after
                        speech. Requires sherpa-onnx: pip install asr2clip[vad].
                        The Silero VAD model (~629 KB) is downloaded on first use.
  --interval SEC        Continuous recording with fixed interval (seconds)
  --silence_threshold PROB
                        VAD speech probability threshold, 0.0-1.0 (default: 0.5)
  --silence_duration SEC
                        Silence duration to trigger transcription (default: 1.5 s)
  -R, --robust          Robust mode for -i file input: split at silence boundaries,
                        quality-check chunks, retry failures, stream output
  -C SEC, --chunk-duration SEC
                        Maximum chunk duration for --robust mode in seconds
                        (default: 180)
  --toggle              Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.

Local ASR server:
  --serve               Start the local ASR API server
  --host HOST           Server bind address (default: 127.0.0.1)
  --port PORT           Server bind port (default: 8000)
  --model-dir MODEL_DIR
                        Path to ASR model directory
  --num-threads NUM_THREADS
                        Inference threads (default: 4)
  --download-model      Download the SenseVoice model and exit
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Audio not captured | Run `asr2clip --list_devices` and select a working device |
| Clipboard not working | Install `xclip` (X11) or `wl-clipboard` (Wayland) |
| API errors | Check your API key and endpoint in config |
| whisper.cpp errors | Run `asr2clip --test -b wcpp`; check binary and model paths |
| Silent audio | Try a different audio device with `--device` |
| Video/audio format rejected | Ensure `ffmpeg` is installed (`apt install ffmpeg` / `brew install ffmpeg`) |
| Preprocessor not found | Run `asr2clip --test` to see which are available and their install commands |
| Pre-processing too slow | Switch `preprocessor_live` to `noisereduce` or `none` in config |

Run `asr2clip --test` (or `asr2clip --test -b <name>`) to diagnose issues. The test
command also verifies that your configured preprocessors are installed and importable.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome any improvements or new features!

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
