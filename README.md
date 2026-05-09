# asr2clip -- Speech-to-Text Clipboard Tool

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[中文](README_zh.md)

This tool records speech, transcribes it, and copies the result to your clipboard. It supports cloud ASR APIs (OpenAI-compatible) as well as fully local, offline transcription via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — no API key required for local use.

## TL;DR

**Cloud (API) path:**
```bash
pip install asr2clip
asr2clip --edit    # add your API key
asr2clip --test    # verify
asr2clip           # record and transcribe
```

**Local offline path (no API key needed):**
```bash
pip install asr2clip
# configure whisper.cpp binary and model path as e.g. 'wcpp' — see Multiple Backends below
asr2clip --test -b wcpp
asr2clip -b wcpp
```

## Prerequisites

- **Python 3.8 or higher**
- **ASR backend** — one of:
  - A cloud API key (OpenAI/Whisper, Groq, SiliconFlow, xinference, or any OpenAI-compatible endpoint)
  - A local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) installation (fully offline, no key needed)

### System Dependencies

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| **ffmpeg** | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| **PortAudio** | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |
| **Clipboard** | Copy to clipboard | Built-in (copykitten) | Built-in | Built-in |

## Installation

```bash
pip install asr2clip

# or in an isolated environment
pipx install asr2clip

# upgrade
pip install --upgrade asr2clip
```

**From source:**
```bash
git clone https://github.com/Oaklight/asr2clip.git
cd asr2clip
pip install -e .
```

## Configuration

```bash
asr2clip --edit          # create/open config in your default editor
asr2clip --print_config  # print all available options
```

Config file is created at `~/.config/asr2clip/config.yaml`. Locations searched in order:
1. `./asr2clip.conf`
2. `~/.config/asr2clip/config.yaml`
3. `~/.config/asr2clip.conf`
4. `~/.asr2clip.conf`

### Cloud API backend (single backend, simplest form)

```yaml
api_base_url: "https://api.openai.com/v1/"
api_key: "YOUR_API_KEY"
model_name: "whisper-1"
# audio_device: "pulse"   # optional, see --list_devices
```

Compatible services: OpenAI, [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), and others.

### Local offline backend (whisper.cpp, no API key)

```yaml
backends:
  wcpp:
    type: whisper_cpp
    binary: ~/path/to/whisper.cpp/build/bin/whisper-cli
    model:  ~/path/to/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin
    # language: auto   # auto-detect language
    # threads: 4
```

```bash
asr2clip --test -b wcpp   # verify
asr2clip -b wcpp          # record and transcribe offline
```

See the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) project for build instructions and model downloads.

### Multiple named backends

Define several backends and switch between them at runtime with `-b`:

```yaml
default_backend: groq
backends:
  groq:
    type: api
    api_base_url: "https://api.groq.com/openai/v1/"
    api_key: "YOUR_GROQ_KEY"
    model_name: "whisper-large-v3-turbo"
  wcpp:
    type: whisper_cpp
    binary: ~/path/to/whisper.cpp/build/bin/whisper-cli
    model:  ~/path/to/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin
```

```bash
asr2clip                       # use default backend (groq)
asr2clip -b wcpp               # use whisper.cpp
asr2clip -b wcpp -i audio.wav  # transcribe file offline
asr2clip --test -b wcpp        # test a specific backend
```

### Audio device

```bash
asr2clip --list_devices        # list available input devices
asr2clip --device pulse        # use a specific device for this run
```

Or set permanently in config:
```yaml
audio_device: "pulse"          # or a device index like 12
```

## Usage

### Basic

```bash
asr2clip                       # record until Ctrl+C, transcribe, copy to clipboard
asr2clip -b wcpp               # same, using local whisper.cpp
asr2clip -i audio.mp3          # transcribe an existing file
asr2clip -i audio.mp3 -b local # transcribe a file offline
asr2clip -o transcript.txt     # also append transcript to a file
```

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

```bash
asr2clip --vad -o ~/meeting.txt          # auto-transcribe on silence (VAD)
asr2clip --interval 60 -o ~/meeting.txt  # transcribe every 60 seconds
```

VAD requires the `vad` extra (`pip install asr2clip[vad]`) and uses the [Silero VAD](https://github.com/snakers4/silero-vad) model via sherpa-onnx. The model (~629 KB) is downloaded automatically on first use.

VAD options:
- `--silence_threshold PROB`: speech probability threshold, 0.0–1.0 (default: 0.5)
- `--silence_duration SEC`: silence duration to trigger transcription (default: 1.5 s)

### Robust long-file transcription

For long recordings, `--robust` splits at silence boundaries, quality-checks each chunk, retries bad chunks, and writes output incrementally:

```bash
asr2clip -i meeting.mp3 -R                    # chunked, quality-checked
asr2clip -i meeting.mp3 -R -C 60              # 60 s chunks instead of default 180
asr2clip -i meeting.mp3 -R -o transcript.txt  # write chunks to file as they complete
asr2clip -i meeting.mp3 -R -b wcpp            # fully offline
```

### CLI reference

```
usage: asr2clip [-h] [-v] [-c FILE] [-q] [-b NAME] [-i FILE] [-o FILE]
                [--test] [--list_devices] [--device DEV] [-e]
                [--generate_config] [--print_config] [--vad] [--interval SEC]
                [--silence_threshold PROB] [--silence_duration SEC] [-R]
                [-C SEC] [--toggle] [--serve] [--host HOST] [--port PORT]
                [--model-dir MODEL_DIR] [--num-threads NUM_THREADS]
                [--download-model]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c FILE, --config FILE
                        Path to configuration file
  -q, --quiet           Quiet mode - only output transcription and errors
  -b NAME, --backend NAME
                        Named backend to use (defined under 'backends:' in config)
  -i FILE, --input FILE
                        Transcribe audio file instead of recording
  -o FILE, --output FILE
                        Append transcripts to file
  --test                Test API configuration and exit
  --list_devices        List available audio input devices
  --device DEV          Audio input device (name or index)
  -e, --edit            Open configuration file in editor
  --generate_config     Create config file at ~/.config/asr2clip/config.yaml
  --print_config        Print template configuration to stdout
  --vad                 Continuous recording with voice activity detection
  --interval SEC        Continuous recording with fixed interval (seconds)
  --silence_threshold PROB
                        VAD speech probability threshold, 0.0-1.0 (default: 0.5)
  --silence_duration SEC
                        Silence duration to trigger transcription (default: 1.5)
  -R, --robust          Robust mode for -i file input: split at silence
                        boundaries, check quality, retry bad chunks,
                        stream output (tail-f friendly)
  -C SEC, --chunk-duration SEC
                        Maximum chunk duration in seconds for --robust mode (default: 180)
  --toggle              Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.

Local ASR server:
  --serve               Start the local ASR API server
  --host HOST           Server bind address (default: 127.0.0.1)
  --port PORT           Server bind port (default: 8000)
  --model-dir MODEL_DIR Path to ASR model directory
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
| whisper.cpp errors | Run `asr2clip --test -b local`; check binary and model paths |
| Silent audio | Try a different audio device with `--device` |

Run `asr2clip --test` (or `asr2clip --test -b <name>`) to diagnose issues.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome any improvements or new features!

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
