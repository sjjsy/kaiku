# asr2clip -- Speech-to-Text Clipboard Tool

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[中文](README_zh.md)

This tool is designed to recognize speech in real-time, convert it to text, and automatically copy the text to the system clipboard. The tool leverages API services for speech recognition and uses Python libraries for audio capture and clipboard management.

## TL;DR

```bash
pip install asr2clip       # Install the package
asr2clip --edit            # Create/edit config file
asr2clip --test            # Test your configuration
asr2clip                   # Start recording and transcribing
```

## Prerequisites

Before you begin, ensure you have the following ready:

- **Python 3.8 or higher**: The tool is written in Python, so you'll need Python installed on your system.
- **API Key**: You will need an API key from a speech recognition service (e.g., **OpenAI/Whisper** API or a compatible ASR API, such as **FunAudioLLM/SenseVoiceSmall** at [siliconflow](https://siliconflow.cn/) or [xinference](https://inference.readthedocs.io/en/latest/)).

### System Dependencies

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| **ffmpeg** | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| **PortAudio** | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |
| **Clipboard** | Copy to clipboard | Built-in (copykitten) | Built-in | Built-in |

## Installation

### Option 1: Install via pip or pipx (Recommended)

```bash
# Install using pip
pip install asr2clip

# Or install using pipx (recommended for isolated environments)
pipx install asr2clip

# Upgrade to latest version
pip install --upgrade asr2clip
```

### Option 2: Install from source

```bash
git clone https://github.com/Oaklight/asr2clip.git
cd asr2clip
pip install -e .
```

## Configuration

### Quick Setup

The easiest way to configure asr2clip is using the built-in editor:

```bash
asr2clip --edit  # Opens config file in your default editor
```

This will create a config file at `~/.config/asr2clip/config.yaml` if it doesn't exist.

### Configuration File

The configuration file uses YAML format. The simplest form:

```yaml
api_base_url: "https://api.openai.com/v1/"  # or other compatible API base URL
api_key: "YOUR_API_KEY"                     # api key for the platform
model_name: "whisper-1"                     # or other compatible model
# quiet: false                              # optional, disable logging
# audio_device: "pulse"                     # optional, audio input device
```

Config file locations (searched in order):
1. `./asr2clip.conf` (current directory)
2. `~/.config/asr2clip/config.yaml`
3. `~/.config/asr2clip.conf` (legacy)
4. `~/.asr2clip.conf` (legacy)

### Multiple Backends

You can define several named backends and switch between them with `-b`:

```yaml
default_backend: groq
backends:
  groq:
    type: api
    api_base_url: "https://api.groq.com/openai/v1/"
    api_key: "YOUR_GROQ_KEY"
    model_name: "whisper-large-v3-turbo"
  local:
    type: whisper_cpp
    binary: ~/path/to/whisper.cpp/build/bin/whisper-cli
    model:  ~/path/to/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin
    # language: auto
    # threads: 4
```

Select at runtime:

```bash
asr2clip -b local -i meeting.mp3   # use whisper.cpp backend
asr2clip -b groq                   # use Groq cloud backend
asr2clip --test -b local           # test whisper.cpp configuration
```

### Test Your Configuration

Before using the tool, verify your setup:

```bash
asr2clip --test
```

### Audio Device Selection

```bash
asr2clip --list_devices    # List all audio input devices
asr2clip --device pulse    # Use specific device
```

Or add to your config file:
```yaml
audio_device: "pulse"  # or device index like 12
```

## Usage

### Basic Usage

```bash
asr2clip                   # Record until Ctrl+C, transcribe, copy to clipboard
asr2clip --vad             # Continuous recording with voice detection
asr2clip -i audio.mp3      # Transcribe an audio file
```

### CLI Options

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

### Toggle Mode

Toggle mode lets you bind a single keyboard shortcut to start and stop recording:

```bash
asr2clip --toggle   # First press: start recording in background
asr2clip --toggle   # Second press: stop, transcribe, copy to clipboard
```

The recording runs as a background process; invoking the command a second time stops it and sends the audio through transcription. A desktop notification is shown on start and finish (requires `notify-send`).

Example awesome WM keybinding:

```lua
awful.key({ modkey }, "r", function()
    awful.spawn("asr2clip --toggle")
end)
```

### Continuous Recording Mode

For long recordings like meetings or lectures, use `--vad` or `--interval`:

```bash
# Continuous with voice activity detection (auto-transcribe on silence)
asr2clip --vad -o ~/meeting.txt

# Continuous with fixed interval (transcribe every 60 seconds)
asr2clip --interval 60 -o ~/meeting.txt
```

In continuous mode:
- Audio is recorded continuously
- Transcription happens automatically (on silence or at interval)
- Press Ctrl+C once to stop (transcribes remaining audio before exit)
- Transcripts are appended to the output file with timestamps

### Voice Activity Detection (VAD)

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) neural network model via sherpa-onnx for reliable speech detection. Requires the `vad` extra:

```bash
pip install asr2clip[vad]
```

```bash
asr2clip --vad                                  # Auto-transcribe on silence
asr2clip --vad --silence_threshold 0.3 --silence_duration 2.0
asr2clip --vad -o ~/meeting.txt
```

VAD options:
- `--silence_threshold`: Speech probability threshold, 0.0-1.0 (default: 0.5). Lower values are more sensitive.
- `--silence_duration`: Seconds of silence to trigger transcription (default: 1.5)

The Silero VAD model (~629 KB) is downloaded automatically on first use.

### Robust Long-File Transcription

For long audio files, `--robust` splits at silence boundaries, quality-checks each chunk, retries bad chunks, and writes output incrementally (suitable for `tail -f`):

```bash
asr2clip -i meeting.mp3 -R                    # Chunked, quality-checked transcription
asr2clip -i meeting.mp3 -R -C 60             # Use 60 s chunks instead of default 180
asr2clip -i meeting.mp3 -R -o transcript.txt  # Stream chunks to file
asr2clip -i meeting.mp3 -R -b local          # Robust mode with whisper.cpp
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Audio not captured | Run `asr2clip --list_devices` and select a working device |
| Clipboard not working | Install `xclip` (X11) or `wl-clipboard` (Wayland) |
| API errors | Check your API key and endpoint in config |
| Silent audio | Try a different audio device with `--device` |

Run `asr2clip --test` to diagnose issues.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome any improvements or new features!

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
