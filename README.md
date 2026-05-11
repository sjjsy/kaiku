# asr2clip -- Speech-to-Text Clipboard Tool — With Everything You'd Want :)

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[中文](README_zh.md)

Record speech, transcribe it, and copy the result to clipboard or a file. Supports cloud and fully-local [ASR backends](#asr-backends), [VAD](#vad-continuous-recording) (for background daemons) and [toggle](#toggle-mode) (for keyboard shortcuts), [noise reduction](#audio-preprocessing-noise-reduction), [speaker diarization](#diarization), and [LLM post-processing](#llm-post-processing).

## TL;DR

**Cloud (API) path:**
```bash
pip install asr2clip
asr2clip --generate_config   # create config with all backend examples
asr2clip --edit              # fill in your API key
asr2clip --test              # verify
asr2clip                     # record and transcribe
```

**Local offline path — sherpa-onnx with VAD support (model auto-downloads):**
```bash
pip install asr2clip[vad]
asr2clip --download-model    # download SenseVoice model on first use
asr2clip --serve &           # start local ASR API server
# configure a backend pointing to http://127.0.0.1:8000/v1/ — see Local ASR server below
asr2clip --test -b sonnx
asr2clip -b sonnx
```

**Local offline path — whisper.cpp (no VAD):**
```bash
pip install asr2clip
# build whisper.cpp and download a model, then configure it in config
asr2clip --generate_config   # shows a wcpp backend example
asr2clip --test -b wcpp
asr2clip -b wcpp
```

## CLI reference

```
usage: asr2clip [-h] [-v] [-q] [-c FILE] [-e] [--generate_config]
                [--print_config] [--test] [--list_devices] [-d DEV] [-p NAME]
                [--toggle] [-b NAME] [-i FILE] [-o FILE] [-l LANG] [-r]
                [-C SEC] [--serve] [--host HOST] [--port PORT]
                [--model-dir MODEL_DIR] [--num-threads NUM_THREADS]
                [--download-model] [--vad] [--interval SEC]
                [--silence_threshold PROB] [--silence_duration SEC] [-D]
                [-s N] [-P NAME] [-M MODEL] [-T NAME]

Record audio and transcribe to clipboard using ASR API

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -q, --quiet           Quiet mode — only output transcription and errors

Setup:
  -c FILE, --config FILE
                        Path to configuration file
  -e, --edit            Open configuration file in editor (creates default
                        config if missing)
  --generate_config     Write config template to
                        ~/.config/asr2clip/config.yaml
  --print_config        Print config template to stdout
  --test                Test backend connectivity and configured
                        preprocessors, then exit

Audio:
  --list_devices        List available audio input devices
  -d DEV, --device DEV  Audio input device (name, ALSA name, or index).
                        Overrides config.
  -p NAME, --preprocessor NAME
                        Audio preprocessor: none, noisereduce, pyrnnoise,
                        deepfilter. Overrides preprocessor_live /
                        preprocessor_file in config.

Transcription:
  -b NAME, --backend NAME
                        ASR backend to use (key under 'asr_backends:' in
                        config). Overrides asr_backend_live /
                        asr_backend_file.
  -i FILE, --input FILE
                        Transcribe an existing audio or video file instead of
                        recording. Supported: wav, mp3, m4a, ogg, flac, aac,
                        opus, wma, mp4, mov, mkv, webm, avi, flv, mvi
  -o FILE, --output FILE
                        Append transcripts to file
  -l LANG, --language LANG
                        Language hint for transcription (ISO-639-1, e.g. 'fi',
                        'en'). Overrides config. Omit to auto-detect.
  -r, --robust          Robust mode for -i file input: split at silence
                        boundaries, quality-check chunks, retry failures,
                        stream output (tail-f friendly).
  -C SEC, --chunk-duration SEC
                        Max chunk duration in seconds for -r/--robust mode
                        (default: 180)
  --toggle              Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.

Local ASR server:
  --serve               Start the local sherpa-onnx ASR API server
  --host HOST           Server bind address (default: 127.0.0.1)
  --port PORT           Server bind port (default: 8000)
  --model-dir MODEL_DIR
                        Path to ASR model directory
  --num-threads NUM_THREADS
                        Inference threads (default: 4)
  --download-model      Download the SenseVoice model and exit

VAD (continuous recording):
  --vad                 Continuous recording with voice activity detection.
                        Transcribes automatically when silence is detected
                        after speech. Requires sherpa-onnx: pip install
                        asr2clip[vad].
  --interval SEC        Continuous recording with fixed interval (seconds)
  --silence_threshold PROB
                        VAD speech probability threshold, 0.0-1.0 (default:
                        0.5)
  --silence_duration SEC
                        Silence duration to trigger transcription (default:
                        1.5 s)

Diarization:
  -D, --diarize         Speaker diarization via WhisperX. Replaces the
                        configured ASR backend for this run. Output:
                        '[HH:MM:SS] SPEAKER_NN: text'. Requires: pip install
                        asr2clip[diarize] and HF_TOKEN env var.
  -s N, --speakers N    Expected number of speakers (hint to pyannote for
                        better accuracy). If omitted, pyannote infers
                        automatically.

Post-processing:
  -P NAME, --post NAME  LLM post-processor name (key in 'postprocessors:'
                        config) or an inline system-prompt string. Requires
                        'postprocessor_backends:' in config. Overrides
                        postprocessor_live / postprocessor_file for this run.
  -M MODEL, --post-model MODEL
                        LLM model used for the post-processing (f. ex. claude-
                        sonnet-4-6). Overrides the post-processor config for
                        this run.
  -T NAME, --template NAME
                        Output template name from 'output_templates:' in
                        config. Controls what is written to clipboard/-o FILE.
                        Overrides the template specified in the prompt
                        definition.

Examples:
  asr2clip --edit                             # create/open config in editor
  asr2clip --test                             # verify backend and preprocessors
  asr2clip                                    # record, transcribe, copy to clipboard
  asr2clip --toggle                           # toggle recording (for keyboard shortcuts)
  asr2clip --toggle -P solo-restructured      # toggle, and produce LLM-structured memo
  asr2clip -i audio.mp3                       # transcribe an existing file
  asr2clip -i m.mp3 -p deepfilter -r          # neural denoising + chunked transcription
  asr2clip -i m.m4a -D -s 3                   # speaker diarization, 3 speakers
  asr2clip --serve                            # start local sherpa-onnx ASR server
  asr2clip --vad -o meeting.txt               # continuous VAD transcription to file
  asr2clip --interval 60                      # fixed-interval continuous recording

See https://github.com/sjjsy/asr2clip for full documentation and configuration examples.
```

## Prerequisites

**Python 3.8+** and one of:
- Cloud API key ([OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text), [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), or any OpenAI-compatible endpoint)
- Local [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) server (`pip install asr2clip[vad]`, model auto-downloads)
- Local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) binary + model file (fully offline, no key needed)

### System packages

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| [**ffmpeg**](https://ffmpeg.org/) | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| [**PortAudio**](https://www.portaudio.com/) | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |

### Optional Python extras

| Extra | Install | Purpose |
|-------|---------|---------|
| `vad` | `pip install asr2clip[vad]` | VAD continuous recording + local sherpa-onnx ASR server |
| `deepfilter` | `pip install asr2clip[deepfilter]` | DeepFilterNet3 best-quality noise reduction |
| `noisereduce` | `pip install asr2clip[noisereduce]` | Spectral noise reduction (scipy) |
| `pyrnnoise` | `pip install asr2clip[pyrnnoise]` | RNNoise GRU noise reduction (scipy) |
| `enhance` | `pip install asr2clip[enhance]` | All three noise reduction options |
| `diarize` | `pip install asr2clip[diarize]` | Speaker diarization via WhisperX |

## Installation

```bash
pip install asr2clip

# or in an isolated environment
pipx install asr2clip

# upgrade
pip install --upgrade asr2clip
```

All extras: Noise reduction options + VAD and the local sherpa-onnx ASR server:
```bash
pip install asr2clip[enhance,vad]
```

### From source

```bash
git clone https://github.com/Oaklight/asr2clip.git
cd asr2clip
pip install -e .
```

Note: Audio preprocessing (`-p`) is not yet applied in VAD/interval continuous mode.

## Setup

Setup commands manage your configuration file and verify that configured backends and preprocessors are working.

| Flag | Description |
|------|-------------|
| `-c FILE` | Path to a specific configuration file |
| `-e / --edit` | Open config in editor (creates default if missing) |
| `--generate_config` | Write the annotated config template to `~/.config/asr2clip/config.yaml` |
| `--print_config` | Print the config template to stdout |
| `--test` | Test backend connectivity and preprocessor availability, then exit |
| `-q / --quiet` | Suppress informational output; only print the transcript and errors |

### Setup commands

```bash
asr2clip --generate_config   # write a fully annotated config with all backend examples
asr2clip --edit              # create/open config in your default editor
asr2clip --print_config      # print the annotated template to stdout
asr2clip --test              # verify backend connectivity and preprocessors
asr2clip --test -b wcpp      # test a specific backend
```

### Config file

Config file is created at `~/.config/asr2clip/config.yaml`. Locations searched in order:
1. `./asr2clip.conf`
2. `~/.config/asr2clip/config.yaml`
3. `~/.config/asr2clip.conf`
4. `~/.asr2clip.conf`

Note: The created config file embeds mostly commented-out configuration options along with brief explanations for most features.
See [`asr2clip.conf.example`](asr2clip.conf.example) in the repo for a complete, current example with all backend and feature documentation.
The following sections tackle some of these in more detail where relevant.

## Audio

| Flag | Description |
|------|-------------|
| `--list_devices` | List available audio input devices with names and indices |
| `-d DEV / --device DEV` | Audio input device (name, ALSA name, or index). Overrides `audio_device` in config. |
| `-p NAME / --preprocessor NAME` | Audio preprocessor: `none`, `noisereduce`, `pyrnnoise`, `deepfilter`. Overrides config. |

### Audio device

```bash
asr2clip --list_devices            # list available input devices with names and indices
asr2clip -d "plughw:Snowball"      # use a specific ALSA device for this run
```

### System audio routing (recommended)

On Linux, set `audio_device` to `"pulse"` or `"pipewire"` and select the active microphone in `pavucontrol` or your desktop sound settings.

```yaml
audio_device: "pulse"              # PulseAudio routes to whichever mic is set as default input
```

### Targeting a specific device directly

```yaml
audio_device: "plughw:Snowball"    # ALSA plughw — format conversion included
audio_device: 3                    # device index from --list_devices
```

| Value | System | Notes |
|-------|--------|-------|
| `"pulse"` | [PulseAudio](https://www.freedesktop.org/wiki/Software/PulseAudio/) (Linux) | Recommended; configure mic via `pavucontrol` |
| `"pipewire"` | [PipeWire](https://pipewire.org/) (Linux) | Recommended on modern Linux |
| `"plughw:Snowball"` | [ALSA](https://alsa-project.org/) (Linux) | Direct USB mic access with format conversion |
| `"hw:2,0"` | ALSA (Linux) | Raw direct access, card 2 device 0 |
| `3` | Any | Device index from `--list_devices` |
| `"BlackHole 2ch"` | macOS | Virtual routing device |

### Audio preprocessing (noise reduction)

asr2clip can denoise audio before transcription — useful in noisy environments (café, open-plan office) or when the ASR backend produces errors or hallucinations from poor signal quality.

### Available preprocessors

| Name | Technology | Extra dependencies | Best for | Project |
|------|-----------|-------------------|----------|---------|
| `none` | — | none (default) | clean recordings | — |
| `noisereduce` | Spectral subtraction | scipy | steady hum, fan noise | [GitHub](https://github.com/timsainb/noisereduce) |
| `pyrnnoise` | Mozilla RNNoise GRU | scipy | babble, non-stationary noise | [GitHub](https://github.com/g-node/pyrnnoise) |
| `deepfilter` | DeepFilterNet3 neural | torch + Rust wheel | best quality overall | [GitHub](https://github.com/Rikorose/DeepFilterNet) |

### `noisereduce` vs `pyrnnoise`

`noisereduce` (spectral subtraction) handles *stationary* noise — constant hum, fan, electrical interference. `pyrnnoise` (neural GRU) handles *non-stationary* noise — babble, footsteps, intermittent sounds. Both require scipy; neither needs a GPU. `pyrnnoise` internally operates at 48 kHz and performs a 16→48→16 kHz round-trip when used at 16 kHz — `noisereduce` avoids this.

### Installing preprocessors

```bash
pip install asr2clip[noisereduce]   # spectral subtraction
pip install asr2clip[pyrnnoise]     # RNNoise GRU
pip install asr2clip[deepfilter]    # DeepFilterNet3
pip install asr2clip[enhance]       # all three
```

### Preprocessor usage

```bash
asr2clip -p deepfilter              # denoise live recording with DeepFilterNet
asr2clip -p noisereduce             # spectral denoising
asr2clip -p deepfilter -i talk.mp4  # denoise video file before transcription
asr2clip --test                     # also checks that configured preprocessors are available
```

### Preprocessor config

Set different preprocessors for live vs. file transcription:

```yaml
preprocessor_live: noisereduce      # no resampling overhead for live 16 kHz audio
preprocessor_file: deepfilter       # best quality for longer file transcription
```

`asr2clip --generate_config` probes your system and writes these keys automatically with the best available option. Override for a single run with `-p`.

### Loudness normalisation

All three preprocessors apply a loudnorm pass after cleaning (RMS → −20 dBFS, peak ceiling −0.1 dBFS) to ensure the ASR backend receives a consistently strong, unclipped signal.

## Transcription

| Flag | Description |
|------|-------------|
| `-b NAME / --backend NAME` | ASR backend to use (key under `asr_backends:` in config). Overrides `asr_backend_live` / `asr_backend_file`. |
| `-i FILE / --input FILE` | Transcribe an existing audio or video file instead of recording. |
| `-o FILE / --output FILE` | Append transcripts to file. |
| `-l LANG / --language LANG` | Language hint (ISO-639-1, e.g. `fi`, `en`). Overrides config. Omit to auto-detect. |
| `-r / --robust` | Robust mode for `-i` file input: split at silence boundaries, quality-check chunks, retry. |
| `-C SEC / --chunk-duration SEC` | Max chunk duration in seconds for `-r/--robust` mode (default: 180). |
| `--toggle` | Toggle recording on/off with repeated invocations. Designed for keyboard shortcuts. |

```bash
asr2clip                           # record until Ctrl+C, transcribe, copy to clipboard
asr2clip -l fi -b wcpp             # Finnish, using local whisper.cpp backend
asr2clip -i audio.mp3              # transcribe an existing audio file
asr2clip -i meeting.mp4            # transcribe from a video file (audio extracted automatically)
asr2clip -o transcript.txt         # also append transcript to a file
```

### Supported input formats

- Audio: `wav`, `mp3`, `m4a`, `ogg`, `flac`, `aac`, `opus`, `wma`
- Video: `mp4`, `mov`, `mkv`, `webm`, `avi`, `flv`, `mvi`

Requires [ffmpeg](https://ffmpeg.org/) on PATH for non-WAV input. Video streams are discarded automatically; basic spectral cleaning (highpass 200 Hz + lowpass 3 kHz + loudnorm) is applied during conversion.

### Language support

Use `-l LANG` with an [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) code (e.g. `fi`, `fr`, `de`, `ja`) to force a specific language. Omit to auto-detect.

Whisper models distinguish between *high-resource* languages (English, Spanish, French, German, Portuguese, Italian, Japanese, Korean, Chinese) — which have abundant training data and near-human accuracy — and *lower-resource* languages with more variable results. Finnish (`fi`), for example, is well-handled by Whisper large-v3 and scores around 15% WER in benchmarks, which is usable for most purposes. Rare languages may require a language hint to avoid misdetection.

Backend-specific notes:
- **whisper.cpp** with `ggml-large-v3-turbo` offers the best multilingual accuracy of any local setup and is the recommended choice for non-English.
- **Groq** (`whisper-large-v3-turbo`) provides equivalent accuracy over API at very low latency — good for live recording in any supported language.
- **SenseVoice** (via sherpa-onnx) is exceptional for Chinese, Japanese, Korean and emotion/event detection, but its language coverage is narrower than Whisper.
- **OpenAI API** (`whisper-1`) supports the same 99 languages as the Whisper model family. Most cloud backends default to English if no hint is provided.

### File and clipboard output

`-o FILE` appends each transcript to the specified file, with a timestamp header prepended to each entry. The file is created if it does not exist.

Continuous and chunked modes integrate with `-o` differently:
- In [robust/chunked mode (`-r`)](#robust-long-file-transcription), chunks are written incrementally as they complete — `tail -f meeting.txt` will show the transcript growing in real time.
- In [VAD mode (`--vad`)](#vad-continuous-recording) and [interval mode (`--interval`)](#vad-continuous-recording), each transcribed utterance is written immediately after the silence boundary triggers.

Clipboard size limit: when a transcript exceeds ~4 000 characters and `-o FILE` is specified, the file path is copied to clipboard instead of the full text.

### ASR backends

ASR (Automatic Speech Recognition) converts spoken audio into text. asr2clip supports several ASR backends, from cloud APIs to fully offline local inference.

All backends are defined under `asr_backends:`. Name them whatever you like and switch at runtime with `-b NAME`. `asr_backend_live` and `asr_backend_file` set the default backend for each mode; `-b` overrides both.

`asr2clip --generate_config` writes a fully annotated config with every supported backend type — uncomment and fill in the one(s) you want.

A minimal working config (cloud API):

```yaml
asr_backends:
  openai:
    type: api
    api_base_url: "https://api.openai.com/v1/"
    api_key: "YOUR_API_KEY"
    model_name: "whisper-1"
asr_backend_live: openai
asr_backend_file: openai
```

A multi-backend example with per-mode defaults:

```yaml
asr_backends:
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
asr_backend_live: groq    # live/toggle/VAD recording
asr_backend_file: wcpp    # -i file transcription
```

```bash
asr2clip                       # live recording → groq (asr_backend_live)
asr2clip -i audio.wav          # file transcription → wcpp (asr_backend_file)
asr2clip -b sonnx              # override for this run (both modes)
asr2clip --test                # tests both live and file backends if they differ
asr2clip --test -b groq        # test a specific backend
```

### Supported backend types

| `type` | Description | Requires |
|--------|-------------|---------|
| `api` | Any OpenAI-compatible HTTP endpoint | API key or local server |
| `whisper_cpp` | whisper.cpp binary via subprocess | whisper.cpp build + `.bin` model file |

Compatible cloud services for `type: api`: [OpenAI](https://platform.openai.com/docs/guides/speech-to-text), [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), and others.

**Speaker diarization as a backend:** `-D/--diarize` activates [WhisperX](https://github.com/m-bain/whisperX) as an alternative transcription path for file processing. It replaces `asr_backend_file` for that run and produces speaker-attributed output — no `type:` entry needed; see [Diarization](#diarization).

#### Local ASR: whisper.cpp vs. sherpa-onnx

Both provide fully offline, no-API-key ASR. Here is how to choose:

| | whisper.cpp | sherpa-onnx (via `--serve`) |
|---|---|---|
| **What it is** | C++ reimplementation of OpenAI Whisper | ONNX-runtime inference with Python bindings |
| **ASR models** | Whisper family (GGML quantised) | Whisper, [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), paraformer, zipformer, and more |
| **Integration** | Subprocess call to external C++ binary | Python-native; exposes a local HTTP API |
| **Setup** | Build C++ from source; download `.bin` model manually | `pip install asr2clip[vad]` + `asr2clip --download-model` |
| **Model auto-download** | No | Yes |
| **VAD support** | No | Yes (built-in via sherpa-onnx) |
| **Dev activity** | Mature, stable | Very active (k2-fsa / Next-gen Kaldi team) |

**When to choose whisper.cpp:** You already have it built, or you prefer GGML-quantised Whisper models without extra Python ML packages.

**When to choose sherpa-onnx:** You want zero C++ build steps, you already installed `asr2clip[vad]` (sherpa-onnx is already there), or you want VAD support or models beyond the Whisper family.

See [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for build instructions and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for its model zoo.

### Robust long-file transcription

For long recordings, `-r`/`--robust` splits at silence boundaries, quality-checks each chunk, retries bad chunks, and writes output incrementally:

```bash
asr2clip -i meeting.mp3 -r                    # chunked, quality-checked
asr2clip -i m.mp3 -rC 60                      # 60 s chunks instead of default 180
asr2clip -i m.mp3 -ro transcript.txt          # write chunks to file as they complete (tail -f)
asr2clip -i m.mp3 -rb wcpp -l fi -o t.txt     # fully offline, Finnish language
asr2clip -i m.mp3 -rP group -T full -o t.txt  # LLM meeting notes + original transcript
```

Long transcripts often exceed the clipboard size limit; using `-o FILE` is recommended.

### Toggle mode

Toggle mode lets you bind a single keyboard shortcut to start and stop recording. The recording runs as a background process; the second invocation stops it, transcribes, and copies to clipboard. A desktop notification is shown on start and finish (requires [`notify-send`](https://man.archlinux.org/man/notify-send.1) on Linux).

```bash
asr2clip --toggle                             # first press: start recording in background
asr2clip --toggle                             # second press: stop, transcribe, copy to clipboard
asr2clip -b wcpp --toggle                     # same, using local whisper.cpp
asr2clip --toggle -P solo-restructured        # toggle → structured personal memo
```

Toggle mode requires a POSIX system (Linux, macOS). Example awesome WM keybinding:

```lua
awful.key({ modkey }, "r", function()
    awful.spawn("asr2clip --toggle")
end)
```

## Local ASR server

`asr2clip` can run a local OpenAI-compatible ASR API server backed by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

| Flag | Description |
|------|-------------|
| `--serve` | Start the local sherpa-onnx ASR API server |
| `--download-model` | Download the SenseVoice model and exit |
| `--host HOST` | Server bind address (default: 127.0.0.1) |
| `--port PORT` | Server bind port (default: 8000) |
| `--model-dir DIR` | Path to ASR model directory |
| `--num-threads N` | Inference threads (default: 4) |

```bash
pip install asr2clip[vad]
asr2clip --download-model                     # download SenseVoice model (~1 GB, once)
asr2clip --serve                              # start server at 127.0.0.1:8000
```

Corresponding config backend:
```yaml
asr_backends:
  sonnx:
    type: api
    api_base_url: "http://127.0.0.1:8000/v1/"
    model_name: "SenseVoiceSmall"
```

## VAD (continuous recording)

VAD (Voice Activity Detection) classifies audio frames as speech or silence, enabling hands-free continuous transcription that triggers automatically at the end of each utterance.

| Flag | Description |
|------|-------------|
| `--vad` | Continuous recording with voice activity detection. Transcribes when silence is detected after speech. Requires `pip install asr2clip[vad]`. |
| `--interval SEC` | Continuous recording with fixed interval (seconds). |
| `--silence_threshold PROB` | Speech probability threshold, 0.0–1.0 (default: 0.5); lower = more sensitive. |
| `--silence_duration SEC` | How long silence must last to trigger transcription (default: 1.5 s). |

### Continuous recording modes

| Mode | Flag | Trigger | Use case |
|------|------|---------|----------|
| Voice Activity Detection | `--vad` | Silence after speech | Meetings, dictation, any unscripted speech |
| Fixed interval | `--interval SEC` | Every N seconds | Lectures, podcasts with predictable pauses |

```bash
asr2clip --vad -o ~/meeting.txt          # auto-transcribe when silence is detected
asr2clip --interval 60 -o ~/meeting.txt  # transcribe every 60 seconds
```

VAD requires [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):

```bash
pip install asr2clip[vad]
```

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) model (~629 KB, downloads automatically on first use). No internet connection required after the first run.

## Diarization

Speaker diarization attributes each spoken segment to a speaker label, producing a transcript where every turn is tagged `[HH:MM:SS] SPEAKER_NN: text`. [WhisperX](https://github.com/m-bain/whisperX) replaces the configured ASR backend for the run — no double transcription.

Speaker name substitution (SPEAKER_00 → real names) is intentionally left to the calling assistant or post-processor.

| Flag | Description |
|------|-------------|
| `-D / --diarize` | Speaker diarization via WhisperX. Replaces the configured ASR backend for this run. |
| `-s N / --speakers N` | Expected number of speakers (hint to pyannote for better accuracy). If omitted, pyannote infers automatically. |

### Diarization setup

```bash
pip install asr2clip[diarize]
# Accept the pyannote licence at https://huggingface.co/pyannote/speaker-diarization-3.1
# then set your HuggingFace token:
export HF_TOKEN=hf_...
# or add to config: diarize_hf_token: "hf_..."
```

**HF_TOKEN** is your [HuggingFace access token](https://huggingface.co/settings/tokens). You must also accept the license for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace before the model can be downloaded. The token is only needed on first run; the model is cached locally thereafter.

### Diarization usage

```bash
asr2clip -i meeting.m4a -D              # diarize: SPEAKER_NN-attributed transcript
asr2clip -i meeting.m4a -D -s 3         # hint: 3 speakers (improves accuracy)
asr2clip -i meeting.m4a -D -P group     # diarize + LLM meeting notes
asr2clip --toggle -D                    # live toggle recording with diarization
```

### Diarization config

Config hints (optional, help [pyannote](https://github.com/pyannote/pyannote-audio)):

```yaml
diarize_hf_token: "hf_..."
diarize_min_speakers: 2
diarize_max_speakers: 6
```

## LLM post-processing

LLM post-processing passes the finished transcript to a language model with user-customizable instructions (prompts) to return refined output output — with simple legibility improvements or significant restructuring (summaries, meeting notes, action items) — to the clipboard (or output file). All post-processing options are defined in the user's config.

| Flag | Description |
|------|-------------|
| `-P NAME / --post NAME` | LLM post-processor name (key in `postprocessors:` config) or an inline system-prompt string. Overrides `postprocessor_live` / `postprocessor_file`. |
| `-M MODEL / --post-model MODEL` | LLM model used for post-processing. Overrides the post-processor config for this run. |
| `-T NAME / --template NAME` | Output template name from `output_templates:` in config. Controls what is written to clipboard / `-o FILE`. |

### Supported LLM backends

| Backend type | What it covers |
|---|---|
| `openai_compat` | [Ollama](https://ollama.com/) (local), [Groq](https://console.groq.com/), [Anthropic API](https://www.anthropic.com/api), [OpenAI](https://platform.openai.com/), any OpenAI-compatible endpoint |
| `claude_code` | [Claude Code](https://claude.ai/code) CLI — uses your CC session/subscription, no per-token billing |

### Post-processing setup

```yaml
# ~/.config/asr2clip/config.yaml
postprocessor_backends:
  ollama:
    type: openai_compat
    api_base_url: "http://localhost:11434/v1/"
    api_key: "ollama"
    model: "qwen3:14b"
  cc:
    type: claude_code
    model: "claude-haiku-4-5-20251001"
```

### Postprocessor configuration

Each prompt under `postprocessors:` can have the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `prompt:` | string | System prompt text. Required unless `extends:` is used. |
| `extends:` | string | Name of another prompt to inherit from. Inherits `prompt`, `backend`, `model`, and `context_path` from parent. |
| `extra:` | string | Text appended to the inherited prompt (only with `extends:`). |
| `backend:` | string | Which `postprocessor_backends:` entry to use. Overrides inherited backend. |
| `model:` | string | Override the backend's default model for this prompt. |
| `template:` | string | Output template name from `output_templates:` section (default: `default`). Controls final output format. |
| `context_path:` | list of strings | File glob patterns to inject as context. Glob patterns are expanded and combined with inherited patterns when using `extends:`. |

**Inheritance behavior:** When a prompt uses `extends:`, it inherits all parent fields. Child fields override parent fields, except `context_path:` which accumulates (both parent and child patterns are expanded and combined).

Example with inheritance:

```yaml
postprocessors:
  solo-base:
    backend: groq
    prompt: |
      You are a professional transcript scribe ...

  solo-enhanced:
    extends: solo-base            # inherits backend, prompt
    extra: |
      Also improve grammar and style ...

  solo-private:
    extends: solo-enhanced        # inherits backend, prompt + extra
    backend: ollama               # override: use local model for privacy
    context_path:
      - "~/.asr2clip/context/private.md"  # accumulates with parent's context
```

### Built-in prompts

Shipped in the config template, ready to use:

**Personal dictation prompts:**

| Name | Purpose |
|------|---------|
| `solo-base` | Correct errors and clean up a personal single-speaker transcript |
| `solo-enhanced` | Improve quality, fix grammar and word choice while honoring the author's style |
| `solo-restructured` | Restructure a personal dictation into a structured memo with sections |
| `solo-private` | Like `solo-restructured` but defaults to a local offline model to ensure privacy |

**Meeting/group discussion prompts:**

| Name | Purpose |
|------|---------|
| `group-base` | Correct errors and clean up a group discussion transcript |
| `group-enhanced` | Improve quality of group transcript while honoring each speaker's style |
| `group-restructured` | Restructure a group discussion into a meeting memo with summary, decisions, action items |
| `group-private` | Like `group-restructured` but defaults to a local offline model to ensure privacy |

```bash
asr2clip --toggle -P solo-enhanced          # toggle → improved personal transcript
asr2clip --toggle -P solo-restructured      # toggle → structured personal memo
asr2clip -i meeting.m4a -D -P group-restructured  # diarize + meeting memo
asr2clip --toggle -P "List action items."   # inline system prompt
```

### Prompt inheritance

`extends:` is used by the built-in prompts and is available for your own:

```yaml
postprocessors:
  solo-base:
    prompt: |
      You are a professional personal transcript scribe.
      ...
    template: bare

  solo-restructured:
    extends: solo-base
    extra: |
      Produce a concise, structured memo ...

  solo-private:
    extends: solo-restructured
    # backend: ollama    # local model for sensitive content
```

### Output templates

Output templates control what ends up in clipboard / `-o FILE`. The output format depends on your template and LLM instructions — can be Markdown, [ReStructuredText](https://docutils.sourceforge.io/rst.html), plain text, or any text-based format. Two templates are shipped; select with `-T NAME`:

```yaml
output_templates:
  bare: "{result}"
  full: |
    {result}

    ---
    *Transcript from {duration_s:.0f}s recording post-processed at {datetime} with asr2clip ({backend}, {prompt_name}, {model})*

    ## Original transcript

    {transcript}
```

```bash
asr2clip -i m.mp3 -r -P group -T full      # meeting notes + full transcript appended
```

Available placeholders: `{result}` `{transcript}` `{date}` `{datetime}` `{prompt_name}` `{model}` `{backend}` `{duration_s}`

Set `postprocessor_live` / `postprocessor_file` to apply a prompt automatically for every recording without passing `-P`.

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
| Preprocessing too slow | Switch `preprocessor_live` to `noisereduce` or `none` in config |
| Post-processor not found | Check `postprocessors:` in config; name must match exactly |
| Post-processor backend error | Check `postprocessor_backends:` in config; verify API key and URL |
| Diarization fails | Ensure `pip install asr2clip[diarize]`, `HF_TOKEN` is set, and pyannote licence is accepted |

Run `asr2clip --test` (or `asr2clip --test -b <name>`) to diagnose issues.

## Contributing

Fork the repository and submit a pull request. Any improvements or new features are welcome! :)

## License

GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Related projects

asr2clip operates within a four-stage pipeline:

```
[Audio capture] → [ASR / transcription] → [LLM post-processing] → [Output: clipboard / file]
     stage 1            stage 2                   stage 3                     stage 4
```

The tables below cover the ecosystem at each pipeline stage and compare competing end-user tools. asr2clip covers stages 1–3 in a single composable CLI.

### ASR engines

ASR engines convert audio to text. asr2clip is a frontend: it delegates transcription to an ASR backend, supporting two locally-run backends (whisper.cpp, sherpa-onnx) and any OpenAI-compatible HTTP endpoint for cloud or self-hosted services. The engines listed as integrated below are ones asr2clip directly calls or supports as backends; the others are libraries or specialized tools that require custom wrappers to use.

| Project | License | Stars | Best for | In asr2clip |
|---------|---------|-------|----------|-------------|
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT | 80k+ | Gold standard; 99 languages; most widely reproduced | Via API (`whisper-1`) or indirectly through sherpa-onnx and whisper.cpp |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | MIT | 80k+ | Fully offline; best CPU performance; GGML-quantised models | **Yes** — `type: whisper_cpp` backend; subprocess call |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 15k+ | 4× faster than Whisper; identical accuracy; INT8/FP16 via CTranslate2 | Not directly; used internally by WhisperX and Meetily |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2 | 4k+ | ONNX inference; multi-model-family; model auto-download; Python-native | **Yes** — `--serve` local server; `type: api` backend |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD | 13k+ | Whisper + word-level timestamps + speaker diarization in one pipeline | **Yes** — via `--diarize` flag |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Apache-2 | 6k+ | Emotion + language event detection; excellent CJK | Via sherpa-onnx default model; also SiliconFlow API |
| [Vosk](https://github.com/alphacep/vosk-api) | Apache-2 | 8k+ | Lightweight; 20+ languages; embedded and low-RAM devices | No — lower accuracy than Whisper family |
| [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Apache-2 | (NeMo) | 3 380× faster than real-time; English only; GPU | No — English-only; GPU-dependent; no multilingual support |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | Apache-2 | 9k+ | Research platform; fine-tuning; custom model training | No — research library, not a drop-in backend |
| [Coqui STT](https://github.com/coqui-ai/STT) | MPL-2 | 5k+ | DeepSpeech successor; trainable on custom data | No — lower quality than Whisper; limited community activity |
| [Kaldi](https://github.com/kaldi-asr/kaldi) | Apache-2 | 14k+ | Enterprise/research; highly configurable; steep setup | No — complex; not a practical CLI backend |

### Audio preprocessing (noise reduction)

Audio preprocessing cleans the signal before transcription. asr2clip integrates all three libraries below as optional extras (`pip install asr2clip[enhance]`); they run in a pipeline with loudness normalisation applied after cleaning.

| Project | Technology | License | Best for | In asr2clip |
|---------|-----------|---------|----------|-------------|
| [noisereduce](https://github.com/timsainb/noisereduce) | Spectral subtraction | MIT | Stationary noise: fans, AC, electrical hum | **Yes** — `pip install asr2clip[noisereduce]` |
| [pyrnnoise](https://github.com/g-node/pyrnnoise) | Mozilla RNNoise GRU | GPL-3 | Non-stationary noise: crowd, babble, footsteps | **Yes** — `pip install asr2clip[pyrnnoise]` |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | DeepFilterNet3 neural net | MIT | Best quality overall; speech naturalness; medium CPU | **Yes** — `pip install asr2clip[deepfilter]` |
| [RNNoise](https://github.com/xiph/rnnoise) | Xiph GRU | BSD | Original Mozilla RNNoise (C library) | No — pyrnnoise wraps this at the Python layer |
| [SpeechBrain enhance](https://github.com/speechbrain/speechbrain) | Encoder-decoder neural | Apache-2 | Research-grade speech separation and denoising | No — heavy ML framework dependency; not practical as a live preprocessor |

### Voice Activity Detection

VAD classifies audio frames as speech or silence, enabling automatic segment boundaries without user interaction. asr2clip uses Silero VAD (bundled in sherpa-onnx) for both the `--vad` continuous mode and the silence-split inside `--robust`.

| Project | License | Stars | Notes | In asr2clip |
|---------|---------|-------|-------|-------------|
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | 14k+ | 629 KB model; enterprise-grade; ONNX + PyTorch; auto-downloads | **Yes** — via sherpa-onnx in `--vad` and `--robust` |
| [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) | BSD | 1k+ | Google's classic GMM-based VAD; very fast, lower accuracy | No — less accurate than Silero; not integrated |
| [pyannote VAD](https://github.com/pyannote/pyannote-audio) | MIT | 6k+ | Neural VAD embedded in the pyannote diarization pipeline | Indirectly — activated during `--diarize` via WhisperX |

### Speaker diarization

Speaker diarization labels each segment with a speaker identity ("who said what"). asr2clip's `--diarize` flag activates a complete diarization pipeline through WhisperX; pyannote.audio is used internally by WhisperX for the actual speaker embedding and clustering.

| Project | License | Stars | Notes | In asr2clip |
|---------|---------|-------|-------|-------------|
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT | 6k+ | De-facto OSS standard; speaker embedding + clustering; requires HF token for model download | Via WhisperX (`--diarize`) |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD | 13k+ | faster-whisper + word alignment + pyannote; all-in-one | **Yes** — `--diarize` (`pip install asr2clip[diarize]`) |
| [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) | MIT | 2k+ | faster-whisper + pyannote script pipeline | No — WhisperX provides equivalent functionality with an active upstream |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Apache-2 | 13k+ | Fastest GPU diarization; English and enterprise focus | No — GPU-heavy; no practical CLI integration path |

### Desktop audio capture and transcription tools

These are end-user tools that combine audio capture, ASR, and transcript output — the closest category to asr2clip itself.

| Project | Type | Platform | License | Live | File | Toggle | VAD | Offline | Diarize | LLM post | Notes |
|---------|------|----------|---------|------|------|--------|-----|---------|---------|----------|-------|
| **asr2clip** (this) | CLI | Linux, macOS | AGPL-3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full pipeline; scriptable; multi-backend; video input |
| [Turbo Whisper](https://github.com/knowall-ai/turbo-whisper) | GUI | Linux | MIT | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | faster-whisper-large-v3-turbo; global hotkey; no file mode; PPA install |
| [Whispering](https://github.com/braden-w/whispering) | GUI/tray | Any | MIT | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | Cross-platform (snap/exe); local or cloud API; minimal UI |
| [Superwhisper](https://superwhisper.com/) | GUI | macOS, Windows, iOS | Proprietary | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | Partial | Premium dictation app; polished UX; no Linux |
| [Meetily](https://github.com/Zackriya-Solutions/meetily) | Desktop app | macOS, Windows | MIT | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | 11.9k★; Rust backend; Ollama summaries; no Linux |
| [Screenpipe](https://github.com/screenpipe/screenpipe) | Agent layer | Any | MIT | ✓ (always-on) | ✓ | ✗ | ✓ | ✓ | ✗ | Via MCP | 18.6k★; ambient 24/7 recording; MCP server; not a CLI tool |

### SaaS meeting assistants

Commercial cloud services that join calls automatically or process uploaded recordings. Included for context — these require trusting a third party with your audio.

| Service | Platform | Bot-less | Offline | Privacy | Notable | vs asr2clip |
|---------|----------|----------|---------|---------|---------|-------------|
| [Fathom](https://fathom.video/) | Web (Zoom/Meet/Teams) | No | No | Cloud (US) | Free tier; calendar-integrated; good UX | Cloud-only; no file processing; no Linux CLI |
| [Jamie](https://meetjamie.ai/) | macOS, Windows | Yes | No | GDPR (EU) | Best Finnish quality; bot-free desktop app | €24+/mo; macOS/Windows only |
| [Fireflies.ai](https://fireflies.ai/) | Web | No | No | Cloud | 100 languages; CRM sync; "Ask Fred" AI queries | Cloud; BIPA lawsuit 2025; data outside EU |
| [Granola](https://granola.ai/) | macOS | Yes | Partial | Cloud for AI | Calendar-integrated; note editor; bot-free | macOS only; AI processing requires cloud |
| [Otter.ai](https://otter.ai/) | Web, mobile | No | No | Cloud (US) | Real-time captions; widely known | Class-action lawsuit 2025; weak Finnish |
| [Soniox](https://soniox.com/) | API | — | No | Cloud (US) | Best Finnish WER (10.6%); 56 languages; developer API | API service, not an end-user tool |
| [Krisp](https://krisp.ai/) | App + SDK | Yes | Partial | Cloud for AI | Industry-leading noise suppression + transcription | Proprietary; subscription; not scriptable |

### asr2clip as an open source contribution

The speech-to-text tool landscape in 2026 has a sharp divide: powerful Python libraries (faster-whisper, WhisperX, pyannote) that require programming to use, and polished end-user apps (Superwhisper, Meetily, Granola) that are macOS/Windows-only or cloud-dependent. Linux users who want local, private, keyboard-shortcut-driven transcription with the full power of the Whisper ecosystem face a gap. Turbo Whisper and Whispering address the simplest dictation case but lack file transcription, noise reduction, robustness for long recordings, and any programmable post-processing. asr2clip fills this gap as a single composable CLI that exposes the full four-stage pipeline without requiring the user to write any code.

Beyond Linux, asr2clip's value is its scriptability and composability. Every feature — backend, preprocessor, language, diarization, post-processor, output template — is a flag or config key. This makes it naturally callable from shell scripts, Makefiles, cron jobs, and AI coding agents: one invocation covers the full audio → transcript → structured-memo pipeline that would otherwise require stitching together three or four Python libraries. The support for both local (whisper.cpp, sherpa-onnx) and cloud (OpenAI, Groq, SiliconFlow) backends with a unified interface means the same command works offline on a laptop and in a cloud pipeline on a headless server.

The most capable competing open-source project, Meetily, is architecturally similar in ambition — local-first, offline, Whisper-backed, with LLM summaries — but is a GUI-only desktop app for macOS and Windows with no Linux support and no CLI surface. Screenpipe is a different paradigm (always-on ambient capture) rather than a competing tool. This leaves asr2clip as currently the most complete open-source, Linux-native, CLI-accessible speech processing pipeline — a category with no direct competition and clear utility for developers, power users, and autonomous AI agents that need to process human speech.
