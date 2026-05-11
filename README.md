# asr2clip -- Speech-to-Text Clipboard Tool — With Everything You'd Want :)

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[中文](README_zh.md)

Record speech, transcribe it, and copy the result to clipboard or a file. Supports cloud and fully-local [ASR backends](#asr-backends), [VAD](#vad-continuous-recording) (for background daemons) and [toggle](#toggle-mode) (for keyboard shortcuts) recording modes, [noise reduction](#audio-preprocessing-noise-reduction), [speaker diarization](#diarization), and [LLM post-processing](#llm-post-processing).

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
  --toggle              Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.
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
  asr2clip                                    # record, transcribe, copy to clipboard
  asr2clip --vad -o meeting.txt               # continuous VAD transcription to file
  asr2clip --interval 60                      # fixed-interval continuous recording
  asr2clip -i audio.mp3                       # transcribe an existing file
  asr2clip --toggle                           # toggle recording (for keyboard shortcuts)
  asr2clip -p deepfilter --toggle             # toggle with DeepFilterNet noise reduction
  asr2clip -i m.mp3 -r                        # robust chunked long-file transcription
  asr2clip -i m.m4a -D -s 3                  # speaker diarization, 3 speakers
  asr2clip --toggle -P solo-restructured      # toggle → LLM-structured memo
  asr2clip --serve                            # start local sherpa-onnx ASR server
  asr2clip --edit                             # create/open config in editor
  asr2clip --test                             # verify backend and preprocessors

See https://github.com/sjjsy/asr2clip for full documentation and configuration examples.
```

## Prerequisites

**Python 3.8+** and one of:
- Cloud API key (OpenAI Whisper, Groq, SiliconFlow, xinference, or any OpenAI-compatible endpoint)
- Local [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) server (`pip install asr2clip[vad]`, model auto-downloads)
- Local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) binary + model file (fully offline, no key needed)

**System packages:**

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| **ffmpeg** | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| **PortAudio** | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |

**Optional Python extras** (install only what you need):

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

**From source:**
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

```bash
asr2clip --generate_config   # write a fully annotated config with all backend examples
asr2clip --edit              # create/open config in your default editor
asr2clip --print_config      # print the annotated template to stdout
asr2clip --test              # verify backend connectivity and preprocessors
asr2clip --test -b wcpp      # test a specific backend
```

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

**Recommended — use system audio routing:**

On Linux, set `audio_device` to `"pulse"` or `"pipewire"` and select the active microphone in `pavucontrol` or your desktop sound settings.

```yaml
audio_device: "pulse"              # PulseAudio routes to whichever mic is set as default input
```

**Targeting a specific device directly:**

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

### Audio preprocessing (noise reduction)

asr2clip can denoise audio before transcription — useful in noisy environments (café, open-plan office) or when the ASR backend produces errors or hallucinations from poor signal quality.

**Available preprocessors:**

| Name | Technology | Extra dependencies | Best for | Project |
|------|-----------|-------------------|----------|---------|
| `none` | — | none (default) | clean recordings | — |
| `noisereduce` | Spectral subtraction | scipy | steady hum, fan noise | [GitHub](https://github.com/timsainb/noisereduce) |
| `pyrnnoise` | Mozilla RNNoise GRU | scipy | babble, non-stationary noise | [GitHub](https://github.com/g-node/pyrnnoise) |
| `deepfilter` | DeepFilterNet3 neural | torch + Rust wheel | best quality overall | [GitHub](https://github.com/Rikorose/DeepFilterNet) |

**`noisereduce` vs `pyrnnoise`:** `noisereduce` (spectral subtraction) handles *stationary* noise — constant hum, fan, electrical interference. `pyrnnoise` (neural GRU) handles *non-stationary* noise — babble, footsteps, intermittent sounds. Both require scipy; neither needs a GPU. `pyrnnoise` internally operates at 48 kHz and performs a 16→48→16 kHz round-trip when used at 16 kHz — `noisereduce` avoids this.

**Installation:**
```bash
pip install asr2clip[noisereduce]   # spectral subtraction
pip install asr2clip[pyrnnoise]     # RNNoise GRU
pip install asr2clip[deepfilter]    # DeepFilterNet3
pip install asr2clip[enhance]       # all three
```

**Usage:**
```bash
asr2clip -p deepfilter              # denoise live recording with DeepFilterNet
asr2clip -p noisereduce             # spectral denoising
asr2clip -p deepfilter -i talk.mp4  # denoise video file before transcription
asr2clip --test                     # also checks that configured preprocessors are available
```

**Config** (set different preprocessors for live vs. file transcription):
```yaml
preprocessor_live: noisereduce      # no resampling overhead for live 16 kHz audio
preprocessor_file: deepfilter       # best quality for longer file transcription
```

`asr2clip --generate_config` probes your system and writes these keys automatically with the best available option. Override for a single run with `-p`.

**Loudness normalisation:** All three preprocessors apply a loudnorm pass after cleaning (RMS → −20 dBFS, peak ceiling −0.1 dBFS) to ensure the ASR backend receives a consistently strong, unclipped signal.

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

**Supported input formats:**
- Audio: `wav`, `mp3`, `m4a`, `ogg`, `flac`, `aac`, `opus`, `wma`
- Video: `mp4`, `mov`, `mkv`, `webm`, `avi`, `flv`, `mvi`

Requires `ffmpeg` on PATH for non-WAV input. Video streams are discarded automatically; basic spectral cleaning (highpass 200 Hz + lowpass 3 kHz + loudnorm) is applied during conversion.

Notes:
- **Language:** Many backends auto-detect language but some default to English. Use `-l LL` (ISO-639-1: `fi`, `fr`, `de`) to force a language.
- **Clipboard size limit:** When a transcript exceeds ~4000 characters and `-o FILE` is specified, the file path is copied to clipboard instead of the full text.

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

**Supported `type` values:**

| `type` | Description | Requires |
|--------|-------------|---------|
| `api` | Any OpenAI-compatible HTTP endpoint | API key or local server |
| `whisper_cpp` | whisper.cpp binary via subprocess | whisper.cpp build + `.bin` model file |

Compatible cloud services for `type: api`: OpenAI, [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), and others.

**Speaker diarization as a backend:** `-D/--diarize` activates [WhisperX](https://github.com/m-bain/whisperX) as an alternative transcription path for file processing. It replaces `asr_backend_file` for that run and produces speaker-attributed output — no `type:` entry needed; see [Diarization](#diarization).

#### Local ASR: whisper.cpp vs. sherpa-onnx

Both provide fully offline, no-API-key ASR. Here is how to choose:

| | whisper.cpp | sherpa-onnx (via `--serve`) |
|---|---|---|
| **What it is** | C++ reimplementation of OpenAI Whisper | ONNX-runtime inference with Python bindings |
| **ASR models** | Whisper family (GGML quantised) | Whisper, SenseVoice, paraformer, zipformer, and more |
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
asr2clip -i meeting.mp3 -r -C 60              # 60 s chunks instead of default 180
asr2clip -i meeting.mp3 -r -o transcript.txt  # write chunks to file as they complete (tail -f)
asr2clip -i m.mp3 -r -b wcpp -l fi -o o.txt   # fully offline, Finnish language
asr2clip -i m.mp3 -r -P group -T full         # LLM meeting notes + original transcript
```

Long transcripts often exceed the clipboard size limit; using `-o FILE` is recommended.

### Toggle mode

Toggle mode lets you bind a single keyboard shortcut to start and stop recording. The recording runs as a background process; the second invocation stops it, transcribes, and copies to clipboard. A desktop notification is shown on start and finish (requires `notify-send` on Linux).

```bash
asr2clip --toggle                         # first press: start recording in background
asr2clip --toggle                         # second press: stop, transcribe, copy to clipboard
asr2clip -b wcpp --toggle                 # same, using local whisper.cpp
asr2clip --toggle -P solo-restructured    # toggle → structured personal memo
```

Toggle mode requires a POSIX system (Linux, macOS). Example awesome WM keybinding:

```lua
awful.key({ modkey }, "r", function()
    awful.spawn("asr2clip --toggle")
end)
```

## Local ASR server

`asr2clip` can run a local OpenAI-compatible ASR API server backed by sherpa-onnx.

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
asr2clip --download-model          # download SenseVoice model (~1 GB, once)
asr2clip --serve                   # start server at 127.0.0.1:8000
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

Two continuous modes are available:

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

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) model (~629 KB, downloads automatically on first use). No internet connection required after the first run.

## Diarization

Speaker diarization attributes each spoken segment to a speaker label, producing a transcript where every turn is tagged `[HH:MM:SS] SPEAKER_NN: text`. [WhisperX](https://github.com/m-bain/whisperX) replaces the configured ASR backend for the run — no double transcription.

Speaker name substitution (SPEAKER_00 → real names) is intentionally left to the calling assistant or post-processor.

| Flag | Description |
|------|-------------|
| `-D / --diarize` | Speaker diarization via WhisperX. Replaces the configured ASR backend for this run. |
| `-s N / --speakers N` | Expected number of speakers (hint to pyannote for better accuracy). If omitted, pyannote infers automatically. |

**Setup:**

```bash
pip install asr2clip[diarize]
# Accept the pyannote licence at https://huggingface.co/pyannote/speaker-diarization-3.1
# then set your HuggingFace token:
export HF_TOKEN=hf_...
# or add to config: diarize_hf_token: "hf_..."
```

**HF_TOKEN** is your [HuggingFace access token](https://huggingface.co/settings/tokens). You must also accept the license for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace before the model can be downloaded. The token is only needed on first run; the model is cached locally thereafter.

**Usage:**

```bash
asr2clip -i meeting.m4a -D              # diarize: SPEAKER_NN-attributed transcript
asr2clip -i meeting.m4a -D -s 3        # hint: 3 speakers (improves accuracy)
asr2clip -i meeting.m4a -D -P group    # diarize + LLM meeting notes
asr2clip --toggle -D                    # live toggle recording with diarization
```

Config hints (optional, help pyannote):

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

**Supported LLM backends:**

| Backend type | What it covers |
|---|---|
| `openai_compat` | Ollama (local), Groq, Anthropic API, OpenAI, any OpenAI-compatible endpoint |
| `claude_code` | Claude Code CLI — uses your CC session/subscription, no per-token billing |

**Setup:**

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

**Built-in prompts** (shipped in the config template, ready to use):

| Name | Purpose |
|------|---------|
| `solo-base` | Correct errors and clean up a personal single-speaker transcript |
| `solo-enhanced` | Improve quality, fix grammar and word choice while honoring the author's style |
| `solo-restructured` | Restructure a personal dictation into a structured memo with sections |
| `solo-private` | Like `solo-restructured` with privacy handling (names/locations omitted) |
| `group` | Meeting notetaker: summary, decisions, action items, open questions |

```bash
asr2clip --toggle -P solo-enhanced          # toggle → improved transcript
asr2clip --toggle -P solo-restructured      # toggle → structured personal memo
asr2clip -i meeting.m4a -D -P group        # diarize + meeting notes
asr2clip --toggle -P "List action items."   # inline system prompt
```

**Prompt inheritance with `extends:`** — used by the built-in prompts and available for your own:

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

**Output templates** control what ends up in clipboard / `-o FILE`. Two are shipped; select with `-T NAME`:

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
