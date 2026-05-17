# kaiku -- Modular Voice (SST) Pipeline Prototype :)

Record speech, transcribe it, and copy the result to clipboard or a file. Supports cloud and fully local [ASR backends](#asr-backends), [VAD](#vad-continuous-recording) (for background daemons) and [toggle](#toggle-mode) (for keyboard shortcuts), [noise reduction](#audio-preprocessing-noise-reduction), [speaker diarization](#diarization) (as an ASR backend), and [AI post-processing](#post-processing-with-ai-models).

Originally forked from the elegant but minimalist speech to clipboard tool [asr2clip](https://github.com/Oaklight/asr2clip) which is also available from [![PyPI](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/). The word "kaiku" is Finnish for "echo".

Jump to the [Related projects](#related-projects) section at the end to understand the landscape of ASR related tooling and why this project was developed for the open source community.

## TL;DR

**Cloud (API) path:**
```bash
pip3 install kaiku
kaiku --generate-config   # create config with all backend examples
kaiku --edit              # fill in your API key
kaiku --test              # verify
kaiku                     # record and transcribe
```

**Local offline path — sherpa-onnx with VAD support (model auto-downloads):**
```bash
pip3 install kaiku[vad]
kaiku --download-model    # download SenseVoice model on first use
kaiku --serve &           # start local ASR API server
# configure a backend pointing to http://127.0.0.1:8000/v1/ — see Local ASR server below
kaiku --test -b sonnx
kaiku -b sonnx
```

**Local offline path — whisper.cpp (no VAD):**
```bash
pip3 install kaiku
# build whisper.cpp and download a model, then configure it in config
kaiku --generate-config   # shows a wcpp backend example
kaiku --test -b wcpp
kaiku -b wcpp
```

## CLI reference

```
usage: kaiku [-h] [-v] [-q] [-c FILE] [-e] [--generate-config]
                [--print-config] [--test] [-x NAME] [--list-devices] [-d DEV]
                [-i FILE] [-p NAME] [-b NAME] [-l LANG] [-r] [-C SEC] [-g]
                [--serve] [--host HOST] [--port PORT] [--model-dir MODEL_DIR]
                [--num-threads NUM_THREADS] [--download-model] [--vad]
                [--interval SEC] [--silence-threshold PROB]
                [--silence-duration SEC] [-s N] [-P NAME] [-M MODEL] [-o FILE]
                [-T NAME] [-z]

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
  --generate-config     Write config template to
                        ~/.config/kaiku/config.yaml
  --print-config        Print config template to stdout
  --test                Test backend connectivity and configured
                        preprocessors, then exit
  -x NAME, --preset NAME
                        Pipeline preset name (key under 'presets:' in config).
                        Presets define complete pipelines: ASR backend,
                        preprocessor, post-processor. Optional if
                        'default_preset' is set in config; CLI overrides still
                        work (-b, -p, -P).

Audio:
  --list-devices        List available audio input devices
  -d DEV, --device DEV  Audio input device (name, ALSA name, or index).
                        Overrides config.
  -i FILE, --input FILE
                        Transcribe an existing audio or video file instead of
                        recording. Supported: wav, mp3, m4a, ogg, flac, aac,
                        opus, wma, mp4, mov, mkv, webm, avi, flv, mvi
  -p NAME, --preprocessor NAME
                        Audio preprocessor: none, noisereduce, pyrnnoise,
                        deepfilter. Overrides the preprocessor in the selected
                        preset.

Transcription:
  -b NAME, --backend NAME
                        ASR backend to use (key under 'asr_backends:' in
                        config). Overrides the backend in the selected preset.
  -l LANG, --language LANG
                        Language hint for transcription (ISO-639-1, e.g. 'fi',
                        'en'). Overrides config. Omit to auto-detect.
  -r, --robust          Robust mode for -i file input: split at silence
                        boundaries, quality-check chunks, retry failures,
                        stream output (tail-f friendly).
  -C SEC, --chunk-duration SEC
                        Max chunk duration in seconds for -r/--robust mode
                        (default: 180)
  -g, --toggle          Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.

Local ASR server:
  --serve               Start the local sherpa-onnx ASR API server
  --host HOST           Server bind address (default: 127.0.0.1 or
                        local_asr.host in config)
  --port PORT           Server bind port (default: 8000)
  --model-dir MODEL_DIR
                        Path to ASR model directory
  --num-threads NUM_THREADS
                        Inference threads (default: 4)
  --download-model      Download the SenseVoice model and exit

VAD (continuous recording):
  --vad                 Continuous recording with voice activity detection.
                        Transcribes automatically when silence is detected
                        after speech. Requires sherpa-onnx: pip3 install
                        kaiku[vad].
  --interval SEC        Continuous recording with fixed interval (seconds)
  --silence-threshold PROB
                        VAD speech probability threshold, 0.0-1.0 (default:
                        0.5)
  --silence-duration SEC
                        Silence duration to trigger transcription (default:
                        1.5 s)

Diarization:
  -s N, --speakers N    Speaker count hint for diarization backends (type:
                        whisperx, type: mock-diarize). Ignored by all other
                        backends. Selects a diarization backend in your preset
                        or with -b / --backend; see 'asr_backends:' in config.
                        If omitted, the backend uses its own default or auto-
                        detects speaker count.

Post-processing:
  -P NAME, --post NAME  AI post-processor name (key in 'postprocessors:'
                        config) or an inline system-prompt string. Requires
                        'postprocessor_backends:' in config. Overrides the
                        post-processor in the selected preset.
  -M MODEL, --post-model MODEL
                        AI model used for the post-processing (f. ex. claude-
                        sonnet-4-6). Overrides the post-processor config for
                        this run.

Output:
  -o FILE, --output FILE
                        Append transcripts to file
  -T NAME, --template NAME
                        Output template name from 'output_templates:' in
                        config. Controls what is written to clipboard/-o FILE.
                        Overrides the template specified in the prompt
                        definition.
  -z, --no-clipboard    Do not copy the transcript (or a file path) to the
                        system clipboard. Stdout and -o output behave as
                        usual.

Examples:
  kaiku --edit                             # create/open config in editor
  kaiku --test                             # verify backend and preprocessors
  kaiku                                    # record, transcribe, copy to clipboard
  kaiku --toggle                           # toggle recording (for keyboard shortcuts)
  kaiku --toggle -P solo-restructure       # toggle, and produce AI-structured memo
  kaiku -i audio.mp3                       # transcribe an existing file
  kaiku -i m.mp3 -p deepfilter -r          # neural denoising + chunked transcription
  kaiku -i meeting.m4a -b whisperx -s 3    # speaker diarization, 3-speaker hint
  kaiku --serve                            # start local sherpa-onnx ASR server
  kaiku --vad -o meeting.txt               # continuous VAD transcription to file
  kaiku --interval 60                      # fixed-interval continuous recording

See https://github.com/sjjsy/kaiku for full documentation and configuration examples.
```

## Prerequisites

**Python 3.8+** and one of:
- Cloud API key ([OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text), [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), or any OpenAI-compatible endpoint)
- Local [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) server (`pip3 install kaiku[vad]`, model auto-downloads)
- Local [whisper.cpp](https://github.com/ggerganov/whisper.cpp) binary + model file (fully offline, no key needed)

### System packages

| Dependency | Purpose | Linux | macOS | Windows |
|------------|---------|-------|-------|---------|
| [**ffmpeg**](https://ffmpeg.org/) | Audio format conversion | `apt install ffmpeg` | `brew install ffmpeg` | [Download](https://ffmpeg.org/download.html) |
| [**PortAudio**](https://www.portaudio.com/) | Audio recording | `apt install libportaudio2` | `brew install portaudio` | Included with sounddevice |

### Optional Python extras

| Extra | Install | Purpose |
|-------|---------|---------|
| `vad` | `pip3 install kaiku[vad]` | VAD continuous recording + local sherpa-onnx ASR server |
| `deepfilter` | `pip3 install kaiku[deepfilter]` | DeepFilterNet3 best-quality noise reduction |
| `noisereduce` | `pip3 install kaiku[noisereduce]` | Spectral noise reduction (scipy) |
| `pyrnnoise` | `pip3 install kaiku[pyrnnoise]` | RNNoise GRU noise reduction (scipy) |
| `enhance` | `pip3 install kaiku[enhance]` | All three noise reduction options |
| `diarize` | `pip3 install kaiku[diarize]` | Speaker diarization via WhisperX |

## Installation

```bash
pip3 install kaiku

# or in an isolated environment
pipx install kaiku

# upgrade
pip3 install --upgrade kaiku
```

All extras: Noise reduction options + VAD and the local sherpa-onnx ASR server:
```bash
pip3 install kaiku[enhance,vad]
```

Note: Audio preprocessing (`-p`) is not (yet) applied in VAD/interval continuous mode.

### From source

```bash
git clone https://github.com/Oaklight/kaiku.git
cd kaiku
pip3 install -e .
```

## Setup

Setup commands manage your configuration file and verify that configured backends and preprocessors are working.

| Flag | Description |
|------|-------------|
| `-c FILE` | Path to a specific configuration file |
| `-e / --edit` | Open config in editor (creates default if missing) |
| `--generate-config` | Write the annotated config template to `~/.config/kaiku/config.yaml` |
| `--print-config` | Print the config template to stdout |
| `--test` | Test backend connectivity and preprocessor availability, then exit |
| `-x NAME / --preset NAME` | Pipeline preset (key under `presets:`). Optional if `default_preset` is set in config. |
| `-q / --quiet` | Suppress informational output; only print the transcript and errors |

### Setup commands

```bash
kaiku --generate-config   # write a fully annotated config with all backend examples
kaiku --edit              # create/open config in your default editor
kaiku --print-config      # print the annotated template to stdout
kaiku --test              # verify backend connectivity and preprocessors
kaiku --test -b wcpp      # test a specific backend
```

### Config file

Config file is created at `~/.config/kaiku/config.yaml`. Locations searched in order:
1. `./kaiku.conf`
2. `~/.config/kaiku/config.yaml`
3. `~/.config/kaiku.conf`
4. `~/.kaiku.conf`

Note: The created config file embeds partially commented-out configuration options along with brief explanations for most features.
See [`kaiku.conf.example`](kaiku.conf.example) in the repo for a complete, current example with all backend and feature documentation.
The config file template is **not usable immediately**: You must update it based on your setup and needs.
The following sections tackle some of these in more detail where relevant.

## Audio

| Flag | Description |
|------|-------------|
| `--list-devices` | List available audio input devices with names and indices |
| `-d DEV / --device DEV` | Audio input device (name, ALSA name, or index). Overrides `audio_device` in config. |
| `-p NAME / --preprocessor NAME` | Audio preprocessor: `none`, `noisereduce`, `pyrnnoise`, `deepfilter`. Overrides config. |

### Audio device

```bash
kaiku --list-devices            # list available input devices with names and indices
kaiku -d "plughw:Snowball"      # use a specific ALSA device for this run
```

### System audio routing (recommended)

On Linux, set `audio_device` to `"pulse"` or `"pipewire"` and select the active microphone in `pavucontrol` or your desktop sound settings.

```yaml
audio_device: "pulse"              # PulseAudio routes to whichever mic is set as default input
```

### Targeting a specific device directly

```yaml
audio_device: "plughw:Snowball"    # ALSA plughw — format conversion included
audio_device: 3                    # device index from --list-devices
```

| Value | System | Notes |
|-------|--------|-------|
| `"pulse"` | [PulseAudio](https://www.freedesktop.org/wiki/Software/PulseAudio/) (Linux) | Recommended; configure mic via `pavucontrol` |
| `"pipewire"` | [PipeWire](https://pipewire.org/) (Linux) | Recommended on modern Linux |
| `"plughw:Snowball"` | [ALSA](https://alsa-project.org/) (Linux) | Direct USB mic access with format conversion |
| `"hw:2,0"` | ALSA (Linux) | Raw direct access, card 2 device 0 |
| `3` | Any | Device index from `--list-devices` |
| `"BlackHole 2ch"` | macOS | Virtual routing device |

### Audio preprocessing (noise reduction)

Audio preprocessing enhances a recording before transcription by filtering unwanted signal content.
**Noise reduction** is a key preprocessing technique that removes background sound — café chatter, fan hum, keyboard clicks — while preserving speech intelligibility.
Preprocessing is useful in noisy environments or when your [ASR backend](#asr-backends) struggles with poor signal quality, producing errors or hallucinations.
kaiku provides three noise reduction libraries, each with different strengths depending on noise type and available compute resources.

### Available preprocessors

| Name | Technology | Dependencies | Strengths | Weaknesses |
|------|-----------|------------|-----------|-----------|
| `none` | — | none (default) | No overhead; baseline for clean recordings | No noise reduction; ASR errors in noisy conditions |
| [`noisereduce`](https://github.com/timsainb/noisereduce) | Spectral subtraction | scipy only | Low CPU; scipy-only install; excellent for stationary noise (hum, AC, fans); live recording friendly | Ineffective on crowd/babble noise; may remove speech texture; limited to repeating noise patterns |
| [`pyrnnoise`](https://github.com/g-node/pyrnnoise) | Mozilla RNNoise GRU | scipy only | Handles non-stationary noise (crowd, footsteps, babble) better than spectral; learned noise patterns; no special hardware needed | 16→48→16 kHz resampling overhead on 16 kHz audio; slower than noisereduce |
| [`deepfilter`](https://github.com/Rikorose/DeepFilterNet) | DeepFilterNet3 neural | torch + Rust wheel | Best overall quality; handles mixed noise types; preserves speech naturalness and dynamics; recommended for important recordings | Highest CPU usage; heaviest dependencies; slowest option; overkill for live dictation |

### Choosing noise reduction by noise type and ASR backend

**ASR backend noise robustness:** Different backends have built-in resilience to noise due to their training data. Whisper models (including [`whisper.cpp`](#local-asr-whisper.cpp-vs-sherpa-onnx) and cloud APIs using `whisper-large-v3-turbo` like Groq) are trained on diverse YouTube videos containing natural background noise, making them inherently robust to non-stationary sounds. SenseVoice (via [`sherpa-onnx`](#local-asr-whisper.cpp-vs-sherpa-onnx)) is also fairly noise-tolerant. However, all backends still benefit from preprocessing in truly noisy settings (café, open office, street noise).

**Preprocessing strategy by scenario:**

1. **Clean or quiet environment** (home office, studio): Skip preprocessing (`none`) and let your backend's training handle any minor noise. Saves CPU and latency.

2. **Steady, repeating background noise** (office AC, ceiling fan, electrical hum): Use [`noisereduce`](https://github.com/timsainb/noisereduce). Spectral subtraction is highly effective on stationary patterns, adds minimal latency, and works well with any [ASR backend](#asr-backends). Ideal for live toggle recording.

3. **Variable, crowd noise** (meetings, café, office chatter): Use [`pyrnnoise`](https://github.com/g-node/pyrnnoise). Its neural GRU approach learns diverse noise patterns better than spectral methods. Note the 16→48→16 kHz resampling overhead on standard 16 kHz recordings — acceptable for file transcription, less ideal for live recording. Pairs well with Whisper-based backends, which are already trained on YouTube's ambient noise.

4. **Highest quality regardless of noise type** (important interviews, archival, transcription service): Use [`deepfilter`](https://github.com/Rikorose/DeepFilterNet). The deepest neural processing delivers the cleanest output, but requires substantial CPU and time. Reserve this for offline file processing where latency and compute cost can be amortized.

**Platform-specific guidance:**
- **Live toggle recording on CPU-constrained devices** (`--toggle`): Prefer `noisereduce` over `pyrnnoise` or `deepfilter` to minimize latency and system load.
- **Local [`whisper.cpp`](#local-asr-whisper.cpp-vs-sherpa-onnx) backend**: The backend is fast but already somewhat noise-robust; consider `none` in quiet settings or `noisereduce` for steady background. Avoid `deepfilter` on CPU systems.
- **Cloud API** (Groq, OpenAI): Optional preprocessing for mildly noisy audio; use preprocessing aggressively for café-grade noise to maximize transcription quality.
- **Speaker diarization** (`-b whisperx`): Preprocessing before diarization is recommended in noisy settings — speaker separation depends on clear voice boundaries, which noise obscures.

### Loudness normalisation

To complete the audio enhancement after noise reduction with any of the three preprocessors, `kaiku` applies a loudnorm pass (RMS → −20 dBFS, peak ceiling −0.1 dBFS) to ensure the ASR backend receives a consistently strong, unclipped signal.

### Installing preprocessors

```bash
pip3 install kaiku[noisereduce]   # spectral subtraction
pip3 install kaiku[pyrnnoise]     # RNNoise GRU
pip3 install kaiku[deepfilter]    # DeepFilterNet3
pip3 install kaiku[enhance]       # all three
```

### Preprocessor configuration

The preprocessor choice is determined using the first field of each [preset](#presets) spec. Override it for a single run with `-p NAME`.

### Preprocessor usage

```bash
kaiku -p deepfilter              # denoise live recording with DeepFilterNet
kaiku -p noisereduce             # spectral denoising
kaiku -p deepfilter -i talk.mp4  # denoise video file before transcription
kaiku --test                     # also checks that configured preprocessors are available
```

## Transcription

| Flag | Description |
|------|-------------|
| `-b NAME / --backend NAME` | ASR backend to use (key under `asr_backends:` in config). Overrides the preset's `asr_backend`. |
| `-i FILE / --input FILE` | Transcribe an existing audio or video file instead of recording. |
| `-o FILE / --output FILE` | Append transcripts to file. |
| `-l LANG / --language LANG` | Language hint (ISO-639-1, e.g. `fi`, `en`). Overrides config. Omit to auto-detect. |
| `-r / --robust` | Robust mode for `-i` file input: split at silence boundaries, quality-check chunks, retry. |
| `-C SEC / --chunk-duration SEC` | Max chunk duration in seconds for `-r/--robust` mode (default: 180). |
| `-g / --toggle` | Toggle recording: first invocation starts, second stops and transcribes. |
| `-z / --no-clipboard` | Do not copy transcript (or file path) to the system clipboard; stdout and `-o` unchanged. |

```bash
kaiku                  # record until Ctrl+C, transcribe, copy to clipboard
kaiku -l fi         # Finnish, using local whisper.cpp backend (privacy preset)
kaiku -i audio.mp3  # transcribe an existing audio file
kaiku -i meeting.mp4  # transcribe from a video file (audio extracted automatically)
kaiku -x speed -o transcript.txt   # preset + append transcript to a file
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
- In [robust/chunked mode (`-r`)](#robust-long-file-transcription), raw chunk text is appended to `-o` FILE as each chunk finishes (`tail -f` shows progress). After all chunks, the file is read back, optionally post-processed, formatted, and **replaced** with that final text only (no timestamped append).
- In [VAD mode (`--vad`)](#vad-continuous-recording) and [interval mode (`--interval`)](#vad-continuous-recording), each transcribed utterance is written immediately after the silence boundary triggers.

Clipboard size limit: when a transcript exceeds ~4 000 characters and `-o FILE` is specified, the file path is copied to clipboard instead of the full text.

### ASR backends

ASR (Automatic Speech Recognition) converts spoken audio into text. kaiku supports several ASR backends, from cloud APIs to fully offline local inference. Speaker diarization backends (`type: whisperx`) live here too — they replace the regular transcription step and produce speaker-attributed output.

#### ASR backend configuration

ASR backends are defined under `asr_backends:` and referenced by name from [presets](#presets) or overridden per-run with `-b NAME`. `kaiku --generate-config` writes a fully annotated config with every supported backend type.

```yaml
asr_backends:
  openai:
    type: api
    api_base_url: "https://api.openai.com/v1/"
    api_key: "YOUR_API_KEY"
    model_name: "whisper-1"
  groq:
    type: api
    api_base_url: "https://api.groq.com/openai/v1/"
    api_key: "YOUR_GROQ_KEY"
    model_name: "whisper-large-v3-turbo"
```

```bash
kaiku -b groq -i audio.wav          # use groq backend for this run
kaiku --test -b openai              # test a specific backend
```

#### Supported backend types

| `type` | Description | Requires |
|--------|-------------|---------|
| `api` | Any OpenAI-compatible HTTP endpoint ([OpenAI](https://platform.openai.com/docs/guides/speech-to-text), [Groq](https://console.groq.com/), [SiliconFlow](https://siliconflow.cn/), [xinference](https://inference.readthedocs.io/en/latest/), etc.) | API key or local server |
| `whisper_cpp` | whisper.cpp binary via subprocess | whisper.cpp build + `.bin` model file |
| `whisperx` | WhisperX speaker diarization — ASR + word alignment + speaker attribution in one pass; output: `[HH:MM:SS] SPEAKER_NN: text` | `pip3 install kaiku[diarize]`, HF token |
| `mock` | Fixed-response mock for testing and demos | None — no credentials needed |
| `mock-fwd` | Duration-proportional transcript mock (forward word order) | None |
| `mock-bwd` | Duration-proportional transcript mock (reverse word order) | None |
| `mock-diarize` | Mock diarization with round-robin speaker assignment | None |

#### Mock backends for testing

The mock backends return transcripts without making API calls or running external processes. Useful for development, demos, and CI pipelines:

```yaml
asr_backends:
  demo:
    type: mock
    response: "The quick brown fox jumps over the lazy dog"
    latency_ms: 100              # Optional: simulate network delay
  mock-fwd:
    type: mock-fwd
    transcript_path: "test_data/group-2p-2.txt"   # source for word pool
  mock-dia-2:
    type: mock-diarize
    speaker_count: 2
    transcript_path: "test_data/group-2p-2.txt"
```

```bash
kaiku -b demo -i dummy_audio.wav         # Returns mock transcript instantly
kaiku --test -b demo                     # No credentials needed
kaiku -i audio.wav -b mock-fwd           # Duration-proportional words from transcript
kaiku -i audio.wav -b mock-dia-2         # Mock diarization, 2 speakers
```

#### Local ASR: whisper.cpp vs. sherpa-onnx

Both provide fully offline, no-API-key ASR. Here is how to choose:

| | whisper.cpp | sherpa-onnx (via `--serve`) |
|---|---|---|
| **What it is** | C++ reimplementation of OpenAI Whisper | ONNX-runtime inference with Python bindings |
| **ASR models** | Whisper family (GGML quantised) | Whisper, [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), paraformer, zipformer, and more |
| **Language coverage** | 99 languages — full OpenAI Whisper training set | Varies by model; SenseVoice default covers ~50 languages (strongest for CJK); Whisper models via sherpa-onnx add 99 |
| **Multilingual quality** | Best local option for European and other non-English languages; `large-v3-turbo` recommended | SenseVoice leads for Chinese, Japanese, Korean and handles emotion/event detection; weaker for many European languages |
| **Python ML deps** | None — single binary + model file | Yes — ONNX runtime and sherpa-onnx Python packages |
| **Integration** | Subprocess call to external C++ binary | Python-native; exposes a local HTTP API |
| **Setup** | Build C++ from source; download `.bin` model manually | `pip3 install kaiku[vad]` + `kaiku --download-model` |
| **Model auto-download** | No | Yes |
| **VAD support** | No | Yes (built-in via sherpa-onnx) |
| **Dev activity** | Mature, stable | Very active (k2-fsa / Next-gen Kaldi team) |

**When to choose whisper.cpp:** Your primary language is non-English — especially European languages (Finnish, German, French, etc.) where `ggml-large-v3-turbo` delivers the best local accuracy of any backend. Also the right choice when you want no Python ML package dependencies: the binary and model file are self-contained, with nothing added to your Python environment.

**When to choose sherpa-onnx:** You are transcribing Chinese, Japanese, or Korean (SenseVoice is the stronger choice there), you want model auto-download and zero C++ build steps, you already installed `kaiku[vad]` (sherpa-onnx is already present), or you need VAD support or access to models beyond the Whisper family.

See [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for build instructions and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for its model zoo.

### Robust long-file transcription

For long recordings, `-r`/`--robust` splits at silence boundaries, quality-checks each chunk, retries bad chunks, streams raw chunk text to `-o` FILE as it goes, then overwrites FILE with the post-processed, template-formatted result:

```bash
kaiku -i meeting.mp3 -r                               # chunked, quality-checked
kaiku -i m.mp3 -rC 60                                 # 60 s chunks instead of default 180
kaiku -i m.mp3 -ro transcript.txt                     # tail -f during chunks; final file is formatted output only
kaiku -i m.mp3 -l fi -o t.txt                         # fully offline, Finnish language
kaiku -i m.mp3 -rP group-restructure -T bare -o t.md  # AI meeting memo without the transcript
```

Long transcripts often exceed the clipboard size limit; using `-o FILE` is recommended.

### Toggle mode

Toggle mode lets you bind a single keyboard shortcut to start and stop recording. The recording runs as a background process; the second invocation stops it, transcribes, and copies to clipboard. A desktop notification is shown on start and finish (requires [`notify-send`](https://man.archlinux.org/man/notify-send.1) on Linux).

```bash
kaiku --toggle                        # first press: start recording in background
kaiku --toggle                        # second press: stop, transcribe, copy to clipboard
kaiku --toggle                      # toggle with fully offline transcription
kaiku --toggle -P solo-restructure    # toggle → structured personal memo
```

Example awesome WM keybinding:

```lua
awful.key({ modkey }, "r", function()
    awful.spawn("kaiku --toggle")
end)
```

#### Requirements for toggle mode to work

Toggle mode requires a POSIX system (Linux or macOS). Windows is not supported: the recorder subprocess relies on `signal.pause()` (POSIX-only) and `SIGTERM` for graceful stop-and-write; the Windows equivalents would require reimplementing IPC with Win32 named pipes or events.

**Recorder backends** — Toggle mode needs audio to be captured in the background, and currently the alternative recorder backends are:

| Recorder | Platform | Requirement |
|---|---|---|
| `sounddevice` (default) | cross-platform | already a dependency |
| `arecord` | Linux / ALSA only | `alsa-utils` system package |

From these, `sounddevice` is used by default.
If you want direct ALSA access, set `recorder: arecord` in config.

## Local ASR server

`kaiku` can run a local OpenAI-compatible ASR API server backed by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

| Flag | Description |
|------|-------------|
| `--serve` | Start the local sherpa-onnx ASR API server |
| `--download-model` | Download the SenseVoice model and exit |
| `--host HOST` | Server bind address (default: 127.0.0.1) |
| `--port PORT` | Server bind port (default: 8000) |
| `--model-dir DIR` | Path to ASR model directory |
| `--num-threads N` | Inference threads (default: 4) |

```bash
pip3 install kaiku[vad]
kaiku --download-model                     # download SenseVoice model (~1 GB, once)
kaiku --serve                              # start server at 127.0.0.1:8000
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
| `--vad` | Continuous recording with voice activity detection. Transcribes when silence is detected after speech. Requires `pip3 install kaiku[vad]`. |
| `--interval SEC` | Continuous recording with fixed interval (seconds). |
| `--silence-threshold PROB` | Speech probability threshold, 0.0–1.0 (default: 0.5); lower = more sensitive. |
| `--silence-duration SEC` | How long silence must last to trigger transcription (default: 1.5 s). |

### Continuous recording modes

| Mode | Flag | Trigger | Use case |
|------|------|---------|----------|
| Voice Activity Detection | `--vad` | Silence after speech | Meetings, dictation, any unscripted speech |
| Fixed interval | `--interval SEC` | Every N seconds | Lectures, podcasts with predictable pauses |

```bash
kaiku --vad -o ~/meeting.txt          # auto-transcribe when silence is detected
kaiku --interval 60 -o ~/meeting.txt  # transcribe every 60 seconds
```

VAD requires [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):

```bash
pip3 install kaiku[vad]
```

VAD uses the [Silero VAD](https://github.com/snakers4/silero-vad) model (~629 KB, downloads automatically on first use). No internet connection required after the first run.

## Diarization

Speaker diarization attributes each spoken segment to a speaker label, producing a transcript where every turn is tagged `[HH:MM:SS] SPEAKER_NN: text`.
Diarization is implemented as an ASR backend (`type: whisperx`) — it replaces the regular transcription step entirely, so there is no double pass.

Speaker name substitution (SPEAKER_00 → real names) is intentionally left to the calling assistant or post-processor.

| Flag | Description |
|------|-------------|
| `-b whisperx` | WhisperX diarization backend (`type: whisperx`). |
| `-b` + `mock-diarize` | Mock diarization backend from your config (testing / CI). |
| `-s N / --speakers N` | Speaker count hint for `whisperx` and `mock-diarize`. Ignored by other backends. |

### Diarization setup

```bash
pip3 install kaiku[diarize]
# Accept the pyannote licence at https://huggingface.co/pyannote/speaker-diarization-3.1
# then set your HuggingFace token:
export HF_TOKEN=hf_...
```

**HF_TOKEN** is your [HuggingFace access token](https://huggingface.co/settings/tokens). You must also accept the license for [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace before the model can be downloaded. The token is only needed on first run; the model is cached locally thereafter.

### Diarization config

Define the backend among your [ASR backends](#asr-backends) and reference it in a [preset](#presets) or pass `-b whisperx`:

```yaml
asr_backends:
  whisperx:
    type: whisperx
    hf_token: "hf_..."       # or set HF_TOKEN env var
    min_speakers: 2          # optional hint to pyannote
    max_speakers: 6          # optional hint to pyannote
```

### Diarization usage

```bash
kaiku -i meeting.m4a -b whisperx           # diarize: SPEAKER_NN-attributed transcript
kaiku -i meeting.m4a -b whisperx -s 3      # hint: 3 speakers (improves accuracy)
kaiku -i meeting.m4a -b whisperx -P group  # diarize + LLM meeting notes
```

## Post-processing (with AI models)

Post-processing refines transcripts by passing them through an artificial intelligence system with custom instructions.
Use this to fix transcription errors, improve grammar, condense transcripts, or restructure them into consistently formatted memos with essential information extracted.
The feature is especially valuable for frequent dictators (researchers, journalists, managers) and teams with important discussions, decisions, tasks and timelines.

| Flag | Description |
|------|-------------|
| `-P NAME / --post NAME` | LLM post-processor name (key in `postprocessors:` config) or an inline system-prompt string. Overrides the preset's post-processor. |
| `-M MODEL / --post-model MODEL` | LLM model used for post-processing. Overrides the post-processor config for this run. |
| `-T NAME / --template NAME` | Output template name from `output_templates:` in config. Controls what is written to clipboard / `-o FILE`. |

### Mock post-processor for testing

The mock post-processor analyzes prompts and transcripts without making API calls, returning linguistic statistics. Useful for testing post-processing workflows, demos, and CI pipelines without credentials:

```yaml
postprocessor_backends:
  mock:
    type: mock
    model: Claude-Opus         # Required; becomes "Claude-Opus" in the signature

postprocessors:
  analyze:
    backend: mock
    prompt: "Enhance this transcript"
```

The output shows analysis of both the system prompt and the input transcript:

```
Prompt analyzed: longest=Enhance, shortest=this, most_frequent=enhance, lines=1, words=3, chars=24
Transcript analyzed: longest=transcription, shortest=a, most_frequent=the, lines=2, words=42, chars=245
*Yours truly, Claude-Opus
*
```

Each analysis includes:
- **longest**: longest word in the text
- **shortest**: shortest word in the text
- **most_frequent**: most common word (case-insensitive)
- **lines**: number of lines
- **words**: total word count
- **chars**: total character count

Usage:

```bash
kaiku -P analyze -i audio.mp3         # Analyze with mock
kaiku --test -P analyze                              # Test without credentials
```

### Post-processors (prompt templates)

Six post-processor specs are provided in the config template as examples; each is a starting point that requires configuration:
- **Setup a `postprocessor_backends:` entry** (Ollama, Groq, Anthropic API, OpenAI, Claude Code, or any OpenAI-compatible endpoint)
- **Assign that backend to the prompt** (via the `backend:` field) or set a default with `postprocessor_urgent` / `postprocessor_casual`
- **Update the context file list** via `context_path:` to help the LLM understand context if required; Delete it if no extra context neeeded.

Examples below; see the full configuration section for all available fields.

#### Personal dictation prompts

| Name | Purpose |
|------|---------|
| `solo-enhance` | Improve quality, fix grammar and word choice while honoring the author's style |
| `solo-restructure` | Restructure a personal dictation into a structured memo with sections |
| `solo-private` | Like `solo-restructure` but defaults to a local offline model to ensure privacy |

#### Group discussion prompts

| Name | Purpose |
|------|---------|
| `group-enhance` | Improve quality of group transcript while honoring each speaker's style |
| `group-restructure` | Restructure a group discussion into a meeting memo with summary, decisions, action items |
| `group-private` | Like `group-restructure` but defaults to a local offline model to ensure privacy |

Tips:
* To get richer and more accurate memos from meetings and debates, use a [diarization backend](#diarization) (`-b whisperx`) to attribute segments to speakers before post-processing.
* Add context files (see configuration below) to help the model better connect the dots between each individual participant and the contents discussed.

#### Post-processor usage examples

```bash
kaiku --toggle -P solo-enhance          # toggle → improved personal transcript
kaiku --toggle -P solo-restructure      # toggle → structured personal memo
kaiku -i meeting.m4a -b whisperx -P group-restructure  # diarize + meeting memo
kaiku --toggle -P "List action items."   # inline system prompt
```

### Supported AI backends

| Backend type | What it covers |
|---|---|
| `openai_compat` | [Ollama](https://ollama.com/) (local), [Groq](https://console.groq.com/), [Anthropic API](https://www.anthropic.com/api), [OpenAI](https://platform.openai.com/), any OpenAI-compatible endpoint |
| `claude_code` | [Claude Code](https://claude.ai/code) CLI — uses your CC session/subscription, no per-token billing |

### Post-processor backend setup

```yaml
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

### Post-processor configuration

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

**Context file formatting:** Files specified in `context_path:` are injected into the LLM prompt with an index and clear delimiters. Each file appears with its name and visual separators, making the context easily scannable for the LLM.

Example with inheritance:

```yaml
postprocessors:
  solo-base:
    backend: groq
    prompt: |
      You are a professional transcript scribe ...
    context_path:
      - "~/.kaiku/context/personal.md"     # Context file to help the LLM "read between the lines"

  solo-enhance:
    extends: solo-base            # inherits backend, prompt
    extra: |
      Also improve grammar and style ...

  solo-private:
    extends: solo-enhance        # inherits backend, prompt + extra
    backend: ollama               # override: use local model for privacy
    context_path:
      - "~/.kaiku/context/private-*.md"  # accumulates with parent's context
```

Note: The `solo-base` and `group-base` are not intended to be used directly. Instead, they provide definitions that are shared by other single-speaker and group discussion post-processors, respectively, through inheritance.

## Output templates

Output templates control what ends up in clipboard / `-o FILE`.
The output format depends on your template and potentially your LLM instructions — can be Markdown, [ReStructuredText](https://docutils.sourceforge.io/rst.html), plain text, or any text-based format.
Two templates are shipped; select with `-T NAME`:

```yaml
output_templates:
  raw: "{transcript}"                         # No post processing
  bare: "{result}"                            # Only the AI models output
  full: |                                     # Both the AI output and the transcript on a Markdown template
    {result}

    ---
    *Transcript from {duration_s:.0f}s recording post-processed at {datetime} with kaiku ({backend}, {prompt_name}, {model})*

    ## Original transcript

    {transcript}
```

```bash
kaiku -i m.mp3 -r -P group -T full      # meeting notes + full transcript appended
```

Available placeholders: `{result}` `{transcript}` `{date}` `{datetime}` `{prompt_name}` `{model}` `{backend}` `{duration_s}`

Set `postprocessor_urgent` / `postprocessor_casual` to apply a prompt automatically for every recording without passing `-P`.

## Presets

Presets are atomic pipeline definitions — each specifies exactly which preprocessor, ASR backend, and post-processor to use. Pick one preset per run; kaiku handles the rest.

**Format:** `preset_name: [preprocessor, asr_backend, postprocessor, description]`

- **Preprocessor** — audio denoising: `none`, `noisereduce`, `pyrnnoise`, or `deepfilter`
- **ASR backend** — any key from `asr_backends:` (including diarization backends like `type: whisperx`)
- **Postprocessor** — `none` or any key from `postprocessors:`
- **Description** — user-facing label shown by `--list` / `--test`

All four fields are required; this keeps every processing stage explicit.

```yaml
asr_backends:
  groq:
    type: api
    api_base_url: "https://api.groq.com/openai/v1/"
    api_key: "YOUR_GROQ_KEY"
    model_name: "whisper-large-v3-turbo"
  wcpp:
    type: whisper_cpp
    binary_path: "/usr/local/bin/whisper-cli"
    model_path: "/models/ggml-large-v3-turbo.bin"

presets:
  speed:     [     none,       groq,            none,  "Fast transcription with minimal processing"]
  quality:   [ deepfilter,     groq,            none,  "High-accuracy with neural denoising"]
  privacy:   [     none,       wcpp,            none,  "Fully offline, local transcription"]
  balanced:  [ noisereduce,    groq,            none,  "Good balance of speed, quality, and cost"]
  memo:      [ noisereduce,    groq,    solo-enhance,  "Transcription + LLM memo enhancement"]
```

```bash
kaiku --preset speed                  # record & transcribe, copied to clipboard
kaiku --preset privacy --toggle       # toggle mode with fully offline transcription
kaiku --preset quality -i audio.mp3   # transcribe file
kaiku --preset speed -b wcpp          # override backend for this run
kaiku --preset memo -P group-enhance  # override post-processor for this run
```

**Default preset:** set `default_preset: speed` in config to omit `--preset` on every invocation. Flags `-b`, `-p`, and `-P` still override individual stages.

```yaml
default_preset: speed
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Audio not captured | Run `kaiku --list-devices` and select a working device |
| Clipboard not working | Install `xclip` (X11) or `wl-clipboard` (Wayland) |
| API errors | Check your API key and endpoint in config |
| whisper.cpp errors | Run `kaiku --test -b wcpp`; check binary and model paths |
| Silent audio | Try a different audio device with `--device` |
| Video/audio format rejected | Ensure `ffmpeg` is installed (`apt install ffmpeg` / `brew install ffmpeg`) |
| Preprocessor not found | Run `kaiku --test` to see which are available and their install commands |
| Preprocessing too slow | Switch `preprocessor_urgent` to `noisereduce` or `none` in config |
| Post-processor not found | Check `postprocessors:` in config; name must match exactly |
| Post-processor backend error | Check `postprocessor_backends:` in config; verify API key and URL |
| Diarization fails | Ensure `pip3 install kaiku[diarize]`, `HF_TOKEN` is set, pyannote licence accepted, and backend `type: whisperx` is in `asr_backends:` |

Run `kaiku --test` (or `kaiku --test -b <name>`) to diagnose issues.

## Contributing

Fork the repository and submit a pull request.
Any improvements or new features are welcome! :)

### Testing

Development relies on a small but powerful **black-box E2E** suite: the real CLI runs as a subprocess and assertions cover exit codes, stdout, stderr, and files — see [`tests/README.md`](tests/README.md) for the full strategy, log-shape notes, and scenario index.

```bash
pytest tests/ -v
```

**Disclaimer**: Not all features have been properly tested and only on a legacy Ubuntu 20.04 environment. More testing and hardening will be done by June 2026.

## License

GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Related projects

kaiku operates within a four-stage pipeline:

```
[Audio capture] → [ASR / transcription] → [Post-processing] → [Output: clipboard / file]
     stage 1            stage 2                stage 3                 stage 4
```

The tables below cover the ecosystem at each pipeline stage and compare competing end-user tools. kaiku covers the whole pipeline in a single powerful CLI.

### Audio preprocessing (noise reduction)

Audio preprocessing cleans the signal before transcription. kaiku integrates all three libraries below as optional extras (`g install kaiku[enhance]`); they run in a pipeline with loudness normalisation applied after cleaning.

| Project | Technology | License | Best for | In kaiku |
|---------|-----------|---------|----------|-------------|
| [noisereduce](https://github.com/timsainb/noisereduce) | Spectral subtraction | MIT | Stationary noise: fans, AC, electrical hum | **Yes** — `pip3 install kaiku[noisereduce]` |
| [pyrnnoise](https://github.com/g-node/pyrnnoise) | Mozilla RNNoise GRU | GPL-3 | Non-stationary noise: crowd, babble, footsteps | **Yes** — `pip3 install kaiku[pyrnnoise]` |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | DeepFilterNet3 neural net | MIT | Best quality overall; speech naturalness; medium CPU | **Yes** — `pip3 install kaiku[deepfilter]` |
| [RNNoise](https://github.com/xiph/rnnoise) | Xiph GRU | BSD | Original Mozilla RNNoise (C library) | No — pyrnnoise wraps this at the Python layer |
| [SpeechBrain enhance](https://github.com/speechbrain/speechbrain) | Encoder-decoder neural | Apache-2 | Research-grade speech separation and denoising | No — heavy ML framework dependency; not practical as a live preprocessor |

### Voice Activity Detection

VAD classifies audio frames as speech or silence, enabling automatic segment boundaries without user interaction. kaiku uses Silero VAD (bundled in sherpa-onnx) for both the `--vad` continuous mode and the silence-split inside `--robust`.

| Project | License | Stars | Notes | In kaiku |
|---------|---------|-------|-------|-------------|
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | 14k+ | 629 KB model; enterprise-grade; ONNX + PyTorch; auto-downloads | **Yes** — via sherpa-onnx in `--vad` and `--robust` |
| [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) | BSD | 1k+ | Google's classic GMM-based VAD; very fast, lower accuracy | No — less accurate than Silero; not integrated |
| [pyannote VAD](https://github.com/pyannote/pyannote-audio) | MIT | 6k+ | Neural VAD embedded in the pyannote diarization pipeline | Indirectly — activated by the `whisperx` ASR backend |

### ASR engines

ASR engines convert audio to text. kaiku is a frontend: it delegates transcription to an ASR backend, supporting two locally-run backends (whisper.cpp, sherpa-onnx) and any OpenAI-compatible HTTP endpoint for cloud or self-hosted services. The engines listed as integrated below are ones kaiku directly calls or supports as backends; the others are libraries or specialized tools that require custom wrappers to use.

| Project | License | Stars | Best for | In kaiku |
|---------|---------|-------|----------|-------------|
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT | 80k+ | Gold standard; 99 languages; most widely reproduced | Via API (`whisper-1`) or indirectly through sherpa-onnx and whisper.cpp |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | MIT | 80k+ | Fully offline; best CPU performance; GGML-quantised models | **Yes** — `type: whisper_cpp` backend; subprocess call |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 15k+ | 4× faster than Whisper; identical accuracy; INT8/FP16 via CTranslate2 | Not directly; used internally by WhisperX and Meetily |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2 | 4k+ | ONNX inference; multi-model-family; model auto-download; Python-native | **Yes** — `--serve` local server; `type: api` backend |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD | 13k+ | Whisper + word-level timestamps + speaker diarization in one pipeline | **Yes** — `type: whisperx` backend (`pip3 install kaiku[diarize]`) |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | Apache-2 | 6k+ | Emotion + language event detection; excellent CJK | Via sherpa-onnx default model; also SiliconFlow API |
| [Vosk](https://github.com/alphacep/vosk-api) | Apache-2 | 8k+ | Lightweight; 20+ languages; embedded and low-RAM devices | No — lower accuracy than Whisper family |
| [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Apache-2 | (NeMo) | 3 380× faster than real-time; English only; GPU | No — English-only; GPU-dependent; no multilingual support |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | Apache-2 | 9k+ | Research platform; fine-tuning; custom model training | No — research library, not a drop-in backend |
| [Coqui STT](https://github.com/coqui-ai/STT) | MPL-2 | 5k+ | DeepSpeech successor; trainable on custom data | No — lower quality than Whisper; limited community activity |
| [Kaldi](https://github.com/kaldi-asr/kaldi) | Apache-2 | 14k+ | Enterprise/research; highly configurable; steep setup | No — complex; not a practical CLI backend |

### Speaker diarization

Speaker diarization labels each segment with a speaker identity ("who said what"). kaiku treats diarization as an ASR backend (`type: whisperx`); select it with `-b whisperx` or via a preset. WhisperX handles transcription and speaker attribution in one pass; pyannote.audio is used internally for speaker embedding and clustering.

| Project | License | Stars | Notes | In kaiku |
|---------|---------|-------|-------|-------------|
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT | 6k+ | De-facto OSS standard; speaker embedding + clustering; requires HF token for model download | Via WhisperX (`type: whisperx`) |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD | 13k+ | faster-whisper + word alignment + pyannote; all-in-one | **Yes** — `type: whisperx` backend (`pip3 install kaiku[diarize]`) |
| [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) | MIT | 2k+ | faster-whisper + pyannote script pipeline | No — WhisperX provides equivalent functionality with an active upstream |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Apache-2 | 13k+ | Fastest GPU diarization; English and enterprise focus | No — GPU-heavy; no practical CLI integration path |

### Desktop audio capture and transcription tools

These are end-user tools that combine audio capture, ASR, and transcript output — the closest category to kaiku itself.

| Project | Type | Platform | License | Live | File | Toggle | VAD | Offline | Diarize | LLM post | Notes |
|---------|------|----------|---------|------|------|--------|-----|---------|---------|----------|-------|
| **kaiku** (this) | CLI | Linux, macOS | AGPL-3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Full pipeline; scriptable; multi-backend; video input |
| [Turbo Whisper](https://github.com/knowall-ai/turbo-whisper) | GUI | Linux | MIT | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | faster-whisper-large-v3-turbo; global hotkey; no casual mode; PPA install |
| [Whispering](https://github.com/braden-w/whispering) | GUI/tray | Any | MIT | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | Cross-platform (snap/exe); local or cloud API; minimal UI |
| [Superwhisper](https://superwhisper.com/) | GUI | macOS, Windows, iOS | Proprietary | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | Partial | Premium dictation app; polished UX; no Linux |
| [Meetily](https://github.com/Zackriya-Solutions/meetily) | Desktop app | macOS, Windows | MIT | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | 11.9k★; Rust backend; Ollama summaries; no Linux |
| [Screenpipe](https://github.com/screenpipe/screenpipe) | Agent layer | Any | MIT | ✓ (always-on) | ✓ | ✗ | ✓ | ✓ | ✗ | Via MCP | 18.6k★; ambient 24/7 recording; MCP server; not a CLI tool |

### Agentic voice-to-text frameworks

For AI agents a streaming architecture is often preferred: continuous listening, local wake-word detection, and real-time VAD. This allows hands-free triggering of agentic actions and conversation that feels natural. In contrast, `kaiku` serves as a **high-precision bridge**: It is currently the most complete CLI-native pipeline for when an agent (or developer) needs to process a specific audio file or a manual "push-to-talk" segment with maximum control over all the processing stages. The table below lists some of the most notable choices for AI assistants:

| Project | Primary Tech | Wake Word | VAD | Agent Integration | vs. **kaiku** |
| --- | --- | --- | --- | --- | --- |
| [LiveKit Agents](https://github.com/livekit/agents) | Python / Rust | Optional | Silero | WebSocket / WebRTC | **Full Agent Framework:** High-performance streaming for voice-to-voice; `kaiku` is a CLI tool for text generation. |
| [Wyoming Satellite](https://www.google.com/search?q=https://github.com/home-assistant/wyoming-satellite) | Python | open-wakeword | Silero | Wyoming Protocol | **Smart Home Focus:** Designed as a background daemon for Home Assistant; `kaiku` is a foreground productivity tool. |
| [Rhasspy 3](https://github.com/rhasspy/rhasspy3) | Modular (C++/Python) | Porcupine / Snowboy | WebRTC / Silero | MQTT / Unix Sockets | **Deeply Modular:** Can swap every component; `kaiku` is more integrated and opinionated for CLI users. |
| [LocalAI](https://github.com/mudler/LocalAI) | Go / C++ | Yes (via API) | Yes | OpenAI-compatible API | **The Server Hub:** Acts as an all-in-one local API server; `kaiku` acts as a client that can call such servers. |
| [Whisper Mic](https://www.google.com/search?q=https://github.com/davabase/whisper_mic) | Python | No | Silero | Stdout / Text stream | **Simple Loop:** A continuous transcription script; lacks `kaiku`’s noise reduction, diarization, and post-processing. |
| [Leon](https://github.com/leon-ai/leon) | Node.js / Python | Yes | Yes | Custom SDK / Web | **Full Assistant:** Includes skills, memory, and UI; `kaiku` is a specialized "sensor" for such an assistant. |

### SaaS meeting assistants

Commercial cloud services that join calls automatically or process uploaded recordings. Included for context — these require trusting a third party with your audio.

| Service | Platform | Bot-less | Offline | Privacy | Notable | vs kaiku |
|---------|----------|----------|---------|---------|---------|-------------|
| [Fathom](https://fathom.video/) | Web (Zoom/Meet/Teams) | No | No | Cloud (US) | Free tier; calendar-integrated; good UX | Cloud-only; no file processing; no Linux CLI |
| [Jamie](https://meetjamie.ai/) | macOS, Windows | Yes | No | GDPR (EU) | Best Finnish quality; bot-free desktop app | €24+/mo; macOS/Windows only |
| [Fireflies.ai](https://fireflies.ai/) | Web | No | No | Cloud | 100 languages; CRM sync; "Ask Fred" AI queries | Cloud; BIPA lawsuit 2025; data outside EU |
| [Granola](https://granola.ai/) | macOS | Yes | Partial | Cloud for AI | Calendar-integrated; note editor; bot-free | macOS only; AI processing requires cloud |
| [Otter.ai](https://otter.ai/) | Web, mobile | No | No | Cloud (US) | Real-time captions; widely known | Class-action lawsuit 2025; weak Finnish |
| [Soniox](https://soniox.com/) | API | — | No | Cloud (US) | Best Finnish WER (10.6%); 56 languages; developer API | API service, not an end-user tool |
| [Krisp](https://krisp.ai/) | App + SDK | Yes | Partial | Cloud for AI | Industry-leading noise suppression + transcription | Proprietary; subscription; not scriptable |

### kaiku as an open source contribution

The speech-to-text tool landscape in 2026 has a sharp divide: powerful Python libraries (faster-whisper, WhisperX, pyannote) that require programming to use, and polished end-user apps (Superwhisper, Meetily, Granola) that are macOS/Windows-only or cloud-dependent. Linux users who want local, private, keyboard-shortcut-driven transcription with the full power of the Whisper ecosystem face a gap. Turbo Whisper and Whispering address the simplest dictation case but lack file transcription, noise reduction, robustness for long recordings, and any programmable post-processing. kaiku fills this gap as a single composable CLI that exposes the full four-stage pipeline without requiring the user to write any code.

Beyond Linux, kaiku's value is its scriptability and composability. Every feature — backend, preprocessor, language, diarization, post-processor, output template — is a flag or config key. This makes it naturally callable from shell scripts, Makefiles, cron jobs, and AI coding agents: one invocation covers the full audio → transcript → structured-memo pipeline that would otherwise require stitching together three or four Python libraries. The support for both local (whisper.cpp, sherpa-onnx) and cloud (OpenAI, Groq, SiliconFlow) backends with a unified interface means the same command works offline on a laptop and in a cloud pipeline on a headless server.

The most capable competing open-source project, Meetily, is architecturally similar in ambition — local-first, offline, Whisper-backed, with LLM summaries — but is a GUI-only desktop app for macOS and Windows with no Linux support and no CLI surface. Screenpipe is a different paradigm (always-on ambient capture) rather than a competing tool. This leaves kaiku as currently the most complete open-source, Linux-native, CLI-accessible speech processing pipeline — a category with no direct competition and clear utility for developers, power users, and autonomous AI agents that need to process human speech.
