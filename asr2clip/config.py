"""Configuration management for asr2clip."""

from __future__ import annotations

import os
import subprocess
import sys

from asr2clip._vendor.yaml import yaml

# Default paths to search for config file (in order of priority)
CONFIG_PATHS = [
    "asr2clip.conf",  # Current directory
    os.path.expanduser("~/.config/asr2clip/config.yaml"),  # XDG style
    os.path.expanduser("~/.config/asr2clip.conf"),  # Legacy
    os.path.expanduser("~/.asr2clip.conf"),  # Home directory
]

# Default config location for new configs
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/asr2clip/config.yaml")

_CONFIG_TEMPLATE = """
# asr2clip configuration — uncomment and fill in one backend to start.
# Test with: asr2clip --test -b <name>   Switch at runtime: asr2clip -b <name>

## ── Audio input ───────────────────────────────────────────────────────────────
# "pulse"/"pipewire" uses whichever mic is set as default in system settings.
# Configure active mic in pavucontrol or your desktop sound panel.
# To force a specific device, run `asr2clip --list_devices` for names and indices.
# audio_device: "pulse"                       # PulseAudio (recommended on Linux)
# audio_device: "pipewire"                    # PipeWire (recommended on modern Linux)
# audio_device: "plughw:Snowball"             # ALSA device name — bypasses system mixer
# audio_device: 3                             # device index from --list_devices

## ── Audio preprocessing (noise reduction) ────────────────────────────────────
# Options: none, noisereduce, pyrnnoise, deepfilter
#   none        — no preprocessing, zero latency, no extra dependencies
#   noisereduce — spectral subtraction; best for stationary noise (fan, AC)
#                 install: pip install asr2clip[noisereduce]
#   pyrnnoise   — Mozilla RNNoise GRU; best for non-stationary noise (babble)
#                 install: pip install asr2clip[pyrnnoise]
#   deepfilter  — DeepFilterNet3, best quality overall, medium CPU
#                 install: pip install asr2clip[deepfilter]
# Override at runtime: asr2clip -p deepfilter -i meeting.mp4
preprocessor_live: none                       # applied to live/toggle; consider noisereduce
preprocessor_file: none                       # applied to -i FILE; consider deepfilter

## ── ASR backends ──────────────────────────────────────────────────────────────
asr_backends:
  # openai:                                   # OpenAI Whisper API
  #   type: api
  #   api_base_url: "https://api.openai.com/v1/"
  #   api_key: "sk-..."
  #   model_name: "whisper-1"
  #   # language: en                          # ISO-639-1; omit to auto-detect
  # groq:                                     # Groq — fast, free tier available
  #   type: api
  #   api_base_url: "https://api.groq.com/openai/v1/"
  #   api_key: "YOUR_GROQ_KEY"
  #   model_name: "whisper-large-v3-turbo"
  # siliconflow:                              # SiliconFlow
  #   type: api
  #   api_base_url: "https://api.siliconflow.com/v1/"
  #   api_key: "YOUR_API_KEY"
  #   model_name: "FunAudioLLM/SenseVoiceSmall"
  # xinference:                               # xinference or other self-hosted endpoint
  #   type: api
  #   api_base_url: "http://localhost:9997/v1/"
  #   api_key: "any"                          # required field, value not checked
  #   model_name: "SenseVoiceSmall"
  # sonnx:                                    # local sherpa-onnx server (pip install asr2clip[vad])
  #   type: api
  #   api_base_url: "http://127.0.0.1:8000/v1/"
  #   model_name: "SenseVoiceSmall"
  # wcpp:                                     # whisper.cpp — build from source, fully offline
  #   type: whisper_cpp
  #   binary: ~/path/to/whisper.cpp/build/bin/whisper-cli
  #   model:  ~/path/to/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin
  #   # language: auto                        # 'auto' or ISO-639-1 (e.g. fi, en)
  #   # threads: 4

## Default ASR backend selection (override with -b NAME):
asr_backend_live: sonnx                       # backend for live/toggle/VAD recording
asr_backend_file: wcpp                        # backend for file transcription (-i FILE)

## ── Diarization ───────────────────────────────────────────────────────────────
## Speaker diarization via WhisperX. Requires: pip install asr2clip[diarize]
## Enable per run with -D / --diarize. Replaces the configured ASR backend.
## Output: "[HH:MM:SS] SPEAKER_NN: text" — name substitution is left to the caller.
# diarize_hf_token: "hf_..."                  # HuggingFace token for pyannote model download
#                                             # or set HF_TOKEN env var
# diarize_min_speakers: 2                     # optional hint to pyannote; omit for auto
# diarize_max_speakers: 6

## ── LLM post-processing ───────────────────────────────────────────────────────
# Post-processors send the transcript to an LLM and return structured output.
# Enable a post-processor for a mode, or pass -P NAME per run.
# Requires postprocessor_backends and postprocessors sections below.

## LLM backends for post-processing:
# postprocessor_backends:
#   ollama:                                   # Ollama — free, private, fully offline
#     type: openai_compat
#     api_base_url: "http://localhost:11434/v1/"
#     api_key: "ollama"                       # Ollama ignores the key value
#     model: "qwen3:14b"
#   groq:                                     # Groq — fast, generous free tier
#     type: openai_compat
#     api_base_url: "https://api.groq.com/openai/v1/"
#     api_key: "YOUR_GROQ_KEY"
#     model: "llama-3.3-70b-versatile"
#   anthropic:                                # Anthropic API (OpenAI-compatible endpoint)
#     type: openai_compat
#     api_base_url: "https://api.anthropic.com/v1/"
#     api_key_env: ANTHROPIC_API_KEY          # reads key from environment variable
#     model: "claude-haiku-4-5-20251001"
#   cc:                                       # Claude Code CLI — uses your CC subscription
#     type: claude_code                       # no api_key needed; uses active CC session
#     model: "claude-haiku-4-5-20251001"

## Prompt library — add as many prompts as you like.
# Each prompt is available via -P NAME or as the postprocessor_live/file default.
# Fields per prompt:
#   prompt:        system prompt text (required unless 'extends' is used)
#   extends:       inherit system prompt from another named prompt in this config
#   extra:         text appended after the inherited prompt (used with extends)
#   backend:       which postprocessor_backends entry to use (default: first defined)
#   model:         override the backend's default model for this prompt
#   template:      output template name from output_templates below (default: "default")
#   context_path:  list of file glob patterns; contents are injected into the LLM request
postprocessors:
  solo-base:
    prompt: |
      You are a professional personal transcript scribe.
      Your client used a transcriber tool that may have produced sub-standard quality.
      The recording should only contain your client's (= author/user) voice.
#   backend: groq
    template: bare

  solo-enhanced:
    extends: solo-base
    extra: |
      The content may be intended to constitute a message, an email, a command, a personal diary or note entry, or an unstructured thinking-aloud segment, or something else.
      Produce an improved transcript that honors the author's original choice of words, structures and flow but fix misspellings, misinterpretations and grammatical mistakes, reduce unnecessary words and repetition and otherwise improve the quality and legibility of the transcript.

  solo-restructured:
    extends: solo-base
    extra: |
      The content may be from a long unstructured dictation or thinking-aloud session for some purpose (which may be deducible from the transcript itself).
      Produce a concise, structured memo that maximizes legibility and usefulness.
      The following provides a template. Omit sections that would have no clear content.

      # YYYY-MM-DD HH:SS Solo Notes on <Topic>
      Tags: <tags>

      ## Summary
      <2-3 sentences>

      ## Key points
      <Bulleted list>

      ## Open questions
      <Bulleted list>

      ## Progress and action items
      <[x] Action that is already done
      [/] Action that is in progress (by person A, due by date B, if specified)
      [ ] Action that is to be done but not yet started>
#   context_path:
#     - "~/.asr2clip/context/personal.md"     # include personal context info to improve reasoning

  solo-private:                               # extends solo-restructured with privacy instructions
    extends: solo-restructured
#   backend: ollama                           # use local ollama model for sensitive content
#   context_path:
#     - "~/.asr2clip/context/personal.md"
#     - "~/.asr2clip/context/private.md"      # include private info to improve reasoning

  group:
    prompt: |
      You are a professional meeting notetaker.
      Your team used a transcriber tool that may have produced sub-standard quality.
      Produce a concise, structured, legible memo that promotes flow for the whole team.
      The following provides a template. Omit sections that would have no clear content.

      # YYYY-MM-DD HH:SS Meeting on <Topic>
      Tags: <tags>

      ## Summary
      <2-6 sentences, depending on length and breadth of discussion>

      ## Key points
      <Bulleted list; Important to get all essential discussion contributions covered>

      ## Decision made
      <Bulleted list>

      ## Open questions
      <Bulleted list; flag minority positions explicitly; unresolved disagreements are more important than resolved ones>

      ## Progress and action items
      <[x] Action that is already done
      [/] Action that is in progress (by person A, due by date B, if specified)
      [ ] Action that is to be done but not yet started>
#   context_path:
#     - "~/project_x/README.md"
#     - "~/project_x/todo.md"
#   backend: anthropic
#   model: claude-sonnet-4-6
    template: full

## Which post-processor to apply automatically (none = disabled):
postprocessor_live: none                      # toggle / VAD / single live recording
postprocessor_file: none                      # -i file transcription

## ── Output templates ──────────────────────────────────────────────────────────
## Named templates controlling what gets copied to clipboard / written to -o FILE.
## Placeholders: {result} {transcript} {date} {datetime}
##               {prompt_name} {model} {backend} {duration_s}
output_templates:
  bare: "{result}"
  full: |
    {result}

    ---

    *Transcript from {duration_s:.0f}s recording post-processed at {datetime} with asr2clip ({backend}, {prompt_name}, {model})*

    ## Original transcript

    {transcript}

## ── Other parameters ──────────────────────────────────────────────────────────
# quiet: false                                # true = only output transcription and errors
# org_id:                                     # OpenAI organization ID (rarely needed)
"""


def find_config_path(config_file: str | None = None) -> str | None:
    """Find the configuration file path.

    Args:
        config_file: Optional path to a specific config file.

    Returns:
        Path to the config file if found, None otherwise.
    """
    if config_file and os.path.exists(config_file):
        return config_file

    for path in CONFIG_PATHS:
        if os.path.exists(path):
            return path

    return None


def read_config(config_file: str) -> dict:
    """Read and parse the configuration file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        SystemExit: If the config file is not found or cannot be read.
    """
    config_path = find_config_path(config_file)

    if config_path is None:
        print("Configuration file not found.")
        print("\nSearched locations:")
        for path in CONFIG_PATHS:
            print(f"  - {path}")
        print("\nTo create a new configuration file, run:")
        print("    asr2clip --generate_config")
        print("\nOr create and edit directly:")
        print("    asr2clip --edit")
        sys.exit(1)

    try:
        with open(config_path) as file:
            config = yaml.load(file.read())
            # Handle legacy config format
            if "asr_model" in config and len(config) == 1:
                return config["asr_model"]
            return config
    except Exception as e:
        print(f"Could not read configuration file {config_path}: {e}")
        sys.exit(1)


def open_in_editor(config_file: str | None = None):
    """Open the configuration file in the system's default editor.

    Args:
        config_file: Optional path to a specific config file.

    Raises:
        SystemExit: If no suitable editor is found.
    """
    config_path = find_config_path(config_file)

    # If no config exists, create a default one
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        config_dir = os.path.dirname(config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        with open(config_path, "w") as f:
            f.write(_CONFIG_TEMPLATE.strip() + "\n")
        print(f"Created new config file: {config_path}")

    # Determine which editor to use
    editors_to_try = []
    if os.getenv("EDITOR"):
        editors_to_try.append(os.getenv("EDITOR"))

    if os.name == "nt":  # Windows
        editors_to_try.append("notepad")
    else:  # Unix-like
        editors_to_try.extend(["nano", "vi", "vim"])

    for editor in editors_to_try:
        try:
            subprocess.run([editor, config_path], check=True)
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Failed to open editor '{editor}': {e}")
            sys.exit(1)

    print(f"No suitable editor found. Please edit manually: {config_path}")
    sys.exit(1)


def generate_config(
    output_path: str | None = None, force: bool = False, print_only: bool = False
) -> str | None:
    """Generate a template configuration file.

    The preprocessor section is built dynamically by probing installed libraries,
    so the generated file reflects what is actually available on this system.

    Args:
        output_path: Path to write the config file. If None, uses DEFAULT_CONFIG_PATH.
        force: If True, overwrite existing file.
        print_only: If True, print to stdout instead of writing to file.

    Returns:
        Path to the generated config file, or None if print_only.
    """
    content = _CONFIG_TEMPLATE.strip() + "\n"

    if print_only:
        print(content)
        return None

    if output_path is None:
        output_path = DEFAULT_CONFIG_PATH

    # Check if file already exists
    if os.path.exists(output_path) and not force:
        print(f"Config file already exists: {output_path}")
        print("Use --edit to modify it, or delete it first to regenerate.")
        return output_path

    # Create directory if needed
    config_dir = os.path.dirname(output_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    # Write config file
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Created config file: {output_path}")
    print("\nEdit it with your API credentials:")
    print("    asr2clip --edit")
    print("\nOr edit directly:")
    print(f"    $EDITOR {output_path}")

    return output_path


def resolve_preprocessor_config(
    config: dict,
    cli_override: str | None = None,
    mode: str = "live",
) -> str:
    """Return the effective preprocessor name for the given mode.

    Args:
        config: Full configuration dictionary.
        cli_override: Name supplied via -P/--preprocessor flag (takes priority).
        mode: 'live' for microphone recording, 'file' for -i file input.

    Returns:
        Preprocessor name string ('none', 'noisereduce', 'pyrnnoise', 'deepfilter').
    """
    if cli_override:
        return cli_override
    key = "preprocessor_live" if mode == "live" else "preprocessor_file"
    # Fall back to single 'preprocessor' key, then to 'none'
    return config.get(key, config.get("preprocessor", "none"))


def resolve_backend_name(
    config: dict,
    backend_name: str | None = None,
    mode: str = "live",
) -> str | None:
    """Return the effective backend name for the given mode.

    Args:
        config: Full configuration dictionary.
        backend_name: CLI override (takes priority).
        mode: 'live' for microphone recording, 'file' for -i file input.

    Returns:
        Backend name string, or None if config has no asr_backends section.
    """
    backends = config.get("asr_backends")
    if not backends:
        return None
    if backend_name:
        return backend_name
    mode_key = "asr_backend_live" if mode == "live" else "asr_backend_file"
    return config.get(mode_key) or next(iter(backends))


def resolve_backend_config(
    config: dict,
    backend_name: str | None = None,
    mode: str = "live",
) -> dict:
    """Return the effective backend config dict for transcription.

    Supports two formats:

    Named backends format:
        asr_backends:
          groq:
            type: api
            api_key: ...
          local:
            type: whisper_cpp
            binary: ...
            model: ...
        asr_backend_live: groq    # live/toggle/VAD recording
        asr_backend_file: local   # -i file transcription

    Args:
        config: Full configuration dictionary.
        backend_name: Override which named backend to use (from --backend flag).
        mode: 'live' for microphone recording, 'file' for -i file input.

    Returns:
        A flat config dict recognised by transcribe_with_config().

    Raises:
        SystemExit: If the requested backend name is not defined.
    """
    backends = config.get("asr_backends")
    if not backends:
        print(
            "Error: config has no 'asr_backends:' section.\n"
            "Run 'asr2clip --generate_config' to create a fresh config, for example:\n\n"
            "  asr_backends:\n"
            "    groq:\n"
            "      type: api\n"
            "      api_key: YOUR_KEY\n"
            "      ...\n"
            "  asr_backend_live: groq\n"
            "  asr_backend_file: groq\n"
        )
        import sys
        sys.exit(1)

    name = resolve_backend_name(config, backend_name, mode)
    if name not in backends:
        available = ", ".join(backends)
        print(f"Error: backend '{name}' not found in config. Available: {available}")
        import sys
        sys.exit(1)

    defn = backends[name]
    btype = defn.get("type", "api")

    if btype == "api":
        return {
            "backend": "api",
            "api_key": defn.get("api_key", os.environ.get("OPENAI_API_KEY")),
            "api_base_url": defn.get("api_base_url", "https://api.openai.com/v1/"),
            "model_name": defn.get("model_name", "whisper-1"),
            "org_id": defn.get("org_id", os.environ.get("OPENAI_ORG_ID")),
            "language": defn.get("language"),
        }
    elif btype == "whisper_cpp":
        return {
            "backend": "whisper_cpp",
            "whisper_cpp": {k: v for k, v in defn.items() if k != "type"},
        }
    else:
        print(f"Error: unknown backend type '{btype}' for backend '{name}'.")
        import sys
        sys.exit(1)


def list_backends(config: dict) -> list[str]:
    """Return the names of all configured ASR backends."""
    return list(config.get("asr_backends", {}).keys())


def get_api_config(config: dict) -> tuple[str, str, str, str | None]:
    """Extract API configuration from config dictionary.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (api_key, api_base_url, model_name, org_id).
    """
    api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY"))
    api_base_url = config.get("api_base_url", "https://api.openai.com/v1")
    model_name = config.get("model_name", "whisper-1")
    org_id = config.get("org_id", os.environ.get("OPENAI_ORG_ID"))
    return api_key, api_base_url, model_name, org_id


def get_audio_device(config: dict, cli_device: str | None = None) -> str | int | None:
    """Get audio device from config or CLI argument.

    Args:
        config: Configuration dictionary.
        cli_device: Optional device specified via CLI.

    Returns:
        Audio device name, index, or None for default.
    """
    device = cli_device if cli_device is not None else config.get("audio_device", None)

    # Convert device to int if it's a numeric string
    if device is not None and isinstance(device, str) and device.isdigit():
        device = int(device)

    return device
