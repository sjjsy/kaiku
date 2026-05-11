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

_CONFIG_TEMPLATE_STATIC = """
# asr2clip configuration — uncomment and fill in one backend to start.
# Test with: asr2clip --test -b <name>   Switch at runtime: asr2clip -b <name>
backends:
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
default_backend: openai

# quiet: false                               # true = only output transcription and errors
# org_id:                                    # OpenAI organization ID (rarely needed)

# Audio input: "pulse"/"pipewire" uses whichever mic is set as default in system settings.
# Configure active mic in pavucontrol or your desktop sound panel.
# To force a specific device, run `asr2clip --list_devices` for names and indices.
# audio_device: "pulse"                      # PulseAudio (recommended on Linux)
# audio_device: "pipewire"                   # PipeWire (recommended on modern Linux)
# audio_device: "plughw:Snowball"            # ALSA device name — bypasses system mixer
# audio_device: 3                            # device index from --list_devices
"""


def _build_preprocessor_section() -> str:
    """Generate the preprocessor config block, probing installed libraries."""
    from .preprocessors import LATENCY_ORDER, QUALITY_ORDER, probe_available

    available = probe_available()
    not_installed = [p for p in QUALITY_ORDER if p not in available]

    detected_str = ", ".join(available) if available else "none"
    not_installed_str = ", ".join(not_installed) if not_installed else "all installed"

    best_file = next((p for p in QUALITY_ORDER if p in available), "none")
    best_live = next((p for p in LATENCY_ORDER if p in available), "none")

    lines = [
        "",
        "# --- Audio pre-processing ---",
        f"# Detected on this system: {detected_str}",
        f"# Not installed: {not_installed_str}",
        "#",
        "# Options and trade-offs:",
        "#   none        — no pre-processing, zero latency, no extra dependencies",
        "#   noisereduce — spectral subtraction (scipy), low CPU, no resampling needed at 16 kHz",
        "#                 best for stationary noise (fan, AC); install: pip install asr2clip[noisereduce]",
        "#   pyrnnoise   — Mozilla RNNoise GRU; requires 16→48→16 kHz resampling (scipy)",
        "#                 best for non-stationary noise (babble); install: pip install asr2clip[pyrnnoise]",
        "#   deepfilter  — DeepFilterNet3, best quality, medium CPU, torch-based (no scipy)",
        "#                 install: pip install asr2clip[deepfilter]",
        "#",
    ]

    if not available:
        lines.append(
            "# No enhancement libraries detected. Install one above to enable pre-processing."
        )
        lines.append("# Switch to 'none' if you notice added latency during recording.")
    elif best_live != best_file:
        lines.append(
            "# Using lower-latency option for live recordings, higher-quality for files."
        )
        lines.append("# Switch preprocessor_live to 'none' if recording latency increases.")
    else:
        lines.append("# Switch to 'none' if you notice added latency during recording.")

    lines += [
        "#",
        f"preprocessor_live: {best_live}   # applied to live recordings (toggle, single-shot, VAD)",
        f"preprocessor_file: {best_file}   # applied to file transcription (-i)",
        "#",
        "# Override at runtime:  asr2clip -P deepfilter -i meeting.mp4",
    ]

    return "\n".join(lines)


def _build_full_template() -> str:
    """Return the full config template with a dynamic preprocessor section."""
    return _CONFIG_TEMPLATE_STATIC.rstrip() + "\n" + _build_preprocessor_section() + "\n"


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
            f.write(_build_full_template().strip() + "\n")
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
    content = _build_full_template().strip() + "\n"

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


def resolve_backend_config(config: dict, backend_name: str | None = None) -> dict:
    """Return the effective backend config dict for transcription.

    Supports two formats:

    New (named backends):
        backends:
          groq:
            type: api
            api_key: ...
          local:
            type: whisper_cpp
            binary: ...
            model: ...
        default_backend: groq

    Legacy (flat, backward-compatible):
        backend: whisper_cpp   # or omit for api
        api_key: ...
        whisper_cpp:
          binary: ...

    Args:
        config: Full configuration dictionary.
        backend_name: Override which named backend to use (from --backend flag).

    Returns:
        A flat config dict recognised by transcribe_with_config().

    Raises:
        SystemExit: If the requested backend name is not defined.
    """
    backends = config.get("backends")
    if not backends:
        if backend_name:
            print(
                f"Error: --backend '{backend_name}' specified but config has no named backends.\n"
                "Add a 'backends:' section to your config file, for example:\n\n"
                "  default_backend: groq\n"
                "  backends:\n"
                "    groq:\n"
                "      type: api\n"
                "      api_key: YOUR_KEY\n"
                "      ...\n"
                "    local:\n"
                "      type: whisper_cpp\n"
                "      binary: ~/path/to/whisper-cli\n"
                "      model:  ~/path/to/model.bin\n"
            )
            import sys
            sys.exit(1)
        return config

    name = backend_name or config.get("default_backend") or next(iter(backends))
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
    """Return the names of all configured backends, or [] for legacy configs."""
    return list(config.get("backends", {}).keys())


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
