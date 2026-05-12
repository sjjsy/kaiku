"""Configuration management for asr2clip."""

from __future__ import annotations

import os
import sys

from asr2clip._vendor.yaml import yaml
from asr2clip.utils import run_subprocess

# Default paths to search for config file (in order of priority)
CONFIG_PATHS = [
    "asr2clip.conf",  # Current directory
    os.path.expanduser("~/.config/asr2clip/config.yaml"),  # XDG style
    os.path.expanduser("~/.config/asr2clip.conf"),  # Legacy
    os.path.expanduser("~/.asr2clip.conf"),  # Home directory
]

# Default config location for new configs
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/asr2clip/config.yaml")


def _load_config_template() -> str:
    """Load config template from asr2clip.conf.example file."""
    # Try to find the template file relative to this module
    module_dir = os.path.dirname(__file__)
    template_path = os.path.join(os.path.dirname(module_dir), "asr2clip.conf.example")

    if os.path.exists(template_path):
        with open(template_path) as f:
            return f.read()

    # Fallback: empty template if file not found
    return "# asr2clip configuration template not found\n"


_CONFIG_TEMPLATE = _load_config_template()


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
            run_subprocess([editor, config_path], check=True)
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


def resolve_recorder_config(
    config: dict,
    cli_override: str | None = None,
) -> str:
    """Return the effective recorder name for toggle mode.

    Args:
        config: Full configuration dictionary.
        cli_override: Name supplied via a future --recorder flag (takes priority).

    Returns:
        Recorder name: 'auto', 'sounddevice', or 'arecord'.
    """
    if cli_override:
        return cli_override
    return config.get("recorder", "auto")


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
