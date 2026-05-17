"""Configuration management for kaiku."""

from __future__ import annotations

import os
import re
import sys

from kaiku._vendor.yaml import yaml
from kaiku.utils import run_subprocess, warning

# Default paths to search for config file (in order of priority)
CONFIG_PATHS = [
    "kaiku.conf",  # Current directory
    os.path.expanduser("~/.config/kaiku/config.yaml"),  # XDG style
    os.path.expanduser("~/.config/kaiku.conf"),  # Legacy
    os.path.expanduser("~/.kaiku.conf"),  # Home directory
]

# Default config location for new configs
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/kaiku/config.yaml")


def _load_config_template() -> str:
    """Load config template bundled with the installed package."""
    import importlib.resources as resources

    try:
        return resources.read_text("kaiku", "kaiku.conf.example", encoding="utf-8")
    except (FileNotFoundError, OSError):
        return "# kaiku configuration template not found\n"


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


def read_config(config_file: str | None = None) -> tuple[str, dict]:
    """Read and parse the configuration file.

    Args:
        config_file: Path from ``-c``/``--config``, or ``None`` to use
            :data:`CONFIG_PATHS` discovery (first existing file wins).

    Returns:
        ``(absolute_path, config_dict)``. The path is the file that was read;
        use it in logs so users see which file supplies keys such as
        ``clipboard_max_chars`` (a common confusion is editing XDG
        ``~/.config/kaiku/config.yaml`` while a ``./kaiku.conf`` in the
        current working directory takes precedence).

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
        print("    kaiku --generate-config")
        print("\nOr create and edit directly:")
        print("    kaiku --edit")
        sys.exit(1)

    resolved_abs = os.path.abspath(config_path)

    try:
        with open(config_path, encoding="utf-8") as file:
            source = file.read()
        raw = yaml.load(source)
        if not isinstance(raw, dict):
            raw = {}
    except Exception as e:
        print(f"Could not read configuration file {config_path}: {e}")
        sys.exit(1)

    # Verify whether all top level keys were parsed by YAML, because a comment breaking a
    # text block can break the document, and this check might save the user some headache
    top_level_keys: set[str] = set()
    for line in source.splitlines():
        if line and line[0] not in " \t#" and (m := re.match(r"^([A-Za-z_][\w-]*)\s*:", line)):
            top_level_keys.add(m.group(1))
    if missing := sorted(top_level_keys - raw.keys()):
        warning(f"Config {resolved_abs}: top-level key(s) in file but not loaded by YAML: {', '.join(missing)}")
        warning(f"The issue might be a comment line '# ...' breaking a text block ': |'. Double check recommended!")
    return resolved_abs, raw


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
    print("    kaiku --edit")
    print("\nOr edit directly:")
    print(f"    $EDITOR {output_path}")

    return output_path
