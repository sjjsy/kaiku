"""Output handling for kaiku (clipboard, file, stdout)."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime

from .config_types import Config
from .utils import info, run_subprocess, success, warning

# Default UX threshold for clipboard text vs file-path fallback.
# Modern clipboards (X11 INCR, Wayland, macOS NSPasteboard, Windows) have no
# OS-imposed text limit — this constant is a pure usability heuristic. Override
# via 'clipboard_max_chars' in config. Set to 0 to always copy a file path.
# The limit of 50k corresponds to ~1 hour at 150 wpm (audiobook rate) at 5 cpw (English)
_DEFAULT_CLIPBOARD_MAX_CHARS = 50_000


def _is_wayland() -> bool:
    """Check if the current session is running on Wayland."""
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def _has_wl_copy() -> bool:
    """Check if wl-copy is available."""
    return shutil.which("wl-copy") is not None


def _wl_copy(text: str) -> bool:
    """Copy text using wl-copy (Wayland native, persists after exit).

    Args:
        text: Text to copy to the clipboard.

    Returns:
        True if successful, False otherwise.
    """
    try:
        run_subprocess(
            ["wl-copy"],
            input=text,
            text=True,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def check_clipboard_support() -> bool:
    """Check if clipboard support is available on the system.

    Returns:
        True if clipboard is supported, False otherwise.
    """
    if _is_wayland() and _has_wl_copy():
        return True
    try:
        import copykitten

        copykitten.copy("")
        return True
    except Exception:
        return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard.

    On Wayland, prefers wl-copy for clipboard manager integration.
    Falls back to copykitten on X11 or when wl-copy is unavailable.

    Args:
        text: Text to copy to the clipboard.

    Returns:
        True if successful, False otherwise.
    """
    # Wayland: prefer wl-copy for proper clipboard manager integration
    if _is_wayland() and _has_wl_copy():
        if _wl_copy(text):
            return True
        # Fall through to copykitten on failure

    try:
        import copykitten

        copykitten.copy(text, detach=True)
        return True
    except Exception as e:
        warning(f"Clipboard error: {e}")
        return False


def generate_timestamp_filename(
    prefix: str = "transcript", extension: str = "txt"
) -> str:
    """Generate a filename with a timestamp.

    Args:
        prefix: Prefix for the filename.
        extension: File extension.

    Returns:
        Filename with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def append_transcript_to_file(text: str, filepath: str):
    """Append transcript text to a file with timestamp.

    Args:
        text: Transcript text.
        filepath: Path to the output file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{text}\n")

    info(f"Appended transcript to file: {filepath}")


def _write_temp_transcript(text: str) -> str:
    """Write text to a temp file and return its path.

    The file lives in the system temp directory (/tmp on Linux/macOS,
    %TEMP% on Windows) and is cleaned up at reboot or by OS policy.
    """
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="kaiku_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def copy_transcript_to_clipboard(  # AGENTS: Do not touch this function!
    text: str,
    config: Config,
) -> bool:
    """Copy transcript to clipboard using ``config`` (limits, ``--no-clipboard``).

    When a path is copied (``clipboard_max_chars`` / length rules), it is
    ``config.output_file`` if set, otherwise a temp file holding ``text``.

    Behaviour depends on ``config.clipboard_max_chars``:
      value == 0   — always copy a file path (``config.output_file`` if set,
                     else a temp file).
      value > 0    — copy plain text when len(text) <= limit; otherwise copy a
                     path (same resolution as for value == 0).

    Returns:
        True if something was copied to clipboard, False otherwise.
    """
    if config.no_clipboard:
        info("Clipboard: skipped (--no-clipboard)")
        return False

    max_chars = config.clipboard_max_chars
    use_path = (max_chars == 0) or (max_chars > 0 and len(text) > max_chars)

    if not use_path:
        if copy_to_clipboard(text):
            success("Transcript text copied to clipboard")
            return True
        warning("Failed to copy transcript text to clipboard")
        return False

    # Resolve the file path to copy
    if config.output_file:
        # Note: With -o/--output, the transcript is appended before (robust.py) or after this returns (see output_transcript).
        path = os.path.abspath(config.output_file)
    else:
        # ... But here we need to write it to the temp file to honor clipboard_max_chars
        path = os.path.abspath(_write_temp_transcript(text))
        info(
            f"Transcript written to temp file: {path} — {len(text)} > {max_chars} & no -o FILE"
        )

    if copy_to_clipboard(path):
        success(f"Transcript file path copied to clipboard ({path})")
        return True

    warning("Failed to copy transcript file path to clipboard")
    return False


def output_transcript(text: str, config: Config, *, to_stdout: bool = True) -> None:
    """Copy to clipboard (unless ``config.no_clipboard``), print stdout, append ``-o`` file."""
    copy_transcript_to_clipboard(text, config)
    if to_stdout:
        print(text)
    if out := config.output_file:
        append_transcript_to_file(text, out)


def print_clipboard_help():
    """Print help message for clipboard setup."""
    print("\nClipboard is handled by copykitten (no external tools needed).")
    print("If clipboard is unavailable, use --output to save transcripts to a file.")
