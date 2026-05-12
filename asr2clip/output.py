"""Output handling for asr2clip (clipboard, file, stdout)."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime

from .utils import log, print_success, run_subprocess, warning

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
        text: Text to copy.

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
        text: Text to copy to clipboard.

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
    """Generate a filename with timestamp.

    Args:
        prefix: Prefix for the filename.
        extension: File extension.

    Returns:
        Filename with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def append_transcript_to_file(text: str, filepath: str):
    """Append transcript text to a file with timestamp.

    Args:
        text: Transcript text to append.
        filepath: Path to the output file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{text}\n")

    log(f"Appended transcript to {filepath}")


def _write_temp_transcript(text: str) -> str:
    """Write text to a temp file and return its path.

    The file lives in the system temp directory (/tmp on Linux/macOS,
    %TEMP% on Windows) and is cleaned up at reboot or by OS policy.
    """
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="asr2clip_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def copy_transcript_to_clipboard(
    text: str,
    output_file: str | None = None,
    max_chars: int = _DEFAULT_CLIPBOARD_MAX_CHARS,
) -> bool:
    """Copy transcript to clipboard.

    Behaviour depends on max_chars (configured via 'clipboard_max_chars'):
      max_chars == 0   — always copy a file path (output_file if given, else a
                         temp file in the system temp directory).
      max_chars > 0    — copy the text when len(text) <= max_chars; otherwise
                         copy the file path (output_file) when one exists, or
                         copy the full text when no file path is available.

    Args:
        text: Transcript text.
        output_file: Path to the output file, if one was written.
        max_chars: Character threshold; 0 = always use file path.

    Returns:
        True if something was copied to clipboard, False otherwise.
    """
    use_path = (max_chars == 0) or (max_chars > 0 and len(text) > max_chars)

    if not use_path:
        if copy_to_clipboard(text):
            print_success("Copied to clipboard")
            return True
        warning("Failed to copy to clipboard")
        return False

    # Resolve the file path to copy
    if output_file:
        path = os.path.abspath(output_file)
    elif max_chars == 0:
        path = _write_temp_transcript(text)
        log(f"Transcript written to temp file (clipboard_max_chars=0): {path}")
    else:
        # max_chars > 0 but text too long and no output file — copy full text anyway
        if copy_to_clipboard(text):
            print_success("Copied to clipboard")
            return True
        warning("Failed to copy to clipboard")
        return False

    if copy_to_clipboard(path):
        log(f"File path copied to clipboard: {path}")
        return True
    warning("Failed to copy file path to clipboard")
    return False


def output_transcript(
    text: str,
    to_clipboard: bool = True,
    to_stdout: bool = True,
    to_file: str | None = None,
    max_clipboard_chars: int = _DEFAULT_CLIPBOARD_MAX_CHARS,
):
    """Output transcript to various destinations.

    Args:
        text: Transcript text to output.
        to_clipboard: Whether to copy to clipboard.
        to_stdout: Whether to print to stdout.
        to_file: Optional file path to append transcript to.
        max_clipboard_chars: Passed to copy_transcript_to_clipboard.
    """
    if to_clipboard:
        copy_transcript_to_clipboard(text, to_file if to_file else None, max_clipboard_chars)

    if to_stdout:
        print(text)

    if to_file:
        append_transcript_to_file(text, to_file)


def print_clipboard_help():
    """Print help message for clipboard setup."""
    print("\nClipboard is handled by copykitten (no external tools needed).")
    print("If clipboard is unavailable, use --output to save transcripts to a file.")
