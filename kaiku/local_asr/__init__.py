"""Local ASR server providing an OpenAI-compatible transcription API."""

_INSTALL_HINT = (
    "Local ASR server dependencies are not installed. "
    "Install them with: pip install kaiku[vad]"
)


def check_deps() -> None:
    """Verify that server dependencies are available.

    Raises:
        ImportError: If any required dependency is missing.
    """
    missing = []
    for pkg in ("fastapi", "uvicorn", "sherpa_onnx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise ImportError(f"{_INSTALL_HINT}\n  Missing: {', '.join(missing)}")


def cli_main() -> None:
    """Entry point for the ``kaiku-serve`` console script."""
    check_deps()
    from .app import run_server_cli

    run_server_cli()
