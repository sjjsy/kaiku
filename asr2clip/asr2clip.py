#!/usr/bin/env python3
"""ASR to Clipboard - Record audio and transcribe to clipboard.

This is the main CLI entry point for asr2clip.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .audio import (
    convert_audio_to_wav,
    get_audio_duration,
    list_audio_devices,
    load_wav,
    record_audio,
    save_audio,
)
from .config import (
    generate_config,
    get_api_config,
    get_audio_device,
    open_in_editor,
    read_config,
    resolve_backend_config,
    resolve_backend_name,
    resolve_preprocessor_config,
)
from .daemon import continuous_recording
from .output import (
    check_clipboard_support,
    output_transcript,
    print_clipboard_help,
)
from .transcribe import test_transcription, transcribe_audio, transcribe_with_config
from .utils import (
    info,
    log,
    print_key_value,
    print_separator,
    safe_unlink,
    set_verbose,
    setup_logging,
    setup_signal_handlers,
    warning,
)


def test_config(
    backend_config: dict,
    full_config: dict | None = None,
    preprocessor_override: str | None = None,
) -> bool:
    """Test the configuration by checking backend connectivity and preprocessor availability.

    Args:
        backend_config: Resolved backend configuration dictionary.
        full_config: Full (unresolved) configuration dictionary for preprocessor checks.
        preprocessor_override: Name supplied via -P/--preprocessor flag.

    Returns:
        True if all configured components are accessible.
    """
    from .utils import print_error, print_success

    backend = backend_config.get("backend", "api")
    info(f"Testing configuration (backend: {backend})...")
    print_separator()

    if backend == "whisper_cpp":
        from .backends.whisper_cpp import WhisperCppConfig, test as wc_test
        cfg = WhisperCppConfig.from_config(backend_config)
        backend_ok = wc_test(cfg)
    else:
        api_key, api_base_url, model_name, org_id = get_api_config(backend_config)
        print_key_value("API Base URL", api_base_url)
        print_key_value("Model", model_name)
        masked_key = f"{'*' * 8}...{api_key[-4:] if len(api_key) > 4 else '****'}"
        print_key_value("API Key", masked_key)
        print_separator()
        backend_ok = test_transcription(api_key, api_base_url, model_name, org_id)

    if full_config is None:
        return backend_ok

    # Check preprocessors
    from .preprocessors import check_preprocessor_available

    print_separator()
    info("Checking pre-processors...")

    cfg_live = resolve_preprocessor_config(full_config, preprocessor_override, "live")
    cfg_file = resolve_preprocessor_config(full_config, preprocessor_override, "file")

    pp_ok = True
    for label, name in [("live", cfg_live), ("file", cfg_file)]:
        avail, hint = check_preprocessor_available(name)
        if avail:
            print_success(f"Preprocessor ({label}): {name}")
        else:
            print_error(
                f"Preprocessor ({label}): {name} — NOT AVAILABLE  ({hint})"
            )
            pp_ok = False

    return backend_ok and pp_ok


def process_recording(
    config: dict,
    device: str | int | None = None,
    output_file: str | None = None,
    language: str | None = None,
    preprocessor=None,
):
    """Record audio, transcribe, and output the result.

    Args:
        config: Configuration dictionary.
        device: Audio device name or index.
        output_file: Optional file to append transcript to.
        preprocessor: AudioPreprocessor instance to apply before transcription.
    """
    import time
    from .preprocessors import NonePreprocessor

    # Check clipboard support
    if not check_clipboard_support():
        warning("Clipboard support may not be available.")
        print_clipboard_help()

    setup_signal_handlers(daemon_mode=False)

    log("Recording... Press Ctrl+C to stop (press twice to cancel)")

    t0 = time.time()
    audio_data = record_audio(device=device)

    duration = get_audio_duration(audio_data)
    if duration < 0.1:
        log("Recording too short or empty. Exiting.")
        sys.exit(0)

    info(f"Recorded {duration:.1f}s of audio ({time.time() - t0:.1f}s elapsed)")

    if preprocessor is not None and not isinstance(preprocessor, NonePreprocessor):
        log(f"Pre-processing audio with {preprocessor.name}...")
        t_pre = time.time()
        audio_data = preprocessor.process(audio_data, 16000)
        info(f"Pre-processing completed in {time.time() - t_pre:.2f}s")

    log("Processing...")
    temp_path = save_audio(audio_data)

    try:
        t1 = time.time()
        text = transcribe_with_config(temp_path, config, language=language)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if text.strip():
            output_transcript(
                text,
                to_clipboard=True,
                to_stdout=True,
                to_file=output_file,
            )
        else:
            log("No speech detected in the recording.")

    finally:
        safe_unlink(temp_path)


def process_file(
    config: dict,
    input_file: str,
    output_file: str | None = None,
    language: str | None = None,
    preprocessor=None,
):
    """Transcribe an existing audio or video file.

    Args:
        config: Configuration dictionary.
        input_file: Path to the audio/video file.
        output_file: Optional file to append transcript to.
        preprocessor: AudioPreprocessor instance to apply before transcription.
    """
    import time
    from .preprocessors import NonePreprocessor

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    log(f"Processing file: {input_file}")

    if not input_file.lower().endswith(".wav"):
        t0 = time.time()
        log("Converting to WAV format...")
        temp_path = convert_audio_to_wav(input_file)
        info(f"Conversion completed in {time.time() - t0:.1f}s")
        cleanup_temp = True
    else:
        temp_path = input_file
        cleanup_temp = False

    # Apply pre-processing (load → process → re-save) when a real preprocessor is active
    if preprocessor is not None and not isinstance(preprocessor, NonePreprocessor):
        try:
            audio_data, sr = load_wav(temp_path)
            log(f"Pre-processing audio with {preprocessor.name}...")
            t_pre = time.time()
            audio_data = preprocessor.process(audio_data, sr)
            info(f"Pre-processing completed in {time.time() - t_pre:.2f}s")
            if cleanup_temp:
                os.unlink(temp_path)
            temp_path = save_audio(audio_data, sr)
            cleanup_temp = True
        except Exception as e:
            warning(f"Pre-processing failed ({e}), using audio as-is.")

    try:
        t1 = time.time()
        text = transcribe_with_config(temp_path, config, language=language)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if text.strip():
            output_transcript(
                text,
                to_clipboard=True,
                to_stdout=True,
                to_file=output_file,
            )
        else:
            log("No speech detected in the audio file.")

    finally:
        if cleanup_temp:
            safe_unlink(temp_path)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for asr2clip.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe to clipboard using ASR API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  asr2clip                             # Single recording (Ctrl+C to stop)
  asr2clip --vad                       # Continuous with voice detection
  asr2clip --interval 60               # Continuous with fixed interval
  asr2clip -i audio.mp3                # Transcribe existing audio file
  asr2clip -i meeting.mp4              # Transcribe from video (audio extracted automatically)
  asr2clip -d pulse                    # Record using PulseAudio default input (overrides config)
  asr2clip -d plughw:Snowball          # Record using a specific ALSA device (stable across reboots)

With output file:
  asr2clip --vad -o meeting.txt        # Save transcripts to file
  asr2clip -i audio.mp3 -o out.txt     # Transcribe file and save

Setup:
  asr2clip --edit                      # Create/edit configuration
  asr2clip --generate_config           # Create new config file (probes available enhancers)
  asr2clip --test                      # Test default backend and configured preprocessors
  asr2clip --list_devices              # List audio devices
  asr2clip --print_config              # Print config template (shows all options)

Audio pre-processing (noise reduction before transcription):
  asr2clip -P deepfilter               # Record + denoise with DeepFilterNet3 (best quality)
  asr2clip -P pyrnnoise                # Record + denoise with RNNoise (lowest latency)
  asr2clip -P noisereduce              # Record + spectral subtraction
  asr2clip -P none                     # Disable pre-processing for this run
  asr2clip -P deepfilter -i m.mp4      # File transcription with denoising
  asr2clip -P deepfilter -i m.mp4 -R   # Robust chunked + denoised
  asr2clip --test                      # Checks backend AND both configured preprocessors

Local ASR server (sherpa-onnx):
  asr2clip --serve                     # Start local ASR server on :8000
  asr2clip --serve --port 9000         # Start on custom port
  asr2clip --download-model            # Download SenseVoice model

Toggle mode (useful as a keyboard shortcut with WM keybinding):
  asr2clip --toggle                    # First call: start recording in background
  asr2clip --toggle                    # Second call: stop, transcribe, copy to clipboard
  asr2clip -P deepfilter --toggle      # Toggle mode with DeepFilterNet denoising

When whisper.cpp is installed and configured as 'backends.wcpp' with type 'whisper_cpp' in config:
  asr2clip -b wcpp --test                      # Test the local offline transcription configuration
  asr2clip -b wcpp                             # Single recording via whisper.cpp
  asr2clip -b wcpp -i a.wav                    # Transcribe file with whisper.cpp
  asr2clip -b wcpp -P deepfilter -i m.mp4      # Offline + denoised video transcription

Robust long-file transcription:
  asr2clip -i m.mp3 -R                         # Chunked transcription with quality checking
  asr2clip -i m.mp3 -R -C 60                   # Use 60 s chunks instead of default 180
  asr2clip -i m.mp3 -R -o t.txt                # Stream chunks to file (tail -f)
  asr2clip -i m.mp3 -R -b wcpp                 # Robust mode with whisper.cpp
  asr2clip -i m.mp3 -R -b wcpp -l fi           # Robust mode, Finnish, offline
  asr2clip -i m.mkv -R -P deepfilter -o t.txt  # Video + denoise + robust
""",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"asr2clip {__version__}"
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode - only output transcription and errors",
    )
    parser.add_argument(
        "-b",
        "--backend",
        metavar="NAME",
        default=None,
        help=(
            "Named backend to use (defined under 'backends:' in config). "
            "Overrides default_backend_live / default_backend_file in config."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        help=(
            "Transcribe an existing audio or video file instead of recording. "
            "Supported: wav, mp3, m4a, ogg, flac, aac, opus, wma, "
            "mp4, mov, mkv, webm, avi, flv, mvi"
        ),
        default=None,
    )
    parser.add_argument(
        "-P",
        "--preprocessor",
        metavar="NAME",
        default=None,
        help=(
            "Audio pre-processor to apply before transcription. "
            "Choices: none, noisereduce, pyrnnoise, deepfilter. "
            "Overrides preprocessor_live / preprocessor_file in config."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Append transcripts to file",
        default=None,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test backend connectivity and configured preprocessors, then exit",
    )
    parser.add_argument(
        "--list_devices", action="store_true", help="List available audio input devices"
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="DEV",
        help="Audio input device (name, ALSA name, or index). Overrides config.",
        default=None,
    )
    parser.add_argument(
        "-l", "--language",
        metavar="LANG",
        default=None,
        help=(
            "Language hint for transcription (ISO-639-1, e.g. 'fi', 'en'). "
            "Overrides config. Omit to auto-detect."
        ),
    )
    parser.add_argument(
        "-e", "--edit", action="store_true", help="Open configuration file in editor"
    )
    parser.add_argument(
        "--generate_config",
        action="store_true",
        help="Create config file at ~/.config/asr2clip/config.yaml",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print template configuration to stdout",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help=(
            "Continuous recording with voice activity detection (VAD). "
            "Transcribes automatically when silence is detected after speech. "
            "Requires sherpa-onnx: pip install asr2clip[vad]. "
            "The Silero VAD model (~629 KB) is downloaded automatically on first use."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        metavar="SEC",
        default=None,
        help="Continuous recording with fixed interval (seconds)",
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        metavar="PROB",
        default=None,
        help="VAD speech probability threshold, 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--silence_duration",
        type=float,
        metavar="SEC",
        default=1.5,
        help="Silence duration to trigger transcription (default: 1.5)",
    )

    parser.add_argument(
        "-R",
        "--robust",
        action="store_true",
        help=(
            "Robust mode for -i file input: split at silence boundaries, "
            "check quality, retry bad chunks, stream output (tail-f friendly)."
        ),
    )
    parser.add_argument(
        "-C",
        "--chunk-duration",
        type=int,
        metavar="SEC",
        default=180,
        help="Maximum chunk duration in seconds for --robust mode (default: 180)",
    )
    parser.add_argument(
        "--toggle",
        action="store_true",
        help=(
            "Toggle recording: first call starts, second call stops and transcribes. "
            "Designed for keyboard shortcuts (e.g. awesome WM keybinding)."
        ),
    )

    # Local ASR server
    server_group = parser.add_argument_group("Local ASR server")
    server_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the local ASR API server",
    )
    server_group.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1)",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port (default: 8000)",
    )
    server_group.add_argument(
        "--model-dir",
        default=None,
        help="Path to ASR model directory",
    )
    server_group.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Inference threads (default: 4)",
    )
    server_group.add_argument(
        "--download-model",
        action="store_true",
        help="Download the SenseVoice model and exit",
    )

    return parser


def _validate_args(args: argparse.Namespace):
    """Validate and resolve argument defaults.

    Args:
        args: Parsed arguments namespace (modified in-place).

    Raises:
        SystemExit: On invalid argument combinations.
    """
    if args.silence_threshold is None:
        args.silence_threshold = 0.5


def main():
    """Main entry point for asr2clip."""
    parser = _build_parser()
    args = parser.parse_args()

    # Handle --serve (local ASR server) — before validation
    if args.serve:
        from .local_asr import check_deps

        check_deps()
        from .local_asr.app import run_server

        run_server(
            host=args.host,
            port=args.port,
            model_dir=args.model_dir,
            num_threads=args.num_threads,
        )
        return

    # Handle --download-model — before validation
    if args.download_model:
        from .local_asr import check_deps

        check_deps()
        from .local_asr.model_registry import create_registry

        registry = create_registry(model_dir=args.model_dir)
        default_cfg = registry.get_default_model()
        registry.download_model(default_cfg)
        return

    _validate_args(args)

    # Handle --generate_config and --print_config
    if args.print_config:
        generate_config(print_only=True)
        return

    if args.generate_config:
        generate_config()
        return

    if args.edit:
        open_in_editor(args.config)
        return

    if args.list_devices:
        list_audio_devices()
        return

    # Read configuration
    config = read_config(args.config)

    # Set up logging
    quiet = args.quiet or config.get("quiet", False)
    setup_logging(verbose=not quiet)
    set_verbose(not quiet)

    # Resolve per-mode backend configs (live vs file, like preprocessor_live/file)
    backend_config_live = resolve_backend_config(config, args.backend, "live")
    backend_config_file = resolve_backend_config(config, args.backend, "file")

    live_name = resolve_backend_name(config, args.backend, "live")
    file_name = resolve_backend_name(config, args.backend, "file")
    if live_name and live_name == file_name:
        info(f"Using backend: {live_name}")
    elif live_name and file_name:
        info(f"Using backend (live): {live_name}  (file): {file_name}")
    elif live_name:
        info(f"Using backend: {live_name}")

    if args.test:
        if live_name == file_name or file_name is None:
            success = test_config(backend_config_live, config, args.preprocessor)
        else:
            info("--- live backend ---")
            ok_live = test_config(backend_config_live, config, args.preprocessor)
            info("--- file backend ---")
            ok_file = test_config(backend_config_file, config, args.preprocessor)
            success = ok_live and ok_file
        sys.exit(0 if success else 1)

    device = get_audio_device(config, args.device)

    # Resolve preprocessors once; pass objects into each code path
    from .preprocessors import make_preprocessor
    preprocessor_live = make_preprocessor(
        resolve_preprocessor_config(config, args.preprocessor, "live")
    )
    preprocessor_file = make_preprocessor(
        resolve_preprocessor_config(config, args.preprocessor, "file")
    )

    if preprocessor_live.name != "none":
        info(f"Live preprocessor: {preprocessor_live.name}")
    if preprocessor_file.name != "none":
        info(f"File preprocessor: {preprocessor_file.name}")

    if args.toggle:
        from .toggle import toggle_recording
        toggle_recording(
            backend_config_live, device, args.output,
            language=args.language,
            preprocessor=preprocessor_live,
        )
        return

    # File transcription takes priority over continuous modes
    if args.input:
        if args.robust:
            from .robust import process_file_robust
            process_file_robust(
                backend_config_file, args.input, args.output,
                chunk_duration=args.chunk_duration,
                language=args.language,
                preprocessor=preprocessor_file,
            )
        else:
            process_file(
                backend_config_file, args.input, args.output,
                language=args.language,
                preprocessor=preprocessor_file,
            )
        return

    if args.vad or args.interval is not None:
        api_key, api_base_url, model_name, org_id = get_api_config(backend_config_live)
        if args.vad:
            try:
                __import__("sherpa_onnx")
            except ImportError:
                print(
                    "Error: VAD requires sherpa-onnx.\n"
                    "Install with: pip install asr2clip[vad]"
                )
                sys.exit(1)
        interval = args.interval if args.interval is not None else 30.0
        continuous_recording(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
            org_id=org_id,
            device=device,
            interval=interval,
            output_file=args.output,
            vad_enabled=args.vad,
            silence_threshold=args.silence_threshold,
            silence_duration=args.silence_duration,
        )
        return

    process_recording(
        backend_config_live, device, args.output,
        language=args.language,
        preprocessor=preprocessor_live,
    )


if __name__ == "__main__":
    main()
