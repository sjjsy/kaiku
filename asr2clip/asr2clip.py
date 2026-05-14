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
    open_in_editor,
)
from .config_types import Config
from .daemon import continuous_recording
from .output import (
    check_clipboard_support,
    output_transcript,
    print_clipboard_help,
)
from .transcribe import test_openai_compat_connection, test_transcription, transcribe
from .utils import (
    error,
    info,
    print_key_value,
    print_separator,
    safe_unlink,
    set_verbose,
    setup_logging,
    setup_signal_handlers,
    warning,
)


def test_config(config: Config) -> bool:
    """Test the configuration by checking backend connectivity and preprocessor availability.

    Args:
        config: Resolved Config instance.

    Returns:
        True if all configured components are accessible.
    """
    from .utils import error, success

    info(f"Testing configuration (backend: {config.asr_backend.name})...")
    print_separator()

    if config.asr_backend.type == "whisper_cpp":
        from .backends.whisper_cpp import WhisperCppConfig, test as wc_test
        cfg = WhisperCppConfig(
            binary=config.asr_backend.binary,
            model=config.asr_backend.model,
            threads=config.asr_backend.threads or 4,
        )
        backend_ok = wc_test(cfg)
    elif config.asr_backend.type in ("mock", "mock-fwd", "mock-bwd", "mock-diarize"):
        success(
            f"Backend '{config.asr_backend.name}' ({config.asr_backend.type}) — no API connectivity check"
        )
        backend_ok = True
    else:
        print_key_value("API Base URL", config.asr_backend.api_base_url or "")
        print_key_value("Model", config.asr_backend.model_name or "")
        ak = config.asr_backend.api_key or ""
        masked_key = f"{'*' * 8}...{ak[-4:] if ak and len(ak) > 4 else '****'}"
        print_key_value("API Key", masked_key)
        print_separator()
        backend_ok = test_transcription(config)

    # Check preprocessor
    from .preprocessors import check_preprocessor_available

    print_separator()
    info("Checking preprocessors...")

    avail, hint = check_preprocessor_available(config)
    if avail:
        success(f"Preprocessor: {config.preprocessor}")
        pp_ok = True
    else:
        error(f"Preprocessor: {config.preprocessor} — NOT AVAILABLE  ({hint})")
        pp_ok = False

    return backend_ok and pp_ok


def _test_postprocessors(config: Config) -> bool:
    """Check that post-processor backends are reachable. Returns True if all OK."""
    import shutil
    from .postprocessors import _resolve_backend, _resolve_prompt
    from .utils import error, success

    print_separator()
    info("Checking post-processors...")

    if not config.postprocessor.name or config.postprocessor.name == "none":
        success("No post-processors configured to check")
        return True

    try:
        resolved = _resolve_prompt(config.postprocessor.name, config._config_dict)
        backend_cfg = _resolve_backend(
            config._config_dict, resolved["backend_name"], config.postprocessor.model_override
        )
    except SystemExit:
        error(f"Post-processor '{config.postprocessor.name}' — config error (see above)")
        return False

    ok = True
    btype = backend_cfg["type"]
    model = backend_cfg.get("model", "")
    model_note = f"  model: {model}" if model else ""

    if btype == "claude_code":
        if shutil.which("claude"):
            success(f"Post-processor '{config.postprocessor.name}': claude_code{model_note}")
            success("  claude CLI: found")
        else:
            error(f"Post-processor '{config.postprocessor.name}': claude_code{model_note}")
            error("  claude CLI: NOT FOUND — install from https://claude.ai/code")
            ok = False

    elif btype == "openai_compat":
        api_base = backend_cfg.get("api_base_url", "")
        api_key = backend_cfg.get("api_key", "sk-none")
        success(f"Post-processor '{config.postprocessor.name}': openai_compat{model_note}")
        print_key_value("  Endpoint", api_base)
        if not test_openai_compat_connection(api_key, api_base, model, None):
            ok = False

    elif btype == "mock":
        success(f"Post-processor '{config.postprocessor.name}': mock (no credentials needed)")

    else:
        error(f"Post-processor '{config.postprocessor.name}' — unknown backend type '{btype}'")
        ok = False

    return ok


def _test_clipboard() -> bool:
    """Check clipboard availability. Returns True if OK."""
    from .utils import error, success
    import shutil, os

    print_separator()
    info("Checking clipboard...")

    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        success("wl-copy available (Wayland)")
        return True
    try:
        import copykitten
        copykitten.copy("")
        success("copykitten available (X11 / fallback)")
        return True
    except Exception as e:
        error(f"Clipboard unavailable: {e}")
        error("  Install wl-copy (Wayland) or ensure copykitten works on X11")
        return False


def _test_diarization(config: Config) -> None:
    """Check diarization dependencies. Prints warnings; never blocks overall success."""
    import importlib.util
    from .utils import success, warning

    print_separator()
    info("Checking diarization (optional)...")

    if importlib.util.find_spec("whisperx") is None:
        warning("whisperx: not installed  →  pip install asr2clip[diarize]")
        warning("  (backend type 'whisperx' will not work until installed)")
        return

    success("whisperx: installed")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        masked = f"{'*' * 8}...{hf_token[-4:]}" if len(hf_token) > 4 else "****"
        success(f"HuggingFace token (HF_TOKEN): {masked}")
    else:
        warning(
            "HuggingFace token: not set\n"
            "  Set HF_TOKEN env var or add 'hf_token: hf_...' to the whisperx\n"
            "  entry in asr_backends."
        )


def process_recording(config: Config):
    """Record audio, transcribe, and output the result."""
    import time
    from .postprocessors import NonePostProcessor, format_output, PostMetadata, make_postprocessor, resolve_output_template
    from .preprocessors import NonePreprocessor, make_preprocessor

    if not check_clipboard_support():
        warning("Clipboard support may not be available.")
        print_clipboard_help()

    if config.recorder.device.mock_source:
        audio_data, _sr = load_wav(config.recorder.device.mock_source)
        duration = get_audio_duration(audio_data)
        info(f"Mock device: loaded {duration:.1f}s from {config.recorder.device.mock_source}")
    else:
        setup_signal_handlers(daemon_mode=False)
        info("Recording... Press Ctrl+C to stop (press twice to cancel)")
        t0 = time.time()
        audio_data = record_audio(device=config.recorder.device.get_spec(config.recorder.name))
        duration = get_audio_duration(audio_data)
        if duration < 0.1:
            error("Recording too short or empty. Exiting.")
            sys.exit(0)
        info(f"Recorded {duration:.1f}s of audio ({time.time() - t0:.1f}s elapsed)")

    preprocessor = make_preprocessor(config)
    if not isinstance(preprocessor, NonePreprocessor):
        t_pre = time.time()
        audio_data = preprocessor.process(audio_data, 16000)
        info(f"Preprocessing completed in {time.time() - t_pre:.2f}s")

    temp_path = save_audio(audio_data)

    try:
        t1 = time.time()
        transcript = transcribe(temp_path, config)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if not transcript.strip():
            info("No speech detected in the recording.")
            return

        from datetime import date
        postprocessor = make_postprocessor(config)
        template = resolve_output_template(config)
        metadata = PostMetadata(
            date=date.today().isoformat(),
            duration_s=duration,
            language=config.language or "auto",
            prompt_name=postprocessor.name,
            diarized=config.asr_backend.type in ("whisperx", "mock-diarize"),
            source="recording",
        )
        if not isinstance(postprocessor, NonePostProcessor):
            info(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
        else:
            result = transcript
        final = format_output(
            template, result=result, transcript=transcript,
            metadata=metadata, model=postprocessor.model,
            backend=postprocessor.backend_type,
        )

        output_transcript(final, config)

    finally:
        safe_unlink(temp_path)


def process_file(config: Config):
    """Transcribe an existing audio or video file."""
    import time
    from .postprocessors import NonePostProcessor, PostMetadata, format_output, make_postprocessor, resolve_output_template
    from .preprocessors import NonePreprocessor, make_preprocessor

    if not config.input_file or not os.path.exists(config.input_file):
        print(f"File not found: {config.input_file}")
        sys.exit(1)

    info(f"Processing file: {config.input_file}")

    if config.input_file.lower().endswith(".txt"):
        with open(config.input_file, encoding="utf-8") as fh:
            transcript = fh.read()
        if not transcript.strip():
            info("Transcript file is empty.")
            return
        from datetime import date
        postprocessor = make_postprocessor(config)
        template = resolve_output_template(config)
        metadata = PostMetadata(
            date=date.today().isoformat(),
            duration_s=0.0,
            language=config.language or "auto",
            prompt_name=postprocessor.name,
            diarized=False,
            source="file",
        )
        if not isinstance(postprocessor, NonePostProcessor):
            info(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
        else:
            result = transcript
        final = format_output(
            template, result=result, transcript=transcript,
            metadata=metadata, model=postprocessor.model,
            backend=postprocessor.backend_type,
        )
        output_transcript(final, config)
        return

    if not config.input_file.lower().endswith(".wav"):
        t0 = time.time()
        info("Converting to WAV format...")
        temp_path = convert_audio_to_wav(config.input_file)
        info(f"Conversion completed in {time.time() - t0:.1f}s")
        cleanup_temp = True
    else:
        temp_path = config.input_file
        cleanup_temp = False

    preprocessor = make_preprocessor(config)
    if not isinstance(preprocessor, NonePreprocessor):
        try:
            audio_data, sr = load_wav(temp_path)
            t_pre = time.time()
            audio_data = preprocessor.process(audio_data, sr)
            info(f"Preprocessing completed in {time.time() - t_pre:.2f}s")
            if cleanup_temp:
                os.unlink(temp_path)
            temp_path = save_audio(audio_data, sr)
            cleanup_temp = True
        except Exception as e:
            warning(f"Preprocessing failed ({e}), using audio as-is.")

    try:
        t1 = time.time()
        transcript = transcribe(temp_path, config)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if not transcript.strip():
            info("No speech detected in the audio file.")
            return

        try:
            import wave as _wave
            with _wave.open(temp_path) as wf:
                duration_s = wf.getnframes() / wf.getframerate()
        except Exception:
            duration_s = 0.0

        from datetime import date
        postprocessor = make_postprocessor(config)
        template = resolve_output_template(config)
        metadata = PostMetadata(
            date=date.today().isoformat(),
            duration_s=duration_s,
            language=config.language or "auto",
            prompt_name=postprocessor.name,
            diarized=config.asr_backend.type in ("whisperx", "mock-diarize"),
            source="file",
        )
        if not isinstance(postprocessor, NonePostProcessor):
            info(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
        else:
            result = transcript
        final = format_output(
            template, result=result, transcript=transcript,
            metadata=metadata, model=postprocessor.model,
            backend=postprocessor.backend_type,
        )

        output_transcript(final, config)

    finally:
        if cleanup_temp:
            safe_unlink(temp_path)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for asr2clip."""
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe to clipboard using ASR API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  asr2clip --edit                             # create/open config in editor
  asr2clip --test                             # verify backend and preprocessors
  asr2clip                                    # record, transcribe, copy to clipboard
  asr2clip --toggle                           # toggle recording (for keyboard shortcuts)
  asr2clip --toggle -P solo-restructure       # toggle, and produce AI-structured memo
  asr2clip -i audio.mp3                       # transcribe an existing file
  asr2clip -i m.mp3 -p deepfilter -r          # neural denoising + chunked transcription
  asr2clip -i meeting.m4a -b whisperx -s 3    # speaker diarization, 3-speaker hint
  asr2clip --serve                            # start local sherpa-onnx ASR server
  asr2clip --vad -o meeting.txt               # continuous VAD transcription to file
  asr2clip --interval 60                      # fixed-interval continuous recording

See https://github.com/sjjsy/asr2clip for full documentation and configuration examples.
""",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"asr2clip {__version__}"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode — only output transcription and errors",
    )

    # Setup
    setup_group = parser.add_argument_group("Setup")
    setup_group.add_argument(
        "-c", "--config", metavar="FILE",
        help="Path to configuration file",
        default=None,
    )
    setup_group.add_argument(
        "-e", "--edit", action="store_true",
        help="Open configuration file in editor (creates default config if missing)",
    )
    setup_group.add_argument(
        "--generate-config", action="store_true",
        help="Write config template to ~/.config/asr2clip/config.yaml",
    )
    setup_group.add_argument(
        "--print-config", action="store_true",
        help="Print config template to stdout",
    )
    setup_group.add_argument(
        "--test", action="store_true",
        help="Test backend connectivity and configured preprocessors, then exit",
    )
    setup_group.add_argument(
        "-x", "--preset", metavar="NAME",
        default=None,
        help=(
            "Pipeline preset name (key under 'presets:' in config). "
            "Presets define complete pipelines: ASR backend, preprocessor, post-processor. "
            "Optional if 'default_preset' is set in config; CLI overrides still work (-b, -p, -P)."
        ),
    )

    # Audio
    audio_group = parser.add_argument_group("Audio")
    audio_group.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices",
    )
    audio_group.add_argument(
        "-d", "--device", metavar="DEV",
        help="Audio input device (name, ALSA name, or index). Overrides config.",
        default=None,
    )
    audio_group.add_argument(
        "-i", "--input", metavar="FILE",
        help=(
            "Transcribe an existing audio or video file instead of recording. "
            "Supported: wav, mp3, m4a, ogg, flac, aac, opus, wma, "
            "mp4, mov, mkv, webm, avi, flv, mvi"
        ),
        default=None,
    )
    audio_group.add_argument(
        "-p", "--preprocessor", metavar="NAME",
        default=None,
        help=(
            "Audio preprocessor: none, noisereduce, pyrnnoise, deepfilter. "
            "Overrides the preprocessor in the selected preset."
        ),
    )

    # Transcription
    trans_group = parser.add_argument_group("Transcription")
    trans_group.add_argument(
        "-b", "--backend", metavar="NAME",
        default=None,
        help=(
            "ASR backend to use (key under 'asr_backends:' in config). "
            "Overrides the backend in the selected preset."
        ),
    )
    trans_group.add_argument(
        "-l", "--language", metavar="LANG",
        default=None,
        help=(
            "Language hint for transcription (ISO-639-1, e.g. 'fi', 'en'). "
            "Overrides config. Omit to auto-detect."
        ),
    )
    trans_group.add_argument(
        "-r", "--robust", action="store_true",
        help=(
            "Robust mode for -i file input: split at silence boundaries, "
            "quality-check chunks, retry failures, stream output (tail-f friendly)."
        ),
    )
    trans_group.add_argument(
        "-C", "--chunk-duration", type=int, metavar="SEC",
        default=None,
        help="Max chunk duration in seconds for -r/--robust mode (default: 180)",
    )
    trans_group.add_argument(
        "-g", "--toggle", action="store_true",
        help=(
            "Toggle recording: first call starts, second call stops and transcribes. "
            "Designed for keyboard shortcuts."
        ),
    )

    # Local ASR server
    server_group = parser.add_argument_group("Local ASR server")
    server_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the local sherpa-onnx ASR API server",
    )
    server_group.add_argument(
        "--host",
        default=None,
        help="Server bind address (default: 127.0.0.1 or local_asr.host in config)",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=None,
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
        default=None,
        help="Inference threads (default: 4)",
    )
    server_group.add_argument(
        "--download-model",
        action="store_true",
        help="Download the SenseVoice model and exit",
    )

    # VAD (continuous recording)
    vad_group = parser.add_argument_group("VAD (continuous recording)")
    vad_group.add_argument(
        "--vad", action="store_true",
        help=(
            "Continuous recording with voice activity detection. "
            "Transcribes automatically when silence is detected after speech. "
            "Requires sherpa-onnx: pip install asr2clip[vad]."
        ),
    )
    vad_group.add_argument(
        "--interval", type=float, metavar="SEC",
        default=None,
        help="Continuous recording with fixed interval (seconds)",
    )
    vad_group.add_argument(
        "--silence-threshold", type=float, metavar="PROB",
        default=None,
        help="VAD speech probability threshold, 0.0-1.0 (default: 0.5)",
    )
    vad_group.add_argument(
        "--silence-duration", type=float, metavar="SEC",
        default=None,
        help="Silence duration to trigger transcription (default: 1.5 s)",
    )

    # Diarization
    diarize_group = parser.add_argument_group("Diarization")
    diarize_group.add_argument(
        "-s", "--speakers", type=int, metavar="N",
        default=None,
        help=(
            "Speaker count hint for diarization backends (type: whisperx, type: mock-diarize). "
            "Ignored by all other backends. "
            "Selects a diarization backend in your preset or with -b / --backend; "
            "see 'asr_backends:' in config. "
            "If omitted, the backend uses its own default or auto-detects speaker count."
        ),
    )

    # Post-processing
    post_group = parser.add_argument_group("Post-processing")
    post_group.add_argument(
        "-P", "--post", metavar="NAME",
        default=None,
        help=(
            "AI post-processor name (key in 'postprocessors:' config) "
            "or an inline system-prompt string. "
            "Requires 'postprocessor_backends:' in config. "
            "Overrides the post-processor in the selected preset."
        ),
    )
    post_group.add_argument(
        "-M", "--post-model", metavar="MODEL",
        default=None,
        help=(
            "AI model used for the post-processing (f. ex. claude-sonnet-4-6). "
            "Overrides the post-processor config for this run."
        ),
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", metavar="FILE",
        help="Append transcripts to file",
        default=None,
    )
    output_group.add_argument(
        "-T", "--template", metavar="NAME",
        default=None,
        help=(
            "Output template name from 'output_templates:' in config. "
            "Controls what is written to clipboard/-o FILE. "
            "Overrides the template specified in the prompt definition."
        ),
    )
    output_group.add_argument(
        "-z", "--no-clipboard", action="store_true",
        help=(
            "Do not copy the transcript (or a file path) to the system clipboard. "
            "Stdout and -o output behave as usual."
        ),
    )

    return parser


def main():
    """Main entry point for asr2clip."""
    parser = _build_parser()
    args = parser.parse_args()

    quiet = args.quiet
    setup_logging(verbose=not quiet)
    set_verbose(not quiet)

    # Handle --generate-config and --print-config
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

    config = Config.from_file(args.config, args)

    if args.robust and not args.input:
        error(
            "-r/--robust requires -i/--input: chunked transcription applies only to "
            "audio/video files, not live microphone recording, --toggle, or VAD/interval mode."
        )
        sys.exit(1)

    if args.serve:
        from .local_asr import check_deps
        check_deps()
        from .local_asr.app import run_server
        run_server(config)
        return

    if args.download_model:
        from .local_asr import check_deps
        check_deps()
        from .local_asr.model_registry import create_registry
        registry = create_registry(
            config_path=config.local_asr.models_config_path, model_dir=config.local_asr.model_dir
        )
        registry.download_model(registry.get_default_model())
        return

    if args.test:
        ok_asr = test_config(config)
        ok_post = _test_postprocessors(config)
        ok_clip = _test_clipboard()
        _test_diarization(config)
        print_separator()
        checks_ok = ok_asr and ok_post and ok_clip
        info("All checks passed." if checks_ok else "Some checks failed — see details above.")
        sys.exit(0 if checks_ok else 1)

    if args.toggle:
        from .toggle import toggle_recording
        toggle_recording(config)
        return

    if args.input:
        if args.robust:
            from .robust import process_file_robust
            process_file_robust(config)
        else:
            process_file(config)
        return

    if args.vad or args.interval is not None:
        if args.vad:
            try:
                __import__("sherpa_onnx")
            except ImportError:
                print(
                    "Error: VAD requires sherpa-onnx.\n"
                    "Install with: pip install asr2clip[vad]"
                )
                sys.exit(1)
        continuous_recording(config)
        return

    process_recording(config)


if __name__ == "__main__":
    main()
