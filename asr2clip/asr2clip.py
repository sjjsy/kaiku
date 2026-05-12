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
    mode: str = "both",
) -> bool:
    """Test the configuration by checking backend connectivity and preprocessor availability.

    Args:
        backend_config: Resolved backend configuration dictionary.
        full_config: Full (unresolved) configuration dictionary for preprocessor checks.
        preprocessor_override: Name supplied via -P/--preprocessor flag.
        mode: Which preprocessor mode(s) to check — "live", "file", or "both".

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

    # Check preprocessors — only for the modes relevant to this backend invocation
    from .preprocessors import check_preprocessor_available

    print_separator()
    info("Checking preprocessors...")

    modes_to_check = (
        [("live", "live"), ("file", "file")] if mode == "both"
        else [("live", "live")] if mode == "live"
        else [("file", "file")]
    )

    pp_ok = True
    for label, m in modes_to_check:
        name = resolve_preprocessor_config(full_config, preprocessor_override, m)
        avail, hint = check_preprocessor_available(name)
        if avail:
            print_success(f"Preprocessor ({label}): {name}")
        else:
            print_error(f"Preprocessor ({label}): {name} — NOT AVAILABLE  ({hint})")
            pp_ok = False

    return backend_ok and pp_ok


def _test_postprocessors(config: dict, post_override: str | None, model_override: str | None) -> bool:
    """Check that post-processor backends are reachable. Returns True if all OK."""
    import shutil
    from .postprocessors import resolve_postprocessor_config
    from .postprocessors import _resolve_backend, _resolve_prompt
    from .utils import print_error, print_success

    print_separator()
    info("Checking post-processors...")

    ok = True
    seen: set = set()

    for mode in ("live", "file"):
        name = resolve_postprocessor_config(config, post_override, mode)
        label = f"({mode})"

        if name in (None, "none"):
            print_success(f"Post-processor {label}: none")
            continue

        # Deduplicate: if live and file use the same named processor, only check once
        if name in seen:
            print_success(f"Post-processor {label}: {name}  (same as above)")
            continue
        seen.add(name)

        # Resolve backend type
        try:
            resolved = _resolve_prompt(name, config)
            backend_cfg = _resolve_backend(config, resolved["backend_name"], model_override)
        except SystemExit:
            print_error(f"Post-processor {label}: {name} — config error (see above)")
            ok = False
            continue

        btype = backend_cfg["type"]
        model = backend_cfg.get("model", "")
        model_note = f"  model: {model}" if model else ""

        if btype == "claude_code":
            if shutil.which("claude"):
                print_success(f"Post-processor {label}: {name}  (claude_code{model_note})")
                print_success("  claude CLI: found")
            else:
                print_error(f"Post-processor {label}: {name}  (claude_code{model_note})")
                print_error("  claude CLI: NOT FOUND — install from https://claude.ai/code")
                ok = False

        elif btype == "openai_compat":
            from .transcribe import test_transcription
            api_base = backend_cfg.get("api_base_url", "")
            api_key = backend_cfg.get("api_key", "sk-none")
            org_id = None
            print_success(f"Post-processor {label}: {name}  (openai_compat{model_note})")
            print_key_value("  Endpoint", api_base)
            reachable = test_transcription(api_key, api_base, model, org_id)
            if not reachable:
                ok = False
        else:
            print_error(f"Post-processor {label}: {name} — unknown backend type '{btype}'")
            ok = False

    return ok


def _test_clipboard() -> bool:
    """Check clipboard availability. Returns True if OK."""
    from .utils import print_error, print_success
    import shutil, os

    print_separator()
    info("Checking clipboard...")

    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        print_success("wl-copy available (Wayland)")
        return True
    try:
        import copykitten
        copykitten.copy("")
        print_success("copykitten available (X11 / fallback)")
        return True
    except Exception as e:
        from .utils import print_error
        print_error(f"Clipboard unavailable: {e}")
        print_error("  Install wl-copy (Wayland) or ensure copykitten works on X11")
        return False


def _test_diarization(config: dict) -> None:
    """Check diarization dependencies. Prints warnings; never blocks overall success."""
    import importlib.util
    from .utils import print_success, print_warning

    print_separator()
    info("Checking diarization (optional)...")

    if importlib.util.find_spec("whisperx") is None:
        print_warning("whisperx: not installed  →  pip install asr2clip[diarize]")
        print_warning("  (--diarize will not work until installed)")
        return

    print_success("whisperx: installed")

    hf_token = config.get("diarize_hf_token") or os.environ.get("HF_TOKEN")
    if hf_token:
        masked = f"{'*' * 8}...{hf_token[-4:]}" if len(hf_token) > 4 else "****"
        print_success(f"HuggingFace token: {masked}")
    else:
        print_warning(
            "HuggingFace token: not set\n"
            "  Set HF_TOKEN env var or add 'diarize_hf_token: hf_...' to config.\n"
            "  (--diarize will fail until token is provided)"
        )


def process_recording(
    config: dict,
    device: str | int | None = None,
    output_file: str | None = None,
    language: str | None = None,
    preprocessor=None,
    postprocessor=None,
    template_str: str = "{result}",
):
    """Record audio, transcribe, and output the result."""
    import time
    from .postprocessors import NonePostProcessor, format_output, PostMetadata
    from .preprocessors import NonePreprocessor

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
        log(f"Preprocessing audio with {preprocessor.name}...")
        t_pre = time.time()
        audio_data = preprocessor.process(audio_data, 16000)
        info(f"Preprocessing completed in {time.time() - t_pre:.2f}s")

    log("Processing...")
    temp_path = save_audio(audio_data)

    try:
        t1 = time.time()
        transcript = transcribe_with_config(temp_path, config, language=language)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if not transcript.strip():
            log("No speech detected in the recording.")
            return

        result = transcript
        if postprocessor is not None and not isinstance(postprocessor, NonePostProcessor):
            from datetime import date
            metadata = PostMetadata(
                date=date.today().isoformat(),
                duration_s=duration,
                language=language or "auto",
                prompt_name=postprocessor.name,
                diarized=False,
                source="file",
            )
            log(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
            final = format_output(
                template_str, result=result, transcript=transcript,
                metadata=metadata, model=postprocessor.model,
                backend=postprocessor.backend_type,
            )
        else:
            final = result

        output_transcript(final, to_clipboard=True, to_stdout=True, to_file=output_file)

    finally:
        safe_unlink(temp_path)


def process_file(
    config: dict,
    input_file: str,
    output_file: str | None = None,
    language: str | None = None,
    preprocessor=None,
    postprocessor=None,
    template_str: str = "{result}",
    diarize: bool = False,
    num_speakers: int | None = None,
):
    """Transcribe an existing audio or video file."""
    import time
    from .postprocessors import NonePostProcessor, PostMetadata, format_output
    from .preprocessors import NonePreprocessor

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    log(f"Processing file: {input_file}")

    if input_file.lower().endswith(".txt"):
        with open(input_file, encoding="utf-8") as fh:
            transcript = fh.read()
        if not transcript.strip():
            log("Transcript file is empty.")
            return
        result = transcript
        if postprocessor is not None and not isinstance(postprocessor, NonePostProcessor):
            from datetime import date
            metadata = PostMetadata(
                date=date.today().isoformat(),
                duration_s=0.0,
                language=language or "auto",
                prompt_name=postprocessor.name,
                diarized=False,
                source="file",
            )
            log(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
            final = format_output(
                template_str, result=result, transcript=transcript,
                metadata=metadata, model=postprocessor.model,
                backend=postprocessor.backend_type,
            )
        else:
            final = result
        output_transcript(final, to_clipboard=True, to_stdout=True, to_file=output_file)
        return

    if not input_file.lower().endswith(".wav"):
        t0 = time.time()
        log("Converting to WAV format...")
        temp_path = convert_audio_to_wav(input_file)
        info(f"Conversion completed in {time.time() - t0:.1f}s")
        cleanup_temp = True
    else:
        temp_path = input_file
        cleanup_temp = False

    if preprocessor is not None and not isinstance(preprocessor, NonePreprocessor):
        try:
            audio_data, sr = load_wav(temp_path)
            log(f"Preprocessing audio with {preprocessor.name}...")
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
        is_diarized = False
        if diarize:
            from .diarize import DiarizationError, run_diarization
            try:
                log("Running diarization (WhisperX)...")
                transcript = run_diarization(
                    temp_path, config, language=language, num_speakers=num_speakers
                )
                is_diarized = True
            except DiarizationError as e:
                print(f"Diarization error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            transcript = transcribe_with_config(temp_path, config, language=language)
        info(f"Transcription completed in {time.time() - t1:.1f}s")

        if not transcript.strip():
            log("No speech detected in the audio file.")
            return

        try:
            import wave as _wave
            with _wave.open(temp_path) as wf:
                duration_s = wf.getnframes() / wf.getframerate()
        except Exception:
            duration_s = 0.0

        result = transcript
        if postprocessor is not None and not isinstance(postprocessor, NonePostProcessor):
            from datetime import date
            metadata = PostMetadata(
                date=date.today().isoformat(),
                duration_s=duration_s,
                language=language or "auto",
                prompt_name=postprocessor.name,
                diarized=is_diarized,
                source="file",
            )
            log(f"Post-processing with '{postprocessor.name}'...")
            t_post = time.time()
            result = postprocessor.process(transcript, metadata=metadata)
            info(f"Post-processing completed in {time.time() - t_post:.1f}s")
            final = format_output(
                template_str, result=result, transcript=transcript,
                metadata=metadata, model=postprocessor.model,
                backend=postprocessor.backend_type,
            )
        else:
            final = result

        output_transcript(final, to_clipboard=True, to_stdout=True, to_file=output_file)

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
  asr2clip -i m.m4a -D -s 3                   # speaker diarization, 3 speakers
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
        "--generate_config", action="store_true",
        help="Write config template to ~/.config/asr2clip/config.yaml",
    )
    setup_group.add_argument(
        "--print_config", action="store_true",
        help="Print config template to stdout",
    )
    setup_group.add_argument(
        "--test", action="store_true",
        help="Test backend connectivity and configured preprocessors, then exit",
    )

    # Audio
    audio_group = parser.add_argument_group("Audio")
    audio_group.add_argument(
        "--list_devices", action="store_true",
        help="List available audio input devices",
    )
    audio_group.add_argument(
        "-d", "--device", metavar="DEV",
        help="Audio input device (name, ALSA name, or index). Overrides config.",
        default=None,
    )
    audio_group.add_argument(
        "-p", "--preprocessor", metavar="NAME",
        default=None,
        help=(
            "Audio preprocessor: none, noisereduce, pyrnnoise, deepfilter. "
            "Overrides preprocessor_live / preprocessor_file in config."
        ),
    )

    # Transcription
    trans_group = parser.add_argument_group("Transcription")
    trans_group.add_argument(
        "-b", "--backend", metavar="NAME",
        default=None,
        help=(
            "ASR backend to use (key under 'asr_backends:' in config). "
            "Overrides asr_backend_live / asr_backend_file."
        ),
    )
    trans_group.add_argument(
        "-i", "--input", metavar="FILE",
        help=(
            "Transcribe an existing audio or video file instead of recording. "
            "Supported: wav, mp3, m4a, ogg, flac, aac, opus, wma, "
            "mp4, mov, mkv, webm, avi, flv, mvi"
        ),
        default=None,
    )
    trans_group.add_argument(
        "-o", "--output", metavar="FILE",
        help="Append transcripts to file",
        default=None,
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
        default=180,
        help="Max chunk duration in seconds for -r/--robust mode (default: 180)",
    )
    trans_group.add_argument(
        "--toggle", action="store_true",
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
        "--silence_threshold", type=float, metavar="PROB",
        default=None,
        help="VAD speech probability threshold, 0.0-1.0 (default: 0.5)",
    )
    vad_group.add_argument(
        "--silence_duration", type=float, metavar="SEC",
        default=1.5,
        help="Silence duration to trigger transcription (default: 1.5 s)",
    )

    # Diarization
    diarize_group = parser.add_argument_group("Diarization")
    diarize_group.add_argument(
        "-D", "--diarize", action="store_true",
        help=(
            "Speaker diarization via WhisperX. "
            "Replaces the configured ASR backend for this run. "
            "Output: '[HH:MM:SS] SPEAKER_NN: text'. "
            "Requires: pip install asr2clip[diarize] and HF_TOKEN env var."
        ),
    )
    diarize_group.add_argument(
        "-s", "--speakers", type=int, metavar="N",
        default=None,
        help=(
            "Expected number of speakers (hint to pyannote for better accuracy). "
            "If omitted, pyannote infers automatically."
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
            "Overrides postprocessor_live / postprocessor_file for this run."
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
    post_group.add_argument(
        "-T", "--template", metavar="NAME",
        default=None,
        help=(
            "Output template name from 'output_templates:' in config. "
            "Controls what is written to clipboard/-o FILE. "
            "Overrides the template specified in the prompt definition."
        ),
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
            ok_asr = test_config(backend_config_live, config, args.preprocessor, mode="both")
        else:
            info("--- live backend ---")
            ok_live = test_config(backend_config_live, config, args.preprocessor, mode="live")
            info("--- file backend ---")
            ok_file = test_config(backend_config_file, config, args.preprocessor, mode="file")
            ok_asr = ok_live and ok_file

        ok_post = _test_postprocessors(config, args.post, args.post_model)
        ok_clip = _test_clipboard()
        _test_diarization(config)  # informational only; never blocks success

        print_separator()
        success = ok_asr and ok_post and ok_clip
        if success:
            info("All checks passed.")
        else:
            info("Some checks failed — see details above.")
        sys.exit(0 if success else 1)

    device = get_audio_device(config, args.device)

    # Resolve preprocessors
    from .preprocessors import make_preprocessor
    preprocessor_live = make_preprocessor(
        resolve_preprocessor_config(config, args.preprocessor, "live")
    )
    preprocessor_file = make_preprocessor(
        resolve_preprocessor_config(config, args.preprocessor, "file")
    )

    if preprocessor_live.name == preprocessor_file.name:
        if preprocessor_live.name != "none":
            info(f"Preprocessor: {preprocessor_live.name}")
    else:
        if preprocessor_live.name != "none":
            info(f"Live preprocessor: {preprocessor_live.name}")
        if preprocessor_file.name != "none":
            info(f"File preprocessor: {preprocessor_file.name}")

    # Resolve post-processors and output templates
    from .postprocessors import (
        make_postprocessor,
        resolve_output_template,
        resolve_postprocessor_config,
    )
    post_name_live = resolve_postprocessor_config(config, args.post, "live")
    post_name_file = resolve_postprocessor_config(config, args.post, "file")

    postprocessor_live = make_postprocessor(post_name_live, config, args.post_model)
    postprocessor_file = make_postprocessor(post_name_file, config, args.post_model)
    template_live = resolve_output_template(config, post_name_live, args.template)
    template_file = resolve_output_template(config, post_name_file, args.template)

    if postprocessor_live.name != "none":
        info(f"Live post-processor: {postprocessor_live.name}")
    if postprocessor_file.name != "none":
        info(f"File post-processor: {postprocessor_file.name}")

    if args.toggle:
        from .toggle import toggle_recording
        toggle_recording(
            backend_config_live, device, args.output,
            language=args.language,
            preprocessor=preprocessor_live,
            postprocessor=postprocessor_live,
            template_str=template_live,
            diarize=args.diarize,
            num_speakers=args.speakers,
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
                postprocessor=postprocessor_file,
                template_str=template_file,
            )
        else:
            process_file(
                backend_config_file, args.input, args.output,
                language=args.language,
                preprocessor=preprocessor_file,
                postprocessor=postprocessor_file,
                template_str=template_file,
                diarize=args.diarize,
                num_speakers=args.speakers,
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
        postprocessor=postprocessor_live,
        template_str=template_live,
    )


if __name__ == "__main__":
    main()
