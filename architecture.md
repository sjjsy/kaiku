# asr2clip Architecture

## Pipeline overview

Four-stage pipeline: audio capture → preprocessor → ASR/transcription → LLM post-processing → output.

```
YAML config + CLI args
        │
        ▼
  Config.from_file()          ← single authoritative instance
        │
        ├─ .asr_backend     → ASRBackendConfig
        ├─ .preprocessor    → PreprocessorConfig
        ├─ .postprocessor   → PostprocessorConfig
        ├─ .recorder        → RecorderConfig
        ├─ .diarization     → DiarizationConfig
        └─ .output          → OutputConfig
              │
              ▼
  main() distributes config to:
        ├─ toggle_recording(...)     toggle.py
        ├─ process_file(...)         asr2clip.py
        ├─ process_recording(...)    asr2clip.py
        ├─ process_file_robust(...)  robust.py
        └─ continuous_recording(...) daemon.py
```

## Config system

### Design

`Config` is a **lazy-loading coordinator** created once at startup from YAML + CLI flags. It is the single authoritative source of truth for all configuration. Downstream functions receive the `Config` object — never the raw config dict.

- `Config.from_file(path, preset_name, cli_overrides)` — primary entry point; handles file reading, preset resolution, and error reporting in one place.
- `Config.resolve(config_dict, preset, cli_overrides)` — lower-level constructor used in tests and toggle mode.
- Each sub-config class (`ASRBackendConfig`, etc.) has a `resolve(config_dict, ...)` classmethod that owns all defaults, env var fallbacks, validation, and logging for its domain.

### CLI override precedence

```
CLI flag (-b, -p, -P, -M, -d)
    > preset component (asr_backend, preprocessor, postprocessor from preset definition)
    > config-level default (default_preset, audio_device, etc.)
```

This is enforced in `Config` properties:
```python
# Example: ASR backend
backend_name = self._cli_overrides.backend or self._preset.asr_backend
```

### Current design gap (incomplete migration)

`Config` is created correctly in `main()`, but downstream processing functions (`process_file`, `process_recording`, `process_file_robust`, `toggle_recording`, `daemon.py`) still accept `config: dict` (raw dict), not `config: Config`. The CLI backend override is re-injected via an explicit `backend: str | None` parameter as a stopgap.

The core internal function `_transcribe_with_config_mode()` reconstructs a `Config` internally from a raw dict because transcription functions still accept dicts. It should disappear entirely once all callers pass `config: Config`.

### Target design

Pass the root `Config` object everywhere. No function below `main()` should accept or forward a raw dict or decomposed parameters — they access what they need from `config`:

```python
# All processing functions accept Config, not dict
def process_file(config: Config, input_file: str, ...) -> None:
    transcript = transcribe(input_file, config, language=language)

# Transcription functions accept Config and use config.asr_backend internally
def transcribe(path: str, config: Config, ...) -> str:
    if config.asr_backend.type == "whisper_cpp":
        return _transcribe_whisper_cpp(path, config.asr_backend, ...)
    return _transcribe_api(path, config.asr_backend, ...)
```

At this point `_transcribe_with_config_mode` and its internal Config reconstruction disappear entirely, as do the `backend: str | None` passthrough parameters.

## Module map

| File | Responsibility |
|---|---|
| `asr2clip.py` | CLI entry point, argument parsing, top-level orchestration |
| `config.py` | YAML reading, config template generation (`_CONFIG_TEMPLATE`) |
| `config_types.py` | All config resolution classes (`Config`, `ASRBackendConfig`, etc.) |
| `transcribe.py` | HTTP transcription API calls, retry logic; `transcribe_casual/urgent` |
| `toggle.py` | Lock-file toggle recording; calls `transcribe_urgent` |
| `robust.py` | Chunked transcription for long files; calls `transcribe_casual` |
| `daemon.py` | Continuous/VAD recording loop; direct API calls |
| `audio.py` | Audio I/O, device enumeration, `DeviceInfo` dataclass |
| `recorders/` | Pluggable recorder backends (`sounddevice`, `arecord`) |
| `preprocessors/` | Pluggable noise-reduction backends |
| `postprocessors/` | Pluggable LLM post-processing |
| `backends/` | Low-level ASR backend implementations (`whisper_cpp`, `mock`) |
| `diarize.py` | WhisperX diarization wrapper |
| `output.py` | Clipboard and file output, `_DEFAULT_CLIPBOARD_MAX_CHARS` |

## Remaining migration items

| File | Current state | Target |
|---|---|---|
| `transcribe.py` | `transcribe_casual/urgent(path, config_dict, ..., backend=None)` — rebuilds `Config` internally | Accept `config: Config`; delete `_transcribe_with_config_mode` |
| `asr2clip.py` | `process_file/process_recording(config_dict, ..., backend=None)` | Accept `config: Config`; use `config.asr_backend` directly |
| `toggle.py` | `toggle_recording(config_dict, ..., backend=None)` | Accept `config: Config` |
| `robust.py` | `process_file_robust(config_dict, ..., backend=None)` | Accept `config: Config` |
| `postprocessors/__init__.py` | `make_postprocessor(name, config_dict, ...)` | Accept `PostprocessorConfig + PostprocessorBackendConfig` |
| `diarize.py` | `run_diarization(path, config_dict, ...)` | Accept `config: Config`, use `config.diarization` |
| `daemon.py` | `continuous_recording(api_key, api_base_url, ...)` — bypasses Config entirely | Accept `config: Config` |

## Config logging contract

Every config decision log line must answer *why* that value was chosen:

- `info(f"Using backend: {name} (CLI override -b)")` or `info(f"Using backend: {name} (from preset 'speed')")`
- Same for preprocessor, postprocessor, device, recorder.

Currently `PreprocessorConfig.resolve()` and `PostprocessorConfig.resolve()` log "(CLI override)" even when called with the preset's value, because `Config` passes `cli_override or preset_value` as a single parameter. Fix: log the source before delegating, remove source-attribution from sub-config resolve methods.
