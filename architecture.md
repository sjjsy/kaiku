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
        ├─ toggle_recording(...)    toggle.py
        ├─ process_file(...)        asr2clip.py
        ├─ process_recording(...)   asr2clip.py
        ├─ process_file_robust(...) robust.py
        └─ continuous_recording(...)daemon.py
```

## Config system

### Design

`Config` is a **lazy-loading coordinator** created once at startup from YAML + CLI flags. It is the single authoritative source of truth for all configuration. Downstream functions receive what they need — never the raw config dict.

- `Config.from_file(path, preset_name, cli_overrides)` — primary entry point; handles file reading, preset resolution, and error reporting in one place.
- `Config.resolve(config_dict, preset, cli_overrides)` — lower-level constructor for testing and toggle mode.
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

`Config` is created correctly in `main()`, but downstream processing functions (`process_file`, `process_recording`, `process_file_robust`, `toggle_recording`) still accept `config: dict` (raw dict), not `config: Config`. The CLI override is re-injected via an explicit `backend: str | None` parameter threaded through the call chain.

This is a stopgap. The correct final design is to accept `config: Config` everywhere (dependency injection), eliminating both `config_dict` parameters and the redundant `backend` passthrough.

**The core internal function** `_transcribe_with_config_mode()` reconstructs a `Config` internally from the raw dict. This exists because transcription functions still accept dicts. It should disappear once all callers pass `config: Config`.

### Target design (next migration)

```python
# All processing functions accept Config, not dict
def process_file(config: Config, input_file: str, ...) -> None:
    transcript = transcribe_audio(input_file, config.asr_backend, language=language)

# Transcription functions accept ASRBackendConfig, not a dict or name
def transcribe_audio(path: str, backend: ASRBackendConfig, ...) -> str:
    if backend.type == "whisper_cpp":
        return _transcribe_whisper_cpp(path, backend, ...)
    return _transcribe_api(path, backend, ...)
```

At this point `_transcribe_with_config_mode` and its internal Config reconstruction disappear entirely.

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

## Known issues and open questions

### `PreprocessorConfig.resolve()` logging is misleading

The method parameter is named `cli_override` but `Config.preprocessor` passes it the value `cli_overrides.preprocessor or preset.preprocessor` — so the preset value comes in through the "CLI override" path and is logged as "CLI override". The method doesn't distinguish between the two sources.

**Fix:** Split the parameter or pass source information so logging says "(CLI override)" vs "(from preset 'quality')".

### `postprocessors/__init__.py` still uses raw dict

`make_postprocessor()`, `_resolve_backend()`, and `_resolve_prompt()` all accept `config_dict: dict`. They should eventually accept `Config` or at least `PostprocessorConfig + PostprocessorBackendConfig`.

### `diarize.py` still uses raw dict

`run_diarization(audio_path, config, ...)` accepts `config: dict`. Should accept `DiarizationConfig`.

### `daemon.py` bypasses Config entirely

`continuous_recording()` accepts raw API key/URL/model parameters instead of `config: Config`. It was not updated during the config system migration.
