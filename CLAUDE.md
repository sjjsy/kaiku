# asr2clip — Claude Code context

This is Samuel's fork of [Oaklight/asr2clip](https://github.com/Oaklight/asr2clip), a speech-to-clipboard CLI tool.
Fork lives at github.com/sjjsy/asr2clip. AGPL-3.0 licensed.

**Scope/Pipeline:** audio capture → optional preprocessor → ASR → optional LLM post-processing → output (clipboard / `-o FILE`).
- Out of scope: Output routing, prompt engineering, context injection beyond `context_path`, per-speaker naming, and assistant-layer intelligence belong in the calling assistant (ZeroClaw/OpenClaw), not here.

## What this fork adds over upstream

| Feature | Module | Status |
|---|---|---|
| whisper.cpp backend (`-b wcpp`) | `backends/whisper_cpp.py` | ✓ |
| Toggle mode (`--toggle`) | `toggle.py` | ✓ |
| Robust chunked transcription (`-r`) | `robust.py` | ✓ |
| Audio preprocessors (`-p`) | `preprocessors/` | ✓ |
| AI post-processing (`-P NAME`) | `postprocessors/` | ✓ |
| Speaker diarization (`-D`) | `diarize.py` | ✓ |

## Architecture

`Config.from_file()` is the single coordinator: lazy properties (`asr_backend`, `preprocessor`, `recorder`, `postprocessor`, `output`, `diarization`, `local_asr`) each own defaults, env fallbacks, and logging for their domain.

**CLI vs preset:** per-component flags (`-b`, `-p`, `-P`, `-M`, `-d`, and local-ASR flags under “Local ASR server”) override the selected preset; top-level keys in YAML (e.g. `default_preset`, `audio_device`) apply when no flag overrides that slice.

`main()` in `asr2clip.py` loads `Config` once, then dispatches to recording, file, robust, toggle, VAD/daemon, `--test`, `--serve`, or `--download-model`. The local sherpa-onnx server reads bind address, model dir, and thread count from `config.local_asr` (optional `local_asr:` in YAML, merged with CLI; same config file and preset are required as for any other subcommand).

## Design decisions

- **Preset system:** Presets are atomic combinations of all pipeline stages (ASR backend, preprocessor, postprocessor). All stages must be explicitly specified. One preset per run. No mode-based fallback logic.
- **Preset config format (list, not dict):** Presets use compact list format `[preprocessor, asr_backend, postprocessor, description]` to make all fields required and visible as a table.
- **Config resolution lives in `config_types.py` only.** No `config.get()` for behavioral decisions anywhere else in the codebase. If you need a config value in a function, accept `Config` or the appropriate sub-config object as a parameter.

## Key files

- `asr2clip/asr2clip.py` — CLI entry point, `_build_parser()`, `main()`, `process_recording()`, `process_file()`
- `asr2clip/config_types.py` — all config resolution classes (`Config`, `ASRBackendConfig`, `CliOverrides`, etc.)
- `asr2clip/config.py` — YAML file reading and `_CONFIG_TEMPLATE` (template shown by `--generate_config`)
- `asr2clip/toggle.py` — lock-file toggle recording, `_transcribe_and_output()`
- `asr2clip/robust.py` — chunked transcription, `process_file_robust()`
- `asr2clip/postprocessors/` — post-processing package
- `asr2clip/preprocessors/` — noise-reduction package
- `asr2clip/diarize.py` — WhisperX diarization
- `AGENTS.md` — short pointer for AI agents (see also this file)
- `now.md` — active work items and upcoming tasks
- `todo.md` — gitignored future ideas and deferred features

## Post-processing system

- All prompts are **user-defined in config** — no hardcoded prompts.
- `postprocessors/__init__.py`: `make_postprocessor()`, `resolve_output_template()`, `format_output()`
- Prompt resolution supports `extends:` + `extra:` inheritance (user-defined only, circular guard in `_resolve_prompt`)
- Per-prompt `backend:`, `model:`, `template:`, `context_path:` fields
- Two backend types: `openai_compat` (Ollama, Groq, Anthropic, OpenAI) and `claude_code` (subprocess to `claude -p`)
- Template placeholders: `{result}` `{transcript}` `{date}` `{datetime}` `{prompt_name}` `{model}` `{backend}` `{duration_s}`

## Diarization

- `diarize.py`: `run_diarization(audio_path, config, language, num_speakers) → str`
- Uses WhisperX. Optional dep: `pip install asr2clip[diarize]`
- Output: `[HH:MM:SS] SPEAKER_NN: text` — name substitution intentionally left to caller
- `--diarize` / `-D` replaces the configured ASR backend entirely for that run
- Requires `HF_TOKEN` env var or `diarize_hf_token:` in config

## Flag conventions

Lowercase = earlier/basic feature. Uppercase = later/advanced feature.

| Short | Long | Since |
|---|---|---|
| `-b` | `--backend` | upstream |
| `-i` | `--input` | upstream |
| `-o` | `--output` | upstream |
| `-l` | `--language` | upstream |
| `-d` | `--device` | upstream |
| `-q` | `--quiet` | upstream |
| `-e` | `--edit` | upstream |
| `-p` | `--preprocessor` | fork (was -P upstream) |
| `-r` | `--robust` | fork (was -R upstream) |
| `-C` | `--chunk-duration` | fork |
| `-P` | `--post` | fork new |
| `-M` | `--post-model` | fork new |
| `-T` | `--template` | fork new |
| `-D` | `--diarize` | fork new |
| `-s` | `--speakers` | fork new |

---

## Development principles

These are strict rules. Follow them always, even under time pressure.

### 1. Config contract

`Config` is created once in `main()` via `Config.from_file()`. It is the single authoritative configuration object for a run.

- **Do** pass `Config` (or its sub-config properties like `ASRBackendConfig`) as function parameters.
- **Do not** pass `config._config_dict` downstream. Keep YAML-shape knowledge inside `config_types.py` and the postprocessor prompt resolver — not in random call sites.
- **Do not** call `config_dict.get("some_key")` anywhere outside `config_types.py` to make a behavioral decision. Move that logic into the appropriate Config class.
- **Do not** create a second `Config` or `PresetConfig` inside a transcription or processing function. Configuration is resolved once at startup.

### 2. Fail fast, no cleverness

Prefer explicit errors over smart fallbacks. Make the code crash close to the real problem.

- If a config key is required, raise `ValueError` immediately with the key name and what was expected.
- Never silently fall back to a default that hides a misconfiguration. Log explicitly when using a default.
- No magic keys in dicts (like `_preset_for_testing`). Tests use real config fixtures.
- No `try/except` that catches broad `Exception` and continues as if nothing happened.

### 3. Zero tolerance for dead code

When changing something, update ALL references immediately and delete the old version.

- No deprecated functions with "will remove later" comments.
- No compatibility shims, fallback branches, or dual-path logic.
- No commented-out code.
- Deleted test files must have their imports removed from conftest and other test files too.

### 4. Minimal complexity

- Centralize behavioral defaults and overrides in `Config` and related typed configs — keep orchestration functions thin.
- Prefer **fewer** classes and **fewer** functions when a slightly larger unit still reads clearly; do not split purely for ceremony.
- Minimize parameter lists: pass `Config` (or the one sub-config a callee needs), not parallel CLI/backend strings.
- No one-off utilities unless used in ≥ 3 places (then consider a method on the owning type instead).
- The `mode: str` string-dispatch pattern remains a smell — prefer separate entrypoints or explicit dispatch.

### 5. Test quality

Tests must catch real bugs. 220 passing tests are worthless if `asr2clip -b wcpp` uses the wrong backend.

**What good tests look like here:**
- **Unit tests** verify that individual Config classes resolve correctly given a specific config dict + CLI overrides. They do not mock internal resolution logic.
- **Integration tests** verify the full flow from YAML string → `Config.from_file()` → correct sub-config properties. They use realistic config fixtures (YAML strings or fixture files), not hand-constructed dicts.
- **E2E tests** run `transcribe_casual(path, config)` with the mock ASR backend and verify the output. They do not mock `transcribe_casual` itself.

**Required test for every non-trivial config behavior:**
```python
# Example: CLI backend override
config = Config.from_file(fixture_path, preset_name="speed", cli_overrides=CliOverrides(backend="wcpp"))
assert config.asr_backend.name == "wcpp"   # NOT "groq" from preset
assert config.asr_backend.type == "whisper_cpp"
```

**Anti-patterns to avoid:**
- Testing that function A calls function B with parameter C — test behavior, not implementation.
- Mocking `transcribe_casual` in tests that are supposed to test config resolution through transcription.
- Using `_preset_for_testing` or other magic test-only dict keys.

### 6. Logging contract

Every config decision log line must answer: *why* was this chosen?

- ❌ `info(f"Using backend: groq")`
- ✅ `info(f"Using backend: groq (from preset 'speed')")` or `info(f"Using backend: groq (CLI override -b)")`

Apply this to preprocessor, postprocessor, device, and recorder decisions too.

### 7. Atomic commits

Each commit must:
- Compile and pass all tests at that exact state.
- Do one thing completely (fix, refactor, feature, or test — not mixed).
- Have a message that describes the behavioral change, not the file change.

---

## Commit message conventions

Format: `TYPE: Description` (imperative mood, lowercase description, ≤50 char title).

| Type | Use for |
|------|---------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `refactor:` | Code refactoring (no behavior change) |
| `perf:` | Performance improvement |
| `test:` | Add/update tests (not test fixes) |
| `docs:` | Documentation, README, docstrings |
| `meta:` | Project structure, `.gitignore`, `CLAUDE.md`, license, tooling |
| `chore:` | Dependencies, build config, CI/CD |
| `style:` | Code formatting only |

Examples:
- `feat: add toggle mode with lock-file protocol`
- `fix: pass backend CLI override to process_file and process_recording`
- `refactor: accept Config object in process_file instead of config dict`
- `meta: rewrite CLAUDE.md with strict development principles`

---

## Tooling conventions

- **Multi-file string replacement:** Use `sed -i 's/old/new/g' file1 file2 ...` (or `grep -rl pattern . | xargs sed -i ...`) instead of the Edit tool for mechanical substitutions that don't need surrounding context.
- **CLI reference in README:** Always run `asr2clip --help` and paste the exact output verbatim.
- **Inline comments in bash code blocks:** Pad to column 46 before `#`. Apply to epilog in `_build_parser()` and all bash example blocks in README.
- **Argument group order in `_build_parser()`:** Setup → Audio → Transcription → Local ASR server → VAD → Diarization → Post-processing.

---

## README structure

Sections mirror CLI argument groups: Setup → Audio → Transcription → Local ASR server → VAD → Diarization → AI post-processing. Within each h2: brief motivation → options table (from `--help`) → relevant config excerpt. Do not duplicate content already in the config template or `--help` output.

---

## Upstream engagement

Contact made 2026-05-12 via GitHub Issue #16 (Oaklight/asr2clip). Awaiting response on PR interest.

**Best PR candidates** (if accepted, in order):
1. Toggle mode (`--toggle`) — most self-contained, useful on all platforms
2. Robust transcription (`-r`) — independent feature
3. whisper.cpp backend (`-b wcpp`) — self-contained new backend
4. Preprocessors (`-p`) — independent audio preprocessing

**Fork naming:** Tool has outgrown the original name. If PRs declined → rename fork as independent project. See `todo.md`.
