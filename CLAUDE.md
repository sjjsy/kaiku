# kaiku — Claude Code context

This is Samuel's fork of [Oaklight/kaiku](https://github.com/Oaklight/kaiku), a speech-to-clipboard CLI tool.
Fork lives at [sjjsy/kaiku](https://github.com/sjjsy/kaiku). AGPL-3.0 licensed.

**Scope/Pipeline:** audio capture → optional preprocessor → ASR → optional LLM post-processing → output (clipboard / `-o FILE`).
- Out of scope: Output routing, prompt engineering, context injection beyond `context_path`, per-speaker naming, and assistant-layer intelligence belong in the calling assistant (ZeroClaw/OpenClaw), not here.

## What this fork adds over upstream

| Feature | Module | Status |
|---|---|---|
| whisper.cpp backend (`-b wcpp`) | `backends/whisper_cpp.py` | ✓ |
| Speaker diarization (`-d whisperx`) | `backends/whisperx.py` | ✓ |
| Toggle mode (`-g`) | `toggle.py` | ✓ |
| Robust chunked transcription (`-r`) | `robust.py` | ✓ |
| Audio preprocessors (`-p`) | `preprocessors/` | ✓ |
| AI post-processing (`-P NAME`) | `postprocessors/` | ✓ |

## Design decisions

- **The Config object** from `Config.from_file()` is the single coordinator: lazy properties (`asr_backend`, `preprocessor`, `recorder`, `postprocessor`, `output`, `diarization`, `local_asr`) each own defaults, env fallbacks, and logging for their domain. Avoid defaults and parameter definition outside Config classes.
- `main()` in `kaiku.py` loads `Config` once, then dispatches to recording, file, robust, toggle, VAD/daemon, `--test`, `--serve`, or `--download-model`. The local sherpa-onnx server reads bind address, model dir, and thread count from `config.local_asr` (optional `local_asr:` in YAML, merged with CLI; same config file and preset are required as for any other subcommand).
- **Config resolution lives in `config_types.py` only.** No `config.get()` for behavioral decisions anywhere else in the codebase. If you need a config value in a function, accept `Config` or the appropriate sub-config object as a parameter.
- **CLI vs preset:** per-component flags (`-b`, `-p`, `-P`, `-M`, `-d`, and local-ASR flags under “Local ASR server”) override the selected preset; top-level keys in YAML (e.g. `default_preset`, `audio_device`) apply when no flag overrides that slice.
- **Preset system:** Presets are atomic combinations of all pipeline stages (ASR backend, preprocessor, postprocessor). All stages must be explicitly specified. One preset per run. No mode-based fallback logic.
- **Preset config format (list, not dict):** Presets use compact list format `[preprocessor, asr_backend, postprocessor, description]` to make all fields required and visible as a table.

## Key files

- `kaiku/kaiku.py` — CLI entry point, `_build_parser()`, `main()`, `process_recording()`, `process_file()`
- `kaiku/config_types.py` — all config resolution classes (`Config`, `ASRBackendConfig`, etc.)
- `kaiku/config.py` — YAML file reading and `_CONFIG_TEMPLATE` (template shown by `--generate-config`)
- `kaiku/toggle.py` — lock-file toggle recording, `_transcribe_and_output()`
- `kaiku/robust.py` — chunked transcription, `process_file_robust()`
- `kaiku/postprocessors/` — post-processing package
- `kaiku/preprocessors/` — noise-reduction package
- `kaiku/diarize.py` — WhisperX diarization
- `AGENTS.md` — short pointer for AI agents (see also this file)
- `now.md` — active work items and upcoming tasks
- `todo.md` — gitignored future ideas and deferred features
- `tests/README.md` — E2E strategy, invocation budget, scenario index, gaps

## Development principles

### 1. Config contract

`Config` is created once in `main()` via `Config.from_file()`. It is the single authoritative configuration object for a run.

- **Do** pass `Config` as a function parameter to avoid having to pass its contents separately.
- **Do not** pass `config._config_dict` downstream. Keep YAML-shape knowledge inside `config_types.py` and the postprocessor prompt resolver — not in random call sites.
- **Do not** call `config_dict.get("some_key")` anywhere outside `config_types.py` to make a behavioral decision. Move that logic into the appropriate Config class.
- **Do not** create a second `Config` or `PresetConfig` inside a transcription or processing function. Configuration is resolved once at startup.
- **Do not** create variables that copy `Config` properties and then pass those forward to downstream functions instead of the `Config` object.

### 2. Fail fast, no cleverness

Prefer explicit errors over smart fallbacks. Make the code crash close to the real problem.

- If a config key is required, raise `ValueError` immediately with the key name and what was expected.
- No behavioural defaults allowed outside Config class modules
- Never silently fall back to a default that hides a misconfiguration. Log explicitly when using a default.
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
- Minimize the number of new local variables used in functions and those passed across them.
- Minimize parameter lists: pass `Config` (or the one sub-config a callee needs), not parallel CLI/backend strings.
- No one-off utilities and helper functions unless used in ≥ 3 places; Even then consider a method on the owning type instead. This helps improve codebase transparency.
- The `mode: str` string-dispatch pattern remains a smell — prefer separate entrypoints or explicit dispatch.
- Save vertical space: Do not break expressions and function calls into multiple lines except if they are over 130 characters long.

### 5. Adhere to our test strategy

- When testing, treat the tool as a black box and focus on a select few E2E tests that
  leverage built-in mock devices and processors in the pipeline and real I/O rather than
  dozens of unit and integration tests that mostly create development inertia while
  producing little user value.
- Run E2E tests before every push and major commit to identify whether something has broken.
- Do not change E2E tests without being explicitly asked to, and when asked, read
  `tests/README.md` first.

### 6. Logging contract

- Every major progression in the pipeline and I/O step deserves a one line log output.
- Every parameter resolution decision (in Config) deserves a log line that must also answer *why* was this chosen (CLI arg > CLI preset > Config default preset > built-in Config default). Apply this to preprocessor, postprocessor, device, and recorder decisions too.

### 7. Atomic commits

Each commit must:
- Do one thing completely (fix, refactor, feature, or test — not mixed).
- Have a message that describes the behavioral change, not the file change.

### 8. Important

- Whenever bumping into a comment annotated with `AGENTS`, read it especially carefully!
  If it says "Do not touch this function!" or similar, honor it!
  The short form annotation `ADNT!` means the same thing: Agents Do Not Touch!
- If you consider any instruction or comment in this document or elsewhere unwarranted or
  counterproductive, write a comment or complaint into `AGENT_COMPLAINTS.md` where you
  explain your reasoning.

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

## Documentation and maintainer conventions

- **CLI reference in README:** Always run `kaiku --help` and paste the exact output verbatim.
- **Argument group order in `_build_parser()` also followed in README.md:** Setup → Audio → Transcription → Local ASR server → VAD → Diarization → Post-processing → Output.
  - Each of these have a section of their own: brief motivation → CLI options table → relevant config excerpt. Rather than duplicating content, refer and link to it.
- **Inline comments in bash code blocks:** Pad to column 46 before `#`. Apply to epilog in `_build_parser()` and all bash example blocks in README.
- **Multi-file string replacement:** Use `sed -i 's/old/new/g' file1 file2 ...` (or `grep -rl pattern . | xargs sed -i ...`) instead of the Edit tool for mechanical substitutions that don't need surrounding context.

---
