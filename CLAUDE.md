# asr2clip — Claude Code context

This is Samuel's fork of [Oaklight/asr2clip](https://github.com/Oaklight/asr2clip), a speech-to-clipboard CLI tool.
Fork lives at github.com/sjjsy/asr2clip. AGPL-3.0 licensed.

**Scope/Pipeline:** audio capture → optional preprocessor → ASR → optional LLM post-processing → output (clipboard / `-o FILE`).
- Out of scope: Output routing, prompt engineering, context injection beyond `context_path`, per-speaker naming, and assistant-layer intelligence belong in the calling assistant (ZeroClaw/OpenClaw), not here.

> **Review (merge proposal):** In the opening paragraph, prefer a full `https://github.com/sjjsy/asr2clip` link for consistency with other docs.

## What this fork adds over upstream

| Feature | Module | Status |
|---|---|---|
| whisper.cpp backend (`-b wcpp`) | `backends/whisper_cpp.py` | ✓ |
| Toggle mode (`--toggle`) | `toggle.py` | ✓ |
| Robust chunked transcription (`-r`) | `robust.py` | ✓ |
| Audio preprocessors (`-p`) | `preprocessors/` | ✓ |
| AI post-processing (`-P NAME`) | `postprocessors/` | ✓ |
| Speaker diarization (`-D`) | `diarize.py` | ✓ |

> **Review (merge proposal):**
> - **Toggle:** Current CLI uses **`-g` / `--toggle`** (see `asr2clip --help`); update this row and any README examples that still say `--toggle` only.
> - **Diarization:** The parser has **no `-D` / `--diarize` flag** today. Diarization is selected via an ASR **backend** (`type: whisperx`, `type: mock-diarize`, …) and **`--speakers` / `-s`**. Replace the `(-D)` row with that description, or restore `-D` in the product first, then document it here.

## Architecture

`Config.from_file()` is the single coordinator: lazy properties (`asr_backend`, `preprocessor`, `recorder`, `postprocessor`, `output`, `diarization`, `local_asr`) each own defaults, env fallbacks, and logging for their domain.

**CLI vs preset:** per-component flags (`-b`, `-p`, `-P`, `-M`, `-d`, and local-ASR flags under “Local ASR server”) override the selected preset; top-level keys in YAML (e.g. `default_preset`, `audio_device`) apply when no flag overrides that slice.

`main()` in `asr2clip.py` loads `Config` once, then dispatches to recording, file, robust, toggle, VAD/daemon, `--test`, `--serve`, or `--download-model`. The local sherpa-onnx server reads bind address, model dir, and thread count from `config.local_asr` (optional `local_asr:` in YAML, merged with CLI; same config file and preset are required as for any other subcommand).

> **Review (merge proposal):** Add **`-z` / `--no-clipboard`** to the “per-component flags” sentence so clipboard behaviour is treated like other overrides.

## Design decisions

- **Preset system:** Presets are atomic combinations of all pipeline stages (ASR backend, preprocessor, postprocessor). All stages must be explicitly specified. One preset per run. No mode-based fallback logic.
- **Preset config format (list, not dict):** Presets use compact list format `[preprocessor, asr_backend, postprocessor, description]` to make all fields required and visible as a table.
- **Config resolution lives in `config_types.py` only.** No `config.get()` for behavioral decisions anywhere else in the codebase. If you need a config value in a function, accept `Config` or the appropriate sub-config object as a parameter.

> **Review (merge proposal):** (Optional) Add one line: behavioural dict access for **postprocessor prompt `extends:` / `extra:`** stays in `postprocessors/__init__.py` — already implied by the Key files / resolver exception elsewhere; include here if you want one canonical sentence.

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

> **Review (merge proposal):**
> - **Remove `CliOverrides` from the `config_types.py` bullet** — the codebase no longer defines that class (`config_types.py` states there is no CliOverrides translation layer; CLI args live on `Config` via `Namespace`).
> - **Add `tests/README.md`** — E2E strategy, invocation budget, scenario index, gaps; agents should read it before changing `tests/test_e2e.py`.

## Post-processing system

- All prompts are **user-defined in config** — no hardcoded prompts.
- `postprocessors/__init__.py`: `make_postprocessor()`, `resolve_output_template()`, `format_output()`
- Prompt resolution supports `extends:` + `extra:` inheritance (user-defined only, circular guard in `_resolve_prompt`)
- Per-prompt `backend:`, `model:`, `template:`, `context_path:` fields
- Two backend types: `openai_compat` (Ollama, Groq, Anthropic, OpenAI) and `claude_code` (subprocess to `claude -p`)
- Template placeholders: `{result}` `{transcript}` `{date}` `{datetime}` `{prompt_name}` `{model}` `{backend}` `{duration_s}`

> **Review (merge proposal):** Mention the **`mock`** postprocessor backend type used in tests/examples, so the list is not read as exhaustive of only remote backends.

## Diarization

- `diarize.py`: `run_diarization(audio_path, config, language, num_speakers) → str`
- Uses WhisperX. Optional dep: `pip install asr2clip[diarize]`
- Output: `[HH:MM:SS] SPEAKER_NN: text` — name substitution intentionally left to caller
- `--diarize` / `-D` replaces the configured ASR backend entirely for that run
- Requires `HF_TOKEN` env var or `diarize_hf_token:` in config

> **Review (merge proposal):** Align with the **CLI**: diarization is normally chosen via **preset / `-b`** on a backend with `type: whisperx` or `type: mock-diarize`, with **`-s`** as speaker hint. **Drop or rewrite the `--diarize` / `-D` bullets** unless that flag is added back to `_build_parser()`.

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
| `-x` | `--preset` | fork new |
| `-C` | `--chunk-duration` | fork |
| `-P` | `--post` | fork new |
| `-M` | `--post-model` | fork new |
| `-T` | `--template` | fork new |
| `-D` | `--diarize` | fork new |
| `-s` | `--speakers` | fork new |

> **Review (merge proposal):**
> - Add **`-g` / `--toggle`** and **`-z` / `--no-clipboard`** rows (and optionally `--test`, `--list_devices`, serve/VAD long-options note) from **`asr2clip --help`**.
> - **Reconcile `-D` row** with parser reality (see Diarization proposal above).
> - Add a line: **authoritative flag list = `asr2clip --help`**; this table is a short mnemonic only.

---

## Development principles

These are strict rules. Follow them always, even under time pressure.

### 1. Config contract

`Config` is created once in `main()` via `Config.from_file()`. It is the single authoritative configuration object for a run.

- **Do** pass `Config` as a function parameter to avoid having to pass its contents separately.
- **Do not** pass `config._config_dict` downstream. Keep YAML-shape knowledge inside `config_types.py` and the postprocessor prompt resolver — not in random call sites.
- **Do not** call `config_dict.get("some_key")` anywhere outside `config_types.py` to make a behavioral decision. Move that logic into the appropriate Config class.
- **Do not** create a second `Config` or `PresetConfig` inside a transcription or processing function. Configuration is resolved once at startup.
- **Do not** create variables that copy `Config` properties and then pass those forward to downstream functions instead of the `Config` object.
- **Do** minimize the number of new local variables used in functions and those passed across them.

> **Review (merge proposal):** Add an explicit allowance already implied elsewhere: pass **`Config` or the single sub-config object** a function needs (not parallel copies of fields). Keeps §1 aligned with §4.

### 2. Fail fast, no cleverness

Prefer explicit errors over smart fallbacks. Make the code crash close to the real problem.

- If a config key is required, raise `ValueError` immediately with the key name and what was expected.
- No defaults allowed outside Config class modules
- Never silently fall back to a default that hides a misconfiguration. Log explicitly when using a default.
- No `try/except` that catches broad `Exception` and continues as if nothing happened.

> **Review (merge proposal):**
> - **Tighten “No defaults…”** to: *behavioural* defaults / “why we chose X” live in **`config_types.py`** (and the same family of resolvers), not scattered `if not x: x = …` in call sites. **Third-party library defaults are out of scope.**
> - **Call signatures:** avoid **fake Python defaults** on parameters that are **actually required** — that suggests callers may omit them. Prefer required args / `ValueError` when missing.
> - **Link to E2E:** fewer hidden defaults and fewer “catch-all” branches → regressions tend to surface as **failures in the black-box E2E suite** (`tests/test_e2e.py`) instead of being diluted across many shallow tests.

### 3. Zero tolerance for dead code

When changing something, update ALL references immediately and delete the old version.

- No deprecated functions with "will remove later" comments.
- No compatibility shims, fallback branches, or dual-path logic.
- No commented-out code.
- Deleted test files must have their imports removed from conftest and other test files too.

> **Review (merge proposal):** (None — section is already clear.)

### 4. Minimal complexity

- Centralize behavioral defaults and overrides in `Config` and related typed configs — keep orchestration functions thin.
- Prefer **fewer** classes and **fewer** functions when a slightly larger unit still reads clearly; do not split purely for ceremony.
- Minimize parameter lists: pass `Config` (or the one sub-config a callee needs), not parallel CLI/backend strings.
- No one-off utilities unless used in ≥ 3 places (then consider a method on the owning type instead).
- The `mode: str` string-dispatch pattern remains a smell — prefer separate entrypoints or explicit dispatch.
- Save vertical space: Do not break expressions and function calls into multiple lines except if they are over 130 characters long.

> **Review (merge proposal):** (None — already consistent with the relaxed §2 wording if you merge that proposal.)

### 5. Adhere to our test strategy

- When testing, treat the tool as a black box and focus on a select few E2E tests that
  leverage built-in mock devices and processors in the pipeline and real I/O rather than
  swarms of unit and integration tests that mostly create development inertia while
  producing little user value
- Run E2E tests before every major commit to identify whether something has broken
- Do not change E2E tests without reading `tests/README.md` and explaining to the user why,
  and asking for permission

> **Review (merge proposal):**
> - **Reconcile with §1 / Config tests:** keep **targeted** tests where the contract is resolution (`Config.from_file()`, fixtures) if you still want them; the “swarms” warning is about **low-value** unit tests, not all non-E2E tests.
> - **Push gate vs every commit:** not every local commit must run the full suite; **before push**, run E2E (and whatever else you rely on) until green, then push.
> - **Changing E2E:** simplest enforceable rule for agents — **do not edit `tests/test_e2e.py` unless the user explicitly asked to update E2E tests**; when they do, read **`tests/README.md`** first. (Drop “ask permission” / commit-message-only alternatives if this is enough.)
> - **Cross-link:** add `tests/README.md` to Key files (see proposal above).

### 6. Logging contract

Every config decision log line must answer: *why* was this chosen?

- ❌ `info(f"Using backend: groq")`
- ✅ `info(f"Using backend: groq (from preset 'speed')")` or `info(f"Using backend: groq (CLI override -b)")`

Apply this to preprocessor, postprocessor, device, and recorder decisions too.

> **Review (merge proposal):** Replace the code-fence examples with one abstract rule: at **INFO**, user-visible choices log **source** (preset vs `CLI -b` / `CLI -p`, …); match patterns already in the tree (`success` / `warning`, chunk lines, clipboard skip text). Keeps CLAUDE shorter; agents learn from code.

### 7. Atomic commits

Each commit must:
- Do one thing completely (fix, refactor, feature, or test — not mixed).
- Have a message that describes the behavioral change, not the file change.

> **Review (merge proposal):** Optionally restore a third bullet: **build / tests pass at that commit** — *if* you want every commit green; otherwise keep only the “before push” testing rule from §5 and state here that **small local commits may skip full runs**.

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

> **Review (merge proposal):** (None — table is fine.)

---

## Tooling conventions

- **Multi-file string replacement:** Use `sed -i 's/old/new/g' file1 file2 ...` (or `grep -rl pattern . | xargs sed -i ...`) instead of the Edit tool for mechanical substitutions that don't need surrounding context.
- **CLI reference in README:** Always run `asr2clip --help` and paste the exact output verbatim.
- **Inline comments in bash code blocks:** Pad to column 46 before `#`. Apply to epilog in `_build_parser()` and all bash example blocks in README.
- **Argument group order in `_build_parser()`:** Setup → Audio → Transcription → Local ASR server → VAD → Diarization → Post-processing.

> **Review (merge proposal):**
> - Append **→ Output** to the parser group order (current `_build_parser()` has an Output group, e.g. `-z`).
> - **Clarify:** these bullets are **maintainer rules inside this file** — there is no separate “tooling README.” Optionally fold the next section into one **“Maintainer conventions”** heading to reduce hopping between “Tooling” and “README structure.”

---

## README structure

Sections mirror CLI argument groups: Setup → Audio → Transcription → Local ASR server → VAD → Diarization → AI post-processing. Within each h2: brief motivation → options table (from `--help`) → relevant config excerpt. Do not duplicate content already in the config template or `--help` output.

> **Review (merge proposal):** Add **Output** (or “Output / clipboard”) to the section list so it matches `_build_parser()` and flags like `-z` / `-o` / `-T`.

---

## Upstream engagement

Contact made 2026-05-12 via GitHub Issue #16 (Oaklight/asr2clip). Awaiting response on PR interest.

**Best PR candidates** (if accepted, in order):
1. Toggle mode (`--toggle`) — most self-contained, useful on all platforms
2. Robust transcription (`-r`) — independent feature
3. whisper.cpp backend (`-b wcpp`) — self-contained new backend
4. Preprocessors (`-p`) — independent audio preprocessing

**Fork naming:** Tool has outgrown the original name. If PRs declined → rename fork as independent project. See `todo.md`.

> **Review (merge proposal):** **Remove this entire section** from `CLAUDE.md` and keep upstream / rename notes only in `todo.md` or `now.md` if you want the contract file to stay timeless. Alternatively, shorten to one line: “Upstream contact: issue #16 (2026-05-12).”
