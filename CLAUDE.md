# asr2clip — Claude Code context

This is Samuel's fork of [Oaklight/asr2clip](https://github.com/Oaklight/asr2clip), a speech-to-clipboard CLI tool.
Fork lives at github.com/sjjsy/asr2clip. AGPL-3.0 licensed.

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

Four-stage pipeline: audio capture → ASR/transcription → optional LLM post-processing → output (clipboard / -o FILE).

**Scope:** asr2clip = audio → accurate transcript + optional minimal LLM passthrough.
Output routing, prompt engineering, context injection beyond `context_path`, per-speaker naming, and assistant-layer intelligence belong in the calling assistant (ZeroClaw/OpenClaw), not here.

## Design decisions

- **Preset system:** Presets are atomic combinations of all pipeline stages (ASR backend, preprocessor, postprocessor). All stages must be explicitly specified. One preset per run. No mode-based fallback logic. The codebase is young with a single user, so breaking changes are acceptable and preferred over clutter.
- **Preset config format (list, not dict):** Presets use compact list format `[asr_backend, preprocessor, postprocessor, description]` to make all fields required and visible as a table. This prevents silent omission of required stages.
- **Config system:** No config and CLI override argument resolution logic outside of config_types.py. If a function in another file does config.get(...) to make a behavioral decision, move that logic into the appropriate Config class.

## Post-processing system

- All prompts are **user-defined in config** — no hardcoded prompts. Five prompts are shipped in the config template: `solo-base`, `solo-enhance`, `solo-restructure`, `solo-private`, `group`.
- `postprocessors/__init__.py`: `make_postprocessor()`, `resolve_output_template()`, `format_output()`
- Prompt resolution supports `extends:` + `extra:` inheritance (user-defined only, circular guard in `_resolve_prompt`)
- Per-prompt `backend:`, `model:`, `template:`, `context_path:` fields
- Two backend types: `openai_compat` (covers Ollama, Groq, Anthropic API, OpenAI) and `claude_code` (subprocess to `claude -p`)
- Output templates are named in `output_templates:` config section; applied via `format_output()` at the call site
- Template placeholders: `{result}` `{transcript}` `{date}` `{datetime}` `{prompt_name}` `{model}` `{backend}` `{duration_s}`

## Diarization

- `diarize.py`: `run_diarization(audio_path, config, language, num_speakers) → str`
- Uses WhisperX (Whisper + word-level alignment + pyannote). Optional dep: `pip install asr2clip[diarize]`
- Output: `[HH:MM:SS] SPEAKER_NN: text` — name substitution intentionally left to caller
- `--diarize` / `-D` replaces the configured ASR backend entirely for that run
- Requires `HF_TOKEN` env var or `diarize_hf_token:` in config for pyannote model download

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

## Key files

- `asr2clip/asr2clip.py` — CLI entry point, `_build_parser()`, `main()`, `process_recording()`, `process_file()`
- `asr2clip/toggle.py` — lock-file toggle recording, `_transcribe_and_output()`
- `asr2clip/robust.py` — chunked transcription, `process_file_robust()`
- `asr2clip/config.py` — config loading, `_CONFIG_TEMPLATE` (the template shown by `--generate_config`)
- `asr2clip/postprocessors/` — post-processing package
- `asr2clip/preprocessors/` — existing noise-reduction package (pattern to follow for postprocessors)
- `asr2clip/diarize.py` — WhisperX diarization
- `todo.md` — gitignored active work (upstream contact, marketing plan, deferred features)
- `done.md` — gitignored archive (completed specs by date)

## Commit message conventions

Format: `TYPE: Description` (imperative mood, lowercase description, 50 char title max).

| Type | Use for |
|------|---------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `refactor:` | Code refactoring (no behavior change) |
| `perf:` | Performance improvement |
| `test:` | Add/update tests (not test fixes) |
| `docs:` | Documentation, README, docstrings |
| `meta:` | Project structure, `.gitignore`, `CLAUDE.md`, license, tooling |
| `chore:` | Dependencies, build config, CI/CD (not application code) |
| `style:` | Code formatting only (no logic change) |

Examples:
- `feat: add toggle mode with lock-file protocol`
- `fix: handle stale lock file in toggle mode`
- `meta: consolidate WISHLIST into todo.md and done.md`
- `docs: update CLI reference from --help output`

Use `meta:` for project metadata and infrastructure; `chore:` for build/dependency changes. Prefer semantic commits on feature branches; squash trivial commits before PR.

## Tooling conventions

- **Multi-file string replacement:** Use `sed -i 's/old/new/g' file1 file2 ...` (or `grep -rl pattern . | xargs sed -i ...` for codebase-wide sweeps) instead of the Edit tool. Edit requires the file to be read first and sends the full diff — `sed` is cheaper for mechanical substitutions that don't need surrounding context.

## Documentation conventions

- **CLI reference in README:** Always run `asr2clip --help` and paste the exact output verbatim — never manually maintain a separate version. Capitalization and wording must match the argparse definition exactly.
- **Inline comments in bash code blocks:** Always pad to 46 characters before `#` so comments start at the same column. Format: `command` + spaces to position 46, then `# comment`. Apply to the epilog in `_build_parser()` and all bash example blocks in the README.
- **Argument group order in `_build_parser()`:** Setup → Audio → Transcription → Local ASR server → VAD (continuous recording) → Diarization → Post-processing.
- **Config key names:** ASR backends use `asr_backends:`, `asr_backend_urgent:`, `asr_backend_casual:` (mirrors `postprocessor_backends:` naming). No backward-compat fallbacks.
- **Post-processing prompt names:** Built-in prompts are `solo-base`, `solo-enhance`, `solo-restructure`, `solo-private`, `group`. Output template names are `bare` and `full`. Use these in all examples and documentation.

## README structure

README sections mirror CLI argument groups in order: Setup → Audio → Transcription → Local ASR server → VAD (continuous recording) → Diarization → AI post-processing. Within each h2 section: brief definition/motivation paragraph → options table (flags from `--help`) → relevant config template excerpt. Subsections (h3) cover features within the group (e.g., `### ASR backends` and `### Toggle mode` under `## Transcription`). Do not duplicate content already explained clearly in the config template or the `--help` output.

## Upstream engagement

Contact made 2026-05-12 via GitHub Issue #16 (Oaklight/asr2clip). Awaiting response on PR interest.

**Best PR candidates** (if accepted, in order):
1. Toggle mode (`--toggle`) — most self-contained, useful on all platforms
2. Robust transcription (`-r`) — independent feature
3. whisper.cpp backend (`-b wcpp`) — self-contained new backend
4. Preprocessors (`-p`) — independent audio preprocessing

**Timidly proposed also:**
- AI post-processing (`-P`) — pending upstream feedback
- Speaker diarization (`-D`) — pending upstream feedback
- These features expand scope beyond "ASR to clipboard" and may require separate discussion

**Fork naming:** Tool has outgrown the original name. If PRs accepted → stay under upstream; if declined → rename fork as independent project.

The file `todo.md` provides more info on future plans.

## Core Development Rules (Strict - Follow Always)

- **Fail Fast, No Cleverness**: Prefer explicit errors and simple, short and obvious code over smart fallbacks, auto-detection, or complex logic. Make the code crash close to the real problem. Never add "just in case" handling unless explicitly asked.

- **Zero Tolerance for Dead Code**: Never leave deprecated functions, old config options, compatibility layers, or commented-out code. When changing something, update ALL references in the entire codebase immediately and remove the old version completely.

- **Minimize Complexity**:
  - Keep functions small and focused (single responsibility).
  - Prefer clear, readable code over clever or overly abstract code.
  - Reduce top-level if/else spaghetti — push logic into classes/methods where it belongs.
  - Avoid creating new small utility functions unless they are reused in ≥3 places.

- **Tests First Mindset**: Before implementing or changing features, ensure or create relevant tests. Run the unit tests frequently, the integration tests whenever you think you completed something, and the E2E tests before committing a significant code change.

- **Atomic Changes**: Make changes in small, verifiable steps. After editing files, summarize exactly what was changed and why.

- **Autonomous Execution**: Do not ask for confirmation on every file edit. If the task is clear from the plan and context, proceed and keep going. Only stop if you hit an error, conflict, or need real clarification.

- **Quality Gates**: After any significant change:
  1. Run relevant tests (`python -m pytest ...`)
  2. Check for linting issues
  3. Verify the feature manually if possible
  4. Report status clearly before continuing

## Workflow Rules

- Always start with a short plan (bullet points) before writing code, unless I explicitly say "implement directly".
- Use existing patterns and config system — do not invent new ones.
- Keep the codebase clean and consistent with current architecture (especially the new config class system).
- When in doubt, prefer simplicity over extensibility right now.
