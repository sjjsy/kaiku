# asr2clip — Claude Code context

Samuel's fork of [Oaklight/asr2clip](https://github.com/Oaklight/asr2clip) (AGPL-3.0). This fork: https://github.com/sjjsy/asr2clip

**Pipeline:** capture → optional preprocessor → ASR → optional post-process → clipboard / `-o FILE`  
**Out of scope:** output routing, prompt design beyond `context_path`, per-speaker naming, assistant-layer behaviour (ZeroClaw/OpenClaw).

## Fork vs upstream

| Feature | Where |
|---------|--------|
| whisper.cpp (`-b wcpp`) | `backends/whisper_cpp.py` |
| Toggle (`-g` / `--toggle`) | `toggle.py` |
| Robust (`-r`) | `robust.py` |
| Preprocessors (`-p`) | `preprocessors/` |
| Post-processing (`-P`) | `postprocessors/` |
| Diarization (WhisperX + `mock-diarize` backends, `-s` hint) | `diarize.py` |

## Architecture

- **`Config.from_file()`** — one coordinator per run; lazy properties (`asr_backend`, `preprocessor`, `recorder`, `postprocessor`, …) own defaults, env, and logging.
- **CLI vs preset** — `-b`, `-p`, `-P`, `-M`, `-d`, `-z`, local-ASR flags override the selected preset; top-level YAML (e.g. `default_preset`, `audio_device`) applies when nothing overrides that slice.
- **Presets** — atomic list `[preprocessor, asr_backend, postprocessor, description]`; one preset per run; no mode-based fallbacks.
- **Where config is read** — behavioural YAML usage only in `config_types.py` and in `postprocessors/__init__.py` for prompt `extends:` / `extra:`. Call sites take `Config` or the one sub-config they need — not `config._config_dict`, not `config_dict.get()` for decisions, no second `Config` mid-run.

`main()` in `asr2clip/asr2clip.py` loads `Config` once, then dispatches: record, `-i` file, robust, toggle, VAD/daemon, `--test`, `--serve`, `--download-model`.

## Files to know

`asr2clip.py`, `config_types.py`, `config.py`, `toggle.py`, `robust.py`, `postprocessors/`, `preprocessors/`, `diarize.py`, `AGENTS.md`, **`tests/README.md`** (E2E strategy and scenario index)

## Post-processing (brief)

Prompts live in user YAML; `make_postprocessor`, `resolve_output_template`, `format_output`; `extends:` + `extra:` with circular guard in `_resolve_prompt`; backends include `openai_compat`, `claude_code`, `mock`.

## Diarization (brief)

WhisperX optional install; diarization is selected via **backend** (`type: whisperx` or `mock-diarize`) in preset or `-b`, plus `-s` when relevant; HF token for WhisperX.

## CLI flags

**Source of truth:** `asr2clip --help` (paste into README when you refresh a section). Short-flag overview:

| Short | Long | Notes |
|-------|------|--------|
| `-c` | `--config` | config path |
| `-e` | `--edit` | open config |
| | `--generate_config` `--print_config` `--test` `--list_devices` | setup / verify |
| `-x` | `--preset` | preset name |
| `-d` | `--device` | input device |
| `-i` | `--input` | media or `.txt` transcript |
| `-p` | `--preprocessor` | `none`, `noisereduce`, … |
| `-b` | `--backend` | ASR backend key |
| `-l` | `--language` | hint |
| `-r` | `--robust` | chunked file mode |
| `-C` | `--chunk-duration` | robust chunk length |
| `-g` | `--toggle` | toggle recording |
| `-s` | `--speakers` | diarization hint |
| `-P` | `--post` | post-processor |
| `-M` | `--post-model` | post model override |
| `-o` | `--output` | append transcript file |
| `-T` | `--template` | output template |
| `-z` | `--no-clipboard` | skip clipboard |
| `-q` | `--quiet` | transcript + errors only |

Serve / VAD / `--download-model`: long options only — see `--help`.

---

## Development principles

Strict rules; follow under time pressure.

### 1. Config contract

- One `Config.from_file()` in `main()`.
- Pass **`Config`** (or the single sub-config a callee needs); do not copy fields into parallel locals and pass those instead of `Config`.
- No behavioural decisions from raw dicts outside the allowed modules (above).

### 2. Fail fast; where defaults live

Prefer **explicit errors** and **logs that state intent** over silent recovery. **Behavioural** defaults and “why this value” belong in **`config_types.py`** (and the same family of resolvers) — not ad‑hoc `if not x: x = …` in random call sites.

**Signatures:** do not give **required** parameters **fake Python defaults** just to satisfy the type checker — that suggests the caller may omit them when they must not. Third-party library defaults are not “our” policy.

No broad `except Exception:` that swallows and continues.

### 3. Dead code

Rewrite all references, delete the old path immediately: no “remove later”, no shims, no commented-out blocks. Removing a test file → remove its imports from `conftest.py` and elsewhere.

### 4. Minimal complexity

Thin orchestration; prefer fewer types/functions until clarity suffers; avoid stringly `mode` dispatch when separate entrypoints are clearer. Wrap lines only past **130** columns.

### 5. Tests

- **E2E first for the pipeline** — a **small** black-box suite (`tests/test_e2e.py`) hits real subprocess I/O with mock devices/backends; strategy, checklists, and gaps: **`tests/README.md`**.
- **Why this pairs with §2** — fewer hidden defaults and fewer “smart” catch-all branches mean regressions surface as **failures in those E2E runs** instead of being smeared across many shallow tests.
- **Do not change E2E tests** unless the **user explicitly asked** to update them. When you are asked, read **`tests/README.md`** first.

### 6. Logging

At INFO, every **resolved config choice** the user cares about should record **why** (preset vs `CLI -b` / `CLI -p`, etc.). Match the patterns already used in the tree (`info` / `success` / `warning`, chunk lines, clipboard skip text, …).

### 7. Commits and push

- **Commit:** one coherent behavioural change; message describes behaviour, not filenames.
- **Tests:** not every tiny local commit must run the full suite; batch commits as you like. **Before push**, run what you rely on (at minimum the E2E suite) until green, then push.

---

## Commit message types

`TYPE: description` — imperative, lowercase description, ≤50-char title.

| Type | Use |
|------|-----|
| `feat:` | new capability |
| `fix:` | bug fix |
| `refactor:` | no behaviour change |
| `perf:` | speed |
| `test:` | add/change tests (not drive-by test edits) |
| `docs:` | docs / docstrings |
| `meta:` | `CLAUDE.md`, `.gitignore`, tooling |
| `chore:` | deps, CI |
| `style:` | format only |

---

## Maintainer conventions

These bullets are **part of this repo contract** (there is no separate “tooling README”):

- Mechanical multi-file text replace: `sed` / `grep -rl … | xargs sed` when context-free.
- **User-facing README:** when you refresh a flag section, align tables with **`asr2clip --help`** (paste verbatim per section refresh).
- Bash examples in README and `_build_parser()` epilog: pad `#` comments to **column 46**.
- **`_build_parser()`** argument groups: Setup → Audio → Transcription → Local ASR server → VAD → Diarization → Post-processing → Output.

**README structure:** mirror CLI groups above; do not duplicate the full config template or full `--help` in prose.
