# kaiku test suite

## Running the tests

```bash
pytest tests/ -v          # all tests, verbose
pytest tests/ -q          # quiet (counts only)
pytest tests/ -k robust   # run one class by keyword
```

On first run, the session fixture calls the same download path as
`kaiku --download-fixtures` into a temporary directory and sets
`KAIKU_FIXTURE_DIR` so `-d <basename>` resolves without `mock_devices` in YAML.
Reference transcripts live in
[`fixtures/transcripts/`](fixtures/transcripts/) (plain text, in git).

Project-wide principles (config contract, commits, **when agents may change E2E tests**): [CLAUDE.md](../CLAUDE.md).

---

## Testing strategy

All tests treat kaiku as a **black box**.  E2E tests call the CLI as a
subprocess and assert on **exit code**, **stdout**, **stderr**, and **written
files**.  No internal module is imported; no mocks are injected into the
process.

This is intentional.  The tool's contract lives entirely at the CLI boundary.
A test that passes but allows the wrong backend to be selected is worthless.

### Invocation budget and assertion density

E2E scenarios are organized for **few subprocess invocations** and **many
orthogonal assertions per `CompletedProcess`**.  Each expensive run (long
audio, toggle start/stop) carries a **checklist** of predicates so unrelated
regressions surface in one failure (wrong backend, missing *why* in logs,
broken `-o`, post-processor not invoked, …).

Parametrization that only duplicates CLI runs is avoided; merging means **one**
`subprocess.run` with **ten** stderr/stdout/file checks, not ten tests each
calling the CLI once for a single assert.

The shared helper `tests/test_e2e.py::_run` injects `--no-clipboard` by
default so the suite does not spawn clipboard helpers.  Clipboard behaviour is
covered explicitly in `TestClipboardE2E` with `clipboard=True`.

### Log format in assertions

CLI stderr uses the coloured formatter (`timestamp │ LEVEL │ message` with
`INFO`, `OK`, `WARN`, …).  Tests assert on **stable message substrings**
(clipboard skip text, `Using backend: …`, chunk progress, …), not on ANSI
codes or exact spacing around the level column.

### How to test config resolution

The logging contract in CLAUDE.md requires that every config decision states
*why* a value was chosen — for example `Using backend: wcpp (CLI -b)` rather
than only `Using backend: wcpp`.  Those lines appear in stderr in verbose
mode and are stacked alongside behavioural assertions in the same run:

```python
result = _run("-i", wav, "-b", "wcpp", config=cfg)
assert result.returncode == 0
assert "Using backend: wcpp (CLI -b)" in result.stderr
```

This is a full end-to-end check: the flag reached `Config.from_file` →
`ASRBackendConfig` → the log call.  Importing `Config` in the test is not
required.

### How to test error paths

Exit code alone is sufficient for some errors; add stderr (and absence of
output files) when the message is part of the contract:

```python
result = _run("-i", wav, "--preset", "no-such-preset", config=cfg)
assert result.returncode != 0
assert "no-such-preset" in result.stderr.lower() or "preset" in result.stderr.lower()
```

### Inline config for edge cases

Tests that require a config structure that cannot be expressed via CLI flags
alone write their own minimal YAML inline (see `TestDeviceAbortOnFailure` in
`test_e2e.py`).

---

## E2E scenario index

### `TestPresetAndFilePipeline`

#### `test_default_preset_matches_short_flag` (×2 invocations)

- Same stdout when using YAML `default_preset` vs explicit `-x mock-fwd`
- `Using backend: mock-fwd (preset 'mock-fwd')` in stderr for both runs
- Transcript is not the fixed `-b mock` fox sentence

#### `test_file_input_dense_primary_contract` (×3 invocations)

- **Run 1:** `-i` WAV with `-p none`, `-b mock`, `-T raw`, `-P mock-pp`, `-l`, `-o` — backend and post *why* lines, mock ASR logs, `Using preprocessor: none (CLI -p)`, **no** `Preprocessing completed` timing line for preprocessing (none preprocessor), `Clipboard: skipped (--no-clipboard)`, post-processed stdout/file, `Prompt analyzed` word count for base `mock-pp` (`lines=1, words=8`), transcript analysis includes `words=9` for the fox sentence
- **Run 2:** unknown `-T` name with `-b mock` — fallback still returns canned fox transcript on stdout
- **Run 3:** `-b mock` + `-P mock-pp2` — `mock-pp2` (`extends: mock-pp` + `extra` in example config) yields a **wider** resolved prompt than `mock-pp` alone (assert `lines=3, words=22` on the `Prompt analyzed:` line)

#### `test_quiet_matches_verbose_transcript_with_fewer_logs` (×2 invocations)

- `-b mock`: identical stdout with and without `-q`
- Verbose stderr has many lines; quiet stderr is empty (logger at ERROR)

### `TestPlainTextInput`

#### `test_txt_input_skips_asr_and_postprocesses` (×1 invocation)

- `-i` path ending in `.txt` — stderr shows `Processing file` for that path
- No mock ASR / transcription-completed lines (ASR bypassed)
- `-P mock-pp2` post-processes file contents; stdout contains transcript-analysis tokens from the file text

### `TestSelfCheck`

#### `test_cli_test_mode_passes_with_mock_preset` (×1 invocation)

- `--test` exits 0 with `example_cfg` (mock preset)
- Stderr includes `All checks passed`, mock-backend skip line (no API probe), preprocessor check for `none`, and the “no post-processors configured to check” success path

### `TestConfigErrors`

#### `test_missing_preset_exits_nonzero` (×1 invocation)

- Non-zero exit; stderr mentions missing preset

#### `test_missing_config_exits_nonzero` (×1 invocation)

- Non-zero exit for a non-existent `--config` path

### `TestMockTranscriptBackends`

#### `test_mock_fwd_bwd_and_duration_scaling` (×3 invocations)

- `mock-fwd` vs `mock-bwd` differ on the same short WAV
- Longer WAV yields strictly more words than short WAV with `mock-fwd`

### `TestDemoAudioEarly` (runs first)

#### `test_mock_1p_device_records_jfk_clip` (×1 invocation)

- `demo-1p-011s-en-jfk` device + `mock-fwd` preset — JFK clip
- Stdout is a leading word-for-word prefix of [`demo-1p-011s-en-jfk.txt`](fixtures/transcripts/demo-1p-011s-en-jfk.txt)

#### `test_jfk_file_input_mock_fwd_matches_transcript` (×1 invocation)

- `-i` JFK WAV + `-b mock-fwd` — backend *why* line in stderr
- Same prefix check against `demo-1p-011s-en-jfk.txt`

#### `test_german_demo_device_with_language_flag` (×1 invocation)

- `-d demo-3p-096s-de-eoc` + `-l de` — `Using language: de (CLI -l)` in stderr

### `TestMockRecordingDevices`

#### `test_devices_transcribe_with_resolution_logs_and_duration_ordering` (×3 invocations)

- `demo-1p-011s-en-jfk`, `demo-4p-030s-en-ami`, `demo-3p-096s-de-eoc` devices all exit 0 with non-empty stdout
- `Using mock device: …` resolution lines in stderr
- Word counts increase with clip length (11 s → 30 s → ~96 s)

### `TestDiarizationOptional` (uses `diarization_cfg`, not default `example_cfg`)

#### `test_mock_dia_4p_matches_fixture` / `test_mock_dia_3p_matches_fixture` (×2 invocations)

- `mock-dia-4` / `mock-dia-3` on matching demo WAVs
- Speaker labels and word recall vs `demo-4p-030s-en-ami.txt` / `demo-3p-096s-de-eoc.txt`

#### `test_whisperx_4p_matches_fixture` / `test_whisperx_3p_matches_fixture` (×2 invocations, skipped without addon)

- Requires `pip install kaiku[diarize]` and `HF_TOKEN`
- Compares WhisperX stdout to the diarized fixture transcripts (lower word-recall bar than mock-diarize)

### `TestDeviceAbortOnFailure`

#### `test_unknown_device_name_exits_nonzero` (×1 invocation)

- Recording path with unknown `--device` exits non-zero

#### `test_broken_mock_device_source_exits_nonzero` (×1 invocation)

- Inline YAML with a mock device whose `source_file` is missing — non-zero exit and stderr names the device or source problem

### `TestRobustMode`

#### `test_chunking_stderr_file_and_chunk_duration_flag` (×2 invocations)

- First run: `-r` on `long_speech` with `-b mock`, `-C 20`, `-o` — chunk progress in stderr (≥ 5 `Chunk n/N` lines), non-empty output file overwritten with final `format_output` text only (no `append_transcript_to_file`); if `noisereduce` is importable, the same invocation adds `-p noisereduce` and stderr must contain `Using preprocessor: noisereduce`
- Second run: same file with `-C 10` and `-o` elsewhere — strictly more `Chunk n/N` lines in stderr than the 20 s chunk run

### `TestToggleMode`

#### `test_lifecycle_lock_stdout_stderr_and_output_file` (×2 invocations)

- Start `--toggle`: lock file exists, stdout empty, `Recording started` in stderr
- Stop `--toggle`: lock removed, non-empty stdout, output file populated, stop/transcribe chatter in stderr

### `TestClipboardE2E`

#### `test_clipboard_opt_in_reports_transcript_copied` (×1 invocation)

- `clipboard=True` (no default `--no-clipboard`) — stderr reports transcript copied to clipboard

### Unit tests (`tests/test_no_clipboard_config.py`)

- `Config.no_clipboard` is true when the CLI flag is set and false otherwise

---

## Fixtures

### `fixture_dir` / `example_cfg` (session)

- `download_fixtures()` into a temp dir (same as CLI)
- `example_cfg` copies `kaiku.conf.example`; `KAIKU_FIXTURE_DIR` points at downloaded clips
- E2E-only `mock-dia-4` / `mock-dia-3` backends injected in `diarization_cfg` (not in shipped example config)

### `demo_1p_wav` / `demo_4p_wav` / `demo_3p_wav` / `long_speech` (session)

- Paths under `fixture_dir` after download

### `diarization_cfg` (session)

- Like `example_cfg` plus injected `mock-dia-4` / `mock-dia-3` (and `whisperx` when `HF_TOKEN` is set)
- Used only by `TestDiarizationOptional`

### `long_speech` (session)

- `demo-1p-127s-en-gb0.wav` (~127 s Bush radio address); used only for robust chunking (`-r`), not one of the three tiered device demos

### `silent_wav` (session)

- Generated silent WAV (~2 s) for cheap file-input tests

---

## Mock pipeline inventory (from `kaiku.conf.example`)

### Devices (`mock_devices` config, else fixture dir basename = `-d` name)

- `demo-1p-011s-en-jfk` — ~11 s, one speaker (JFK)
- `demo-4p-030s-en-ami` — ~30 s, AMI meeting (~4 speakers)
- `demo-3p-096s-de-eoc` — ~96 s, German interview
- `demo-1p-127s-en-gb0` — ~127 s, Bush radio address (WAV)
- `demo-2p-023s-en-courtney`, `demo-4p-082s-en-agni`, `demo-3p-051s-fi-metro` — extended clips

After `kaiku --download-fixtures`, try e.g. `kaiku -d demo-1p-011s-en-jfk -x mock-fwd`.

### ASR backends

- `**mock**` — fixed canned fox sentence
- `**mock-fwd` / `mock-bwd**` — duration-scaled word counts from a transcript file (forward vs reverse)
- `**mock-dia-4` / `mock-dia-3**` — mock diarization with four or three speaker labels (E2E only)

### Post-processors

- `**mock-pp**` — mock backend; linguistic analysis of resolved system prompt + transcript
- `**mock-pp2**` — `extends: mock-pp` with additional `extra` prose in config; used in E2E to assert merged prompt statistics (wider `Prompt analyzed: … words=…` than `mock-pp` alone)

---

## Demo clip catalog

Naming: `demo-{Np}-{DDD}s-{lang}-{codename}` (duration zero-padded to three digits).

Audio is **not** in git. Download with `kaiku --download-fixtures` into
`~/.local/share/kaiku/fixtures`; each file is usable as `kaiku -d <basename>` without
editing config. **Transcripts** (reference text for tests and optional ASR checks):

| Transcript | ~Duration | Speakers | Lang |
|------------|-----------|----------|------|
| [demo-1p-011s-en-jfk.txt](fixtures/transcripts/demo-1p-011s-en-jfk.txt) | 11 s | 1 | en |
| [demo-4p-030s-en-ami.txt](fixtures/transcripts/demo-4p-030s-en-ami.txt) | 30 s | ~4 | en |
| [demo-3p-096s-de-eoc.txt](fixtures/transcripts/demo-3p-096s-de-eoc.txt) | 96 s | 3 | de |
| [demo-1p-127s-en-gb0.txt](fixtures/transcripts/demo-1p-127s-en-gb0.txt) | 127 s | 1 | en |
| [demo-2p-023s-en-courtney.txt](fixtures/transcripts/demo-2p-023s-en-courtney.txt) | 23 s | 2 | en |
| [demo-4p-082s-en-agni.txt](fixtures/transcripts/demo-4p-082s-en-agni.txt) | 82 s | 3–4 | en |
| [demo-3p-051s-fi-metro.txt](fixtures/transcripts/demo-3p-051s-fi-metro.txt) | 51 s | 3 | fi |

`SPEAKER_*` lines in some transcripts are approximate fixture labels for diarization tests, not ground truth.

Sources: whisper.cpp JFK sample; DiariZen AMI excerpt; Wikimedia Element of Crime / Bush address / Courtney Love BBC / Agni-III Wikinews / Finnish metro dialogue — see git history and `kaiku/fixtures.py` URLs.

---

## Extended E2E (optional real ASR)

Not part of default `pytest tests/`. Run when you choose a backend:

```bash
KAIKU_EXTENDED_BACKEND=wcpp pytest tests/test_e2e_extended.py -v -m extended
```

- **Non-diarizing backends** (`wcpp`, API names, …): every `demo-1p-*` mock device, with `-l en` on English solos and `-l de` / `-l fi` on German/Finnish clips where applicable; stdout compared to reference transcripts (word-recall threshold).
- **Diarizing backends** (`whisperx`, …): every multi-speaker `demo-*p-*` device; diarized stdout vs reference `.txt`.

Set `KAIKU_EXTENDED_BACKEND` to a key under `asr_backends:` in your config. Requires credentials/models for that backend.

---

## Coverage gaps

Remaining items suitable for future black-box tests (or a separate integration  
tier).  Items covered by the suite since 2026-05-13 are listed under **Covered** below for  
traceability.

### Covered in the current E2E suite

- `**-p none`** — folded into `test_file_input_dense_primary_contract` (`Using preprocessor: none (CLI -p)`; no `Preprocessing completed` from the preprocessor step)
- `**-p noisereduce`** — optional: when `noisereduce` is installed, `TestRobustMode` adds `-p noisereduce` to the long-audio robust run and asserts `Using preprocessor: noisereduce`
- `**postprocessors` `extends` / `extra**` — `mock-pp2` in the example config; prompt-width assertions vs `mock-pp` in the dense file test and `.txt` input test
- `**--test**` — `TestSelfCheck::test_cli_test_mode_passes_with_mock_preset` (requires mock-class backends to skip the API probe; see `test_config` in `kaiku.py`)
- `**-i` `.txt` shortcut** — `TestPlainTextInput::test_txt_input_skips_asr_and_postprocesses`

### Still missing (E2E or integration)

- `**--serve` / `LocalAsrConfig`** — needs optional `sherpa-onnx`; use `pytest.importorskip` and assert the server binds / responds, or keep out of default CI
- **Real ASR backends** — optional `tests/test_e2e_extended.py` (`-m extended`, `KAIKU_EXTENDED_BACKEND=…`); not in default CI
- **VAD / continuous modes** (`--vad`, `--interval`) — heavy deps and long-running processes; optional suite
- `**pyrnnoise` / `deepfilter` preprocessors** — same pattern as `noisereduce`: assert only when the extra is installed

---

