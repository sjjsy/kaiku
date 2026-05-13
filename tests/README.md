# asr2clip test suite

## Running the tests

```bash
pytest tests/ -v          # all tests, verbose
pytest tests/ -q          # quiet (counts only)
pytest tests/ -k robust   # run one class by keyword
```

Test audio files are auto-downloaded into `test_data/` on first run and cached
there permanently (the directory is gitignored).

---

## Testing strategy

All tests treat asr2clip as a **black box**.  Every test calls the CLI as a
subprocess and asserts on **exit code**, **stdout**, **stderr**, and
**written files**.  No internal module is imported; no mock patches are
injected into the process.

This is intentional.  The tool's contract lives entirely at the CLI boundary.
A test that passes but allows the wrong backend to silently be selected is
worthless.

### How to test config resolution

The logging contract in CLAUDE.md requires that every config decision states
*why* a value was chosen — `"Using backend: wcpp (CLI -b)"` not just
`"Using backend: wcpp"`.  Those lines appear in stderr in verbose mode and
are the primary signal for config-resolution tests:

```python
result = _run("-i", wav, "-b", "wcpp", config=cfg)
assert "Using backend: wcpp (CLI -b)" in result.stderr
```

This is a full end-to-end assertion: the CLI flag travelled through
`Config.from_file` → `ASRBackendConfig.name` → the log call.  If any link in
that chain is broken, the assertion fails.  We do not need to import `Config`
to test it.

### How to test error paths

Exit code alone is sufficient for most error paths:

```python
result = _run("-i", wav, "--preset", "no-such-preset", config=cfg)
assert result.returncode != 0
```

When the error message itself is part of the contract (e.g. a missing device
must mention the device name), assert on `result.stderr`:

```python
assert "broken-mock" in result.stderr
```

### Inline config for edge cases

Tests that require a config structure that cannot be expressed via CLI flags
alone write their own minimal YAML inline:

```python
def _cfg_with_two_backends(tmp_path: Path) -> Path:
    cfg = textwrap.dedent("""\
        default_preset: preset-a
        presets:
          preset-a: [none, backend-a, none, ""]
          preset-b: [none, backend-b, none, ""]
        asr_backends:
          backend-a:
            type: mock
            response: "from A"
          backend-b:
            type: mock
            response: "from B"
        postprocessors: {}
    """)
    p = tmp_path / "config.yaml"
    p.write_text(cfg)
    return p
```

---

## Coverage index (37 tests)


| Test class                   | Test                                              | What it verifies                                                                                          |
| ---------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `TestConfigResolution`       | `test_default_preset_runs_without_flag`           | `default_preset: mock-fwd` in example config is valid and produces output                                 |
|                              | `test_short_preset_flag`                          | `-x NAME` is equivalent to `--preset NAME`                                                                |
|                              | `test_backend_flag_overrides_preset`              | `-b mock` overrides the default preset backend                                                            |
|                              | `test_backend_override_logged_with_source`        | stderr contains `"Using backend: mock (CLI -b)"` — tests the full config-resolution log contract          |
|                              | `test_preset_backend_logged_with_source`          | No `-b` flag → stderr mentions `"preset 'mock-fwd'"` as the source                                        |
|                              | `test_backend_bwd_differs_from_fwd`               | `-b mock-fwd` and `-b mock-bwd` produce different output                                                  |
|                              | `test_missing_preset_exits_nonzero`               | Unknown `--preset` name exits non-zero                                                                    |
|                              | `test_missing_config_exits_nonzero`               | Missing config file path exits non-zero                                                                   |
| `TestFileTranscription`      | `test_transcribe_to_stdout`                       | `-i FILE` writes transcript to stdout                                                                     |
|                              | `test_transcribe_to_output_file`                  | `-o FILE` appends transcript to a file                                                                    |
|                              | `test_language_flag_accepted`                     | `-l LANG` flag is accepted without error                                                                  |
|                              | `test_quiet_flag_reduces_output`                  | `-q` produces fewer stderr lines than default                                                             |
| `TestTranscriptMockBackends` | `test_mock_forward_returns_words_from_transcript` | `-b mock-fwd` draws words from the configured transcript file                                             |
|                              | `test_mock_backward_differs_from_forward`         | `-b mock-bwd` reverses word order relative to `-b mock-fwd`                                               |
|                              | `test_longer_audio_returns_more_words`            | Longer audio duration yields proportionally more words from mock-fwd                                      |
| `TestMockDevices`            | `test_mock_jfk_device_produces_output`            | `--device mock-jfk` loads `jfk-11s-1p.wav` and transcribes it                                             |
|                              | `test_mock_group_device_produces_output`          | `--device mock-group` loads `group-30s-4p.wav` and transcribes it                                         |
|                              | `test_mock_group_longer_than_jfk`                 | Longer mock device clip produces more words than shorter one                                              |
| `TestMockDiarization`        | `test_mock_diarize_2_produces_two_speakers`       | `-b mock-dia-2` produces exactly 2 distinct speaker labels                                                |
|                              | `test_mock_diarize_3_produces_three_speakers`     | `-b mock-dia-3` produces exactly 3 distinct speaker labels                                                |
|                              | `test_speakers_flag_overrides_speaker_count`      | `-s 1` collapses output to a single speaker regardless of backend default                                 |
|                              | `test_preset_triggers_diarization`                | A preset with `asr_backend: mock-dia-2` triggers diarization without `-b`                                 |
| `TestOutputTemplates`        | `test_raw_template_returns_transcript`            | `-T raw` passes through the transcript unchanged                                                          |
|                              | `test_bare_template_accepted`                     | `-T bare` flag is accepted without error                                                                  |
|                              | `test_unknown_template_falls_back_gracefully`     | Unknown `-T` name falls back and still produces transcript output                                         |
| `TestMockPostprocessor`      | `test_mock_postprocessor_runs`                    | `-P mock-pp` post-processor runs and produces non-empty output                                            |
|                              | `test_post_flag_overrides_no_postprocessor`       | `-P mock-pp` activates post-processing even when the active preset has none                               |
| `TestDeviceAbortOnFailure`   | `test_unknown_device_name_exits_nonzero`          | Requesting a non-existent device aborts in the recording path (no `-i`) instead of silently using another |
|                              | `test_broken_mock_device_source_exits_nonzero`    | A mock device with a missing source file aborts with an error message                                     |
| `TestRobustMode`             | `test_robust_exits_zero`                          | `-r` on a long audio file with `-b mock` exits 0                                                          |
|                              | `test_robust_produces_multiple_chunks`            | `-r -C 20` on the ~3.5 min fixture produces ≥ 5 chunks in stdout                                          |
|                              | `test_robust_chunk_progress_logged`               | Chunk progress lines (`Chunk N/M`) appear in stderr                                                       |
|                              | `test_robust_output_file_appended`                | `-r -o FILE` appends each chunk to the file as it completes                                               |
|                              | `test_robust_short_chunk_duration_flag`           | `-C 10` produces more chunks than `-C 20` on the same file                                                |
| `TestToggleMode`             | `test_first_toggle_creates_lock`                  | First `--toggle` invocation creates the lock file and exits 0                                             |
|                              | `test_second_toggle_removes_lock_and_transcribes` | Second `--toggle` removes the lock and writes a transcript to stdout                                      |
|                              | `test_toggle_stdout_is_empty_on_start`            | Start invocation produces no transcript output                                                            |
|                              | `test_toggle_with_output_file`                    | `--toggle -o FILE` appends the transcript to the file on stop                                             |


---

## Fixture summary


| Fixture       | Scope   | Source                  | Purpose                                                                        |
| ------------- | ------- | ----------------------- | ------------------------------------------------------------------------------ |
| `example_cfg` | session | `asr2clip.conf.example` | Patched with absolute `test_data/` paths and `default_preset: mock-fwd`        |
| `jfk_wav`     | session | whisper.cpp samples     | 11 s JFK speech (auto-downloaded)                                              |
| `group_wav`   | session | diarizen-tutorial       | 30 s group conversation, 4 speakers (auto-downloaded)                          |
| `long_speech` | session | Wikimedia Commons       | ~3.5 min George W. Bush radio address, 1 speaker, OGA format (auto-downloaded) |
| `silent_wav`  | session | generated               | 2 s silent WAV for tests that need a valid file but don't care about content   |


---

## Mock pipeline inventory (from `asr2clip.conf.example`)


| Stage             | Name         | Type           | Behaviour                                               |
| ----------------- | ------------ | -------------- | ------------------------------------------------------- |
| **Device**        | `mock-jfk`   | `mock`         | Serves `test_data/jfk-11s-1p.wav` (~11 s, 1 speaker)    |
| **Device**        | `mock-group` | `mock`         | Serves `test_data/group-30s-4p.wav` (~30 s, 4 speakers) |
| **ASR**           | `mock`       | `mock`         | Returns fixed fox sentence                              |
| **ASR**           | `mock-fwd`   | `mock-fwd`     | Returns N words from transcript (forward), N ∝ duration |
| **ASR**           | `mock-bwd`   | `mock-bwd`     | Returns N words from transcript (reverse), N ∝ duration |
| **ASR**           | `mock-dia-2` | `mock-diarize` | Round-robin 2-speaker diarization over transcript lines |
| **ASR**           | `mock-dia-3` | `mock-diarize` | Round-robin 3-speaker diarization over transcript lines |
| **Postprocessor** | `mock-pp`    | `mock`         | Returns canned post-processed string                    |


---

## Coverage gaps

Ordered by impact on correctness.  All gaps can be tested with the same
black-box subprocess strategy.

### Still missing from E2E tests


| Usage pattern                        | Gap                                     | Suggested assertion                                                            |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------ |
| `-p noisereduce -i FILE`             | Preprocessor flag never exercised       | Exit 0; stderr contains `"Preprocessing audio with noisereduce"`               |
| `--test`                             | Self-check command not covered          | Exit 0; stdout contains `"All checks passed"` or exit 1 with failure detail    |
| `-i file.txt`                        | `.txt` shortcut (bypass ASR) not tested | stdout equals the file content                                                 |
| `--serve` / `LocalAsrConfig`         | Needs optional `sherpa-onnx` dep        | Skip with `pytest.importorskip("sherpa_onnx")`; assert server starts           |
| `postprocessors extends/extra`       | Inheritance chain not tested            | Requires inline config with a prompt that extends another; assert final output |
| Real backends (groq, wcpp, whisperx) | Credential/hardware gated               | Separate `pytest -m integration` suite; skip by default                        |


---

