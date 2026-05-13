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

All tests treat asr2clip as a **black box**.  E2E tests call the CLI as a
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
broken ``-o``, post-processor not invoked, …).

Parametrization that only duplicates CLI runs is avoided; merging means **one**
``subprocess.run`` with **ten** stderr/stdout/file checks, not ten tests each
calling the CLI once for a single assert.

The shared helper ``tests/test_e2e.py::_run`` injects ``--no-clipboard`` by
default so the suite does not spawn clipboard helpers.  Clipboard behaviour is
covered explicitly in ``TestClipboardE2E`` with ``clipboard=True``.

### How to test config resolution

The logging contract in CLAUDE.md requires that every config decision states
*why* a value was chosen — for example ``Using backend: wcpp (CLI -b)`` rather
than only ``Using backend: wcpp``.  Those lines appear in stderr in verbose
mode and are stacked alongside behavioural assertions in the same run:

```python
result = _run("-i", wav, "-b", "wcpp", config=cfg)
assert result.returncode == 0
assert "Using backend: wcpp (CLI -b)" in result.stderr
```

This is a full end-to-end check: the flag reached ``Config.from_file`` →
``ASRBackendConfig`` → the log call.  Importing ``Config`` in the test is not
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
alone write their own minimal YAML inline (see ``TestDeviceAbortOnFailure`` in
``test_e2e.py``).

---

## E2E scenario index

Each row is one **test method**; a method may run the CLI **more than once**
only when the contract requires it (e.g. compare ``-C 10`` vs ``-C 20``, or
toggle start then stop).  The **Checklist** column lists the main assertion
groups, not every individual ``assert``.

| Class | Method | Invocations (typ.) | Checklist |
| ----- | ------ | ------------------ | --------- |
| ``TestPresetAndFilePipeline`` | ``test_default_preset_matches_short_flag`` | 2 | Same stdout for default YAML preset vs ``-x mock-fwd``; ``Using backend: mock-fwd (preset 'mock-fwd')`` in stderr for both; transcript not the fixed mock sentence |
| | ``test_file_input_dense_primary_contract`` | 3 | **(1)** ``-i -b mock -T raw -P mock-pp -l -o``: backend/post/mock ASR logs, clipboard skip line, post-processed stdout, ``words=9`` on fox transcript, output file contains analysis **(2)** unknown ``-T`` fallback still returns canned mock transcript **(3)** ``-P mock-pp`` alone: preset backend + CLI post *why*, post-processed output |
| | ``test_quiet_matches_verbose_transcript_with_fewer_logs`` | 2 | ``-b mock``: identical stdout with and without ``-q``; quieter stderr |
| ``TestConfigErrors`` | ``test_missing_preset_exits_nonzero`` | 1 | Non-zero exit; stderr mentions missing preset |
| | ``test_missing_config_exits_nonzero`` | 1 | Non-zero exit for missing config path |
| ``TestMockTranscriptBackends`` | ``test_mock_fwd_bwd_and_duration_scaling`` | 3 | ``mock-fwd`` vs ``mock-bwd`` differ on same file; longer audio yields more words than short |
| ``TestMockRecordingDevices`` | ``test_devices_transcribe_with_resolution_logs_and_duration_ordering`` | 2 | Both mock devices exit 0 with non-empty stdout; ``Using mock device: …`` in stderr; group clip more words than JFK |
| ``TestMockDiarization`` | ``test_speaker_counts_and_cli_override`` | 3 | ``mock-dia-2`` / ``mock-dia-3`` speaker counts; ``-s 1`` collapses to one speaker |
| ``TestDeviceAbortOnFailure`` | ``test_unknown_device_name_exits_nonzero`` | 1 | Recording path aborts on unknown ``--device`` |
| | ``test_broken_mock_device_source_exits_nonzero`` | 1 | Mock device with missing ``source_file`` aborts with actionable stderr |
| ``TestRobustMode`` | ``test_chunking_stderr_file_and_chunk_duration_flag`` | 2 | ``-r`` with ``-o``: chunk progress in stderr, ≥ 5 chunk paragraphs in file; shorter ``-C`` yields strictly more chunks than longer ``-C`` |
| ``TestToggleMode`` | ``test_lifecycle_lock_stdout_stderr_and_output_file`` | 2 | Start: lock exists, empty stdout, ``Recording started`` in stderr; stop: lock gone, transcript on stdout and ``-o`` file, stop/transcribe logs |
| ``TestClipboardE2E`` | ``test_clipboard_opt_in_reports_transcript_copied`` | 1 | With ``clipboard=True``, stderr reports transcript copied |

Unit tests: ``tests/test_no_clipboard_config.py`` (``Config.no_clipboard`` from argparse).

---

## Fixture summary

| Fixture       | Scope   | Source                  | Purpose                                                                        |
| ------------- | ------- | ----------------------- | ------------------------------------------------------------------------------ |
| ``example_cfg`` | session | ``asr2clip.conf.example`` | Patched with absolute ``test_data/`` paths and ``default_preset: mock-fwd``        |
| ``jfk_wav``     | session | whisper.cpp samples     | 11 s JFK speech (auto-downloaded)                                              |
| ``group_wav``   | session | diarizen-tutorial       | 30 s group conversation, 4 speakers (auto-downloaded)                          |
| ``long_speech`` | session | Wikimedia Commons       | ~3.5 min George W. Bush radio address, 1 speaker, OGA format (auto-downloaded) |
| ``silent_wav``  | session | generated               | 2 s silent WAV for tests that need a valid file but don't care about content   |

---

## Mock pipeline inventory (from ``asr2clip.conf.example``)

| Stage             | Name         | Type           | Behaviour                                               |
| ----------------- | ------------ | -------------- | ------------------------------------------------------- |
| **Device**        | ``mock-jfk``   | ``mock``         | Serves ``test_data/jfk-11s-1p.wav`` (~11 s, 1 speaker)    |
| **Device**        | ``mock-group`` | ``mock``         | Serves ``test_data/group-30s-4p.wav`` (~30 s, 4 speakers) |
| **ASR**           | ``mock``       | ``mock``         | Returns fixed fox sentence                              |
| **ASR**           | ``mock-fwd``   | ``mock-fwd``     | Returns N words from transcript (forward), N ∝ duration |
| **ASR**           | ``mock-bwd``   | ``mock-bwd``     | Returns N words from transcript (reverse), N ∝ duration |
| **ASR**           | ``mock-dia-2`` | ``mock-diarize`` | Round-robin 2-speaker diarization over transcript lines |
| **ASR**           | ``mock-dia-3`` | ``mock-diarize`` | Round-robin 3-speaker diarization over transcript lines |
| **Postprocessor** | ``mock-pp``    | ``mock``         | Returns deterministic linguistic analysis of prompt + transcript |

---

## Coverage gaps

Ordered by impact on correctness.  All gaps can be tested with the same
black-box subprocess strategy when added.

### Still missing from E2E tests

| Usage pattern                        | Gap                                     | Suggested assertion                                                            |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------ |
| ``-p noisereduce -i FILE``             | Preprocessor flag never exercised       | Exit 0; stderr contains ``Preprocessing audio with noisereduce``               |
| ``--test``                             | Self-check command not covered          | Exit 0; stdout contains ``All checks passed`` or exit 1 with failure detail    |
| ``-i file.txt``                        | ``.txt`` shortcut (bypass ASR) not tested | stdout equals the file content                                                 |
| ``--serve`` / ``LocalAsrConfig``         | Needs optional ``sherpa-onnx`` dep        | ``pytest.importorskip("sherpa_onnx")``; assert server starts           |
| ``postprocessors`` ``extends``/``extra``       | Inheritance chain not tested            | Requires inline config with a prompt that extends another; assert final output |
| Real backends (groq, wcpp, whisperx) | Credential/hardware gated               | Separate ``pytest -m integration`` suite; skip by default                        |

---
