# asr2clip E2E test suite

All tests invoke the `asr2clip` CLI as a subprocess and assert on exit code and
stdout/stderr. The session-scoped `example_cfg` fixture in `conftest.py` reads
`asr2clip.conf.example` directly, so every change to the example config is
automatically exercised — the suite doubles as a runnable demo of every mock
pipeline stage.

Run with:

```bash
pytest tests/ -v
```

## Coverage index

| Test class | Test | What it verifies |
|---|---|---|
| `TestConfigResolution` | `test_default_preset_runs_without_flag` | `default_preset: mock-fwd` in example config is valid and produces output |
| | `test_backend_flag_overrides_preset` | `-b mock` overrides the default preset backend |
| | `test_backend_bwd_differs_from_fwd` | `-b mock-fwd` and `-b mock-bwd` produce different output |
| | `test_missing_preset_exits_nonzero` | Unknown `--preset` name exits non-zero |
| | `test_missing_config_exits_nonzero` | Missing config file path exits non-zero |
| `TestFileTranscription` | `test_transcribe_to_stdout` | `-i FILE` writes transcript to stdout |
| | `test_transcribe_to_output_file` | `-o FILE` appends transcript to a file |
| | `test_language_flag_accepted` | `-l LANG` flag is accepted without error |
| | `test_quiet_flag_reduces_output` | `-q` produces fewer output lines than default |
| `TestTranscriptMockBackends` | `test_mock_forward_returns_words_from_transcript` | `-b mock-fwd` draws words from the configured transcript file |
| | `test_mock_backward_differs_from_forward` | `-b mock-bwd` reverses word order relative to `-b mock-fwd` |
| | `test_longer_audio_returns_more_words` | Longer audio duration yields proportionally more words from mock-fwd |
| `TestMockDevices` | `test_mock_jfk_device_produces_output` | `--device mock-jfk` loads `jfk-11s-1p.wav` and transcribes it |
| | `test_mock_group_device_produces_output` | `--device mock-group` loads `group-30s-4p.wav` and transcribes it |
| | `test_mock_group_longer_than_jfk` | Longer mock device clip produces more words than shorter one |
| `TestMockDiarization` | `test_mock_diarize_2_produces_two_speakers` | `-b mock-dia-2` produces exactly 2 distinct speaker labels |
| | `test_mock_diarize_3_produces_three_speakers` | `-b mock-dia-3` produces ≤ 3 speaker labels |
| | `test_speakers_flag_overrides_speaker_count` | `-s 1` collapses output to a single speaker regardless of backend default |
| | `test_preset_triggers_diarization` | A preset with `asr_backend: mock-dia-2` triggers diarization without `-b` |
| `TestOutputTemplates` | `test_raw_template_returns_transcript` | `-T raw` passes through the transcript unchanged |
| | `test_bare_template_accepted` | `-T bare` flag is accepted without error |
| | `test_unknown_template_falls_back_gracefully` | Unknown `-T` name falls back and still produces transcript output |
| `TestMockPostprocessor` | `test_mock_postprocessor_runs` | `-P mock-pp` post-processor runs and produces non-empty output |
| | `test_post_flag_overrides_no_postprocessor` | `-P mock-pp` activates post-processing even when the active preset has none |

## Fixture summary

| Fixture | Scope | Purpose |
|---|---|---|
| `example_cfg` | session | Patched copy of `asr2clip.conf.example` with absolute `test_data/` paths and `default_preset: mock-fwd` |
| `silent_wav` | session | 2-second silent WAV for tests that need a valid audio file but don't care about content |

## Mock pipeline inventory (from `asr2clip.conf.example`)

| Stage | Name | Type | Behaviour |
|---|---|---|---|
| **Device** | `mock-jfk` | mock | Serves `test_data/jfk-11s-1p.wav` (~11 s, 1 speaker) |
| **Device** | `mock-group` | mock | Serves `test_data/group-30s-4p.wav` (~30 s, 4 speakers) |
| **ASR** | `mock` | `mock` | Returns fixed fox sentence |
| **ASR** | `mock-fwd` | `mock-fwd` | Returns N words from transcript (forward), N ∝ duration |
| **ASR** | `mock-bwd` | `mock-bwd` | Returns N words from transcript (reverse), N ∝ duration |
| **ASR** | `mock-dia-2` | `mock-diarize` | Round-robin 2-speaker diarization over transcript lines |
| **ASR** | `mock-dia-3` | `mock-diarize` | Round-robin 3-speaker diarization over transcript lines |
| **Postprocessor** | `mock-pp` | `mock` | Returns canned post-processed string |
