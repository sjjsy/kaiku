# Agent notes

Read [CLAUDE.md](CLAUDE.md) first — it is the full project contract (config rules, tests, logging, commits).

- One `Config` from `Config.from_file()` per run; pass `Config` (or resolved sub-configs), not raw YAML dicts.
- Behavioral logic belongs in `config_types.py`, not scattered `config.get()` calls.
- Prefer deleting duplicate paths and dead code over adding wrappers.
- Every config-related log line states *why* a value was chosen.
