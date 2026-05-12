# asr2clip Architecture

## Config System Design

### Current Design (as implemented)

The config system uses a **lazy-loading, classmethod-based resolution pattern**:

```
┌─────────────────────────────┐
│   YAML Config File          │
│  (asr2clip.conf)            │
└──────────────┬──────────────┘
               │ read_config()
               ▼
┌─────────────────────────────┐
│   config_dict (raw dict)    │
│   - asr_backends            │
│   - postprocessors          │
│   - presets                 │
│   - audio_device            │
└──────────────┬──────────────┘
               │ PresetConfig.resolve(config_dict, preset_name)
               ▼
┌─────────────────────────────┐
│   Preset (atomic pipeline)  │
│   - preprocessor name       │
│   - asr_backend name        │
│   - postprocessor name      │
└──────────────┬──────────────┘
               │ Config.resolve(config_dict, preset, cli_overrides)
               ▼
┌─────────────────────────────────────────────────────────┐
│   Config (lazy-loading coordinator)                     │
│   ├─ @property asr_backend → ASRBackendConfig           │
│   ├─ @property preprocessor → PreprocessorConfig        │
│   ├─ @property postprocessor → PostprocessorConfig      │
│   ├─ @property diarization → DiarizationConfig          │
│   └─ _cli_overrides (CliOverrides dataclass)            │
└─────────────────────────────────────────────────────────┘
```

**How it works:**

1. **Preset selection**: User specifies `--preset fast` or config has `default_preset: fast`
2. **Preset resolution**: `PresetConfig.resolve()` reads the preset list from config_dict
3. **Config coordination**: `Config.resolve()` creates the master Config object
4. **Lazy resolution**: Each sub-config (ASR backend, preprocessor, etc.) is resolved on first access
5. **CLI override precedence**: CliOverrides passed to Config, applied when resolving each component

**Code example:**
```python
# In asr2clip.py main()
config_dict = read_config(args.config)
preset_config = PresetConfig.resolve(config_dict, preset_name)
cli_overrides = CliOverrides(backend=args.backend, preprocessor=args.preprocessor, ...)
config = Config.resolve(config_dict, preset_config.preset, cli_overrides)

# Later, when needed:
backend_cfg = config.asr_backend  # Resolved on first access
preprocessor_cfg = config.preprocessor  # Resolved on first access
```

### Design Rationale (Current)

✅ **Advantages:**
- **Explicit dependencies**: Every function knows what config it needs
- **Testability**: Easy to mock individual config classes in tests
- **Late binding**: Config components only resolved if actually used
- **Type safety**: Each component has a specific dataclass (ASRBackendConfig, etc.)
- **Isolation**: Config logic separated into config_types.py

❌ **Disadvantages:**
- **Repeated patterns**: Every resolve() method takes `config_dict` and `cli_overrides`
- **Dual sources of truth**: Both `config_dict` and `Config` exist simultaneously
- **Bare config.get() calls**: Scattered throughout (diarize.py, backends/, postprocessors/)
- **Coupling to dict**: Many functions receive `config_dict` instead of Config object
- **Tedious chaining**: `config_dict` → Preset → Config → sub-config properties

---

## Your Proposal: Stateful Config Singleton

### What You're Suggesting

Instead of passing config around, create **one authoritative Config instance** that:

1. Is initialized once at startup with file + preset + CLI args
2. Becomes the single source of truth for all configuration
3. Is either globally accessible or passed once (not everywhere)
4. Eliminates the need for bare `config.get()` calls

**Pseudocode of proposed design:**
```python
# At startup
config = Config.from_file(
    config_file="~/.config/asr2clip/config.yaml",
    preset_name="fast",
    cli_overrides=CliOverrides(backend="groq", preprocessor="deepfilter")
)
# Config is now fully initialized and immutable

# Throughout the codebase
def diarize(audio_path: str) -> str:
    hf_token = config.diarization.hf_token  # Direct access
    # ... no passing config around

def make_postprocessor(name: str) -> PostProcessor:
    template = config.postprocessor.template  # Direct access
    # ... no config parameter needed

# Or with dependency injection:
def diarize(audio_path: str, cfg: Config) -> str:
    hf_token = cfg.diarization.hf_token  # Single parameter, full config
    # ... cleaner than passing config_dict
```

### Analysis of Your Proposal

✅ **Advantages:**
- **Single source of truth**: One Config object, not scattered dicts + Config
- **Eliminates config.get()**: All access through typed properties
- **Simpler signatures**: Functions don't need `config_dict` + `cli_overrides`
- **Cleaner flow**: Config loaded once, used everywhere
- **More testable**: Mock one Config object instead of dict + overrides
- **Matches real-world patterns**: Most CLI tools work this way (argparse → global state or singleton)

❌ **Disadvantages:**
- **Global state** (if truly global): Makes testing harder, hides dependencies
- **Less explicit**: Functions don't declare what config they need
- **Immutability harder**: Need to ensure Config can't be mutated mid-execution
- **Initialization order**: Config must be created before any component uses it

### Hybrid: Dependency Injection + Stateful Object

The best of both worlds:

```python
# At startup (main.py)
config_dict = read_config(args.config)
preset = PresetConfig.resolve(config_dict, preset_name)
config = Config.from_preset_and_overrides(
    config_dict=config_dict,
    preset=preset.preset,
    cli_overrides=CliOverrides(backend=args.backend, ...)
)

# Pass config ONCE through the call chain
def main():
    run_transcription(config)

def run_transcription(config: Config):
    process_file(config, args.input, args.output)

def process_file(config: Config, input_file: str, output_file: str):
    preprocessor = make_preprocessor(config.preprocessor.name)
    transcript = transcribe_file(config, input_file)
    output_transcript(config, transcript, output_file)

def transcribe_file(config: Config, input_file: str) -> str:
    if should_diarize():
        return run_diarization(config, input_file)  # Pass config once
    return transcribe_casual(config, input_file)

def run_diarization(config: Config, audio_path: str) -> str:
    hf_token = config.diarization.hf_token  # Direct access, typed
    # No more config.get("diarize_hf_token")
```

**This approach:**
- ✅ Has a single, fully-initialized Config object
- ✅ Eliminates bare `config.get()` calls
- ✅ Functions declare their dependency: `config: Config`
- ✅ Easy to test (inject mock Config)
- ✅ No hidden global state
- ✅ Clear initialization order (Config created before use)
- ✅ Functions are more testable (no implicit globals)

---

## Remaining Bare config.get() Calls

Currently there are ~20 remaining bare `config.get()` calls in:

| File | Issue | Fix |
|------|-------|-----|
| `asr2clip.py:127-129` | `config.get("default_preset")` | Load preset in main(), pass to functions |
| `asr2clip.py:69-81` | `backend_config.get("api_key")` | Already isolated (backend-specific dict) |
| `diarize.py:232` | `config.get("diarize_hf_token")` | Pass Config object instead of dict |
| `postprocessors/__init__.py` | `config.get("postprocessors")` | Pass Config, use PostprocessorConfig |
| `backends/*.py` | `config.get("backend_section")` | These are backend-specific (acceptable) |
| `local_asr/app.py` | Internal config building | Not on critical path |

**The real culprits** (high-level code that should use Config):
- `asr2clip.py:127-129` - preset loading logic (move to Config)
- `diarize.py:232` - should use Config.diarization
- `postprocessors/__init__.py` - should use Config properties

---

## Recommended Refactoring

### Phase 1: Make Config the authoritative source
1. Move preset-loading logic from main() into Config.from_file()
2. Create Config once, pass it through call chains
3. Eliminate `config_dict` from function signatures

### Phase 2: Remove bare config.get() calls
1. Update diarize() to accept Config
2. Update postprocessor functions to use Config
3. Move default_preset logic into Config initialization

### Phase 3: Consider optional singleton pattern
If dependency injection becomes verbose, add optional global:
```python
# In config_types.py
_global_config: Optional[Config] = None

def set_global_config(config: Config) -> None:
    global _global_config
    _global_config = config

def get_global_config() -> Config:
    if _global_config is None:
        raise RuntimeError("Config not initialized")
    return _global_config

# In tests/mocks
def reset_global_config() -> None:
    global _global_config
    _global_config = None
```

This way:
- Default: pass Config through (`clean, testable`)
- Optional: use global for leaf functions that don't need to be tested

---

## Conclusion

Your observation is **correct**: the current design has unnecessary complexity from passing `config_dict` around while also maintaining a `Config` object.

**Recommended approach:**
1. Keep Config as a stateful, fully-initialized object ✅
2. Pass it through the call chain (dependency injection) ✅
3. Eliminate all bare `config.get()` calls ✅
4. Optionally add a global for true leaf functions ⚠️ (last resort)

This gives us the best of both worlds:
- Single, authoritative config source
- Explicit dependencies (testable)
- Type safety (typed config properties)
- Clean function signatures
