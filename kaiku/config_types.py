"""Configuration type definitions and resolution classes.

Each Config class owns its domain completely: defaults, env vars, fallbacks,
validation, and logging. All resolution happens lazily via cached_property on
first access, sourced from the argparse Namespace (_args) and the YAML config
dict (_config_dict). No CliOverrides translation layer; _args is stored directly.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .audio import DeviceInfo
from .utils import info, warning

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Preset (plain data container — not a resolver)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    """Pipeline preset: an atomic combination of all processing stages.

    Format in config: [preprocessor, asr_backend, postprocessor, description]
    This order matches the pipeline flow: audio → preprocess → ASR → postprocess → output.
    """
    name: str
    preprocessor: str
    asr_backend: str
    postprocessor: str
    postprocessor_backend: Optional[str] = None
    description: Optional[str] = None

    def validate(self) -> None:
        if not self.preprocessor:
            raise ValueError(f"Preset '{self.name}': preprocessor is required")
        if not self.asr_backend or self.asr_backend == "none":
            raise ValueError(f"Preset '{self.name}': asr_backend is required (cannot be 'none')")
        if not self.postprocessor:
            raise ValueError(f"Preset '{self.name}': postprocessor is required")

    @classmethod
    def from_dict(cls, name: str, defn) -> "Preset":
        if not isinstance(defn, list) or len(defn) != 4:
            raise ValueError(
                f"Preset '{name}' must be a 4-element list: "
                "[preprocessor, asr_backend, postprocessor, description]"
            )
        preprocessor, asr_backend, postprocessor, description = defn
        preset = cls(
            name=name,
            preprocessor=preprocessor,
            asr_backend=asr_backend,
            postprocessor=postprocessor,
            postprocessor_backend=None,
            description=description,
        )
        preset.validate()
        return preset


# ---------------------------------------------------------------------------
# ASRBackendConfig
# ---------------------------------------------------------------------------

class ASRBackendConfig:
    """Lazy-resolved ASR backend configuration.

    Reads from asr_backends[name] in the YAML config dict. Backend name is
    resolved from CLI -b flag (highest priority) or the selected preset.
    All properties are cached on first access.
    """

    def __init__(self, config_dict: dict, preset: Preset, args: Any) -> None:
        self._config_dict = config_dict
        self._preset = preset
        self._args = args

    @functools.cached_property
    def name(self) -> str:
        n = getattr(self._args, "backend", None) or self._preset.asr_backend
        source = "CLI -b" if getattr(self._args, "backend", None) else f"preset '{self._preset.name}'"
        info(f"Using backend: {n} ({source})")
        return n

    @functools.cached_property
    def _defn(self) -> dict:
        backends = self._config_dict.get("asr_backends", {})
        if self.name not in backends:
            raise ValueError(
                f"Backend '{self.name}' not found. Available: {', '.join(backends)}"
            )
        return backends[self.name]

    @functools.cached_property
    def type(self) -> str:
        return self._defn.get("type", "api")

    @functools.cached_property
    def api_key(self) -> Optional[str]:
        if self.type not in ("api",):
            return None
        key = self._defn.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"Backend '{self.name}' requires api_key. "
                "Set in config or OPENAI_API_KEY env var."
            )
        key_source = "config" if self._defn.get("api_key") else "env var OPENAI_API_KEY"
        info(f"  API key: from {key_source}")
        return key

    @functools.cached_property
    def api_base_url(self) -> Optional[str]:
        url = self._defn.get("api_base_url")
        if self.type == "api" and not url:
            raise ValueError(f"Backend '{self.name}' requires api_base_url in config.")
        if url:
            info(f"  URL: {url}")
        return url

    @functools.cached_property
    def model_name(self) -> Optional[str]:
        m = self._defn.get("model_name")
        if self.type == "api" and not m:
            raise ValueError(f"Backend '{self.name}' requires model_name in config.")
        return m

    @functools.cached_property
    def org_id(self) -> Optional[str]:
        return self._defn.get("org_id") or os.environ.get("OPENAI_ORG_ID")

    @functools.cached_property
    def binary(self) -> Optional[str]:
        b = os.path.expanduser(self._defn.get("binary") or "")
        if self.type == "whisper_cpp" and not b:
            raise ValueError(f"Backend '{self.name}' requires binary path in config.")
        if b:
            info(f"  Binary: {b}")
        return b or None

    @functools.cached_property
    def model(self) -> Optional[str]:
        m = os.path.expanduser(self._defn.get("model") or "")
        if self.type == "whisper_cpp" and not m:
            raise ValueError(f"Backend '{self.name}' requires model path in config.")
        if m:
            info(f"  Model: {m}")
        return m or None

    @functools.cached_property
    def threads(self) -> int:
        t = self._defn.get("threads", 4)
        if self.type == "whisper_cpp":
            info(f"  Threads: {t}")
        return t

    @functools.cached_property
    def vad_model(self) -> Optional[str]:
        v = self._defn.get("vad_model")
        if not v:
            return None
        path = os.path.expanduser(v)
        if self.type == "whisper_cpp":
            info(f"  VAD model: {path} (whisper-cli --vad)")
        return path

    @functools.cached_property
    def response(self) -> Optional[str]:
        r = self._defn.get("response")
        if r:
            info(f"  Response: {len(r)} characters")
        return r

    @functools.cached_property
    def latency_ms(self) -> int:
        ms = self._defn.get("latency_ms", 0)
        if ms:
            info(f"  Simulated latency: {ms}ms")
        return ms

    @functools.cached_property
    def transcript_path(self) -> Optional[str]:
        p = self._defn.get("transcript_path")
        if p:
            info(f"  Transcript: {p}")
        return os.path.expanduser(p) if p else None

    @functools.cached_property
    def speaker_count(self) -> Optional[int]:
        v = self._defn.get("speaker_count")
        return int(v) if v is not None else None

    @functools.cached_property
    def hf_token(self) -> Optional[str]:
        token = self._defn.get("hf_token") or os.environ.get("HF_TOKEN")
        if token:
            src = "asr_backends entry" if self._defn.get("hf_token") else "env var HF_TOKEN"
            info(f"  HF token: from {src}")
        return token

    @functools.cached_property
    def min_speakers(self) -> Optional[int]:
        v = self._defn.get("speakers_min")
        return int(v) if v is not None else None

    @functools.cached_property
    def max_speakers(self) -> Optional[int]:
        v = self._defn.get("speakers_max")
        return int(v) if v is not None else None


# ---------------------------------------------------------------------------
# PostprocessorConfig (merges old PostprocessorConfig + PostprocessorBackendConfig)
# ---------------------------------------------------------------------------

class PostprocessorConfig:
    """Lazy-resolved post-processor configuration.

    Reads from postprocessors[name] in the YAML config dict. Name is resolved
    from CLI -P flag (highest priority) or the selected preset.
    """

    def __init__(self, config_dict: dict, preset: Preset, args: Any) -> None:
        self._config_dict = config_dict
        self._preset = preset
        self._args = args

    @functools.cached_property
    def name(self) -> str:
        n = getattr(self._args, "post", None) or self._preset.postprocessor
        source = "CLI -P" if getattr(self._args, "post", None) else f"preset '{self._preset.name}'"
        info(f"Using post-processor: {n} ({source})")
        return n

    @functools.cached_property
    def _defn(self) -> dict:
        if self.name == "none":
            return {}
        postprocessors = self._config_dict.get("postprocessors", {})
        if self.name not in postprocessors:
            available = ", ".join(postprocessors.keys())
            raise ValueError(
                f"Postprocessor '{self.name}' not found. Available: {available}"
            )
        return postprocessors[self.name]

    @functools.cached_property
    def template(self) -> str:
        return self._defn.get("template", "{result}")

    @functools.cached_property
    def backend_name(self) -> str:
        if self.name == "none":
            return "none"
        bn = self._defn.get("backend")
        if not bn:
            backends = self._config_dict.get("postprocessor_backends", {})
            bn = next(iter(backends.keys()), "none") if backends else "none"
        if bn:
            info(f"  Post-processor backend: {bn}")
        model_override = getattr(self._args, "post_model", None) or self._defn.get("model")
        if model_override:
            info(f"  Post-processor model override: {model_override}")
        return bn

    @functools.cached_property
    def model_override(self) -> Optional[str]:
        return getattr(self._args, "post_model", None) or self._defn.get("model")

    @functools.cached_property
    def _backend_defn(self) -> dict:
        if self.backend_name == "none":
            return {}
        backends = self._config_dict.get("postprocessor_backends", {})
        if self.backend_name not in backends:
            available = ", ".join(backends.keys())
            raise ValueError(
                f"Post-processor backend '{self.backend_name}' not found. Available: {available}"
            )
        return backends[self.backend_name]

    @functools.cached_property
    def backend_type(self) -> str:
        return self._backend_defn.get("type", "openai_compat") if self._backend_defn else "none"

    @functools.cached_property
    def backend_api_key(self) -> Optional[str]:
        defn = self._backend_defn
        if not defn:
            return None
        key = defn.get("api_key")
        api_key_env = defn.get("api_key_env")
        if key is None and api_key_env:
            key = os.environ.get(api_key_env)
        if key is None:
            key = os.environ.get("OPENAI_API_KEY")
        return key

    @functools.cached_property
    def backend_api_base_url(self) -> Optional[str]:
        return self._backend_defn.get("api_base_url") if self._backend_defn else None

    @functools.cached_property
    def model(self) -> Optional[str]:
        return self.model_override or (self._backend_defn.get("model") if self._backend_defn else None)


# ---------------------------------------------------------------------------
# RecorderConfig
# ---------------------------------------------------------------------------

class RecorderConfig:
    """Lazy-resolved audio input recorder and device configuration."""

    def __init__(self, config_dict: dict, args: Any) -> None:
        self._config_dict = config_dict
        self._args = args

    @functools.cached_property
    def _resolved(self) -> tuple:
        from .audio import DeviceInfo, resolve_device_preference_order
        from .recorders import PREFERENCE_ORDER, _CLASS_MAP
        from .utils import error

        cli_device = getattr(self._args, "device", None)
        device_spec = cli_device or self._config_dict.get("audio_device") or "auto"
        device_source = "CLI --device" if cli_device else ("config audio_device" if self._config_dict.get("audio_device") else "auto")

        # Check mock_devices before querying real hardware.
        # A mock device spec is any comma-separated token that appears as a key
        # in the config's mock_devices: section.
        mock_devices = self._config_dict.get("mock_devices", {})
        specs = [s.strip() for s in device_spec.split(",")] if isinstance(device_spec, str) else [str(s) for s in device_spec]
        for spec in specs:
            if spec in mock_devices:
                mock_cfg = mock_devices[spec]
                source_file = os.path.expanduser(mock_cfg.get("source_file", ""))
                if not source_file or not os.path.exists(source_file):
                    error(
                        f"Mock device '{spec}' source file not found: {source_file!r}. "
                        "Fix the source_file path in mock_devices config."
                    )
                    sys.exit(1)
                device_info = DeviceInfo(
                    index=-1, name=spec, portaudio_name=None, alsa_name=None,
                    channels=1, sample_rate=16000, mock_source=source_file,
                )
                info(f"Using mock device: {spec} (source: {source_file}) (from mock_devices config)")
                return "mock", device_info

        devices = resolve_device_preference_order(device_spec)

        if not devices:
            if device_spec == "auto":
                error(
                    "No audio input devices found. "
                    "Check microphone permissions or system audio configuration."
                )
            else:
                error(
                    f"None of the requested devices are available ({device_source}: {device_spec!r}). "
                    "Run 'kaiku --list-devices' to see what is available."
                )
            sys.exit(1)

        device_info = devices[0]
        info(f"Using device: {device_info.name}")

        recorder_name = None
        if device_info and device_info.portaudio_name is None and device_info.alsa_name:
            recorder_name = "arecord"
        else:
            for candidate in PREFERENCE_ORDER:
                if candidate in _CLASS_MAP:
                    if _CLASS_MAP[candidate]().is_available():
                        recorder_name = candidate
                        break

        if not recorder_name:
            raise ValueError("No audio recorders available (sounddevice or arecord)")

        info(f"Using recorder: {recorder_name}")
        return recorder_name, device_info

    @property
    def name(self) -> str:
        return self._resolved[0]

    @property
    def device(self) -> DeviceInfo:
        return self._resolved[1]


# ---------------------------------------------------------------------------
# DiarizationConfig
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LocalAsrConfig
# ---------------------------------------------------------------------------

class LocalAsrConfig:
    """Lazy-resolved settings for the bundled sherpa-onnx HTTP server."""

    def __init__(self, config_dict: dict, args: Any) -> None:
        self._config_dict = config_dict
        self._args = args

    @functools.cached_property
    def _raw(self) -> dict:
        raw = self._config_dict.get("local_asr") or {}
        return raw if isinstance(raw, dict) else {}

    @functools.cached_property
    def host(self) -> str:
        return getattr(self._args, "host", None) or self._raw.get("host") or "127.0.0.1"

    @functools.cached_property
    def port(self) -> int:
        v = getattr(self._args, "port", None)
        return int(v) if v is not None else int(self._raw.get("port", 8000))

    @functools.cached_property
    def model_dir(self) -> Optional[str]:
        v = getattr(self._args, "model_dir", None) or self._raw.get("model_dir")
        return os.path.expanduser(str(v)) if v else None

    @functools.cached_property
    def num_threads(self) -> int:
        v = getattr(self._args, "num_threads", None)
        return int(v) if v is not None else int(self._raw.get("num_threads", 4))

    @functools.cached_property
    def models_config_path(self) -> Optional[str]:
        v = (
            getattr(self._args, "local_asr_models_config", None)
            or self._raw.get("models_config")
            or self._raw.get("models_config_path")
        )
        return os.path.expanduser(str(v)) if v else None


# ---------------------------------------------------------------------------
# Config — master coordinator
# ---------------------------------------------------------------------------

class Config:
    """Master config coordinator. Lazy-loads and caches all configuration.

    Single entry point: Config.from_file(config_file, args). Stores the raw
    argparse Namespace as _args; all defaults live here, not in argparse.

    Priority (highest to lowest) for component selection (backend, preprocessor, post):
      1. Direct CLI flag (-b, -p, -P, -M, -d)
      2. CLI-selected preset component
      3. config default_preset component
      4. YAML config definition
      5. Built-in default defined in this file
    """

    def __init__(self, config_dict: dict, preset: Preset, args: Any) -> None:
        self._config_dict = config_dict
        self._preset = preset
        self._args = args

    @classmethod
    def from_file(cls, config_file: str | None, args: Any) -> "Config":
        """Primary entry point. Reads YAML, resolves preset, stores args."""
        from .config import read_config
        from .utils import error, info

        # Fail fast when the caller explicitly provides a path that doesn't exist,
        # rather than silently falling back to the user's default config.
        if config_file and not os.path.exists(config_file):
            error(f"Config file not found: {config_file}")
            sys.exit(1)

        resolved_path, config_dict = read_config(config_file)
        n_params = len(config_dict)

        if config_file:
            info(f"Using configuration file: {resolved_path} (-c/--config) — Provides {n_params} definitions")
        else:
            info(f"Using configuration file: {resolved_path} (default discovery) — Provides {n_params} definitions")

        if not config_dict:
            error("Config file is empty or contains only comments.")
            sys.exit(1)

        preset_name = getattr(args, "preset", None) or config_dict.get("default_preset")
        if not preset_name:
            available = ", ".join(config_dict.get("presets", {}).keys())
            error(
                "No preset selected. Use --preset NAME or set 'default_preset: NAME' in config.\n"
                f"Available presets: {available}"
            )
            sys.exit(1)

        presets = config_dict.get("presets", {})
        if not presets:
            error("No 'presets:' section found in config.")
            sys.exit(1)
        if preset_name not in presets:
            available = ", ".join(presets.keys())
            error(f"Preset '{preset_name}' not found. Available: {available}")
            sys.exit(1)

        try:
            preset = Preset.from_dict(preset_name, presets[preset_name])
        except ValueError as e:
            error(f"Preset error: {e}")
            sys.exit(1)

        desc = (preset.description or "").strip()
        if desc:
            info(f"Using preset: {preset_name} — {desc}")
        else:
            info(f"Using preset: {preset_name}")

        return cls(config_dict, preset, args)

    # --- sub-config objects (lazy, cached) ---

    @functools.cached_property
    def asr_backend(self) -> ASRBackendConfig:
        return ASRBackendConfig(self._config_dict, self._preset, self._args)

    @functools.cached_property
    def postprocessor(self) -> PostprocessorConfig:
        return PostprocessorConfig(self._config_dict, self._preset, self._args)

    @functools.cached_property
    def recorder(self) -> RecorderConfig:
        return RecorderConfig(self._config_dict, self._args)

    @functools.cached_property
    def local_asr(self) -> LocalAsrConfig:
        return LocalAsrConfig(self._config_dict, self._args)

    # --- inline single-value config (no sub-object needed) ---

    @functools.cached_property
    def preprocessor(self) -> str:
        """Preprocessor name: CLI ``-p`` overrides the preset's first field."""
        cli = getattr(self._args, "preprocessor", None)
        n = (cli if cli else self._preset.preprocessor)
        source = "CLI -p" if cli else f"preset '{self._preset.name}'"
        info(f"Using preprocessor: {n} ({source})")
        return n

    @property
    def clipboard_max_chars(self) -> int:
        from .output import _DEFAULT_CLIPBOARD_MAX_CHARS
        return int(self._config_dict.get("clipboard_max_chars", _DEFAULT_CLIPBOARD_MAX_CHARS))

    @property
    def quiet(self) -> bool:
        return bool(self._config_dict.get("quiet", False))

    # --- run-level properties (args pass-through with defaults) ---

    @property
    def input_file(self) -> Optional[str]:
        return getattr(self._args, "input", None)

    @property
    def output_file(self) -> Optional[str]:
        return getattr(self._args, "output", None)

    @property
    def no_clipboard(self) -> bool:
        """True when ``--no-clipboard`` was passed (no wl-copy / copykitten)."""
        return bool(getattr(self._args, "no_clipboard", False))

    @property
    def language(self) -> Optional[str]:
        return getattr(self._args, "language", None)

    @property
    def template(self) -> Optional[str]:
        """CLI -T output template override."""
        return getattr(self._args, "template", None)

    @property
    def num_speakers(self) -> Optional[int]:
        return getattr(self._args, "speakers", None)

    @property
    def robust(self) -> bool:
        return bool(getattr(self._args, "robust", False))

    @property
    def chunk_duration(self) -> int:
        """Max chunk duration in seconds for robust mode. Default: 180."""
        v = getattr(self._args, "chunk_duration", None)
        return int(v) if v is not None else 180

    @property
    def interval(self) -> Optional[float]:
        """Fixed-interval continuous recording in seconds. None = not enabled."""
        return getattr(self._args, "interval", None)

    @property
    def vad(self) -> bool:
        return bool(getattr(self._args, "vad", False))

    @property
    def silence_threshold(self) -> float:
        """VAD silence probability threshold. Default: 0.5."""
        v = getattr(self._args, "silence_threshold", None)
        return float(v) if v is not None else 0.5

    @property
    def silence_duration(self) -> float:
        """Silence duration to trigger transcription. Default: 1.5s."""
        v = getattr(self._args, "silence_duration", None)
        return float(v) if v is not None else 1.5
