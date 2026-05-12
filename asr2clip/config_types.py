"""Configuration type definitions and resolution classes.

Each Config class owns its domain completely: defaults, env vars, fallbacks,
validation, and logging. All resolution happens once at startup and is cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .audio import DeviceInfo
from .utils import info, warning


@dataclass(frozen=True)
class Preset:
    """Pipeline preset: an atomic combination of all processing stages.

    Presets allow users to define complete pipelines in config and select
    one per run. All stages must be explicitly specified — no defaults.

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
        """Validate preset completeness. Raises ValueError if incomplete."""
        if not self.preprocessor:
            raise ValueError(f"Preset '{self.name}': preprocessor is required")
        if not self.asr_backend or self.asr_backend == "none":
            raise ValueError(f"Preset '{self.name}': asr_backend is required (cannot be 'none')")
        if not self.postprocessor:
            raise ValueError(f"Preset '{self.name}': postprocessor is required")

    @classmethod
    def from_dict(cls, name: str, defn) -> "Preset":
        """Create Preset from a config entry (list format only).

        List format: [preprocessor, asr_backend, postprocessor, description]
        """
        if not isinstance(defn, list):
            raise ValueError(
                f"Preset '{name}' must be a 4-element list: "
                "[preprocessor, asr_backend, postprocessor, description]"
            )

        if len(defn) != 4:
            raise ValueError(
                f"Preset '{name}' list must have exactly 4 elements: "
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


@dataclass
class CliOverrides:
    """CLI flag overrides for config values.

    Precedence: Individual component flags override preset components.
    Example: --preset fast --backend groq uses preset 'fast' but overrides
    its ASR backend with 'groq'.
    """
    preset: Optional[str] = None
    backend: Optional[str] = None
    preprocessor: Optional[str] = None
    device: Optional[str] = None
    post: Optional[str] = None
    post_model: Optional[str] = None


@dataclass(frozen=True)
class ASRBackendConfig:
    """Resolved ASR (transcription) backend configuration."""
    name: str
    type: str  # "api", "whisper_cpp", or "mock"
    # API-specific
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    org_id: Optional[str] = None
    # whisper.cpp-specific
    binary: Optional[str] = None
    model: Optional[str] = None
    threads: Optional[int] = None
    # mock-specific
    response: Optional[str] = None
    latency_ms: Optional[int] = None

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        backend_name: str,
    ) -> ASRBackendConfig:
        """Resolve ASR backend configuration.

        Args:
            config_dict: Full configuration dictionary.
            backend_name: Name of the backend (from preset or CLI override).

        All logic: env var fallback, defaults, validation, logging.
        """
        info(f"Using backend: {backend_name}")

        # Fetch backend definition
        backends = config_dict.get("asr_backends", {})
        if backend_name not in backends:
            available = ", ".join(backends.keys())
            raise ValueError(
                f"Backend '{backend_name}' not found. Available: {available}"
            )

        defn = backends[backend_name]
        btype = defn.get("type", "api")

        # Resolve based on type
        if btype == "api":
            api_key = defn.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    f"Backend '{backend_name}' requires api_key. "
                    f"Set in config or OPENAI_API_KEY env var."
                )

            api_base_url = defn.get("api_base_url")
            if not api_base_url:
                raise ValueError(
                    f"Backend '{backend_name}' requires api_base_url in config."
                )

            model_name = defn.get("model_name")
            if not model_name:
                raise ValueError(
                    f"Backend '{backend_name}' requires model_name in config."
                )

            org_id = defn.get("org_id") or os.environ.get("OPENAI_ORG_ID")

            key_source = "config" if defn.get("api_key") else "env var OPENAI_API_KEY"
            info(f"  API key: from {key_source}")
            info(f"  URL: {api_base_url}")

            return cls(
                name=backend_name,
                type="api",
                api_key=api_key,
                api_base_url=api_base_url,
                model_name=model_name,
                org_id=org_id,
            )

        elif btype == "whisper_cpp":
            binary = defn.get("binary")
            if not binary:
                raise ValueError(
                    f"Backend '{backend_name}' requires binary path in config."
                )
            model = defn.get("model")
            if not model:
                raise ValueError(
                    f"Backend '{backend_name}' requires model path in config."
                )
            threads = defn.get("threads", 4)

            info(f"  Binary: {binary}")
            info(f"  Model: {model}")
            info(f"  Threads: {threads}")

            return cls(
                name=backend_name,
                type="whisper_cpp",
                binary=binary,
                model=model,
                threads=threads,
            )

        elif btype == "mock":
            response = defn.get("response")
            latency_ms = defn.get("latency_ms", 0)

            if response:
                info(f"  Response: {len(response)} characters")
            if latency_ms:
                info(f"  Simulated latency: {latency_ms}ms")

            return cls(
                name=backend_name,
                type="mock",
                response=response,
                latency_ms=latency_ms,
            )

        else:
            raise ValueError(f"Unknown backend type: {btype}")


@dataclass(frozen=True)
class PreprocessorConfig:
    """Resolved audio preprocessor configuration."""
    name: str  # "none", "noisereduce", "deepfilter", "pyrnnoise"

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        cli_override: Optional[str] = None,
    ) -> PreprocessorConfig:
        """Resolve preprocessor from CLI override or preset."""
        if cli_override:
            name = cli_override
            info(f"Using preprocessor: {name} (CLI override)")
        else:
            name = "none"
            info(f"Using preprocessor: {name} (from preset)")

        return cls(name=name)


@dataclass(frozen=True)
class RecorderConfig:
    """Resolved audio input recorder configuration."""
    name: str  # "sounddevice", "arecord"
    device: Optional[DeviceInfo] = None

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        cli_device_override: Optional[str] = None,
    ) -> RecorderConfig:
        """Resolve recorder and audio device."""
        from .audio import resolve_device_preference_order
        from .recorders import PREFERENCE_ORDER, _CLASS_MAP

        # Resolve device preference order
        device_spec = cli_device_override or config_dict.get("audio_device") or "auto"
        devices = resolve_device_preference_order(device_spec)

        if not devices:
            if cli_device_override:
                warning(f"None of the devices in '{cli_device_override}' are available.")
            device_info = None
        else:
            device_info = devices[0]
            if device_info:
                info(f"Using device: {device_info.name}")

        # Determine recorder: prefer arecord for ALSA-only devices
        recorder_name = None
        if device_info and device_info.portaudio_name is None and device_info.alsa_name:
            # ALSA-only device, prefer arecord
            recorder_name = "arecord"
        else:
            # Use default preference order (sounddevice first, then arecord)
            for candidate in PREFERENCE_ORDER:
                if candidate in _CLASS_MAP:
                    recorder_class = _CLASS_MAP[candidate]()
                    if recorder_class.is_available():
                        recorder_name = candidate
                        break

        if not recorder_name:
            raise ValueError("No audio recorders available (sounddevice or arecord)")

        info(f"Using recorder: {recorder_name}")
        return cls(name=recorder_name, device=device_info)


@dataclass(frozen=True)
class PostprocessorConfig:
    """Resolved LLM post-processor configuration."""
    name: str
    backend_name: str
    model_override: Optional[str] = None
    template: str = "{result}"

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        cli_post_override: Optional[str] = None,
        cli_model_override: Optional[str] = None,
    ) -> PostprocessorConfig:
        """Resolve post-processor and its backend from CLI override or preset."""
        # Determine postprocessor name
        if cli_post_override:
            post_name = cli_post_override
            info(f"Using post-processor: {post_name} (CLI override)")
        else:
            post_name = "none"
            info(f"Using post-processor: {post_name} (from preset)")

        if post_name == "none":
            return cls(
                name="none",
                backend_name="none",
                template="{result}",
            )

        # Fetch postprocessor definition
        postprocessors = config_dict.get("postprocessors", {})
        if post_name not in postprocessors:
            available = ", ".join(postprocessors.keys())
            raise ValueError(
                f"Postprocessor '{post_name}' not found. Available: {available}"
            )

        defn = postprocessors[post_name]

        # Handle prompt inheritance (extends + extra)
        # For now, just get the base prompt; full inheritance handled elsewhere
        backend_name = defn.get("backend")
        model_override = cli_model_override or defn.get("model")
        template = defn.get("template", "{result}")

        if not backend_name:
            # Use first available backend as default
            backends = config_dict.get("postprocessor_backends", {})
            if backends:
                backend_name = next(iter(backends.keys()))
            else:
                backend_name = "none"

        info(f"  Post-processor backend: {backend_name}")
        if model_override:
            info(f"  Post-processor model override: {model_override}")

        return cls(
            name=post_name,
            backend_name=backend_name,
            model_override=model_override,
            template=template,
        )


@dataclass(frozen=True)
class PostprocessorBackendConfig:
    """Resolved LLM backend for post-processing (Groq, Ollama, Claude Code, etc.)."""
    name: str
    type: str  # "openai_compat", "claude_code"
    # For openai_compat
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        backend_name: str,
        cli_model_override: Optional[str] = None,
    ) -> PostprocessorBackendConfig:
        """Resolve LLM backend configuration for post-processing."""
        if backend_name == "none":
            return cls(name="none", type="none")

        backends = config_dict.get("postprocessor_backends", {})
        if backend_name not in backends:
            available = ", ".join(backends.keys())
            raise ValueError(
                f"Post-processor backend '{backend_name}' not found. Available: {available}"
            )

        defn = backends[backend_name]
        btype = defn.get("type", "openai_compat")

        if btype == "openai_compat":
            api_base_url = defn.get("api_base_url")
            if not api_base_url:
                raise ValueError(
                    f"Post-processor backend '{backend_name}' requires api_base_url."
                )

            api_key = defn.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    f"Post-processor backend '{backend_name}' requires api_key or OPENAI_API_KEY env var."
                )

            model = cli_model_override or defn.get("model")
            if not model:
                raise ValueError(
                    f"Post-processor backend '{backend_name}' requires model."
                )

            key_source = "config" if defn.get("api_key") else "env var"
            info(f"  Post-processor backend API key: from {key_source}")

            return cls(
                name=backend_name,
                type="openai_compat",
                api_base_url=api_base_url,
                api_key=api_key,
                model=model,
            )

        elif btype == "claude_code":
            model = cli_model_override or defn.get("model")
            if not model:
                raise ValueError(
                    f"Post-processor backend '{backend_name}' requires model."
                )
            return cls(
                name=backend_name,
                type="claude_code",
                model=model,
            )

        else:
            raise ValueError(f"Unknown post-processor backend type: {btype}")


@dataclass(frozen=True)
class OutputConfig:
    """Resolved output configuration."""
    clipboard_max_chars: int

    @classmethod
    def resolve(cls, config_dict: dict) -> OutputConfig:
        """Resolve output configuration."""
        from .output import _DEFAULT_CLIPBOARD_MAX_CHARS

        clipboard_max_chars = int(
            config_dict.get("clipboard_max_chars", _DEFAULT_CLIPBOARD_MAX_CHARS)
        )
        info(f"Clipboard max chars: {clipboard_max_chars}")
        return cls(clipboard_max_chars=clipboard_max_chars)


@dataclass(frozen=True)
class DiarizationConfig:
    """Resolved speaker diarization configuration."""
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    @classmethod
    def resolve(cls, config_dict: dict) -> DiarizationConfig:
        """Resolve diarization configuration."""
        hf_token = config_dict.get("diarize_hf_token") or os.environ.get("HF_TOKEN")
        min_speakers = config_dict.get("diarize_min_speakers")
        max_speakers = config_dict.get("diarize_max_speakers")

        if hf_token:
            token_source = (
                "config" if config_dict.get("diarize_hf_token") else "env var HF_TOKEN"
            )
            info(f"Diarization HF token: from {token_source}")

        return cls(
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )


@dataclass(frozen=True)
class PresetConfig:
    """Resolved preset configuration."""
    preset: Preset

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        preset_name: Optional[str] = None,
    ) -> PresetConfig:
        """Resolve a preset from config.

        Args:
            config_dict: Full configuration dictionary.
            preset_name: Name of preset to load. If None, tries to use default preset.

        Returns:
            PresetConfig with validated preset.

        Raises:
            ValueError: If preset not found or invalid.
        """
        presets = config_dict.get("presets", {})

        if not presets:
            raise ValueError(
                "No 'presets:' section found in config. "
                "Add one with at least one preset definition."
            )

        if not preset_name:
            raise ValueError(
                "No preset selected and no default specified. "
                "Use --preset NAME to select a preset, or add 'default_preset: NAME' to config."
            )

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Preset '{preset_name}' not found in config. "
                f"Available presets: {available}"
            )

        defn = presets[preset_name]
        preset = Preset.from_dict(preset_name, defn)

        info(f"Using preset: {preset_name}")
        if preset.description:
            info(f"  {preset.description}")

        return cls(preset=preset)


@dataclass
class Config:
    """Master config coordinator. Lazy-loads and caches domain-specific configs."""

    _config_dict: dict
    _preset: Preset
    _cli_overrides: CliOverrides

    # Cached resolved configs
    _asr_backend: Optional[ASRBackendConfig] = field(default=None, init=False)
    _preprocessor: Optional[PreprocessorConfig] = field(default=None, init=False)
    _recorder: Optional[RecorderConfig] = field(default=None, init=False)
    _postprocessor: Optional[PostprocessorConfig] = field(default=None, init=False)
    _postprocessor_backend: Optional[PostprocessorBackendConfig] = field(
        default=None, init=False
    )
    _output: Optional[OutputConfig] = field(default=None, init=False)
    _diarization: Optional[DiarizationConfig] = field(default=None, init=False)

    @property
    def asr_backend(self) -> ASRBackendConfig:
        """Lazy-load and cache ASR backend config."""
        if self._asr_backend is None:
            # Use CLI override if provided, otherwise use preset's backend
            backend_name = self._cli_overrides.backend or self._preset.asr_backend
            self._asr_backend = ASRBackendConfig.resolve(
                self._config_dict,
                backend_name,
            )
        return self._asr_backend

    @property
    def preprocessor(self) -> PreprocessorConfig:
        """Lazy-load and cache preprocessor config."""
        if self._preprocessor is None:
            # Use preset's preprocessor as override (CLI override takes precedence)
            preprocessor_override = self._cli_overrides.preprocessor or self._preset.preprocessor
            self._preprocessor = PreprocessorConfig.resolve(
                self._config_dict,
                preprocessor_override,
            )
        return self._preprocessor

    @property
    def recorder(self) -> RecorderConfig:
        """Lazy-load and cache recorder config."""
        if self._recorder is None:
            self._recorder = RecorderConfig.resolve(
                self._config_dict,
                self._cli_overrides.device,
            )
        return self._recorder

    @property
    def postprocessor(self) -> PostprocessorConfig:
        """Lazy-load and cache postprocessor config."""
        if self._postprocessor is None:
            # Use preset's postprocessor as override (CLI override takes precedence)
            postprocessor_override = self._cli_overrides.post or self._preset.postprocessor
            self._postprocessor = PostprocessorConfig.resolve(
                self._config_dict,
                postprocessor_override,
                self._cli_overrides.post_model,
            )
        return self._postprocessor

    @property
    def postprocessor_backend(self) -> PostprocessorBackendConfig:
        """Lazy-load and cache postprocessor backend config."""
        if self._postprocessor_backend is None:
            self._postprocessor_backend = PostprocessorBackendConfig.resolve(
                self._config_dict,
                self.postprocessor.backend_name,
                self._cli_overrides.post_model,
            )
        return self._postprocessor_backend

    @property
    def output(self) -> OutputConfig:
        """Lazy-load and cache output config."""
        if self._output is None:
            self._output = OutputConfig.resolve(self._config_dict)
        return self._output

    @property
    def diarization(self) -> DiarizationConfig:
        """Lazy-load and cache diarization config."""
        if self._diarization is None:
            self._diarization = DiarizationConfig.resolve(self._config_dict)
        return self._diarization

    @classmethod
    def resolve(
        cls,
        config_dict: dict,
        preset: Preset,
        cli_overrides: Optional[CliOverrides] = None,
    ) -> Config:
        """Load config and create master Config coordinator.

        Args:
            config_dict: Full configuration dictionary.
            preset: Preset object defining pipeline stages (required).
            cli_overrides: CLI flag overrides.

        Returns:
            Config instance ready for use.
        """
        cli_overrides = cli_overrides or CliOverrides()

        return cls(
            _config_dict=config_dict,
            _preset=preset,
            _cli_overrides=cli_overrides,
        )
