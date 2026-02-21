from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

# Allow only supported H1 policies. Literal gives strict validation at parse time.
PolicyName = Literal["baseline", "best_of_b", "repair_b"]


class GenerationConfig(BaseModel):
    """Decoding options shared across all policy arms."""

    # Hard cap for generated tokens per model call.
    max_new_tokens: int = Field(default=256, ge=1)
    # Keep deterministic by default; reproducibility first in H1.
    do_sample: bool = Field(default=False)
    # If sampling is enabled later, keep temperature within valid range.
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


class RuntimeConfig(BaseModel):
    """Verifier and runtime logging options."""

    # Per-attempt verifier timeout (seconds).
    verifier_timeout_s: float = Field(default=8.0, gt=0.0)
    # Keep this flag so runs can log placement/device details.
    capture_device_map: bool = Field(default=True)


class PathsConfig(BaseModel):
    """Paths used to load tasks and store run artifacts."""

    # Task dataset path used by loader.
    tasks_file: Path
    # Root folder where run artifacts are written.
    runs_root: Path = Field(default=Path("runs"))
    # Root folder for aggregated experiment outputs.
    results_root: Path = Field(default=Path("results/h1"))


class ExperimentConfig(BaseModel):
    """Top-level H1 experiment contract loaded from config files."""

    experiment_id: str
    hypothesis: str = Field(default="H1")
    policy: PolicyName
    # Primary H1 budget: maximum model calls per task.
    budget_B: int = Field(ge=1)
    seed: int = Field(default=0, ge=0)
    model_id: str
    # Optional cap for fast smoke runs.
    max_tasks: int | None = Field(default=None, ge=1)

    # Nested blocks keep config readable and scalable.
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig

    @field_validator("experiment_id", "model_id", "hypothesis")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        """Normalize and reject empty strings for key identifiers."""
        # Strip accidental whitespace from YAML/JSON values.
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value


def _load_raw_config(config_path: Path) -> dict:
    """Read JSON/YAML config as a raw dictionary before validation."""
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install `pyyaml`."
            ) from exc

        # safe_load avoids executing arbitrary YAML objects.
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Normalize empty YAML files to an empty mapping.
        return raw or {}

    raise ValueError(
        f"Unsupported config extension: {suffix}. Use .json, .yaml or .yml."
    )


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load config from disk and return a validated ExperimentConfig object."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_config(path)

    try:
        # Pydantic validates types + constraints + custom validators.
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        # Keep a simple external API: callers handle ValueError for bad configs.
        raise ValueError(f"Invalid experiment config at {path}: {exc}") from exc
