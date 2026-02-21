from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskRecord(BaseModel):
    """Single coding task plus deterministic verifier information."""

    # Unique identifier used in logs and error reporting.
    task_id: str
    # Natural language problem statement sent to the model.
    prompt: str
    # Python test code used by the verifier executor.
    test_code: str
    # Optional expected function name (can be checked by tests/prompts).
    entrypoint: str | None = None

    # Per-task timeout; can override global defaults when needed.
    timeout_s: float = Field(default=8.0, gt=0.0)

    # Optional labels for slicing analyses (e.g., arrays, strings, dp).
    tags: list[str] = Field(default_factory=list)
    # Free-form metadata for dataset bookkeeping.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "prompt", "test_code")
    @classmethod
    def _non_empty_required_text(cls, value: str) -> str:
        """Trim required text fields and reject empty content."""
        # Normalize whitespace-only values so empty checks are reliable.
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("entrypoint")
    @classmethod
    def _normalize_entrypoint(cls, value: str | None) -> str | None:
        """Normalize optional entrypoint, keeping None when blank."""
        if value is None:
            return None
        value = value.strip()
        # Convert blank strings to None to avoid two 'empty' representations.
        return value or None

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        """Drop blank tags while preserving original order."""
        # Stable order helps reproducible reporting and filtering.
        return [tag.strip() for tag in value if tag.strip()]
