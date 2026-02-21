from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reasoninglab.tasks.schema import TaskRecord


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON task records."""
    records: list[dict[str, Any]] = []

    # Parse line-by-line so error messages can include exact line numbers.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            # Ignore blank lines to make hand-edited files friendlier.
            continue

        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")

        records.append(record)

    return records


def _read_yaml(path: Path) -> list[dict[str, Any]]:
    """Read YAML task records from either list or {'tasks': [...]} format."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML task files. Install `pyyaml`."
        ) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []

    # Accept both shapes to keep dataset authoring simple.
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        records = raw["tasks"]
    else:
        raise ValueError(
            f"Invalid YAML task format in {path}. Use a list or {{'tasks': [...]}}."
        )

    # Validate container type early to produce clearer errors before schema parsing.
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid task record at {path}:{idx}: expected object")

    return records


def _read_task_records(path: Path) -> list[dict[str, Any]]:
    """Dispatch task loading based on file extension."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _read_jsonl(path)

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Invalid JSON task format in {path}: expected list")
        return raw

    if suffix in {".yaml", ".yml"}:
        return _read_yaml(path)

    raise ValueError(
        f"Unsupported task extension: {suffix}. Use .jsonl, .json, .yaml or .yml."
    )


def load_tasks(tasks_path: str | Path, limit: int | None = None) -> list[TaskRecord]:
    """Load and validate tasks from disk with stable input order."""
    path = Path(tasks_path)

    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    raw_records = _read_task_records(path)
    tasks: list[TaskRecord] = []

    # Validate each record against TaskRecord schema and keep order unchanged.
    for idx, raw in enumerate(raw_records, start=1):
        try:
            task = TaskRecord.model_validate(raw)
        except ValidationError as exc:
            # Include index and task id when present for faster debugging.
            task_hint = raw.get("task_id") if isinstance(raw, dict) else None
            suffix = f" ({task_hint})" if task_hint else ""
            raise ValueError(f"Invalid task record at {path}:{idx}{suffix}: {exc}") from exc
        tasks.append(task)

    if limit is not None:
        tasks = tasks[:limit]

    return tasks
