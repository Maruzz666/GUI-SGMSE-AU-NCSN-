import json
from pathlib import Path
from typing import Any

from .runtime import prepare_runtime


def append_history(record: dict[str, Any]) -> None:
    path = prepare_runtime() / "history.jsonl"
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_history() -> list[dict[str, Any]]:
    path = prepare_runtime() / "history.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as history_file:
        for line_number, line in enumerate(history_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Historial dañado en {path}, línea {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Registro inválido en {path}, línea {line_number}."
                )
            records.append(record)
    return records
