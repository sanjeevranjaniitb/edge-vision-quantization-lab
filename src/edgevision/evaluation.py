from __future__ import annotations
import csv
from pathlib import Path


def append_csv(row: dict, path: str = "benchmarks/results.csv") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing_fields: list[str] = []
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []

    # Union of existing columns and new row keys, preserving order
    all_fields = list(existing_fields) + [k for k in row if k not in existing_fields]

    rows: list[dict] = []
    if p.exists() and existing_fields:
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    rows.append(row)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
