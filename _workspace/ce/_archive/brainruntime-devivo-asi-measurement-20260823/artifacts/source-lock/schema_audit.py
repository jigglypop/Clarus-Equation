"""Read-only schema receipt for the frozen de Vivo 2017 CSV.

This script intentionally reports identities, nesting, missingness, and numeric
ranges only.  It does not fit a model or compute sleep/wake effect estimates.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


RAW = Path("data/external/devivo2017/synapse_data.csv")


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with RAW.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing_tokens = {"", "na", "nan", "null"}
    missing = {
        column: sum(row[column].strip().lower() in missing_tokens for row in rows)
        for column in columns
    }
    unique = {column: len({row[column] for row in rows}) for column in columns}
    low_cardinality_values = {
        column: sorted({row[column] for row in rows})
        for column in columns
        if unique[column] <= 20
    }
    numeric_ranges: dict[str, dict[str, float]] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            value = row[column].strip()
            if value.lower() in missing_tokens:
                continue
            try:
                values.append(float(value))
            except ValueError:
                values = []
                break
        if values:
            numeric_ranges[column] = {"min": min(values), "max": max(values)}

    condition_rows = Counter(row["condition"] for row in rows)
    condition_mice: dict[str, set[str]] = defaultdict(set)
    condition_dendrites: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        condition = row["condition"]
        condition_mice[condition].add(row["mouse"])
        condition_dendrites[condition].add(
            (row["mouse"], row["location"], row["dendrite_number"])
        )

    payload = {
        "path": RAW.as_posix(),
        "bytes": RAW.stat().st_size,
        "md5": file_hash(RAW, "md5"),
        "sha256": file_hash(RAW, "sha256"),
        "columns": columns,
        "row_count": len(rows),
        "duplicate_full_rows": len(rows) - len({tuple(row[column] for column in columns) for row in rows}),
        "missing_by_column": missing,
        "unique_by_column": unique,
        "values_by_low_cardinality_column": low_cardinality_values,
        "numeric_ranges": numeric_ranges,
        "condition_rows": dict(sorted(condition_rows.items())),
        "condition_mice": {
            key: sorted(value, key=lambda item: int(item))
            for key, value in sorted(condition_mice.items())
        },
        "condition_dendrite_counts": {
            key: len(value) for key, value in sorted(condition_dendrites.items())
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
