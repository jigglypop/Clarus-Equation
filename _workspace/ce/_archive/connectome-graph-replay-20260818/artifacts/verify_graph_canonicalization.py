"""Small, dependency-free witness for the normalization algebra in 11-math.md."""
from __future__ import annotations

import hashlib
import json


def normalize(row: dict[str, object]) -> tuple[str, str, str, int, int]:
    cls, pre, post, weight, ordinal = (
        row["class"], row["pre"], row["post"], row["weight"], row["ordinal"]
    )
    assert cls in {"chemical", "electrical"}
    assert all(isinstance(x, str) and x for x in (pre, post))
    assert type(weight) is int and weight >= 0  # bool is intentionally excluded.
    assert type(ordinal) is int and ordinal >= 0
    a, b = (pre, post) if cls == "chemical" else tuple(sorted((pre, post)))
    return cls, a, b, weight, ordinal


def canonical(rows: list[dict[str, object]]) -> bytes:
    observations = sorted(map(normalize, rows), key=lambda x: (x[0], x[1], x[2], x[4]))
    aggregate: dict[tuple[str, str, str], int] = {}
    for cls, a, b, weight, _ in observations:
        aggregate[(cls, a, b)] = aggregate.get((cls, a, b), 0) + weight
    value = {"aggregate": [{"class": c, "endpoint_a": a, "endpoint_b": b, "multiplicity": w}
                           for (c, a, b), w in sorted(aggregate.items())],
             "observations": [{"class": c, "endpoint_a": a, "endpoint_b": b,
                               "multiplicity": w, "source_record_ordinal": o}
                              for c, a, b, w, o in observations]}
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


rows = [
    {"class": "chemical", "pre": "AVA", "post": "AVB", "weight": 2, "ordinal": 11},
    {"class": "chemical", "pre": "AVB", "post": "AVA", "weight": 3, "ordinal": 12},
    {"class": "electrical", "pre": "AVD", "post": "AVE", "weight": 5, "ordinal": 13},
    {"class": "electrical", "pre": "AVE", "post": "AVD", "weight": 7, "ordinal": 14},
]
assert canonical(rows) == canonical(list(reversed(rows)))
payload = json.loads(canonical(rows))
assert [(x["endpoint_a"], x["endpoint_b"], x["multiplicity"]) for x in payload["aggregate"]] == [
    ("AVA", "AVB", 2), ("AVB", "AVA", 3), ("AVD", "AVE", 12)
]
assert hashlib.sha256(canonical(rows)).hexdigest() == hashlib.sha256(canonical(list(reversed(rows)))).hexdigest()

try:
    normalize({"class": "chemical", "pre": "AVA", "post": "AVB", "weight": True, "ordinal": 0})
except AssertionError:
    pass
else:
    raise AssertionError("bool was accepted as a count")

print("OK: canonicalization witness")
