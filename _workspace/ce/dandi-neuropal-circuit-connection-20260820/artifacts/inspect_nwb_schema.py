"""Outcome-blind NWB schema walker: paths, shapes, dtypes, and attributes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


KEYWORDS = (
    "roi", "segment", "label", "name", "neuron", "position", "centroid",
    "trace", "fluorescence", "response", "calcium", "stim", "optogen",
    "interval", "event", "timestamp", "rate", "confidence", "target", "light",
)


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return [scalar(v) for v in value.reshape(-1)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect(path: Path) -> dict:
    objects = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            low = name.lower()
            attrs = {str(k): scalar(v) for k, v in obj.attrs.items()}
            if isinstance(obj, h5py.Dataset):
                row = {
                    "path": "/" + name,
                    "kind": "dataset",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": attrs,
                }
            else:
                row = {"path": "/" + name, "kind": "group", "attrs": attrs}
            if any(k in low for k in KEYWORDS) or any(k in str(attrs).lower() for k in KEYWORDS):
                objects.append(row)
        f.visititems(visitor)
        root_attrs = {str(k): scalar(v) for k, v in f.attrs.items()}
    return {
        "status": "SCHEMA_ONLY_VALUES_UNOPENED",
        "file": str(path),
        "bytes": path.stat().st_size,
        "root_attrs": root_attrs,
        "matched_objects": objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nwb", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.nwb.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"matched_objects": len(result["matched_objects"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
