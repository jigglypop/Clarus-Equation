"""Inspect a local LINDI index without loading referenced response arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lindi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lindi_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    f = lindi.LindiH5pyFile.from_lindi_file(str(args.lindi_file.resolve()))
    rows = []

    def walk(group, prefix=""):
        for key in group.keys():
            path = f"{prefix}/{key}"
            if path == "/specifications":
                continue
            obj = group[key]
            shape = getattr(obj, "shape", None)
            dtype = getattr(obj, "dtype", None)
            rows.append({
                "path": path,
                "kind": "dataset" if shape is not None else "group",
                "shape": list(shape) if shape is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
            })
            if shape is None and hasattr(obj, "keys"):
                walk(obj, path)

    walk(f)
    result = {"status": "LINDI_SCHEMA_ONLY_VALUES_UNOPENED", "file": str(args.lindi_file), "objects": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"objects": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
