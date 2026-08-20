"""Read only HDF5 metadata from an HTTP-range-capable NWB asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with fsspec.open(args.url, "rb", block_size=4 << 20, cache_type="readahead") as remote:
        with h5py.File(remote, "r") as f:
            def walk(group, prefix=""):
                for key in group.keys():
                    name = f"{prefix}/{key}"
                    if name == "/specifications":
                        continue
                    obj = group[key]
                    is_dataset = isinstance(obj, h5py.Dataset)
                    rows.append({
                        "path": name,
                        "kind": "dataset" if is_dataset else "group",
                        "shape": list(obj.shape) if is_dataset else None,
                        "dtype": str(obj.dtype) if is_dataset else None,
                    })
                    if isinstance(obj, h5py.Group):
                        walk(obj, name)
            walk(f)
    result = {"status": "REMOTE_SCHEMA_ONLY_VALUES_UNOPENED", "url": args.url, "objects": rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"objects": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
