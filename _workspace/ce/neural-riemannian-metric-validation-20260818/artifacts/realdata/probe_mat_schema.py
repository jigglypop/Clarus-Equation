"""Record top-level MATLAB schemas without interpreting scientific outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from scipy.io import whosmat


def hdf5_schema(path: Path) -> list[dict[str, object]]:
    variables: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                variables.append(
                    {
                        "name": name,
                        "shape": list(item.shape),
                        "class": str(item.dtype),
                    }
                )

        handle.visititems(visitor)
    return variables


def inspect_file(path: Path) -> dict[str, object]:
    try:
        variables = [
            {"name": name, "shape": list(shape), "class": matlab_class}
            for name, shape, matlab_class in whosmat(path)
        ]
        file_format = "MATLAB level 5"
    except (NotImplementedError, ValueError, OSError):
        variables = hdf5_schema(path)
        file_format = "MATLAB 7.3/HDF5"
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "format": file_format,
        "variables": variables,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.rglob("*.mat"))
    result = {
        "schema_version": 1,
        "root": root.as_posix(),
        "mat_file_count": len(files),
        "files": [inspect_file(path) for path in files],
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "mat_file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
