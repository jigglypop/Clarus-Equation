"""Inspect one frozen E2SYT NWB exemplar without reading dataset values."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "clarus.e2syt.nwb-schema-inventory.v1"
EXPECTED_BYTES = 1_273_970
EXPECTED_SHA256 = "40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e"
KEYWORDS = (
    "stim",
    "ogen",
    "optogen",
    "event",
    "interval",
    "trial",
    "target",
    "source",
    "neuron",
    "roi",
    "segmentation",
    "timeseries",
    "timestamp",
    "response",
    "control",
    "sham",
    "condition",
    "identity",
    "neuropal",
)
SELECTED_ATTRIBUTE_VALUES = {
    "colnames",
    "comments",
    "description",
    "help",
    "namespace",
    "neurodata_type",
    "nwb_version",
    "object_id",
    "table",
    "unit",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:512]
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _json_scalar(value.item())
    if isinstance(value, np.ndarray) and value.size <= 16:
        return [_json_scalar(item) for item in value.reshape(-1).tolist()]
    return f"<{type(value).__name__}>"


def _attribute_value(obj: h5py.Group | h5py.Dataset, key: str) -> Any:
    value = obj.attrs[key]
    if isinstance(value, (h5py.Reference, h5py.RegionReference)):
        return obj.file[value].name if value else None
    return _json_scalar(value)


def _attributes(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    keys = sorted(str(key) for key in obj.attrs.keys())
    values = {
        key: _attribute_value(obj, key)
        for key in keys
        if key.lower() in SELECTED_ATTRIBUTE_VALUES
    }
    return {"keys": keys, "selected_values": values}


def _dtype_metadata(dtype: np.dtype[Any]) -> dict[str, Any]:
    string_info = h5py.check_string_dtype(dtype)
    reference_info = h5py.check_dtype(ref=dtype)
    vlen_info = h5py.check_dtype(vlen=dtype)
    return {
        "kind": dtype.kind,
        "string_encoding": string_info.encoding if string_info is not None else None,
        "string_length": string_info.length if string_info is not None else None,
        "reference_type": getattr(reference_info, "__name__", None),
        "vlen_base": str(vlen_info) if vlen_info is not None else None,
    }


def inspect(input_path: Path) -> dict[str, Any]:
    size = input_path.stat().st_size
    sha256 = _sha256(input_path)
    if size != EXPECTED_BYTES:
        raise ValueError(f"byte mismatch: expected {EXPECTED_BYTES}, got {size}")
    if sha256 != EXPECTED_SHA256:
        raise ValueError(f"sha256 mismatch: expected {EXPECTED_SHA256}, got {sha256}")

    objects: list[dict[str, Any]] = []
    keyword_hits: dict[str, list[str]] = {keyword: [] for keyword in KEYWORDS}

    with h5py.File(input_path, "r") as nwb:
        root = {
            "path": "/",
            "object_type": "group",
            "children": sorted(str(key) for key in nwb.keys()),
            "attributes": _attributes(nwb),
        }
        objects.append(root)

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            path = f"/{name}"
            record: dict[str, Any] = {
                "path": path,
                "object_type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                "attributes": _attributes(obj),
            }
            if isinstance(obj, h5py.Group):
                record["children"] = sorted(str(key) for key in obj.keys())
            else:
                record.update(
                    {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "dtype_metadata": _dtype_metadata(obj.dtype),
                        "chunks": list(obj.chunks) if obj.chunks is not None else None,
                        "compression": obj.compression,
                        "maxshape": list(obj.maxshape) if obj.maxshape is not None else None,
                    }
                )
            objects.append(record)

            searchable = " ".join(
                [path]
                + record["attributes"]["keys"]
                + [str(value) for value in record["attributes"]["selected_values"].values()]
            ).lower()
            for keyword in KEYWORDS:
                if keyword in searchable:
                    keyword_hits[keyword].append(path)

        nwb.visititems(visitor)

    paths = {record["path"] for record in objects}
    keyword_hits = {
        key: sorted(set(values)) for key, values in keyword_hits.items() if values
    }
    checks = {
        "has_acquisition_group": "/acquisition" in paths,
        "has_analysis_group": "/analysis" in paths,
        "has_intervals_group": "/intervals" in paths,
        "has_processing_group": "/processing" in paths,
        "has_stimulus_group": "/stimulus" in paths,
        "has_trials_table": any(path.startswith("/intervals/trials") for path in paths),
        "has_ogen_path": any("ogen" in path.lower() for path in paths),
        "has_segmentation_path": any("segmentation" in path.lower() for path in paths),
        "has_response_named_path": any("response" in path.lower() for path in paths),
        "has_control_or_sham_named_path": any(
            token in path.lower() for path in paths for token in ("control", "sham")
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_SCHEMA_INVENTORY",
        "scope": "metadata_and_object_schema_only",
        "input": {
            "path": input_path.as_posix(),
            "bytes": size,
            "sha256": sha256,
            "hdf5_signature": "89-48-44-46-0d-0a-1a-0a",
        },
        "environment": {
            "python": platform.python_version(),
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "audit_boundary": {
            "dataset_values_read": False,
            "neural_effect_computed": False,
            "endpoint_selected": False,
        },
        "object_count": len(objects),
        "checks": checks,
        "keyword_hits": keyword_hits,
        "objects": sorted(objects, key=lambda item: item["path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = inspect(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"object_count={report['object_count']}")
    print(json.dumps(report["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
