"""Read only small identity/time/coordinate/stimulus receipt fields from LINDI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lindi
import numpy as np


BASE = "/processing/CalciumActivity"


def clean(value):
    arr = np.asarray(value)
    if arr.dtype.fields:
        return [{name: clean(row[name]) for name in arr.dtype.names} for row in arr]
    if arr.ndim == 0:
        item = arr.item()
        return item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
    out = []
    for item in arr.reshape(-1):
        item = item.item() if isinstance(item, np.generic) else item
        out.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item)
    return out


def attrs(obj):
    return {str(k): clean(v) for k, v in obj.attrs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lindi_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    f = lindi.LindiH5pyFile.from_lindi_file(str(args.lindi_file.resolve()))
    series = f[f"{BASE}/SignalRawFluordNMF/dNMFCalciumImResponseSeries"]
    labels_ds = f[f"{BASE}/NeuronIDs/labels"]
    labels = clean(labels_ds[()])
    mask_ds = f["/processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/voxel_mask"]
    # LINDI's h5py compatibility layer rejects compound selection, while its
    # underlying Zarr reference array preserves the exact four-field rows.
    neuro_mask = clean(mask_ds._zarr_array[:])
    start_ds = series["starting_time"]
    stimuli = f["/intervals/chemical_stimuli"]
    result = {
        "status": "RECEIPTS_ONLY_RESPONSE_VALUES_UNOPENED",
        "file": str(args.lindi_file),
        "series_shape": list(series["data"].shape),
        "series_dtype": str(series["data"].dtype),
        "series_attrs": attrs(series),
        "starting_time": clean(start_ds[()]),
        "starting_time_attrs": attrs(start_ds),
        "label_count": len(labels),
        "labels": labels,
        "labels_sha256": hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "neuro_mask_count": len(neuro_mask),
        "neuro_mask": neuro_mask,
        "stimulus_start_time": clean(stimuli["start_time"][()]),
        "stimulus_stop_time": clean(stimuli["stop_time"][()]),
        "stimulus": clean(stimuli["stimulus"][()]),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "shape": result["series_shape"],
        "label_count": result["label_count"],
        "mask_count": result["neuro_mask_count"],
        "starting_time": result["starting_time"],
        "starting_time_attrs": result["starting_time_attrs"],
        "stimuli": list(zip(result["stimulus_start_time"], result["stimulus_stop_time"], result["stimulus"])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
