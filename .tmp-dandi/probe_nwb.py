from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
from pynwb import NWBHDF5IO

ASSET_ID = "1e4d5403-a8cc-4814-a904-7aff57f8cc4d"
DOWNLOAD = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)
FILE = Path("/tmp/dandi001695_m02_20240312.nwb")


def download() -> None:
    if FILE.exists() and FILE.stat().st_size > 1_000_000:
        return
    with requests.get(DOWNLOAD, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with FILE.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def summarize_value(v):
    if isinstance(v, np.ndarray):
        return {"shape": list(v.shape), "dtype": str(v.dtype), "sample": v.reshape(-1)[:10].tolist()}
    return str(v)[:500]


def main() -> None:
    download()
    out = {"asset_id": ASSET_ID, "download_url": DOWNLOAD, "bytes": FILE.stat().st_size}
    with NWBHDF5IO(str(FILE), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        out["identifier"] = nwb.identifier
        out["session_description"] = nwb.session_description
        out["session_start_time"] = str(nwb.session_start_time)
        out["acquisition"] = list(nwb.acquisition.keys())
        out["processing"] = list(nwb.processing.keys())
        out["intervals"] = list(nwb.intervals.keys())
        out["electrode_groups"] = list(nwb.electrode_groups.keys())
        if nwb.units is not None:
            out["unit_columns"] = list(nwb.units.colnames)
            out["n_units"] = len(nwb.units)
            df = nwb.units.to_dataframe()
            summary = {}
            for col in df.columns:
                if col == "spike_times":
                    lengths = df[col].map(len).to_numpy()
                    summary[col] = {"min_n": int(lengths.min()), "median_n": float(np.median(lengths)), "max_n": int(lengths.max())}
                elif df[col].dtype == object:
                    vals = df[col].astype(str)
                    vc = vals.value_counts().head(30)
                    summary[col] = {str(k): int(v) for k, v in vc.items()}
                else:
                    arr = np.asarray(df[col])
                    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
                    summary[col] = {"dtype": str(arr.dtype), "n_unique": int(len(np.unique(arr))), "sample": arr[:10].tolist()}
                    if finite.size:
                        summary[col].update({"min": float(np.min(finite)), "max": float(np.max(finite)), "median": float(np.median(finite))})
            out["unit_summary"] = summary
        if nwb.electrodes is not None:
            out["electrode_columns"] = list(nwb.electrodes.colnames)
            edf = nwb.electrodes.to_dataframe()
            es = {}
            for col in edf.columns:
                vals = edf[col].astype(str)
                es[col] = {str(k): int(v) for k, v in vals.value_counts().head(30).items()}
            out["electrode_summary"] = es
        interval_summary = {}
        for name, table in nwb.intervals.items():
            try:
                df = table.to_dataframe()
                interval_summary[name] = {"n": len(df), "columns": list(df.columns), "head": df.head(5).astype(str).to_dict(orient="records")}
            except Exception as exc:
                interval_summary[name] = {"error": repr(exc)}
        out["interval_summary"] = interval_summary
    (OUT / "nwb_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:100000])


if __name__ == "__main__":
    main()
