from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.io import whosmat

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
OUT = Path("e17_direct_results")
OUT.mkdir(exist_ok=True)
ZIP = Path("/tmp/e17-etlk5k.zip")
KEYWORDS = [
    "readme", "source", "figure", "fig", "calcium", "glut", "iglu", "dend", "ndnf",
    "behavior", "behav", "neuropixel", "spike", "burst", "rule", "switch", "shaft",
    "spine", "optogen", "chemogen", "et", "pt", "alm"
]


def download() -> str:
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)
    h = hashlib.sha256()
    with ZIP.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_ext(name: str) -> str:
    p = Path(name)
    return p.suffix.lower() or "<none>"


def selected(name: str, size: int) -> bool:
    low = name.lower()
    text_like = clean_ext(name) in {".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".py", ".r"}
    structured = clean_ext(name) in {".mat", ".h5", ".hdf5", ".npy", ".npz", ".xlsx", ".xls"}
    key = any(k in low for k in KEYWORDS)
    return size <= 30 * 1024 * 1024 and (text_like or (structured and key))


def h5_inventory(path: Path) -> dict:
    out = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append({"path": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        f.visititems(visit)
    return {"format": "hdf5", "datasets": out[:500], "n_datasets": len(out)}


def mat_inventory(path: Path) -> dict:
    try:
        variables = whosmat(path)
        return {
            "format": "mat-v5",
            "variables": [{"name": n, "shape": list(s), "class": c} for n, s, c in variables]
        }
    except Exception as exc:
        try:
            return h5_inventory(path)
        except Exception as exc2:
            return {"format": "unknown", "error": repr(exc), "h5_error": repr(exc2)}


def np_inventory(path: Path) -> dict:
    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            return {"format": "npy", "shape": list(a.shape), "dtype": str(a.dtype)}
        z = np.load(path, allow_pickle=False)
        return {"format": "npz", "arrays": {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}}
    except Exception as exc:
        return {"format": "numpy-unknown", "error": repr(exc)}


def inspect() -> dict:
    sha = download()
    extract_root = Path("/tmp/e17-selected")
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir()
    with zipfile.ZipFile(ZIP) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        ext_counts = Counter(clean_ext(i.filename) for i in infos)
        top_counts = Counter(Path(i.filename).parts[0] if Path(i.filename).parts else "" for i in infos)
        candidates = [i for i in infos if selected(i.filename, i.file_size)]
        records = []
        for info in candidates:
            safe = Path(info.filename)
            if any(part == ".." for part in safe.parts):
                continue
            dst = extract_root / safe
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            rec = {"name": info.filename, "size": info.file_size, "extension": clean_ext(info.filename)}
            ext = dst.suffix.lower()
            if ext == ".mat":
                rec["inventory"] = mat_inventory(dst)
            elif ext in {".h5", ".hdf5"}:
                try:
                    rec["inventory"] = h5_inventory(dst)
                except Exception as exc:
                    rec["inventory"] = {"error": repr(exc)}
            elif ext in {".npy", ".npz"}:
                rec["inventory"] = np_inventory(dst)
            elif ext in {".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".py", ".r"} and info.file_size <= 2_000_000:
                try:
                    rec["preview"] = dst.read_text(encoding="utf-8", errors="replace")[:12000]
                except Exception as exc:
                    rec["preview_error"] = repr(exc)
            records.append(rec)
        all_files = [{"name": i.filename, "size": i.file_size, "extension": clean_ext(i.filename)} for i in infos]
    return {
        "status": "COMPLETE",
        "url": URL,
        "sha256": sha,
        "archive_bytes": ZIP.stat().st_size,
        "n_files": len(all_files),
        "extension_counts": dict(ext_counts),
        "top_level_counts": dict(top_counts),
        "all_files": all_files,
        "selected_records": records,
    }


def report(result: dict) -> str:
    lines = [
        "# E17 official archive direct inventory",
        "",
        f"Archive SHA-256: `{result['sha256']}`",
        f"Archive bytes: `{result['archive_bytes']}`; files: `{result['n_files']}`.",
        "",
        "## Extensions",
        "",
    ]
    for k, v in sorted(result["extension_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Candidate files", ""]
    for rec in result["selected_records"]:
        inv = rec.get("inventory", {})
        lines.append(f"- `{rec['name']}` ({rec['size']} bytes): `{inv.get('format', 'text')}`")
        if inv.get("variables"):
            lines.append("  - " + ", ".join(f"{v['name']}:{v['shape']}" for v in inv["variables"][:20]))
        if inv.get("datasets"):
            lines.append("  - " + ", ".join(f"{v['path']}:{v['shape']}" for v in inv["datasets"][:20]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    r = inspect()
    (OUT / "inventory.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(report(r), encoding="utf-8")
    print(report(r))
