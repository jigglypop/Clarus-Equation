from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
ZIP = Path("/tmp/e17-etlk5k.zip")
OUT = Path("e17_direct_results")
OUT.mkdir(exist_ok=True)
TARGET_PREFIX = "Figure2/Data/"


def ensure_zip() -> None:
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)


def describe(x: Any, depth: int = 0) -> Any:
    if depth > 5:
        return {"type": type(x).__name__, "truncated": True}
    if isinstance(x, dict):
        return {"type": "dict", "keys": list(x.keys()), "values": {str(k): describe(v, depth + 1) for k, v in list(x.items())[:20]}}
    if isinstance(x, np.ndarray):
        d = {"type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
        if x.dtype == object:
            flat = x.ravel()
            d["items"] = [describe(v, depth + 1) for v in flat[:10]]
        elif np.issubdtype(x.dtype, np.number):
            finite = x[np.isfinite(x)]
            d["n_finite"] = int(finite.size)
            if finite.size:
                d.update({"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite)), "sample": np.asarray(finite[:20]).tolist()})
        return d
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "length": len(x), "items": [describe(v, depth + 1) for v in x[:10]]}
    if isinstance(x, np.void):
        return {"type": "np.void", "fields": {n: describe(x[n], depth + 1) for n in (x.dtype.names or [])}}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return {"type": type(x).__name__, "repr": repr(x)[:500]}


def main() -> None:
    ensure_zip()
    root = Path("/tmp/e17-fig2-probe")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    records = {}
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.startswith(TARGET_PREFIX) and n.endswith("_dff.mat")]
        for name in names:
            dst = root / Path(name).name
            with z.open(name) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            data = loadmat(dst, simplify_cells=True)
            clean = {k: v for k, v in data.items() if not k.startswith("__")}
            records[name] = describe(clean)
    (OUT / "figure2_probe.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2)[:50000])


if __name__ == "__main__":
    main()
