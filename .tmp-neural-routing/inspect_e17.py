from __future__ import annotations

import csv
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
ROOT = Path("e17_direct_results")
ARCHIVE = ROOT / "e17.zip"
EXTRACT = ROOT / "extract"
ROOT.mkdir(exist_ok=True)

if not ARCHIVE.exists():
    print("Downloading E17 archive", flush=True)
    urllib.request.urlretrieve(URL, ARCHIVE)

records = []
with zipfile.ZipFile(ARCHIVE) as zf:
    for zi in zf.infolist():
        records.append({"path": zi.filename, "size": zi.file_size, "compressed": zi.compress_size})

suffix_counts = {}
for r in records:
    suffix = Path(r["path"]).suffix.lower() or "<none>"
    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

keywords = ["readme", "figure", "fig", "behavior", "behav", "calcium", "glut", "iglu", "ndnf", "neuro", "spine", "shaft", "burst", "rule", "switch"]
matched = []
for r in records:
    low = r["path"].lower()
    if any(k in low for k in keywords):
        matched.append(r)

small_candidates = [r for r in matched if r["size"] <= 5_000_000 and Path(r["path"]).suffix.lower() in {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".npy", ".npz", ".mat", ".xlsx"}]

result = {
    "status": "COMPLETE",
    "archive_bytes": ARCHIVE.stat().st_size,
    "n_files": len(records),
    "suffix_counts": dict(sorted(suffix_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    "keyword_matches": matched,
    "small_candidates": small_candidates,
    "largest_files": sorted(records, key=lambda r: r["size"], reverse=True)[:50],
}

(ROOT / "inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

lines = [
    "# E17 official archive inventory",
    "",
    f"Archive bytes: `{result['archive_bytes']}`",
    f"Files: `{result['n_files']}`",
    "",
    "## Suffix counts",
    "",
]
for k, v in result["suffix_counts"].items():
    lines.append(f"- `{k}`: {v}")
lines += ["", "## Small analysis candidates", ""]
for r in small_candidates:
    lines.append(f"- `{r['path']}` ({r['size']} bytes)")
lines += ["", "## Largest files", ""]
for r in result["largest_files"]:
    lines.append(f"- `{r['path']}` ({r['size']} bytes)")
(ROOT / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Extract only small, likely useful analysis files and readme/code text.
EXTRACT.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as zf:
    for r in records:
        p = r["path"]
        low = p.lower()
        suffix = Path(p).suffix.lower()
        should = (
            r["size"] <= 5_000_000
            and (
                any(k in low for k in keywords)
                or suffix in {".py", ".r", ".m", ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xlsx"}
            )
        )
        if should and not p.endswith("/"):
            try:
                zf.extract(p, EXTRACT)
            except Exception as exc:
                print("skip", p, exc)

# Remove archive before commit; only inventory and selected extracts remain.
ARCHIVE.unlink(missing_ok=True)
print((ROOT / "inventory.md").read_text(encoding="utf-8"))
