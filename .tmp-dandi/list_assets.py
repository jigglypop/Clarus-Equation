from __future__ import annotations

import json
from pathlib import Path

import requests

URL = "https://api.dandiarchive.org/api/dandisets/001695/versions/0.260319.2023/assets/?page_size=100"
OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)


def main() -> None:
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    assets = []
    for item in data.get("results", []):
        assets.append({
            "asset_id": item.get("asset_id"),
            "path": item.get("path"),
            "size": item.get("size"),
            "created": item.get("created"),
            "modified": item.get("modified"),
            "blob": item.get("blob"),
            "zarr": item.get("zarr"),
        })
    out = {"status": "COMPLETE", "url": URL, "count": len(assets), "assets": sorted(assets, key=lambda x: (x.get("size") or 0))}
    (OUT / "assets.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
