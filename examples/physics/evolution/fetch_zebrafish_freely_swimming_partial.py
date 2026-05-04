"""Fetch the partial freely-swimming zebrafish files used by local gates.

The Figshare dataset is about 12 GB in full. The current evolutionary-trace
gates only need two medium archives:

- figure5_S8.7z for free/immobilized region activity
- figure8.7z for perturbation, behavior-frame, and direction gates

This script downloads just those archives by default, verifies their MD5
checksums, and extracts them into the paths expected by the zebrafish gate
scripts. Additional smaller archives can be selected with ``--only`` for
alignment scouting without pulling the full 12 GB dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen


DATASET = "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish"
ARTICLE_API = "https://api.figshare.com/v2/articles/24032118"
ARTICLE_HTML = "https://figshare.com/articles/dataset/figure2_data/24032118"
DEFAULT_ROOT = Path("data/evolution/zebrafish/freely_swimming")
ARCHIVES = {
    "figure5_S8": {
        "name": "figure5_S8.7z",
        "download_url": "https://ndownloader.figshare.com/files/42423444",
        "size": 167862803,
        "md5": "41e16b31d866399245329634a0a620ab",
    },
    "figure8": {
        "name": "figure8.7z",
        "download_url": "https://ndownloader.figshare.com/files/42423435",
        "size": 199764238,
        "md5": "631d1c6a61a9b3eac47cd4adf9ab1463",
    },
    "figure7": {
        "name": "figure7.7z",
        "download_url": "https://ndownloader.figshare.com/files/42423438",
        "size": 281441422,
        "md5": "6abe669e61b4171eedb25ac457a701a0",
    },
    "Others_Supplementary": {
        "name": "Others_Supplementary.7z",
        "download_url": "https://ndownloader.figshare.com/files/42423441",
        "size": 757946424,
        "md5": "366c3477821df3c8b69e635353b5181a",
    },
}
DEFAULT_SELECTED = ["figure5_S8", "figure8"]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_path(root: Path, name: str) -> Path:
    return root / "_archives" / str(ARCHIVES[name]["name"])


def download_archive(name: str, root: Path, *, force: bool) -> dict[str, object]:
    spec = ARCHIVES[name]
    path = archive_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = str(spec["md5"])

    if path.exists() and not force:
        actual = md5_file(path)
        if actual == expected_md5:
            return {"name": name, "path": str(path), "downloaded": False, "md5": actual}
        print(f"{path} exists but MD5 differs; redownloading")

    print(f"Downloading {spec['name']} ({int(spec['size']) / 1024**2:.1f} MiB)")
    with urlopen(str(spec["download_url"]), timeout=120) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)

    actual = md5_file(path)
    if actual != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected_md5}, got {actual}")
    return {"name": name, "path": str(path), "downloaded": True, "md5": actual}


def find_7z() -> str:
    for candidate in ("7z", "7zz", "7za"):
        exe = shutil.which(candidate)
        if exe:
            return exe
    raise RuntimeError("7z/7zz/7za not found; install p7zip before extracting archives")


def extract_archive(path: Path, out_root: Path, *, seven_zip: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([seven_zip, "x", "-y", f"-o{out_root}", str(path)], check=True)


def extract_selected(root: Path, selected: list[str]) -> list[str]:
    seven_zip = find_7z()
    extracted = []
    if "figure5_S8" in selected:
        extract_archive(archive_path(root, "figure5_S8"), root, seven_zip=seven_zip)
        extracted.append(str(root / "figure5_S8"))
    if "figure8" in selected:
        extract_archive(archive_path(root, "figure8"), root, seven_zip=seven_zip)
        extracted.append(str(root / "figure8"))
        nested = root / "figure8" / "c" / "LR.7z"
        if nested.exists():
            extract_archive(nested, root / "figure8" / "c", seven_zip=seven_zip)
            extracted.append(str(root / "figure8" / "c" / "LR"))
    if "figure7" in selected:
        extract_archive(archive_path(root, "figure7"), root, seven_zip=seven_zip)
        extracted.append(str(root / "figure7"))
    if "Others_Supplementary" in selected:
        extract_archive(archive_path(root, "Others_Supplementary"), root, seven_zip=seven_zip)
        extracted.append(str(root / "Others_Supplementary"))
    return extracted


def write_summary(root: Path, payload: dict[str, object]) -> None:
    path = root / "partial_fetch_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(ARCHIVES),
        help=(
            "Archive key to fetch; repeat to fetch multiple. Defaults to "
            "figure5_S8 and figure8 only."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    selected = args.only or DEFAULT_SELECTED
    downloads = [download_archive(name, args.root, force=args.force) for name in selected]
    extracted = [] if args.no_extract else extract_selected(args.root, selected)

    write_summary(
        args.root,
        {
            "dataset": DATASET,
            "article_api": ARTICLE_API,
            "article_html": ARTICLE_HTML,
            "root": str(args.root),
            "selected": selected,
            "downloads": downloads,
            "extracted": extracted,
            "next_gates": [
                "zebrafish_freely_swimming_activity_gate.py",
                "zebrafish_laser_behavior_gate.py",
                "zebrafish_activity_behavior_frame_gate.py",
                "zebrafish_activity_direction_gate.py",
                "zebrafish_continuous_alignment_audit.py",
            ],
        },
    )

    print("Zebrafish freely-swimming partial fetch")
    print(f"  root      = {args.root}")
    print(f"  selected  = {', '.join(selected)}")
    print(f"  extracted = {len(extracted)} paths")


if __name__ == "__main__":
    main()
