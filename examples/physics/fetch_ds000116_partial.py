"""Incrementally cache small ds000116 files for modality-gate work.

This downloader intentionally avoids pulling the full dataset. It fetches:

- root metadata: README, participants.tsv, task JSON files
- events.tsv for a selected subject range
- a manifest of all files, including annexed BOLD/T1w paths and sizes
- an annex include list for the selected subjects

Large files such as ``*_bold.nii.gz`` are git-annex objects on OpenNeuro and are
not downloaded by this script. Use the generated include list with DataLad or
OpenNeuro CLI when you are ready to pull one or two subjects at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET = "ds000116"
SNAPSHOT = "00003"
GRAPHQL_URL = "https://openneuro.org/crn/graphql"
RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000116/master"
DEFAULT_ROOT = Path("data/openneuro/ds000116")
ROOT_FILES = [
    "README",
    "CHANGES",
    "dataset_description.json",
    "participants.tsv",
    "task-auditoryoddballwithbuttonresponsetotargetstimuli_bold.json",
    "task-visualoddballwithbuttonresponsetotargetstimuli_bold.json",
]
TASKS = [
    "auditoryoddballwithbuttonresponsetotargetstimuli",
    "visualoddballwithbuttonresponsetotargetstimuli",
]


def post_graphql(query: str) -> dict:
    payload = json.dumps({"query": query}).encode("utf-8")
    req = Request(
        GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(path: str) -> str:
    with urlopen(f"{RAW_BASE}/{path}", timeout=60) as response:
        return response.read().decode("utf-8")


def write_text_if_needed(path: Path, text: str, *, force: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def snapshot_files() -> list[dict[str, object]]:
    query = f"""
    query {{
      snapshot(datasetId: "{DATASET}", tag: "{SNAPSHOT}") {{
        files(recursive: true) {{ filename size directory annexed }}
      }}
    }}
    """
    result = post_graphql(query)
    return result["data"]["snapshot"]["files"]


def selected_subjects(start: int, count: int) -> list[str]:
    return [f"sub-{idx:02d}" for idx in range(start, start + count)]


def event_paths(subjects: list[str]) -> list[str]:
    paths = []
    for subject in subjects:
        for task in TASKS:
            for run in (1, 2, 3):
                paths.append(f"{subject}/func/{subject}_task-{task}_run-{run:02d}_events.tsv")
    return paths


def annex_paths_for_subjects(files: list[dict[str, object]], subjects: list[str]) -> list[str]:
    prefixes = tuple(f"{subject}/" for subject in subjects)
    out = []
    for row in files:
        filename = str(row["filename"])
        if bool(row.get("annexed")) and filename.startswith(prefixes):
            out.append(filename)
    return sorted(out)


def summarize_manifest(files: list[dict[str, object]]) -> dict[str, object]:
    annexed = [row for row in files if bool(row.get("annexed"))]
    bold = [row for row in files if str(row["filename"]).endswith("_bold.nii.gz")]
    events = [row for row in files if str(row["filename"]).endswith("_events.tsv")]
    return {
        "file_count": len(files),
        "annexed_count": len(annexed),
        "event_count": len(events),
        "bold_count": len(bold),
        "bold_total_gb": round(sum(int(row["size"]) for row in bold) / 1024**3, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--subject-count", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects = selected_subjects(args.start_subject, args.subject_count)
    root = args.out_root
    files = snapshot_files()
    manifest_summary = summarize_manifest(files)
    write_text_if_needed(
        root / "manifest.json",
        json.dumps(
            {
                "dataset": DATASET,
                "snapshot": SNAPSHOT,
                "summary": manifest_summary,
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        force=args.force,
    )

    downloaded = []
    skipped = []
    failed = []
    for rel_path in ROOT_FILES + event_paths(subjects):
        try:
            text = fetch_text(rel_path)
        except (HTTPError, URLError, TimeoutError) as exc:
            failed.append({"path": rel_path, "error": str(exc)})
            continue
        did_write = write_text_if_needed(root / rel_path, text, force=args.force)
        (downloaded if did_write else skipped).append(rel_path)

    annex_paths = annex_paths_for_subjects(files, subjects)
    write_text_if_needed(
        root / "selected_annex_paths.txt",
        "\n".join(annex_paths) + ("\n" if annex_paths else ""),
        force=True,
    )

    payload = {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "out_root": str(root),
        "subjects": subjects,
        "manifest_summary": manifest_summary,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "failed": failed,
        "selected_annex_count": len(annex_paths),
        "selected_annex_total_mb": round(
            sum(
                int(row["size"])
                for row in files
                if bool(row.get("annexed")) and str(row["filename"]) in set(annex_paths)
            )
            / 1024**2,
            2,
        ),
        "next_step": (
            "Use selected_annex_paths.txt with DataLad/OpenNeuro CLI only when "
            "you are ready to pull BOLD/T1w for these subjects."
        ),
    }
    write_text_if_needed(
        root / "partial_fetch_summary.json",
        json.dumps(payload, indent=2, ensure_ascii=False),
        force=True,
    )

    print("ds000116 partial fetch")
    print(f"  out_root             = {root}")
    print(f"  subjects             = {', '.join(subjects)}")
    print(f"  manifest files       = {manifest_summary['file_count']}")
    print(f"  BOLD total           = {manifest_summary['bold_total_gb']} GB")
    print(f"  downloaded small     = {len(downloaded)}")
    print(f"  skipped existing     = {len(skipped)}")
    print(f"  failed               = {len(failed)}")
    print(f"  selected annex files = {len(annex_paths)}")
    print(f"  selected annex total = {payload['selected_annex_total_mb']} MB")
    print(f"Saved: {root / 'partial_fetch_summary.json'}")


if __name__ == "__main__":
    main()
