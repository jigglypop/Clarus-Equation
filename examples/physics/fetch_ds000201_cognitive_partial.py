"""Cache small ds000201 cognitive/arousal event files.

This script downloads only public TSV/JSON text files for:

- PVT behavioral vigilance
- workingmemorytest behavioral working memory
- sleepiness subjective arousal/KSS events

It uses the local ds000201 manifest when present, so subjects with missing
working-memory files are skipped cleanly instead of producing 404s.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DATASET = "ds000201"
SNAPSHOT = "1.0.3"
RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000201/master"
DEFAULT_ROOT = Path("data/openneuro/ds000201")
ROOT_FILES = [
    "task-PVT_beh.json",
    "task-sleepiness_bold.json",
    "task-workingmemorytest_beh.json",
]
TASK_MARKERS = [
    "task-PVT_events.tsv",
    "task-sleepiness_events.tsv",
    "task-workingmemorytest_events.tsv",
]


def fetch_text(path: str) -> str:
    with urlopen(f"{RAW_BASE}/{path}", timeout=60) as response:
        return response.read().decode("utf-8")


def write_text_if_needed(path: Path, text: str, *, force: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def participant_subjects(root: Path) -> list[str]:
    text = (root / "participants.tsv").read_text(encoding="utf-8")
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    return [row["participant_id"] for row in rows if row.get("participant_id")]


def manifest_event_paths(root: Path, subjects: list[str]) -> list[str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("missing data/openneuro/ds000201/manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefixes = tuple(f"{subject}/" for subject in subjects)
    out = []
    for row in payload["files"]:
        filename = str(row["filename"])
        if filename.startswith(prefixes) and any(marker in filename for marker in TASK_MARKERS):
            out.append(filename)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--subject-count", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects_all = participant_subjects(args.out_root)
    subjects = subjects_all[
        max(args.start_subject - 1, 0) : max(args.start_subject - 1, 0) + args.subject_count
    ]
    paths = ROOT_FILES + manifest_event_paths(args.out_root, subjects)

    downloaded = []
    skipped = []
    failed = []
    for rel_path in paths:
        try:
            text = fetch_text(rel_path)
        except (HTTPError, URLError, TimeoutError) as exc:
            failed.append({"path": rel_path, "error": str(exc)})
            continue
        did_write = write_text_if_needed(args.out_root / rel_path, text, force=args.force)
        (downloaded if did_write else skipped).append(rel_path)

    task_counts = {
        "PVT": sum("task-PVT_events.tsv" in path for path in paths),
        "sleepiness": sum("task-sleepiness_events.tsv" in path for path in paths),
        "workingmemorytest": sum("task-workingmemorytest_events.tsv" in path for path in paths),
    }
    summary = {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "out_root": str(args.out_root),
        "subjects": subjects,
        "selected_event_count": len(paths) - len(ROOT_FILES),
        "task_counts": task_counts,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "failed": failed,
    }
    write_text_if_needed(
        args.out_root / "cognitive_partial_fetch_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        force=True,
    )

    print("ds000201 cognitive partial fetch")
    print(f"  out_root         = {args.out_root}")
    print(f"  subjects         = {', '.join(subjects)}")
    print(f"  selected events  = {summary['selected_event_count']}")
    print(f"  PVT              = {task_counts['PVT']}")
    print(f"  sleepiness       = {task_counts['sleepiness']}")
    print(f"  workingmemory    = {task_counts['workingmemorytest']}")
    print(f"  downloaded small = {len(downloaded)}")
    print(f"  skipped existing = {len(skipped)}")
    print(f"  failed           = {len(failed)}")
    print(f"Saved: {args.out_root / 'cognitive_partial_fetch_summary.json'}")


if __name__ == "__main__":
    main()
