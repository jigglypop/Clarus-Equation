"""Scout real public covariance/Fisher targets for the H0 readout law."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


TARGETS = Path(__file__).with_name("h0_real_covariance_targets.json")


def git_head(repo_url: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repo_url, "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.split()[0]


def main() -> int:
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    print("# H0 Real Source Scout Gate")
    print()
    print("| target | expected commit | remote HEAD | status |")
    print("|---|---|---|---|")
    failed = 0
    for target in payload["targets"]:
        remote = git_head(target["repo_url"])
        expected = target["commit"]
        status = "PASS" if remote == expected else "DRIFT"
        if status != "PASS":
            failed += 1
        print(f"| {target['id']} | {expected[:12]} | {remote[:12]} | {status} |")

    try:
        import h5py  # noqa: F401

        h5_status = "available"
    except Exception as exc:  # noqa: BLE001 - dependency diagnostic.
        h5_status = f"missing ({exc})"

    print()
    print(f"h5py = {h5_status}")
    print("Verdict: source targets are registered; HDF5 extraction needs h5py or an equivalent reader.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
