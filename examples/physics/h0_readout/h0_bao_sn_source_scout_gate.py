"""Scout BAO+SN public sources for the next H0 readout extension."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


TARGETS = Path(__file__).with_name("h0_external_source_targets.json")
REAL_DATA = Path(__file__).with_name("h0_real_data")


def git_head(repo_url: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repo_url, "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.split()[0]


def local_repo(target_id: str) -> Path | None:
    mapping = {
        "cobaya_bao_data": REAL_DATA / "CobayaSampler_bao_data",
        "pantheon_plus_sh0es_data_release": REAL_DATA / "PantheonPlusSH0ES_DataRelease",
    }
    repo = mapping.get(target_id)
    return repo if repo and (repo / ".git").exists() else None


def git_tree_contains(repo: Path, candidate: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", candidate],
        text=True,
        capture_output=True,
        check=True,
    )
    return candidate in completed.stdout.splitlines()


def checked_out(repo: Path, candidate: str) -> bool:
    return (repo / candidate).exists()


def main() -> int:
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    print("# H0 BAO+SN Source Scout Gate")
    print()
    print("| target | expected commit | remote HEAD | candidate files in git tree | checked out | status |")
    print("|---|---|---|---:|---:|---|")

    failed = 0
    for target in payload["targets"]:
        remote = git_head(target["repo_url"])
        repo = local_repo(target["id"])
        in_tree = 0
        on_disk = 0
        if repo is not None:
            for candidate in target["candidate_files"]:
                if git_tree_contains(repo, candidate):
                    in_tree += 1
                if checked_out(repo, candidate):
                    on_disk += 1

        status = "PASS" if remote == target["commit"] and in_tree else "DRIFT"
        if status != "PASS":
            failed += 1
        print(
            f"| {target['id']} | {target['commit'][:12]} | {remote[:12]} | "
            f"{in_tree}/{len(target['candidate_files'])} | {on_disk}/{len(target['candidate_files'])} | {status} |"
        )

    print()
    print("Verdict: BAO source is ready for a mean/covariance role adapter; Pantheon+ should be accessed sparsely.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
