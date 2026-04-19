"""5 개 도메인 브랜치를 main 으로부터 생성하고, 비-도메인 파일을
git rm 한 뒤 단일 커밋으로 정리한다.

Windows + 한글 경로 안전. xargs 의존 없음.

사용:
    python scripts/make_branches.py            # 안전: 이미 있으면 스킵
    python scripts/make_branches.py --force    # 기존 도메인 브랜치 삭제 후 재생성
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 본 스크립트를 어디서 호출하든 repo root 에서 동작하도록
ROOT = Path(__file__).resolve().parents[1]

DOMAINS = ("cosmology", "particle", "brain-agi", "engineering", "math-derivations")


def git(*args: str, capture: bool = False) -> str:
    cmd = ["git", "-c", "core.quotepath=false", *args]
    if capture:
        return subprocess.check_output(cmd, cwd=ROOT, encoding="utf-8")
    subprocess.check_call(cmd, cwd=ROOT)
    return ""


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD", capture=True).strip()


def working_tree_dirty() -> bool:
    return bool(git("status", "--porcelain", capture=True).strip())


def branch_exists(name: str) -> bool:
    code = subprocess.call(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
        cwd=ROOT,
    )
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if working_tree_dirty():
        print("ERR: working tree dirty. commit or stash first.", file=sys.stderr)
        return 1
    if current_branch() != "main":
        print(f"ERR: must be on main (current: {current_branch()})", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT / "scripts"))
    import branch_split

    all_files = branch_split._git_ls_files()  # noqa: SLF001
    print(f"main has {len(all_files)} tracked files")

    for d in DOMAINS:
        br = f"domain/{d}"
        if branch_exists(br):
            if not args.force:
                print(f"[skip ] {br} exists (use --force)")
                continue
            print(f"[force] deleting {br}")
            git("branch", "-D", br)

        print(f"[make ] {br}")
        git("checkout", "-b", br, "main")

        keep, remove, unmapped = branch_split.files_to_keep(d, all_files)
        print(f"        keep={len(keep)}  remove={len(remove)}  unmapped={len(unmapped)}")

        if remove:
            CHUNK = 50
            for i in range(0, len(remove), CHUNK):
                git("rm", "-q", "--", *remove[i : i + CHUNK])

        kept = len(branch_split._git_ls_files())  # noqa: SLF001
        git("commit", "-q", "-m", f"branch: prune to {d} domain (kept {kept} files)")
        print(f"        committed, branch now has {kept} files")

        git("checkout", "main")

    print()
    print("== branches ==")
    print(git("branch", capture=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
