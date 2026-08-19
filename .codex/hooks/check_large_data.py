from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


LIMIT_BYTES = 95_000_000
DATA_SUFFIXES = {
    ".zip",
    ".mat",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".pt",
    ".onnx",
    ".parquet",
    ".bin",
    ".exe",
}


class GateError(RuntimeError):
    pass


def _git(root: Path, *arguments: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )
    if result.returncode:
        stderr = result.stderr.strip()
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        raise GateError(f"git {' '.join(arguments)} failed: {stderr}")
    return result.stdout


def _repository_root() -> Path:
    requested = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())).resolve()
    root = str(_git(requested, "rev-parse", "--show-toplevel", text=True)).strip()
    return Path(root).resolve()


def _normalized_path(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        value = os.fsdecode(raw)
    else:
        value = raw
    return value.replace("\\", "/")


def _blob_violation(root: Path, object_name: str, path: str, prefix: str) -> str | None:
    object_type = str(_git(root, "cat-file", "-t", object_name, text=True)).strip()
    if object_type != "blob":
        return None
    normalized = _normalized_path(path)
    suffix = PurePosixPath(normalized).suffix.casefold()
    if normalized.casefold().startswith("_workspace/") and suffix in DATA_SUFFIXES:
        return f"[{prefix}:data-ext] {normalized}"
    size = int(str(_git(root, "cat-file", "-s", object_name, text=True)).strip())
    if size > LIMIT_BYTES:
        return f"[{prefix}:>95MB] {normalized} ({size // 1_048_576}MB)"
    return None


def _scan_staged(root: Path) -> list[str]:
    output = _git(root, "diff", "--cached", "--name-only", "--diff-filter=AM", "-z")
    assert isinstance(output, bytes)
    violations: list[str] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = _normalized_path(raw_path)
        violation = _blob_violation(root, f":{path}", path, "staged")
        if violation:
            violations.append(violation)
    return violations


def _scan_outgoing(root: Path) -> list[str]:
    upstream = str(
        _git(root, "rev-parse", "--abbrev-ref", "@{u}", text=True)
    ).strip()
    output = _git(root, "rev-list", "--objects", f"{upstream}..HEAD")
    assert isinstance(output, bytes)
    violations: list[str] = []
    for line in output.splitlines():
        object_name, separator, raw_path = line.partition(b" ")
        if not separator or not raw_path:
            continue
        path = _normalized_path(raw_path)
        violation = _blob_violation(
            root, object_name.decode("ascii"), path, "push"
        )
        if violation:
            violations.append(violation)
    return violations


def main(arguments: list[str]) -> int:
    if arguments == ["--commit"]:
        request = "git commit"
    elif arguments == ["--push"]:
        request = "git push"
    elif arguments:
        raise GateError("expected --commit, --push, or hook JSON on stdin")
    else:
        request = sys.stdin.read()

    scan_commit = "git commit" in request
    scan_push = "git push" in request
    if not scan_commit and not scan_push:
        return 0

    root = _repository_root()
    violations: list[str] = []
    if scan_commit:
        violations.extend(_scan_staged(root))
    if scan_push:
        violations.extend(_scan_outgoing(root))

    if violations:
        print(
            "BLOCKED: research binary/large-blob payload is present in the commit/push set.",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            "Keep reproducible manifests/code/summaries only; reacquire original data from its documented DOI.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (GateError, OSError, ValueError) as error:
        print(f"BLOCKED: payload gate infrastructure failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
