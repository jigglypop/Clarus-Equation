"""PostToolUse(Write|Edit|MultiEdit) 유도 파일 자동 검증. fail-open.

derivations/**/*.derivation.md 또는 *.formula.md(추측 카드)가 저장되면 verify_derivation.py를 60초 제한으로 돌리고
결과 JSON을 additionalContext로 문맥에 넣는다. 실패해도 차단하지 않는다(exit 0).
이유: 탐색 단계는 fail-open이며, 실패 사실이 문맥에 들어오는 것으로 충분하다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TIMEOUT_SECONDS = 60
VERIFIER = Path(__file__).resolve().with_name("verify_derivation.py")


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def is_derivation(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    return relative.startswith("derivations/") and relative.endswith((".derivation.md", ".formula.md"))


def run_verifier(path: Path, root: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("CLAUDE_PROJECT_DIR", str(root))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(VERIFIER), str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"symbolic": "skipped", "numeric": "skipped", "reason": "timeout"}
    stdout = completed.stdout.strip().splitlines()
    for line in reversed(stdout):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"symbolic": "skipped", "numeric": "skipped", "reason": f"verifier error: {completed.stderr.strip()[:400]}"}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not raw:
        return 0
    root = repo_root()
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    if not is_derivation(path, root):
        return 0
    result = run_verifier(path, root)
    compact = {k: result.get(k) for k in ("symbolic", "numeric", "reason", "artifacts") if result.get(k) is not None}
    failures = [d for d in result.get("details", []) if "fail" in (d.get("symbolic"), d.get("numeric"))]
    summary = "[verify-on-save] " + json.dumps(compact, ensure_ascii=False)
    if failures:
        summary += "\n실패 검사: " + json.dumps(failures[:5], ensure_ascii=False, default=str)
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": summary}},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
