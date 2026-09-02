"""PreToolUse(Write|Edit|MultiEdit) 쓰기 게이트. 결정적 코드, LLM 없음.

규칙
1. 저장소 밖 경로(scratchpad 등)는 통과. 이유: 임시 계산은 저장소 밖에서 한다.
2. 저장소 안은 허용 접두사·허용 루트 파일만 통과, 나머지는 exit 2.
   명세의 문서 루트는 이 저장소에서 은퇴 경로이므로 paper/로 사상했고, 기존 direct 모드가
   고치는 tests/ examples/ experiments/ 등은 허용에 포함했다.
3. ledger/ 쓰기는 호출 주체가 judge이거나 주체를 알 수 없을 때(메인 세션)만 허용.
   주체는 stdin JSON의 agent_type, 없으면 CLAUDE_AGENT_NAME 환경변수.
4. paper/ 밖의 내용에 http(s) URL이 있으면 경고만(stderr, exit 0). 이유: 차단하면 sourcer의
   문헌 인용이 막힌다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ALLOWED_PREFIXES = (
    "ledger/",
    "derivations/",
    "verify/",
    "paper/",
    "scripts/",
    ".claude/",
    ".codex/",
    "_workspace/",
    "tests/",
    "examples/",
    "experiments/",
    "benchmarks/",
    "artifacts/",
)
ALLOWED_ROOT_FILES = frozenset(
    {
        "README.md",
        "AGENTS.md",
        "HARNESS_SPEC.md",
        ".gitignore",
        ".gitattributes",
        "pyproject.toml",
        "requirements-harness.txt",
    }
)
LEDGER_WRITER = "judge"


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def caller_agent(payload: dict) -> str | None:
    for key in ("agent_type", "agent_name", "subagent_type"):
        value = payload.get(key)
        if value:
            return str(value)
    return os.environ.get("CLAUDE_AGENT_NAME") or None


def written_text(tool_input: dict) -> str:
    parts = [str(tool_input.get("content", "")), str(tool_input.get("new_string", ""))]
    for edit in tool_input.get("edits") or []:
        if isinstance(edit, dict):
            parts.append(str(edit.get("new_string", "")))
    return "\n".join(parts)


def decide(payload: dict, root: Path) -> tuple[int, str]:
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("path")
    if not raw:
        return 0, ""
    target = Path(str(raw))
    if not target.is_absolute():
        target = root / target
    try:
        relative = target.resolve().relative_to(root).as_posix()
    except ValueError:
        return 0, ""  # 저장소 밖: 게이트 대상 아님

    allowed = relative.startswith(ALLOWED_PREFIXES) or (
        "/" not in relative and relative in ALLOWED_ROOT_FILES
    )
    if not allowed:
        return 2, (
            f"쓰기 허용 경로 밖: {relative}. 허용: "
            + " ".join(prefix for prefix in ALLOWED_PREFIXES)
            + " 및 루트 파일 "
            + " ".join(sorted(ALLOWED_ROOT_FILES))
        )

    if relative.startswith("ledger/"):
        agent = caller_agent(payload)
        if agent and agent != LEDGER_WRITER:
            return 2, f"ledger/ 는 {LEDGER_WRITER}만 쓴다 (호출 주체: {agent}). judge가 ledger.py validate를 거쳐 직접 쓴다."

    warning = ""
    if not relative.startswith("paper/"):
        text = written_text(tool_input)
        if "http://" in text or "https://" in text:
            warning = f"[write-gate 경고] {relative} 에 URL이 포함됨. 연구 데이터 원본·외부 링크 반출 여부를 확인하라."
    return 0, warning


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
    code, message = decide(payload, repo_root())
    if message:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
