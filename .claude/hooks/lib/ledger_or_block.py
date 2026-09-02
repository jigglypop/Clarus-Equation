"""Stop / SubagentStop 훅: active 질문의 현재 attempt가 원장에 없으면 종료를 막는다.

규칙
1. SubagentStop은 agent_type이 judge일 때만 검사한다. 다른 서브에이전트(prover 등)의
   종료는 통과. 이유: prover가 멈출 때마다 원장을 요구하면 루프가 꼬인다.
2. Stop(메인 세션)은 항상 검사한다. active 질문이 없으면 통과(루프 밖 일반 세션).
3. 항목이 없거나 스키마·등급 위반이면 exit 2 + stderr 메시지.
4. 같은 세션에서 3회 차단하면 4회째는 통과시키고 INCOMPLETE 항목을 자동 생성한다.
   이유: 훅이 세션을 영원히 붙들면 헤드리스 드라이버가 죽는다.
   상태 파일: <tempdir>/harness-block-count-<session_id>
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402

MAX_BLOCKS = 3
LEDGER_WRITER = "judge"


def counter_path(session_id: str) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_") or "unknown"
    return Path(tempfile.gettempdir()) / f"harness-block-count-{safe}"


def read_count(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    event = str(payload.get("hook_event_name", "Stop"))
    agent = payload.get("agent_type") or os.environ.get("CLAUDE_AGENT_NAME")
    if agent and str(agent) != LEDGER_WRITER:
        return 0
    if event == "SubagentStop" and not agent:
        return 0

    root = ledger.repo_root()
    counter = counter_path(str(payload.get("session_id", "unknown")))
    blocked_so_far = read_count(counter)

    if blocked_so_far >= MAX_BLOCKS:
        questions = ledger.load_questions(root)
        current = ledger.active_question(questions)
        if current is not None and int(current.get("attempts", 0)) >= 1:
            qid, attempt = str(current["id"]), int(current["attempts"])
            if ledger.find_entry(qid, attempt, root) is None:
                path = ledger.write_incomplete_entry(qid, attempt, root)
                print(f"[ledger-or-block] 3회 차단 후 통과. {path.name} 자동 생성 (L0/continue).", file=sys.stderr)
            else:
                print("[ledger-or-block] 3회 차단 후 통과. 기존 항목의 스키마 위반은 남아 있다.", file=sys.stderr)
        try:
            counter.unlink()
        except OSError:
            pass
        return 0

    code, message = ledger.check_current(root)
    if code == 0:
        try:
            counter.unlink()
        except OSError:
            pass
        return 0
    counter.write_text(str(blocked_so_far + 1), encoding="utf-8")
    print(f"[ledger-or-block {blocked_so_far + 1}/{MAX_BLOCKS}] {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
