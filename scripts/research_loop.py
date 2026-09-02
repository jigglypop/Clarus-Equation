"""헤드리스 연구 루프 드라이버.

사용: research_loop.py [--max-iters N=20] [--question Q-id] [--dry-run] [--max-turns 60]

반복:
  1. ledger.py next-question  → active 질문(없으면 open 중 priority 최소를 active로)
     escalated/parked/resolved만 남으면 exit 0 (사람 개입 필요)
  2. ledger.py bump-attempt <Q>  → attempts += 1
  3. claude -p "<research-loop 프롬프트>" --max-turns N --output-format json > logs/<Q>-attempt-<N>.json
  4. ledger.py after-attempt <Q> <N>  → 항목 없으면 INCOMPLETE 생성, level/verdict로 상태 전이
  5. --max-iters 도달 시 종료

--dry-run은 3단계를 건너뛰고 어떤 질문이 선택될지만 출력한다(파일 변경 없음).
이유: 드라이버는 결정적 코드여야 하고, claude -p가 비정상 종료해도 4단계가 루프를 잇는다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / ".claude" / "hooks" / "lib"
sys.path.insert(0, str(LIB))
import ledger  # noqa: E402

PROMPT = (
    "research-loop 스킬에 따라 {qid}의 attempt {attempt}을 한 바퀴 돌리고 "
    "(prover 후보→prover 유도→adversary→[sourcer]→judge 판정·원장) "
    "ledger/entries/에 기록한 뒤 종료하라. 오케스트레이터는 직접 유도·판정하지 않는다."
)


def env_for_child() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CLAUDE_PROJECT_DIR", str(REPO_ROOT))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def ledger_cmd(*args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(LIB / "ledger.py"), *args],
        capture_output=capture,
        text=True,
        encoding="utf-8",
        env=env_for_child(),
        check=False,
    )


def select_question(forced: str | None, dry_run: bool) -> str | None:
    if forced:
        questions = ledger.load_questions(REPO_ROOT)
        target = ledger.find_question(questions, forced)
        if target is None:
            print(f"unknown question: {forced}", file=sys.stderr)
            return None
        if target.get("status") in ("resolved",):
            print(f"{forced} is already {target['status']}", file=sys.stderr)
            return None
        if not dry_run:
            for question in questions:
                if question.get("status") == "active" and question is not target:
                    question["status"] = "open"
            target["status"] = "active"
            ledger.save_questions(questions, REPO_ROOT)
        return forced
    args = ["next-question"] + (["--dry-run"] if dry_run else [])
    completed = ledger_cmd(*args)
    output = completed.stdout.strip()
    if completed.returncode != 0 or output == "NONE" or not output:
        return None
    return output


def run_claude(qid: str, attempt: int, max_turns: int, log_path: Path) -> int:
    claude = shutil.which("claude")
    if claude is None:
        print("claude CLI를 PATH에서 찾지 못했다. Claude Code를 설치하거나 PATH를 고쳐라.", file=sys.stderr)
        return 127
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.format(qid=qid, attempt=attempt)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [claude, "-p", prompt, "--max-turns", str(max_turns), "--output-format", "json"],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env_for_child(),
            check=False,
        )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="research-loop")
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--question", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-turns", type=int, default=60)
    args = parser.parse_args(argv)

    for iteration in range(1, args.max_iters + 1):
        qid = select_question(args.question, args.dry_run)
        if qid is None:
            print("선택 가능한 open/active 질문이 없다 (escalated/parked/resolved만 남음). 사람 개입 필요.")
            return 0
        if args.dry_run:
            questions = ledger.load_questions(REPO_ROOT)
            question = ledger.find_question(questions, qid) or {}
            print(f"[dry-run] 다음 질문: {qid} (priority {question.get('priority')}, attempts {question.get('attempts', 0)}): {question.get('title', '')}")
            return 0
        bump = ledger_cmd("bump-attempt", qid)
        if bump.returncode != 0:
            print(bump.stderr, file=sys.stderr)
            return 2
        attempt = int(bump.stdout.strip())
        log_path = REPO_ROOT / "logs" / f"{qid}-attempt-{attempt}.json"
        print(f"[{iteration}/{args.max_iters}] {qid} attempt {attempt} → claude -p (log: {log_path.relative_to(REPO_ROOT)})")
        code = run_claude(qid, attempt, args.max_turns, log_path)
        if code != 0:
            print(f"claude -p 종료 코드 {code}. after-attempt가 INCOMPLETE 항목으로 루프를 잇는다.", file=sys.stderr)
        after = ledger_cmd("after-attempt", qid, str(attempt))
        print(after.stdout.strip() or after.stderr.strip())
        if after.returncode != 0:
            return after.returncode
        if args.question:
            questions = ledger.load_questions(REPO_ROOT)
            forced = ledger.find_question(questions, args.question) or {}
            if forced.get("status") != "active":
                print(f"{args.question} 상태 {forced.get('status')}. 루프 종료.")
                return 0
    print(f"--max-iters {args.max_iters} 도달. 요약: ledger/index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
