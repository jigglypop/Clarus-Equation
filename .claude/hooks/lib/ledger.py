"""연구 원장(ledger/) 읽기·쓰기·검증 공용 라이브러리.

서브커맨드 (모두 결정적 코드, LLM 호출 없음):
  validate <entry.yaml>     항목 스키마 + 증거 등급-근거 일치 검사
  check-current             active 질문의 현재 attempt 항목 존재·유효 검사 (Stop 훅)
  summary                   세션 시작 요약 (40줄 이하)
  next-question [--dry-run] active 질문 id 출력. 없으면 open 중 priority 최소를 active로
  bump-attempt <Q>          attempts += 1 하고 새 attempt 번호 출력
  after-attempt <Q> <N>     항목의 level/verdict로 questions.yaml 상태 전이 (드라이버 4단계)
  incomplete <Q> <N>        INCOMPLETE 항목(L0, continue) 생성
  add-question --id --title [--priority] [--origin]
  reindex                   ledger/index.md 재생성

저장소 루트는 CLAUDE_PROJECT_DIR 환경변수, 없으면 이 파일 위치에서 계산한다.
이유: 훅은 CLAUDE_PROJECT_DIR를 받고, 테스트는 임시 저장소를 가리킬 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

LEVELS = ("L0", "L1", "L2", "L3", "L4")
VERDICTS = ("continue", "pivot", "promote", "park")
PIVOT_STEPS = ("partial", "alt_derivation", "reformulate", "weaken")
CHECK_RESULTS = ("pass", "fail", "skipped")
RELATIONS = ("identical", "special_case", "generalizes", "unrelated")
QUESTION_STATUSES = ("open", "active", "resolved", "parked", "escalated")
REQUIRED_ENTRY_KEYS = (
    "id",
    "question",
    "attempt",
    "timestamp",
    "claim",
    "level",
    "verdict",
    "derivation",
    "verification",
    "adversary",
    "assumptions",
    "next_action",
)
ENTRY_ID_RE = re.compile(r"^E-(\d{8})-(\d{3})$")
QUESTION_ID_RE = re.compile(r"^Q-[A-Za-z0-9-]+$")
ESCALATE_AFTER_LOW = 3


def repo_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def ledger_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "ledger"


def questions_path(root: Path | None = None) -> Path:
    return ledger_dir(root) / "questions.yaml"


def entries_dir(root: Path | None = None) -> Path:
    return ledger_dir(root) / "entries"


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


def load_questions(root: Path | None = None) -> list[dict[str, Any]]:
    path = questions_path(root)
    if not path.is_file():
        return []
    data = _load_yaml(path)
    questions = data.get("questions") if isinstance(data, dict) else None
    return list(questions or [])


def save_questions(questions: list[dict[str, Any]], root: Path | None = None) -> None:
    _dump_yaml(questions_path(root), {"questions": questions})


def find_question(questions: list[dict[str, Any]], qid: str) -> dict[str, Any] | None:
    for question in questions:
        if question.get("id") == qid:
            return question
    return None


def load_entries(root: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    directory = entries_dir(root)
    if not directory.is_dir():
        return []
    result = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = _load_yaml(path)
        except yaml.YAMLError:
            data = {}
        if isinstance(data, dict):
            result.append((path, data))
    return result


def find_entry(
    qid: str, attempt: int, root: Path | None = None
) -> tuple[Path, dict[str, Any]] | None:
    for path, data in load_entries(root):
        if data.get("question") == qid and _as_int(data.get("attempt")) == attempt:
            return path, data
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 증거 등급


def derive_level(entry: dict[str, Any], root: Path | None = None) -> str:
    """evidence-ladder 표를 코드로 재구현한다.

    이유: judge가 매긴 level이 객관 기준을 벗어나면 원장 검증이 실패해야 한다
    (이중 점검). 기호검증 통과지만 반례가 있는 항목은 L3 미만이므로 L2에 둔다.
    """
    derivation = entry.get("derivation")
    if not derivation:
        return "L0"
    if root is not None and not (root / str(derivation)).is_file():
        return "L0"
    verification = entry.get("verification") or {}
    symbolic = verification.get("symbolic", "skipped")
    numeric = verification.get("numeric", "skipped")
    lean = verification.get("lean", "skipped")
    lean_waived = bool(verification.get("lean_waived", False))
    adversary = entry.get("adversary") or {}
    counterexamples = adversary.get("counterexamples") or []
    survived = adversary.get("survived_checks") or []
    sourcer = entry.get("sourcer")
    sourcer_ran = isinstance(sourcer, dict)

    if symbolic == "pass" and not counterexamples and len(survived) >= 3:
        if sourcer_ran and (lean == "pass" or lean_waived):
            return "L4"
        return "L3"
    if symbolic == "pass" or numeric == "pass":
        return "L2"
    return "L1"


# ---------------------------------------------------------------- 스키마 검증


def validate_entry(entry: Any, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(entry, dict):
        return ["entry is not a mapping"]
    missing = [key for key in REQUIRED_ENTRY_KEYS if key not in entry]
    if missing:
        errors.append("missing keys: " + ", ".join(missing))

    entry_id = str(entry.get("id", ""))
    if "id" in entry and not ENTRY_ID_RE.match(entry_id):
        errors.append(f"id must match E-YYYYMMDD-NNN: {entry_id}")
    if "question" in entry and not QUESTION_ID_RE.match(str(entry.get("question"))):
        errors.append(f"question must look like Q-xxxx: {entry.get('question')}")
    if "attempt" in entry and (_as_int(entry.get("attempt")) or 0) < 1:
        errors.append("attempt must be a positive integer")
    if "level" in entry and entry.get("level") not in LEVELS:
        errors.append(f"level must be one of {LEVELS}: {entry.get('level')}")
    if "verdict" in entry and entry.get("verdict") not in VERDICTS:
        errors.append(f"verdict must be one of {VERDICTS}: {entry.get('verdict')}")
    if entry.get("verdict") == "pivot" and entry.get("pivot_step") not in PIVOT_STEPS:
        errors.append(f"verdict pivot requires pivot_step in {PIVOT_STEPS}")
    if "claim" in entry and not str(entry.get("claim") or "").strip():
        errors.append("claim is empty")

    derivation = entry.get("derivation")
    if derivation:
        base = root or repo_root()
        if not (base / str(derivation)).is_file():
            errors.append(f"derivation path does not exist: {derivation}")

    verification = entry.get("verification")
    if verification is not None:
        if not isinstance(verification, dict):
            errors.append("verification must be a mapping")
        else:
            for key in ("symbolic", "numeric", "lean"):
                value = verification.get(key, "skipped")
                if value not in CHECK_RESULTS:
                    errors.append(f"verification.{key} must be in {CHECK_RESULTS}: {value}")

    adversary = entry.get("adversary")
    if adversary is not None:
        if not isinstance(adversary, dict):
            errors.append("adversary must be a mapping")
        else:
            counterexamples = adversary.get("counterexamples")
            if not isinstance(counterexamples, list):
                errors.append("adversary.counterexamples must be a list")
            else:
                for index, item in enumerate(counterexamples):
                    if not isinstance(item, dict) or not {"input", "expected", "observed"} <= set(item):
                        errors.append(
                            f"adversary.counterexamples[{index}] needs input/expected/observed"
                        )
            if not isinstance(adversary.get("survived_checks", []), list):
                errors.append("adversary.survived_checks must be a list")

    sourcer = entry.get("sourcer")
    if isinstance(sourcer, dict):
        for index, item in enumerate(sourcer.get("prior_art") or []):
            if not isinstance(item, dict) or item.get("relation") not in RELATIONS:
                errors.append(f"sourcer.prior_art[{index}].relation must be in {RELATIONS}")
    elif sourcer is not None:
        errors.append("sourcer must be a mapping or null")

    if not isinstance(entry.get("assumptions", []), list):
        errors.append("assumptions must be a list")
    if not isinstance(entry.get("open_questions_spawned", []), list):
        errors.append("open_questions_spawned must be a list")

    if not errors:
        expected = derive_level(entry, root)
        if entry.get("level") != expected:
            errors.append(
                f"level/evidence mismatch: declared {entry.get('level')}, evidence supports {expected}"
            )
    return errors


# ---------------------------------------------------------------- 파일 생성


def next_entry_id(root: Path | None = None, today: _dt.date | None = None) -> str:
    today = today or _dt.date.today()
    stamp = today.strftime("%Y%m%d")
    used = 0
    for _, data in load_entries(root):
        match = ENTRY_ID_RE.match(str(data.get("id", "")))
        if match and match.group(1) == stamp:
            used = max(used, int(match.group(2)))
    return f"E-{stamp}-{used + 1:03d}"


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "-", text).strip("-").lower()
    return slug[:limit] or "entry"


def write_incomplete_entry(qid: str, attempt: int, root: Path | None = None) -> Path:
    """훅이 세션을 영원히 붙들지 못하도록 L0/continue 항목을 대신 쓴다."""
    base = root or repo_root()
    entry_id = next_entry_id(base)
    entry = {
        "id": entry_id,
        "question": qid,
        "attempt": attempt,
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "claim": "(INCOMPLETE) attempt가 원장 기록 없이 종료됨",
        "level": "L0",
        "verdict": "continue",
        "pivot_step": None,
        "derivation": None,
        "verification": {"symbolic": "skipped", "numeric": "skipped", "lean": "skipped"},
        "adversary": {"counterexamples": [], "survived_checks": []},
        "sourcer": None,
        "assumptions": [],
        "open_questions_spawned": [],
        "next_action": "같은 attempt 번호로 다시 돌린다",
        "incomplete": True,
    }
    path = entries_dir(base) / f"INCOMPLETE-{qid}-attempt-{attempt}.yaml"
    _dump_yaml(path, entry)
    return path


# ---------------------------------------------------------------- 상태 전이


def active_question(questions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for question in questions:
        if question.get("status") == "active":
            return question
    return None


def pick_next_question(questions: list[dict[str, Any]]) -> dict[str, Any] | None:
    current = active_question(questions)
    if current is not None:
        return current
    candidates = [q for q in questions if q.get("status") == "open"]
    if not candidates:
        return None
    candidates.sort(key=lambda q: (int(q.get("priority", 5)), str(q.get("id"))))
    chosen = candidates[0]
    chosen["status"] = "active"
    return chosen


def apply_entry_to_question(
    question: dict[str, Any], entry: dict[str, Any], questions: list[dict[str, Any]]
) -> None:
    """드라이버 4단계: level/verdict에 따른 질문 상태 전이."""
    level_index = LEVELS.index(entry.get("level", "L0"))
    if level_index <= 1:
        question["consecutive_low"] = int(question.get("consecutive_low", 0)) + 1
    else:
        question["consecutive_low"] = 0
    verdict = entry.get("verdict")
    tried = list(question.get("pivots_tried") or [])
    if verdict == "pivot" and entry.get("pivot_step") and entry["pivot_step"] not in tried:
        tried.append(entry["pivot_step"])
    question["pivots_tried"] = tried

    if verdict == "promote":
        question["status"] = "resolved"
        for spawned in entry.get("open_questions_spawned") or []:
            if isinstance(spawned, dict):
                sid = spawned.get("id")
                title = spawned.get("title")
                priority = spawned.get("priority", 3)
            else:
                sid, title, priority = str(spawned), None, 3
            if sid and find_question(questions, sid) is None:
                questions.append(
                    {
                        "id": sid,
                        "title": title or f"{entry.get('id')}에서 파생된 하위 질문 (제목 미기입)",
                        "status": "open",
                        "priority": int(priority),
                        "origin": entry.get("id"),
                        "attempts": 0,
                        "consecutive_low": 0,
                        "pivots_tried": [],
                        "notes": "",
                    }
                )
    elif verdict == "park":
        question["status"] = "parked"
        note = f"[{entry.get('id')}] parked: {entry.get('next_action', '')}".strip()
        question["notes"] = ((question.get("notes") or "") + " " + note).strip()
    elif question.get("consecutive_low", 0) >= ESCALATE_AFTER_LOW:
        question["status"] = "escalated"
    elif set(PIVOT_STEPS) <= set(tried) and level_index < 3:
        question["status"] = "parked"
        question["notes"] = ((question.get("notes") or "") + " pivot 4단계 소진, L3 미달").strip()
    else:
        question["status"] = "active"


# ---------------------------------------------------------------- 요약·색인


def summary_lines(root: Path | None = None) -> list[str]:
    questions = load_questions(root)
    entries = load_entries(root)
    lines: list[str] = []
    current = active_question(questions)
    if current is None:
        open_count = sum(1 for q in questions if q.get("status") == "open")
        lines.append(f"[원장] active 질문 없음. open {open_count}개.")
    else:
        lines.append(
            f"[원장] active {current.get('id')} (attempt {current.get('attempts', 0)}, "
            f"consecutive_low {current.get('consecutive_low', 0)}, "
            f"pivots {current.get('pivots_tried') or []}): {current.get('title', '')}"
        )
    recent = sorted(entries, key=lambda item: str(item[1].get("id", "")), reverse=True)[:3]
    for _, data in recent:
        lines.append(
            f"- {data.get('id')} {data.get('question')}#{data.get('attempt')} "
            f"{data.get('level')}/{data.get('verdict')}: {str(data.get('claim', ''))[:70]}"
        )
    escalated = [str(q.get("id")) for q in questions if q.get("status") == "escalated"]
    if escalated:
        lines.append("- escalated (사람 개입 필요): " + ", ".join(escalated))
    return lines[:40]


def reindex(root: Path | None = None) -> Path:
    base = root or repo_root()
    questions = load_questions(base)
    entries = load_entries(base)
    lines = [
        "# 연구 원장 색인",
        "",
        "자동 생성. `ledger.py reindex`가 덮어쓴다. 원장 정본은 `entries/*.yaml`이다.",
        "",
        "## 질문",
        "",
        "| id | status | priority | attempts | low | pivots | title |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in questions:
        lines.append(
            f"| {q.get('id')} | {q.get('status')} | {q.get('priority', '')} | {q.get('attempts', 0)} | "
            f"{q.get('consecutive_low', 0)} | {','.join(q.get('pivots_tried') or [])} | {q.get('title', '')} |"
        )
    lines += [
        "",
        "## 항목",
        "",
        "| id | question | attempt | level | verdict | claim |",
        "|---|---|---|---|---|---|",
    ]
    for _, e in sorted(entries, key=lambda item: str(item[1].get("id", ""))):
        claim = str(e.get("claim", "")).replace("|", "/")
        lines.append(
            f"| {e.get('id')} | {e.get('question')} | {e.get('attempt')} | {e.get('level')} | "
            f"{e.get('verdict')} | {claim[:80]} |"
        )
    path = ledger_dir(base) / "index.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- CLI


def cmd_validate(args: argparse.Namespace) -> int:
    root = repo_root()
    path = Path(args.file)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        print(f"no such entry file: {path}", file=sys.stderr)
        return 2
    try:
        entry = _load_yaml(path)
    except yaml.YAMLError as error:
        print(f"yaml error: {error}", file=sys.stderr)
        return 2
    errors = validate_entry(entry, root)
    if errors:
        print(
            json.dumps({"status": "FAIL", "file": str(path), "errors": errors}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "PASS", "file": str(path), "level": entry.get("level")}, ensure_ascii=False))
    return 0


def check_current(root: Path | None = None) -> tuple[int, str]:
    """(exit_code, message). 0=통과, 2=차단."""
    base = root or repo_root()
    questions = load_questions(base)
    current = active_question(questions)
    if current is None:
        return 0, "active 질문 없음 (루프 밖 세션)"
    attempt = int(current.get("attempts", 0))
    if attempt < 1:
        return 0, f"{current.get('id')} attempt 0 (아직 시작 전)"
    found = find_entry(str(current.get("id")), attempt, base)
    if found is None:
        return 2, (
            f"attempt {attempt}의 원장 항목이 없다. judge를 호출해 ledger/entries/에 "
            f"ledger-format 스키마({current.get('id')}, attempt {attempt})로 기록한 뒤 종료하라."
        )
    path, entry = found
    errors = validate_entry(entry, base)
    if errors:
        return 2, f"원장 항목 {path.name} 스키마 위반: " + "; ".join(errors)
    return 0, f"원장 항목 {path.name} 유효 ({entry.get('level')}/{entry.get('verdict')})"


def cmd_check_current(args: argparse.Namespace) -> int:
    code, message = check_current()
    print(message, file=sys.stderr if code else sys.stdout)
    return code


def cmd_summary(args: argparse.Namespace) -> int:
    for line in summary_lines():
        print(line)
    return 0


def cmd_next_question(args: argparse.Namespace) -> int:
    questions = load_questions()
    chosen = pick_next_question(questions)
    if chosen is None:
        print("NONE")
        return 3
    if not args.dry_run:
        save_questions(questions)
    print(chosen.get("id"))
    return 0


def cmd_bump_attempt(args: argparse.Namespace) -> int:
    questions = load_questions()
    question = find_question(questions, args.question)
    if question is None:
        print(f"unknown question: {args.question}", file=sys.stderr)
        return 2
    question["attempts"] = int(question.get("attempts", 0)) + 1
    save_questions(questions)
    print(question["attempts"])
    return 0


def cmd_after_attempt(args: argparse.Namespace) -> int:
    root = repo_root()
    questions = load_questions(root)
    question = find_question(questions, args.question)
    if question is None:
        print(f"unknown question: {args.question}", file=sys.stderr)
        return 2
    found = find_entry(args.question, args.attempt, root)
    if found is None:
        path = write_incomplete_entry(args.question, args.attempt, root)
        found = (path, _load_yaml(path))
        print(f"no entry; wrote {path.name}", file=sys.stderr)
    path, entry = found
    errors = validate_entry(entry, root)
    if errors:
        print(f"entry invalid, treated as L0/continue: {errors}", file=sys.stderr)
        entry = {**entry, "level": "L0", "verdict": "continue"}
    apply_entry_to_question(question, entry, questions)
    save_questions(questions)
    reindex(root)
    print(
        json.dumps(
            {
                "question": args.question,
                "status": question["status"],
                "level": entry.get("level"),
                "verdict": entry.get("verdict"),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_incomplete(args: argparse.Namespace) -> int:
    print(str(write_incomplete_entry(args.question, args.attempt)))
    return 0


def cmd_add_question(args: argparse.Namespace) -> int:
    questions = load_questions()
    if find_question(questions, args.id) is not None:
        print(f"question exists: {args.id}", file=sys.stderr)
        return 2
    if not QUESTION_ID_RE.match(args.id):
        print(f"question id must look like Q-xxxx: {args.id}", file=sys.stderr)
        return 2
    questions.append(
        {
            "id": args.id,
            "title": args.title,
            "status": "open",
            "priority": args.priority,
            "origin": args.origin or "human",
            "attempts": 0,
            "consecutive_low": 0,
            "pivots_tried": [],
            "notes": "",
        }
    )
    save_questions(questions)
    print(args.id)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    print(str(reindex()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ledger.py")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("file")
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("check-current")
    p.set_defaults(func=cmd_check_current)
    p = sub.add_parser("summary")
    p.set_defaults(func=cmd_summary)
    p = sub.add_parser("next-question")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_next_question)
    p = sub.add_parser("bump-attempt")
    p.add_argument("question")
    p.set_defaults(func=cmd_bump_attempt)
    p = sub.add_parser("after-attempt")
    p.add_argument("question")
    p.add_argument("attempt", type=int)
    p.set_defaults(func=cmd_after_attempt)
    p = sub.add_parser("incomplete")
    p.add_argument("question")
    p.add_argument("attempt", type=int)
    p.set_defaults(func=cmd_incomplete)
    p = sub.add_parser("add-question")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--priority", type=int, default=3)
    p.add_argument("--origin", default=None)
    p.set_defaults(func=cmd_add_question)
    p = sub.add_parser("reindex")
    p.set_defaults(func=cmd_reindex)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
