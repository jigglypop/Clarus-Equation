"""연구 원장(ledger/) 읽기·쓰기·검증 공용 라이브러리.

서브커맨드 (모두 결정적 코드, LLM 호출 없음):
  validate <entry.yaml>     항목 스키마 + 증거 등급-근거 일치 검사
  check-current             active 질문의 현재 attempt 항목 존재·유효 검사 (Stop 훅)
  summary                   세션 시작 요약 (40줄 이하)
  next-question [--dry-run] active 질문 id 출력. 없으면 open 중 priority 최소를 active로
  bump-attempt <Q>          attempts += 1 하고 새 attempt 번호 출력
  after-attempt <Q> <N>     항목의 level/verdict로 questions.yaml 상태 전이 (드라이버 4단계)
  incomplete <Q> <N>        INCOMPLETE 항목(L0, continue) 생성
  add-question --id --title [--priority] [--origin] [--kind]
  reindex                   ledger/index.md 재생성
  card-check <formula.md>   추측 카드(derivations/<Q>/F-NN.formula.md) 프론트매터 계약 검사
  adopt-card <Q> <formula.md>  카드의 식·kill·사다리를 질문 상태로 복사 (after-attempt가 adopt 때 자동 수행)
  ladder <Q>                질문의 증명 사다리와 다음 열린 단 출력

추측 우선 루프(conjecture-first 스킬): 질문은 kind ∈ conjecture|lemma|kill_test. conjecture 질문은
카드 하나(F-NN)와 사다리(≤7단)를 가지며 한 attempt가 한 단을 닫는다. sourcer가 identical/special_case를
두 번 보고하면 force_pivot=conjecture가 자동으로 붙는다. 이유: 재발견은 정지가 아니라 "더 강한 식을
세우라"는 신호이며, 축소만 반복하면 문헌으로 수렴한다.

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
VERDICTS = ("continue", "pivot", "promote", "park", "adopt", "refute")
REDUCTION_STEPS = ("partial", "alt_derivation", "reformulate", "weaken")
EXPANSION_STEPS = ("conjecture", "generalize")
PIVOT_STEPS = EXPANSION_STEPS + REDUCTION_STEPS
CHECK_RESULTS = ("pass", "fail", "skipped")
RELATIONS = ("identical", "special_case", "generalizes", "unrelated")
REDISCOVERY_RELATIONS = ("identical", "special_case")
QUESTION_STATUSES = ("open", "active", "resolved", "parked", "escalated")
QUESTION_KINDS = ("conjecture", "lemma", "kill_test")
CARD_KINDS = ("예측식", "예산식")
LADDER_KINDS = ("보조정리", "외부기존", "수치시험", "예측시험")
LADDER_STATUSES = ("open", "closed", "cited", "dead", "blocked")
LADDER_MAX_STEPS = 7
FORCE_CONJECTURE_AFTER_REDISCOVERIES = 2
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


# ---------------------------------------------------------------- 추측 카드

FRONTMATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)


def load_card(path: Path) -> dict[str, Any]:
    """derivations/<Q>/F-NN.formula.md 프론트매터만 읽는다. 본문은 사람·adversary 몫이다."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def card_errors(card: dict[str, Any]) -> list[str]:
    """추측 카드 계약(conjecture-first 스킬 §1).

    이유: 예측 숫자·극한 복원·사전등록 kill·사다리가 없는 식은 공리 후보가 아니라 낙서다.
    검사는 형식뿐이며 식의 참·거짓·신규성은 adversary·sourcer·judge가 판정한다.
    """
    errors: list[str] = []
    if not card:
        return ["card has no frontmatter"]
    for key in ("question", "card", "kind", "formula", "recovers", "kill", "ladder", "novelty", "verify"):
        if key not in card:
            errors.append(f"card missing key: {key}")
    if card.get("kind") not in CARD_KINDS:
        errors.append(f"card.kind must be one of {CARD_KINDS}: {card.get('kind')}")
    if not str(card.get("formula") or "").strip():
        errors.append("card.formula is empty")
    if card.get("kind") == "예측식":
        predicts = card.get("predicts")
        if not isinstance(predicts, list) or not predicts:
            errors.append("예측식 needs predicts (>=1): 숫자를 지금 적는다")
        else:
            for index, item in enumerate(predicts):
                if not isinstance(item, dict) or "observable" not in item or "value" not in item:
                    errors.append(f"predicts[{index}] needs observable and value")
    if card.get("kind") == "예산식":
        budget = card.get("budget")
        parts = budget.get("parts") if isinstance(budget, dict) else None
        if not isinstance(budget, dict) or not budget.get("total") or not isinstance(parts, list) or len(parts) < 2:
            errors.append("예산식 needs budget: {total, parts: [>=2], defined_on}")
    recovers = card.get("recovers")
    if not isinstance(recovers, list) or not recovers:
        errors.append("recovers (기존 극한 복원) must list >=1 item")
    kill = card.get("kill")
    if not isinstance(kill, list) or len(kill) < 2:
        errors.append("kill (사전등록 반증 조건) must list >=2 items")
    ladder = card.get("ladder")
    if not isinstance(ladder, list) or not ladder:
        errors.append(f"ladder must list 1..{LADDER_MAX_STEPS} steps")
    else:
        if len(ladder) > LADDER_MAX_STEPS:
            errors.append(f"ladder has {len(ladder)} steps; max {LADDER_MAX_STEPS} (사다리를 줄여라, 질문을 쪼개지 마라)")
        for index, step in enumerate(ladder):
            if not isinstance(step, dict) or not str(step.get("claim") or "").strip():
                errors.append(f"ladder[{index}] needs claim")
                continue
            if step.get("kind") not in LADDER_KINDS:
                errors.append(f"ladder[{index}].kind must be one of {LADDER_KINDS}")
    novelty = card.get("novelty")
    if not isinstance(novelty, dict) or not str(novelty.get("ce_specific") or "").strip():
        errors.append("novelty.ce_specific (문헌에 없는 것 한 문장) is required")
    checks = card.get("verify")
    if not isinstance(checks, list) or not checks:
        errors.append("verify block must contain >=1 check (극한 복원 항등식)")
    return errors


def ladder_from_card(card: dict[str, Any]) -> list[dict[str, Any]]:
    ladder: list[dict[str, Any]] = []
    for index, step in enumerate(card.get("ladder") or [], start=1):
        if isinstance(step, dict):
            ladder.append(
                {
                    "step": _as_int(step.get("step")) or index,
                    "claim": str(step.get("claim", "")),
                    "kind": str(step.get("kind", "보조정리")),
                    "status": "open",
                    "entry": None,
                }
            )
        else:
            ladder.append({"step": index, "claim": str(step), "kind": "보조정리", "status": "open", "entry": None})
    return ladder


def adopt_card_into_question(question: dict[str, Any], card_path: Path, root: Path) -> list[str]:
    """verdict adopt: 카드의 식·kill·사다리를 질문의 기계 상태로 복사한다. 실패하면 오류 목록."""
    card = load_card(card_path)
    errors = card_errors(card)
    if errors:
        return errors
    try:
        relative = card_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(card_path).replace("\\", "/")
    question["kind"] = "conjecture"
    question["card"] = relative
    question["card_kind"] = str(card.get("kind"))
    question["card_status"] = "채택"
    question["formula"] = str(card.get("formula"))
    question["kill"] = [str(item) for item in card.get("kill") or []]
    question["ladder"] = ladder_from_card(card)
    question["rediscoveries"] = 0
    question.pop("force_pivot", None)
    return []


def ladder_progress(question: dict[str, Any]) -> tuple[int, int]:
    ladder = question.get("ladder") or []
    done = sum(1 for step in ladder if step.get("status") in ("closed", "cited"))
    return done, len(ladder)


def next_open_step(question: dict[str, Any]) -> dict[str, Any] | None:
    for step in question.get("ladder") or []:
        if step.get("status") in ("open", "blocked"):
            return step
    return None


def _append_note(question: dict[str, Any], note: str) -> str:
    return ((question.get("notes") or "") + " " + note).strip()


def _set_step(question: dict[str, Any], step_no: int, status: str, entry_id: Any) -> None:
    for step in question.get("ladder") or []:
        if _as_int(step.get("step")) == step_no:
            step["status"] = status
            step["entry"] = entry_id


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
    if entry.get("ladder_step") is not None and (_as_int(entry.get("ladder_step")) or 0) < 1:
        errors.append("ladder_step must be a positive integer or null")
    cited = entry.get("ladder_cited")
    if cited is not None:
        if not isinstance(cited, list):
            errors.append("ladder_cited must be a list of {step, ref}")
        else:
            for index, item in enumerate(cited):
                if not isinstance(item, dict) or (_as_int(item.get("step")) or 0) < 1 or not str(item.get("ref") or "").strip():
                    errors.append(f"ladder_cited[{index}] needs step (>=1) and ref")
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

    verdict = entry.get("verdict")
    prior_relations = (
        {item.get("relation") for item in sourcer.get("prior_art") or [] if isinstance(item, dict)}
        if isinstance(sourcer, dict)
        else set()
    )
    rediscovered = bool(prior_relations & set(REDISCOVERY_RELATIONS))
    if verdict == "adopt":
        card_rel = entry.get("card") or entry.get("derivation")
        if not card_rel:
            errors.append("verdict adopt requires card (derivations/<Q>/F-NN.formula.md)")
        else:
            card_path = (root or repo_root()) / str(card_rel)
            if not card_path.is_file():
                errors.append(f"card path does not exist: {card_rel}")
            else:
                errors.extend("card: " + item for item in card_errors(load_card(card_path)))
        if not isinstance(sourcer, dict):
            errors.append("verdict adopt requires the sourcer novelty check (sourcer must be a mapping)")
        elif rediscovered:
            errors.append(
                "verdict adopt forbidden: prior_art relation in "
                f"{REDISCOVERY_RELATIONS} (재발견은 채택 불가; refute 뒤 더 강한 카드로 재추측)"
            )
    if verdict == "refute":
        has_counter = isinstance(adversary, dict) and bool(adversary.get("counterexamples"))
        if not (has_counter or rediscovered or str(entry.get("kill_triggered") or "").strip()):
            errors.append("verdict refute requires a counterexample, a rediscovery relation, or kill_triggered")

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


def _spawn_questions(entry: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    for spawned in entry.get("open_questions_spawned") or []:
        if isinstance(spawned, dict):
            sid = spawned.get("id")
            title = spawned.get("title")
            priority = spawned.get("priority", 3)
            kind = spawned.get("kind", "lemma")
        else:
            sid, title, priority, kind = str(spawned), None, 3, "lemma"
        if sid and find_question(questions, sid) is None:
            questions.append(
                {
                    "id": sid,
                    "title": title or f"{entry.get('id')}에서 파생된 하위 질문 (제목 미기입)",
                    "kind": kind if kind in QUESTION_KINDS else "lemma",
                    "status": "open",
                    "priority": int(priority),
                    "origin": entry.get("id"),
                    "attempts": 0,
                    "consecutive_low": 0,
                    "pivots_tried": [],
                    "notes": "",
                }
            )


def _count_rediscovery(question: dict[str, Any], entry: dict[str, Any]) -> None:
    """sourcer가 identical/special_case를 보고한 attempt를 센다.

    이유: 재발견은 정지 신호가 아니라 "더 강한 식을 세우라"는 신호다. 두 번이면 다음 attempt를
    conjecture로 강제한다. 외부 정리를 인용해 닫는 사다리 단(외부기존)은 재발견이 아니다.
    """
    sourcer = entry.get("sourcer")
    if not isinstance(sourcer, dict) or entry.get("verdict") == "adopt":
        return
    relations = {item.get("relation") for item in sourcer.get("prior_art") or [] if isinstance(item, dict)}
    if not (relations & set(REDISCOVERY_RELATIONS)):
        return
    step_no = _as_int(entry.get("ladder_step"))
    if step_no and question.get("kind") == "conjecture":
        for step in question.get("ladder") or []:
            if _as_int(step.get("step")) == step_no and step.get("kind") == "외부기존":
                return
    question["rediscoveries"] = int(question.get("rediscoveries", 0)) + 1
    if question["rediscoveries"] >= FORCE_CONJECTURE_AFTER_REDISCOVERIES and question.get("kind") != "conjecture":
        question["force_pivot"] = "conjecture"
        question["notes"] = _append_note(
            question,
            f"[{entry.get('id')}] 재발견 {question['rediscoveries']}회 → 다음 attempt는 pivot_step=conjecture 강제(예측식·예산식 카드)",
        )


def apply_entry_to_question(
    question: dict[str, Any],
    entry: dict[str, Any],
    questions: list[dict[str, Any]],
    root: Path | None = None,
) -> None:
    """드라이버 4단계: level/verdict에 따른 질문 상태 전이."""
    base = root or repo_root()
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
    question.setdefault("kind", "lemma")

    for item in entry.get("ladder_cited") or []:
        if isinstance(item, dict) and _as_int(item.get("step")):
            _set_step(question, int(item["step"]), "cited", f"{entry.get('id')} {item.get('ref', '')}".strip())
    _count_rediscovery(question, entry)
    sourcer = entry.get("sourcer")
    rediscovered = isinstance(sourcer, dict) and any(
        isinstance(item, dict) and item.get("relation") in REDISCOVERY_RELATIONS for item in sourcer.get("prior_art") or []
    )

    if verdict == "adopt":
        card_rel = entry.get("card") or entry.get("derivation")
        errors = adopt_card_into_question(question, base / str(card_rel), base) if card_rel else ["no card path"]
        # 일괄 카드 모드: 다른 질문이 이미 active면 채택된 카드는 open 큐에 남긴다(사다리는 열려 있고 next-question이 순서대로 꺼낸다).
        other_active = any(q is not question and q.get("status") == "active" for q in questions)
        question["status"] = "open" if other_active else "active"
        if errors:
            question["notes"] = _append_note(question, f"[{entry.get('id')}] adopt 실패: {'; '.join(errors)}")
        else:
            question["notes"] = _append_note(question, f"[{entry.get('id')}] 카드 채택(예측): {question.get('formula')}")
    elif verdict == "refute":
        step_no = _as_int(entry.get("ladder_step"))
        if step_no:
            _set_step(question, step_no, "dead", entry.get("id"))
        if rediscovered and not entry.get("kill_triggered"):
            # 문헌 재발견으로 죽은 카드: 질문은 살려 두고 더 강한 카드를 요구한다.
            question["status"] = "active"
            question["force_pivot"] = "conjecture"
            if question.get("card"):
                question["card_status"] = "기각(재발견)"
            question["notes"] = _append_note(question, f"[{entry.get('id')}] 카드 재발견 기각 → 더 강한 카드로 재추측")
        elif (
            question.get("kind") == "conjecture"
            and not question.get("card")
            and not step_no
            and not entry.get("kill_triggered")
        ):
            # 채택 전 카드가 반례(P0)로 죽음: 반례는 축소 pivot 신호이지 질문의 죽음이 아니다(pivot-playbook).
            # 사전등록 kill이 발동한 것도 아니므로 질문은 살려 두고 더 강한 카드(F-NN+1)를 요구한다.
            # 일괄 모드: 다른 질문이 active면 open 큐에 둔다(force_pivot은 유지).
            other_active = any(q is not question and q.get("status") == "active" for q in questions)
            question["status"] = "open" if other_active else "active"
            question["force_pivot"] = "conjecture"
            question["notes"] = _append_note(
                question, f"[{entry.get('id')}] 카드 반례 기각(채택 전) → 재추측: {entry.get('next_action', '')}"
            )
        else:
            question["status"] = "parked"
            if question.get("card"):
                question["card_status"] = "기각"
            question["notes"] = _append_note(question, f"[{entry.get('id')}] 기각: {entry.get('next_action', '')}")
    elif verdict == "promote":
        step_no = _as_int(entry.get("ladder_step"))
        if question.get("kind") == "conjecture" and step_no and question.get("ladder"):
            _set_step(question, step_no, "closed", entry.get("id"))
            done, total = ladder_progress(question)
            if done >= total:
                question["status"] = "resolved"
                question["card_status"] = "정리"
                _spawn_questions(entry, questions)
            else:
                question["status"] = "active"
        else:
            question["status"] = "resolved"
            if question.get("card"):
                question["card_status"] = "정리"
            _spawn_questions(entry, questions)
    elif verdict == "park":
        question["status"] = "parked"
        question["notes"] = _append_note(question, f"[{entry.get('id')}] parked: {entry.get('next_action', '')}")
    elif question.get("consecutive_low", 0) >= ESCALATE_AFTER_LOW:
        question["status"] = "escalated"
    elif (
        set(REDUCTION_STEPS) <= set(tried)
        and level_index < 3
        and not (set(EXPANSION_STEPS) & set(tried))
        and question.get("kind") != "conjecture"
    ):
        # 축소만 네 번 했는데 L3가 안 나오면 좁히기를 멈추고 더 강한 식을 세운다.
        question["status"] = "active"
        question["force_pivot"] = "conjecture"
        question["notes"] = _append_note(question, "축소 4단계 소진, L3 미달 → 확장(conjecture) 강제")
    elif set(PIVOT_STEPS) <= set(tried) and level_index < 3:
        question["status"] = "parked"
        question["notes"] = _append_note(question, "pivot 6단계(확장 2+축소 4) 소진, L3 미달")
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
        if current.get("card"):
            done, total = ladder_progress(current)
            step = next_open_step(current)
            step_text = f"다음 단 {step.get('step')}({step.get('kind')}): {str(step.get('claim', ''))[:80]}" if step else "열린 단 없음"
            lines.append(
                f"[추측] {current.get('card')} {current.get('card_kind')} {current.get('card_status')} "
                f"formula={str(current.get('formula', ''))[:60]} 사다리 {done}/{total} — {step_text}"
            )
        elif current.get("kind") == "kill_test":
            lines.append("[추측] kill_test 질문(카드 면제) — 통과 뒤 파생 질문은 예측식·예산식 카드로 시작한다")
        elif current.get("kind", "lemma") != "conjecture":
            lines.append("[추측] 카드 없음 — attempt 시작 전 prover(모드: 추측)로 예측식·예산식 카드를 먼저 세운다(conjecture-first)")
        if current.get("force_pivot"):
            lines.append(
                f"[강제] 다음 attempt는 pivot_step={current['force_pivot']} (재발견 {current.get('rediscoveries', 0)}회 또는 축소 소진)"
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
        "| id | kind | status | priority | attempts | low | pivots | 카드/사다리 | title |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for q in questions:
        if q.get("card"):
            done, total = ladder_progress(q)
            card = f"{q.get('card_status', '')} {done}/{total}".strip()
        else:
            card = ""
        if q.get("force_pivot"):
            card = (card + f" force:{q['force_pivot']}").strip()
        lines.append(
            f"| {q.get('id')} | {q.get('kind', 'lemma')} | {q.get('status')} | {q.get('priority', '')} | {q.get('attempts', 0)} | "
            f"{q.get('consecutive_low', 0)} | {','.join(q.get('pivots_tried') or [])} | {card} | {q.get('title', '')} |"
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
    apply_entry_to_question(question, entry, questions, root)
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
    if args.kind not in QUESTION_KINDS:
        print(f"kind must be one of {QUESTION_KINDS}: {args.kind}", file=sys.stderr)
        return 2
    questions.append(
        {
            "id": args.id,
            "title": args.title,
            "kind": args.kind,
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


def cmd_card_check(args: argparse.Namespace) -> int:
    root = repo_root()
    path = Path(args.file)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        print(f"no such card file: {path}", file=sys.stderr)
        return 2
    card = load_card(path)
    errors = card_errors(card)
    if errors:
        print(json.dumps({"status": "FAIL", "file": str(path), "errors": errors}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "file": str(path),
                "kind": card.get("kind"),
                "formula": card.get("formula"),
                "ladder_steps": len(card.get("ladder") or []),
                "predicts": len(card.get("predicts") or []),
                "kill": len(card.get("kill") or []),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_adopt_card(args: argparse.Namespace) -> int:
    root = repo_root()
    questions = load_questions(root)
    question = find_question(questions, args.question)
    if question is None:
        print(f"unknown question: {args.question}", file=sys.stderr)
        return 2
    path = Path(args.file)
    if not path.is_absolute():
        path = root / path
    errors = adopt_card_into_question(question, path, root)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, ensure_ascii=False), file=sys.stderr)
        return 1
    save_questions(questions, root)
    reindex(root)
    print(json.dumps({"status": "PASS", "question": args.question, "card": question["card"], "ladder": question["ladder"]}, ensure_ascii=False))
    return 0


def cmd_ladder(args: argparse.Namespace) -> int:
    questions = load_questions()
    question = find_question(questions, args.question)
    if question is None:
        print(f"unknown question: {args.question}", file=sys.stderr)
        return 2
    done, total = ladder_progress(question)
    print(
        json.dumps(
            {
                "question": args.question,
                "kind": question.get("kind", "lemma"),
                "card": question.get("card"),
                "card_status": question.get("card_status"),
                "formula": question.get("formula"),
                "progress": f"{done}/{total}",
                "next": next_open_step(question),
                "force_pivot": question.get("force_pivot"),
                "rediscoveries": question.get("rediscoveries", 0),
                "ladder": question.get("ladder") or [],
            },
            ensure_ascii=False,
        )
    )
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
    p.add_argument("--kind", default="lemma")
    p.set_defaults(func=cmd_add_question)
    p = sub.add_parser("reindex")
    p.set_defaults(func=cmd_reindex)
    p = sub.add_parser("card-check")
    p.add_argument("file")
    p.set_defaults(func=cmd_card_check)
    p = sub.add_parser("adopt-card")
    p.add_argument("question")
    p.add_argument("file")
    p.set_defaults(func=cmd_adopt_card)
    p = sub.add_parser("ladder")
    p.add_argument("question")
    p.set_defaults(func=cmd_ladder)
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
