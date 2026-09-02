"""`.claude` 연구 하네스 수용 테스트 (HARNESS_SPEC §11 A1–A8, A10–A12).

훅은 subprocess로 stdin JSON을 넣어 호출한다. 저장소 루트는 CLAUDE_PROJECT_DIR로 임시
디렉터리를 가리킨다. A9(실제 claude -p 실행)는 수동이다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".claude" / "hooks" / "lib"
SPEC_SKILLS = (
    "research-loop",
    "evidence-ladder",
    "pivot-playbook",
    "derivation-style",
    "ledger-format",
    "ko-academic-prose",
)
HAVE_SYMPY = importlib.util.find_spec("sympy") is not None


def _run(script: str, *args: str, stdin: dict | None = None, root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("CLAUDE_AGENT_NAME", None)
    return subprocess.run(
        [sys.executable, "-B", str(LIB / script), *args],
        input=json.dumps(stdin) if stdin is not None else "",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


def _write_questions(root: Path, questions: list[dict]) -> None:
    (root / "ledger" / "entries").mkdir(parents=True, exist_ok=True)
    (root / "ledger" / "questions.yaml").write_text(
        yaml.safe_dump({"questions": questions}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _question(qid: str = "Q-TEST-1", status: str = "active", attempts: int = 1, priority: int = 3) -> dict:
    return {
        "id": qid,
        "title": "모든 n≥1에 대해 Σ_{k=1}^n k = n(n+1)/2",
        "status": status,
        "priority": priority,
        "origin": "human",
        "attempts": attempts,
        "consecutive_low": 0,
        "pivots_tried": [],
        "notes": "",
    }


DERIVATION_WITH_VERIFY = """---
question: Q-TEST-1
attempt: 1
claim: "Σ_{k=1}^n k = n(n+1)/2"
assumptions:
  - "n은 양의 정수"
symbols:
  n: positive integer
verify:
  - type: identity
    lhs: "n*(n+1)/2 + (n+1)"
    rhs: "(n+1)*(n+2)/2"
---

## 유도

$$ S(n+1) = S(n) + (n+1) $$  (S1) 정의 대입
"""

DERIVATION_NO_VERIFY = """---
question: Q-TEST-1
attempt: 1
claim: "Σ_{k=1}^n k = n(n+1)/2"
assumptions: []
symbols:
  n: positive integer
---

## 유도

$$ S(n) = n(n+1)/2 $$  (S1) 주장
"""

DERIVATION_FALSE = """---
question: Q-TEST-1
attempt: 1
claim: "(x+1)^2 = x^2 + 1"
assumptions: []
symbols:
  x: real
verify:
  - type: identity
    lhs: "(x+1)**2"
    rhs: "x**2 + 1"
---

## 유도

$$ (x+1)^2 = x^2 + 1 $$  (S1) 틀린 전개
"""


def _entry(root: Path, **overrides) -> dict:
    derivation = root / "derivations" / "Q-TEST-1" / "attempt-01.derivation.md"
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text(DERIVATION_WITH_VERIFY, encoding="utf-8")
    entry = {
        "id": "E-20260902-001",
        "question": "Q-TEST-1",
        "attempt": 1,
        "timestamp": "2026-09-02T10:00:00+09:00",
        "claim": "Σ_{k=1}^n k = n(n+1)/2",
        "level": "L3",
        "verdict": "promote",
        "pivot_step": None,
        "derivation": "derivations/Q-TEST-1/attempt-01.derivation.md",
        "verification": {"symbolic": "pass", "numeric": "pass", "lean": "skipped"},
        "adversary": {"counterexamples": [], "survived_checks": ["dimension", "n_equals_1", "induction"]},
        "sourcer": None,
        "assumptions": ["n≥1"],
        "open_questions_spawned": [],
        "next_action": "논문 반영",
    }
    entry.update(overrides)
    return entry


def _write_entry(root: Path, entry: dict, name: str = "20260902-001-gauss.yaml") -> Path:
    path = root / "ledger" / "entries" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name in ("ledger/entries", "derivations", "verify", "src"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    _write_questions(tmp_path, [_question(status="open", attempts=0)])
    return tmp_path


# ---------------------------------------------------------------- A1


def test_a1_write_outside_allowed_paths_is_blocked(repo: Path) -> None:
    payload = {"tool_name": "Write", "tool_input": {"file_path": str(repo / "src" / "x.py"), "content": "x = 1"}}
    result = _run("write_gate.py", stdin=payload, root=repo)
    assert result.returncode == 2
    assert "ledger/" in result.stderr and "derivations/" in result.stderr


def test_a1b_allowed_paths_pass_and_ledger_is_owner_gated(repo: Path) -> None:
    ok = _run("write_gate.py", stdin={"tool_input": {"file_path": str(repo / "derivations" / "a.md"), "content": ""}}, root=repo)
    assert ok.returncode == 0
    prover = _run(
        "write_gate.py",
        stdin={"agent_type": "prover", "tool_input": {"file_path": str(repo / "ledger" / "entries" / "x.yaml"), "content": ""}},
        root=repo,
    )
    assert prover.returncode == 2 and "judge" in prover.stderr
    writer = _run(
        "write_gate.py",
        stdin={"agent_type": "judge", "tool_input": {"file_path": str(repo / "ledger" / "entries" / "x.yaml"), "content": ""}},
        root=repo,
    )
    assert writer.returncode == 0
    outside = _run("write_gate.py", stdin={"tool_input": {"file_path": str(Path(tempfile.gettempdir()) / "elsewhere.py"), "content": ""}}, root=repo)
    assert outside.returncode == 0


# ---------------------------------------------------------------- A2–A4


def _save_derivation(repo: Path, text: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    path = repo / "derivations" / "Q-TEST-1" / "attempt-01.derivation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    result = _run("verify_on_save.py", stdin={"tool_name": "Write", "tool_input": {"file_path": str(path), "content": text}}, root=repo)
    hook_result = repo / "verify" / "Q-TEST-1" / "attempt-01" / "hook_result.json"
    data = json.loads(hook_result.read_text(encoding="utf-8")) if hook_result.is_file() else {}
    return result, data


def test_a2_derivation_with_verify_block_is_verified(repo: Path) -> None:
    result, data = _save_derivation(repo, DERIVATION_WITH_VERIFY)
    assert result.returncode == 0
    assert data, "hook_result.json must be written"
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "symbolic" in context
    assert data["numeric"] == "pass"
    assert data["symbolic"] == ("pass" if HAVE_SYMPY else "skipped")


def test_a3_derivation_without_verify_block_is_skipped_not_blocked(repo: Path) -> None:
    result, data = _save_derivation(repo, DERIVATION_NO_VERIFY)
    assert result.returncode == 0
    assert data["symbolic"] == "skipped"
    assert "no verify block" in data.get("reason", "")


def test_a4_false_identity_fails_without_blocking(repo: Path) -> None:
    result, data = _save_derivation(repo, DERIVATION_FALSE)
    assert result.returncode == 0
    assert data["numeric"] == "fail"
    assert data["symbolic"] == ("fail" if HAVE_SYMPY else "skipped")
    assert "fail" in result.stdout


# ---------------------------------------------------------------- A5–A7


def _stop(repo: Path, session: str, event: str = "Stop", agent: str | None = None) -> subprocess.CompletedProcess[str]:
    payload = {"hook_event_name": event, "session_id": session, "stop_hook_active": False}
    if agent:
        payload["agent_type"] = agent
    return _run("ledger_or_block.py", stdin=payload, root=repo)


def _cleanup_counter(session: str) -> None:
    counter = Path(tempfile.gettempdir()) / f"harness-block-count-{session}"
    if counter.exists():
        counter.unlink()


def test_a5_stop_without_ledger_entry_is_blocked(repo: Path) -> None:
    _write_questions(repo, [_question(status="active", attempts=1)])
    session = f"t-{uuid.uuid4().hex}"
    try:
        result = _stop(repo, session)
        assert result.returncode == 2
        assert "judge" in result.stderr
        # 다른 서브에이전트의 SubagentStop은 통과한다.
        assert _stop(repo, session, event="SubagentStop", agent="prover").returncode == 0
        # active 질문이 없으면 통과한다.
        _write_questions(repo, [_question(status="open", attempts=0)])
        assert _stop(repo, session + "b").returncode == 0
    finally:
        _cleanup_counter(session)
        _cleanup_counter(session + "b")


def test_a6_fourth_stop_passes_and_writes_incomplete_entry(repo: Path) -> None:
    _write_questions(repo, [_question(status="active", attempts=2)])
    session = f"t-{uuid.uuid4().hex}"
    try:
        codes = [_stop(repo, session).returncode for _ in range(4)]
        assert codes == [2, 2, 2, 0]
        incomplete = repo / "ledger" / "entries" / "INCOMPLETE-Q-TEST-1-attempt-2.yaml"
        assert incomplete.is_file()
        data = yaml.safe_load(incomplete.read_text(encoding="utf-8"))
        assert data["level"] == "L0" and data["verdict"] == "continue"
        # 항목이 생겼으므로 다음 Stop은 검사 통과.
        assert _stop(repo, session).returncode == 0
    finally:
        _cleanup_counter(session)


def test_a7_level_evidence_mismatch_is_rejected(repo: Path) -> None:
    _write_questions(repo, [_question(status="active", attempts=1)])
    bad = _entry(
        repo,
        adversary={
            "counterexamples": [{"input": "n=0", "expected": "0", "observed": "정의역 밖", "note": ""}],
            "survived_checks": ["dimension", "n_equals_1", "induction"],
        },
    )
    path = _write_entry(repo, bad)
    validate = _run("ledger.py", "validate", str(path), root=repo)
    assert validate.returncode == 1
    assert "mismatch" in validate.stderr and "L2" in validate.stderr
    session = f"t-{uuid.uuid4().hex}"
    try:
        assert _stop(repo, session).returncode == 2
    finally:
        _cleanup_counter(session)
    good = _entry(repo)
    path = _write_entry(repo, good)
    assert _run("ledger.py", "validate", str(path), root=repo).returncode == 0


# ---------------------------------------------------------------- A8


def test_a8_dry_run_picks_lowest_priority_without_changes(repo: Path) -> None:
    _write_questions(
        repo,
        [
            _question("Q-0002", status="open", attempts=0, priority=2),
            _question("Q-0001", status="open", attempts=0, priority=1),
            _question("Q-0003", status="parked", attempts=0, priority=1),
        ],
    )
    before = (repo / "ledger" / "questions.yaml").read_bytes()
    driver = ROOT / "scripts" / "research_loop.py"
    # 드라이버는 저장소 루트를 자기 위치에서 계산하므로 임시 저장소로 복사해 실행한다.
    fake = repo / "repo"
    (fake / "scripts").mkdir(parents=True)
    shutil.copytree(LIB, fake / ".claude" / "hooks" / "lib")
    shutil.copy(driver, fake / "scripts" / "research_loop.py")
    shutil.copytree(repo / "ledger", fake / "ledger")
    result = subprocess.run(
        [sys.executable, "-B", str(fake / "scripts" / "research_loop.py"), "--dry-run"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CLAUDE_PROJECT_DIR": str(fake)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Q-0001" in result.stdout
    assert (fake / "ledger" / "questions.yaml").read_bytes() == before
    assert not (fake / "logs").exists()


# ---------------------------------------------------------------- A10–A12


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return yaml.safe_load(match.group(1))


def test_a10_adversary_cannot_write() -> None:
    front = _frontmatter(ROOT / ".claude" / "agents" / "adversary.md")
    tools = [t.strip() for t in str(front["tools"]).split(",")]
    assert "Write" not in tools and "Edit" not in tools and "MultiEdit" not in tools
    assert "Bash" in tools


def test_a10b_all_loop_agents_have_required_frontmatter() -> None:
    for name in ("prover", "adversary", "sourcer", "judge", "paper-writer"):
        front = _frontmatter(ROOT / ".claude" / "agents" / f"{name}.md")
        assert front["name"] == name
        assert front["description"] and front["tools"] and front["model"] in {"inherit", "opus", "sonnet", "haiku"}


def test_a11_spec_skills_have_trigger_descriptions_and_short_bodies() -> None:
    for name in SPEC_SKILLS:
        path = ROOT / ".claude" / "skills" / name / "SKILL.md"
        front = _frontmatter(path)
        assert front["name"] == name
        assert any(word in front["description"] for word in ("때", "호출", "참조"))
        body = path.read_text(encoding="utf-8").split("---", 2)[2]
        assert len(body.strip().splitlines()) <= 150, f"{name} body too long"


def test_a12_validate_reports_missing_keys(repo: Path) -> None:
    path = _write_entry(repo, {"id": "E-20260902-001", "question": "Q-TEST-1"}, name="20260902-001-bad.yaml")
    result = _run("ledger.py", "validate", str(path), root=repo)
    assert result.returncode != 0
    assert "missing keys" in result.stderr
    for key in ("attempt", "claim", "level", "verdict", "next_action"):
        assert key in result.stderr


# ---------------------------------------------------------------- 상태 전이 단위 검사


def test_after_attempt_transitions(repo: Path) -> None:
    _write_questions(repo, [_question(status="active", attempts=1)])
    _write_entry(repo, _entry(repo, level="L1", verdict="continue", verification={"symbolic": "fail", "numeric": "skipped", "lean": "skipped"}))
    out = _run("ledger.py", "after-attempt", "Q-TEST-1", "1", root=repo)
    assert out.returncode == 0, out.stderr
    q = yaml.safe_load((repo / "ledger" / "questions.yaml").read_text(encoding="utf-8"))["questions"][0]
    assert q["status"] == "active" and q["consecutive_low"] == 1
    assert (repo / "ledger" / "index.md").is_file()

    _write_questions(repo, [_question(status="active", attempts=1)])
    _write_entry(repo, _entry(repo, open_questions_spawned=[{"id": "Q-TEST-2", "title": "하위", "priority": 2}]))
    out = _run("ledger.py", "after-attempt", "Q-TEST-1", "1", root=repo)
    assert out.returncode == 0, out.stderr
    qs = yaml.safe_load((repo / "ledger" / "questions.yaml").read_text(encoding="utf-8"))["questions"]
    assert qs[0]["status"] == "resolved"
    assert qs[1]["id"] == "Q-TEST-2" and qs[1]["status"] == "open"
