"""전체 연구 목표의 관측 단계와 알림 문맥 예산을 함께 보존한다."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".codex" / "hooks" / "goal_reminder.py"
SPEC = importlib.util.spec_from_file_location("ce_goal_reminder_checks", SOURCE)
reminder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reminder)


def test_current_goal_includes_gravity_and_all_cosmological_targets(monkeypatch, capsys):
    monkeypatch.setattr(reminder, "conjecture_line", lambda: None)
    assert reminder.main() == 0
    output = capsys.readouterr().out
    goal = next(line for line in output.splitlines() if line.startswith("[최종 목표]"))
    for required in ("끼임", "접힘", "0D", "공통 계량", "Plebanski/Einstein",
                     "암흑에너지", "암흑물질", "허블 텐션"):
        assert required in goal
    for required in ("공리", "예측식", "반증 시험", "다음 증명 고리"):
        assert required in output
    assert len(output) < 700


def test_long_ledger_and_conjecture_keep_transition_and_output_budget(tmp_path, monkeypatch, capsys):
    ledger = tmp_path / "target.md"
    content = "## 2. 지금 단 하나의 표적\n" + "".join(
        f"| {field} | {'가' * 1000} |\n" for field in reminder.FIELDS
    ) + "## 3. 조건부 결과\n"
    ledger.write_text(content, encoding="utf-8")
    monkeypatch.setattr(reminder, "LEDGER", ledger)
    monkeypatch.setattr(reminder, "conjecture_line", lambda: "[추측] " + "나" * 1000)
    assert reminder.main() == 0
    output = capsys.readouterr().out
    assert output.startswith("[최종 목표]")
    assert "[추측]" in output and "[전환]" in output and "[규율]" in output
    assert len(output) < 700


def test_rejected_card_requests_a_new_candidate_instead_of_an_empty_ladder(tmp_path, monkeypatch):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n- id: Q-0020\n  status: active\n"
        "  card: derivations/Q-0020/F-01.formula.md\n  card_status: 기각\n"
        "  formula: null\n  ladder: []\n  force_pivot: conjecture\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reminder, "QUESTIONS", questions)
    line = reminder.conjecture_line()
    assert "기존 카드 기각" in line
    assert "새 공리·예측식과 반증 시험" in line
    assert "0/0" not in line and "None" not in line and "열린 단 없음" not in line
