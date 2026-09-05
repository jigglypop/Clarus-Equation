"""Selection failures must not masquerade as an exhausted research queue."""

import subprocess

import pytest

from scripts import research_loop


@pytest.mark.parametrize("code,output,error", [(1, "", "broken ledger"), (0, "", "")])
def test_selection_failure_returns_nonzero(monkeypatch, code, output, error):
    monkeypatch.setattr(
        research_loop, "ledger_cmd",
        lambda *args: subprocess.CompletedProcess(args, code, output, error),
    )
    assert research_loop.main(["--dry-run"]) == 2


def test_empty_queue_is_success(monkeypatch):
    monkeypatch.setattr(
        research_loop, "ledger_cmd",
        lambda *args: subprocess.CompletedProcess(args, 0, "NONE\n", ""),
    )
    assert research_loop.main(["--dry-run"]) == 0


def test_unknown_forced_question_is_error_without_writes(monkeypatch):
    monkeypatch.setattr(research_loop.ledger, "load_questions", lambda root: [])
    monkeypatch.setattr(
        research_loop.ledger, "save_questions",
        lambda *args: pytest.fail("invalid question must not modify the queue"),
    )
    assert research_loop.main(["--question", "Q-missing", "--dry-run"]) == 2
