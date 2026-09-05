"""검증 프로세스의 실패가 이전 통과 기록으로 남지 않도록 검사한다."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml


SOURCE = Path(__file__).resolve().parents[1] / ".claude/hooks/lib/verify_on_save.py"
SPEC = importlib.util.spec_from_file_location("ce_save_failure_checks", SOURCE)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


@pytest.mark.parametrize("card", [None, "F-02"])
@pytest.mark.parametrize("outcome", ["timeout", "exit", "invalid", "scalar", "details", "missing", "launch"])
def test_process_failure_replaces_previous_pass(tmp_path, monkeypatch, card, outcome):
    path = tmp_path / "derivations/Q-TEST-1/attempt-01.derivation.md"
    path.parent.mkdir(parents=True)
    front = {"question": "Q-TEST-1", "attempt": 1, "card": card}
    path.write_text("---\n" + yaml.safe_dump(front) + "---\n", encoding="utf-8")
    artifact = tmp_path / "verify/Q-TEST-1" / (card or "attempt-01") / "hook_result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"symbolic":"pass","numeric":"pass"}', encoding="utf-8")

    def run(*args, **kwargs):
        assert kwargs["env"]["CLAUDE_PROJECT_DIR"] == str(tmp_path)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("verifier", 60)
        if outcome == "launch":
            raise OSError("프로세스 시작 실패")
        output = {"invalid": "{", "scalar": '"pass"',
                  "details": '{"symbolic":"pass","numeric":"pass","details":[1]}'}.get(
                      outcome, '{"symbolic":"pass","numeric":"pass"}')
        return subprocess.CompletedProcess([], 1 if outcome == "exit" else 0, output, "중단")

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "다른_저장소"))
    monkeypatch.setattr(hook.subprocess, "run", run)
    result = hook.run_verifier(path, tmp_path)
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved == result
    assert result["symbolic"] == result["numeric"] == "skipped"
    assert result["reason"]


@pytest.mark.parametrize("expression,status", [("0", "pass"), ("1", "fail")])
def test_real_verifier_preserves_successful_process_result(tmp_path, expression, status):
    path = tmp_path / "derivations/Q-TEST-1/F-02.formula.md"
    path.parent.mkdir(parents=True)
    front = {"question": "Q-TEST-1", "card": "F-02",
             "verify": [{"type": "numeric", "expr": expression}]}
    path.write_text("---\n" + yaml.safe_dump(front) + "---\n", encoding="utf-8")
    result = hook.run_verifier(path, tmp_path)
    assert result["numeric"] == status
    saved = json.loads((tmp_path / "verify/Q-TEST-1/F-02/hook_result.json").read_text(encoding="utf-8"))
    assert saved["numeric"] == status


def test_failure_artifact_cannot_escape_verify_directory(tmp_path):
    path = tmp_path / "source.derivation.md"
    path.write_text("---\nquestion: ../outside\nattempt: 1\n---\n", encoding="utf-8")
    result = hook.incomplete_result(path, tmp_path, "timeout")
    assert "artifact_error" in result
    assert not (tmp_path / "outside").exists()
