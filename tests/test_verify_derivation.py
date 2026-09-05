"""검증기가 빈 표본·비정상 수치를 증거로 세지 않고 선언한 정의역을 지킨다."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


SOURCE = Path(__file__).resolve().parents[1]/".claude/hooks/lib/verify_derivation.py"
SPEC = importlib.util.spec_from_file_location("ce_verifier_domain_checks", SOURCE)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def check_file(tmp_path, check, symbols=None):
    path = tmp_path/"attempt-01.derivation.md"
    front = {"question": "Q-TEST-1", "attempt": 1, "symbols": symbols or {}, "verify": [check]}
    path.write_text("---\n"+yaml.safe_dump(front)+"---\n", encoding="utf-8")
    with np.errstate(over="ignore", invalid="ignore"):
        return verifier.verify_file(path)


@pytest.mark.parametrize("symbolic_backend", [False, True])
@pytest.mark.parametrize("expr", ["exp(1000)", "log(-1)"])
def test_nonfinite_result_cannot_pass(tmp_path, monkeypatch, symbolic_backend, expr):
    if symbolic_backend and not verifier.HAVE_SYMPY:
        pytest.skip("선택 기호 의존성이 없다")
    monkeypatch.setattr(verifier, "HAVE_SYMPY", symbolic_backend)
    result = check_file(tmp_path, {"type": "numeric", "expr": expr})
    assert result["numeric"] in {"fail", "skipped"}
    assert result["details"][0].get("reason")


@pytest.mark.parametrize("kind", ["numeric", "inequality"])
@pytest.mark.parametrize("samples", [0, -1])
def test_empty_samples_cannot_verify_false_statement(tmp_path, kind, samples):
    result = check_file(tmp_path, {"type": kind, "lhs": "1", "rhs": "0", "samples": samples})
    assert result["numeric"] == "fail"


@pytest.mark.parametrize("tol", [float("inf"), float("nan"), -1.0])
def test_invalid_tolerance_cannot_verify_statement(tmp_path, tol):
    result = check_file(tmp_path, {"type": "numeric", "lhs": "1", "rhs": "0", "tol": tol})
    assert result["numeric"] == "fail"


def test_infinite_inequality_is_not_valid_numeric_evidence(tmp_path):
    result = check_file(tmp_path, {"type": "inequality", "lhs": "exp(1000)", "rhs": "exp(1000)"})
    assert result["numeric"] == "fail"


@pytest.mark.parametrize("declaration,relation", [("nonnegative integer", ">="), ("nonnegative nonzero integer", ">")])
def test_integer_samples_respect_declared_domain(tmp_path, declaration, relation):
    result = check_file(tmp_path, {"type": "inequality", "lhs": "n", "rhs": "0", "relation": relation}, {"n": declaration})
    assert result["numeric"] == "pass"
    assert result["details"][0]["violations"] == []


def test_finite_true_and_false_equations_still_distinguished(tmp_path):
    true = check_file(tmp_path, {"type": "numeric", "lhs": "(x+1)**2", "rhs": "x*x+2*x+1"}, {"x": "real"})
    false = check_file(tmp_path, {"type": "numeric", "lhs": "(x+1)**2", "rhs": "x*x+1"}, {"x": "real"})
    assert true["numeric"] == "pass"
    assert false["numeric"] == "fail"


def test_partial_verification_is_not_a_complete_pass(tmp_path):
    path = tmp_path/"attempt-01.derivation.md"
    front = {"symbols": {"x": "real"}, "verify": [
        {"type": "numeric", "expr": "0"},
        {"type": "numeric", "expr": "missing(x)"},
    ]}
    path.write_text("---\n"+yaml.safe_dump(front)+"---\n", encoding="utf-8")
    result = verifier.verify_file(path)
    assert [item["numeric"] for item in result["details"]] == ["pass", "skipped"]
    assert result["numeric"] == "skipped"
    assert verifier.aggregate(result["details"]+[{"numeric": "fail"}], "numeric") == "fail"


@pytest.mark.parametrize("kind,setting", [("numeric", "tol"), ("numeric", "samples"),
                                          ("identity", "tol"), ("identity", "samples"),
                                          ("inequality", "samples")])
def test_invalid_settings_replace_previous_pass_artifact(tmp_path, monkeypatch, capsys, kind, setting):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    path = tmp_path/"attempt-01.derivation.md"
    check = {"type": kind, "lhs": "0", "rhs": "0"}
    front = {"question": "Q-TEST-1", "attempt": 1, "verify": [check]}
    def write():
        path.write_text("---\n"+yaml.safe_dump(front)+"---\n", encoding="utf-8")
    write()
    assert verifier.main([str(path)]) == 0
    artifact = tmp_path/"verify/Q-TEST-1/attempt-01/hook_result.json"
    assert json.loads(artifact.read_text(encoding="utf-8"))["numeric"] == "pass"
    check[setting] = "typo"
    write()
    assert verifier.main([str(path)]) == 0
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["numeric"] == "fail"
    assert "설정 오류" in result["details"][0]["reason"]


@pytest.mark.parametrize("symbolic_backend", [False, True])
def test_complex_domain_never_silently_samples_only_reals(tmp_path, monkeypatch, symbolic_backend):
    if symbolic_backend and not verifier.HAVE_SYMPY:
        pytest.skip("선택 기호 의존성이 없다")
    monkeypatch.setattr(verifier, "HAVE_SYMPY", symbolic_backend)
    result = check_file(tmp_path, {"type": "identity", "lhs": "Abs(z)**2", "rhs": "z**2"}, {"z": "complex"})
    assert result["numeric"] == "skipped"
    assert "복소수" in result["details"][0]["reason"]
    assert result["symbolic"] == ("fail" if symbolic_backend else "skipped")


def test_save_hook_reports_incomplete_checks(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("ce_save_hook_review", SOURCE.with_name("verify_on_save.py"))
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    path = tmp_path/"derivations/Q-TEST-1/attempt-01.derivation.md"
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps({"tool_input": {"file_path": str(path)}})))
    monkeypatch.setattr(hook, "run_verifier", lambda *_: {
        "symbolic": "skipped", "numeric": "skipped", "details": [
            {"numeric": "pass"}, {"numeric": "skipped", "reason": "정의역 미지원"}]})
    assert hook.main() == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "미완료 검사 1개" in context
    assert "정의역 미지원" in context
