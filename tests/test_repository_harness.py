from __future__ import annotations

import tomllib

from test_support.paths import (
    CODEX_ROOT,
    PAPER_ROOT,
    Q0020_ROOT,
    REPO_ROOT,
    RESEARCH_CONTRACT_PATH,
    TESTS_ROOT,
    VERIFY_ROOT,
)

ROOT = REPO_ROOT
RESEARCH_CONTRACT = RESEARCH_CONTRACT_PATH

ESSENTIAL_TESTS = {
    "test_canonical_document_policy.py",
    "test_ce_residual_forward_model.py",
    "test_contextual_obstruction.py",
    "test_cosmology.py",
    "test_dimension_joint_candidate.py",
    "test_finite_ctp_diagonal_source_obstruction.py",
    "test_holdout_preregistration.py",
    "test_kinetic_dark_sector_gate.py",
    "test_regge_tent_transfer.py",
    "test_repository_harness.py",
    "test_time_homogeneous_pointer_qca.py",
    "test_zerod_plebanski_closure.py",
}


def _toml(relative_path: str) -> dict:
    with (CODEX_ROOT / relative_path).open("rb") as stream:
        return tomllib.load(stream)


def test_shared_test_paths_are_semantic_and_location_independent() -> None:
    assert PAPER_ROOT == ROOT / "paper"
    assert TESTS_ROOT == ROOT / "tests"
    assert VERIFY_ROOT == ROOT / "verify"
    assert Q0020_ROOT == VERIFY_ROOT / "Q-0020"
    assert CODEX_ROOT == ROOT / ".codex"
    assert RESEARCH_CONTRACT == PAPER_ROOT / "검증_원장" / "연구_목표_계약.md"

    legacy_expression = "Path(__file__).resolve()." + "parents[1]"
    active_tests = TESTS_ROOT.glob("test_*.py")
    assert all(
        legacy_expression not in path.read_text(encoding="utf-8")
        for path in active_tests
    )


def test_only_the_essential_regression_suite_is_active() -> None:
    active_tests = {path.name for path in TESTS_ROOT.glob("test_*.py")}

    assert active_tests == ESSENTIAL_TESTS
    assert not any(TESTS_ROOT.rglob("legacy_*.py"))


def test_root_and_delegated_models_are_explicit() -> None:
    config = _toml("config.toml")

    assert config["model"] == "gpt-6-astra"
    assert config["model_reasoning_effort"] == "medium"
    assert config["agents"]["default_subagent_model"] == "gpt-5.6-sol"
    assert config["agents"]["default_subagent_reasoning_effort"] == "low"

    worker = _toml("agents/worker.toml")
    assert worker["model"] == "gpt-5.6-sol"
    assert worker["model_reasoning_effort"] == "low"


def test_codex_surface_stays_compact() -> None:
    files = {
        path.relative_to(CODEX_ROOT).as_posix()
        for path in CODEX_ROOT.rglob("*")
        if path.is_file()
    }

    assert files == {"README.md", "agents/worker.toml", "config.toml"}


def test_active_instructions_do_not_restore_retired_harnesses() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    codex_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(CODEX_ROOT.rglob("*"))
        if path.is_file()
    )

    assert len(instructions.encode("utf-8")) <= 4096
    for retired in ("CE_RUN", "ce-research", "reality_stone", ".codex/hooks"):
        assert retired not in instructions
        assert retired not in codex_docs


def test_research_charter_keeps_goals_validation_and_assumptions_distinct() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    contract = RESEARCH_CONTRACT.read_text(encoding="utf-8")
    identifiers = (
        "OBJ-01", "OBJ-02", "OBJ-03", "OBJ-04",
        "VAL-01", "VAL-02",
        "ASM-01", "ASM-02", "ASM-03",
    )

    for identifier in identifiers:
        assert instructions.count(identifier) == 1
        assert contract.count(identifier) == 1

    for heading in ("## 연구 목표", "## 검증 계약", "## 후보 가정", "## 반례 분기"):
        assert heading in instructions
    assert "목표는 달성 선언이 아니며, 후보 가정은 증명된 사실이 아니다" in contract
    assert "공동 RMSE" in instructions
    assert "피팅으로 봉합하지 않는다" in instructions


def test_astra_owns_scientific_branch_judgment() -> None:
    codex_readme = (CODEX_ROOT / "README.md").read_text(encoding="utf-8")
    worker = _toml("agents/worker.toml")

    assert "## Astra 판단 루프" in codex_readme
    assert "연구_목표_계약.md" in codex_readme
    assert "가정 채택·기각" in codex_readme
    assert "물리 주장 지위 변경은 Astra 부모에게 반환" in worker["developer_instructions"]
