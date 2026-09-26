from __future__ import annotations

import ast
import re

from test_support.paths import (
    PAPER_ROOT,
    PHYSICS_ROOT,
    REPO_ROOT,
    RESEARCH_CONTRACT_PATH,
    TESTS_ROOT,
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
    "test_rendering_registry.py",
    "test_repository_harness.py",
    "test_time_homogeneous_pointer_qca.py",
    "test_zerod_plebanski_closure.py",
}


def test_shared_test_paths_are_semantic_and_location_independent() -> None:
    assert PAPER_ROOT == ROOT / "paper"
    assert TESTS_ROOT == ROOT / "tests"
    assert PHYSICS_ROOT == ROOT / "examples" / "physics"
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


def test_maintained_code_is_reachable_from_the_regression_suite() -> None:
    pending = list(TESTS_ROOT.glob("test_*.py"))
    reachable = set()
    while pending:
        path = pending.pop()
        if path in reachable:
            continue
        reachable.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + a.name for a in node.names]
            for name in names:
                dependency = ROOT.joinpath(*name.split(".")).with_suffix(".py")
                if dependency.is_file() and dependency not in reachable:
                    pending.append(dependency)

    implementations = {
        path for folder in ("examples", "experiments", "test_support")
        for path in (ROOT / folder).rglob("*.py") if path.name != "__init__.py"
    }
    assert implementations <= reachable, sorted(implementations - reachable)
    assert not any(PAPER_ROOT.rglob("*.py"))
    for retired in ("verify", "ledger", "derivations", "_workspace", "scripts", "artifacts"):
        assert not (ROOT / retired).exists(), f"retired code root restored: {retired}"


def test_active_instructions_do_not_restore_retired_harnesses() -> None:
    instructions = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert len(instructions.encode("utf-8")) <= 4096
    for retired in ("CE_RUN", "ce-research", "reality_stone", ".codex/hooks"):
        assert retired not in instructions


def test_research_charter_keeps_goals_and_validation_distinct() -> None:
    instructions = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    contract = RESEARCH_CONTRACT.read_text(encoding="utf-8")
    identifiers = (
        "OBJ-01", "OBJ-02", "OBJ-03", "OBJ-04",
        "VAL-01", "VAL-02",
    )

    for identifier in identifiers:
        assert instructions.count(identifier) == 1
        assert contract.count(identifier) == 1

    for heading in ("## 연구 목표", "## 검증 계약"):
        assert heading in instructions
    assert "목표는 달성 선언이 아니며, 후보 가정은 증명된 사실이 아니다" in contract
    assert "공동 RMSE" in instructions


def test_instruction_paths_exist() -> None:
    instructions = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    roots = ("paper/", "examples/", "experiments/", "tests/", "test_support/", "benchmarks/")
    paths = [
        token.rstrip("/")
        for token in re.findall(r"`([^`\s]+)`", instructions)
        if token.startswith(roots)
    ]

    assert paths
    missing = [path for path in paths if not any(ROOT.glob(path))]
    assert not missing, missing
