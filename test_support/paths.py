"""Stable semantic paths used by active repository tests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_REPOSITORY_ENTRIES = (
    (REPO_ROOT / "pyproject.toml", "file"),
    (REPO_ROOT / "tests", "directory"),
    (REPO_ROOT / "verify", "directory"),
)

for _path, _kind in _REQUIRED_REPOSITORY_ENTRIES:
    _exists = _path.is_file() if _kind == "file" else _path.is_dir()
    if not _exists:
        raise RuntimeError(
            f"test_support.paths resolved an invalid repository root: "
            f"expected {_kind} at {_path}"
        )

CODEX_ROOT = REPO_ROOT / ".codex"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
PAPER_ROOT = REPO_ROOT / "paper"
TESTS_ROOT = REPO_ROOT / "tests"
VERIFY_ROOT = REPO_ROOT / "verify"

PREREGISTRATION_ROOT = EXPERIMENTS_ROOT / "preregistration"
Q0020_ROOT = VERIFY_ROOT / "Q-0020"
RESEARCH_CONTRACT_PATH = PAPER_ROOT / "검증_원장" / "연구_목표_계약.md"


__all__ = (
    "CODEX_ROOT",
    "EXPERIMENTS_ROOT",
    "PAPER_ROOT",
    "PREREGISTRATION_ROOT",
    "Q0020_ROOT",
    "REPO_ROOT",
    "RESEARCH_CONTRACT_PATH",
    "TESTS_ROOT",
    "VERIFY_ROOT",
)
