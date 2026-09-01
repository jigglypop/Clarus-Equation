from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PAPER_ROOT = "paper"
RETIRED_DOCUMENT_ROOT = "do" + "cs"
ACTIVE_TEXT_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path(".codex"),
    Path(".claude"),
    Path("paper"),
    Path("examples"),
    Path("experiments"),
    Path("tests"),
    Path("benchmarks"),
)
ACTIVE_PYTHON_PATHS = (Path("examples"), Path("experiments"), Path("tests"))
RETIRED_RUNTIME_MODULES = ("reality_stone", "clarus")
TEXT_SUFFIXES = frozenset(
    {
        ".cmd",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".rs",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
IGNORED_PARTS = frozenset({".git", ".tmp", ".venv", "__pycache__", "target"})
REQUIRED_PATHS = (
    Path("paper/README.md"),
    Path(".codex/README.md"),
    Path(".codex/config.toml"),
    Path(".codex/hooks/python.cmd"),
    Path("tests/test_canonical_document_policy.py"),
    Path(".codex/skills/ce-explanation-planner/SKILL.md"),
    Path(".codex/skills/ce-explanation-planner/agents/openai.yaml"),
    Path(".codex/agents/ce-explanation-planner.md"),
    Path(".codex/agents/ce-explanation-planner.toml"),
    Path(".codex/prompts/ce-explain-plan.md"),
    Path(".codex/harnesses/explanation_first_planner.md"),
)
REQUIRED_AGENT_REFERENCES = (
    "paper/README.md",
    ".codex/README.md",
    ".codex/hooks/python.cmd",
)
PLANNER_MARKERS = ("LaTeX", "비유", "[정리]", "[공리]", "증명 경로")
PLANNER_ALIGNMENT_PATHS = (
    Path(".codex/skills/ce-explanation-planner/SKILL.md"),
    Path(".codex/agents/ce-explanation-planner.md"),
    Path(".codex/agents/ce-explanation-planner.toml"),
    Path(".codex/prompts/ce-explain-plan.md"),
    Path(".codex/harnesses/explanation_first_planner.md"),
)
PLANNER_ALIGNMENT_MARKERS = ("목표 계약", "완료 조건", "목표 이탈", "복귀 행동")


def iter_active_text_files(
    root: Path, relative_paths: Iterable[Path] = ACTIVE_TEXT_PATHS
) -> tuple[Path, ...]:
    files: set[Path] = set()
    for relative in relative_paths:
        candidate = root / relative
        if candidate.is_file():
            if candidate.suffix.casefold() in TEXT_SUFFIXES:
                files.add(candidate)
            continue
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            if (
                path.is_file()
                and path.suffix.casefold() in TEXT_SUFFIXES
                and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
            ):
                files.add(path)
    return tuple(sorted(files))


def find_retired_path_references(
    root: Path, relative_paths: Iterable[Path] = ACTIVE_TEXT_PATHS
) -> list[str]:
    patterns = (
        f"{RETIRED_DOCUMENT_ROOT}/",
        f"{RETIRED_DOCUMENT_ROOT}{chr(92)}",
    )
    violations: list[str] = []
    for path in iter_active_text_files(root, relative_paths):
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in patterns):
                violations.append(
                    f"{path.relative_to(root)}:{line_number}: retired path reference"
                )
    return violations


def find_retired_runtime_imports(
    root: Path, relative_paths: Iterable[Path] = ACTIVE_PYTHON_PATHS
) -> list[str]:
    """Reject imports from the package removed during the repository split."""

    violations: list[str] = []
    for relative in relative_paths:
        candidate = root / relative
        if candidate.is_file():
            paths: Iterable[Path] = (candidate,)
        elif candidate.is_dir():
            paths = candidate.rglob("*.py")
        else:
            paths = ()
        for path in paths:
            if IGNORED_PARTS.intersection(path.relative_to(root).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                modules: tuple[str, ...] = ()
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = (node.module,)
                if any(
                    module == retired or module.startswith(f"{retired}.")
                    for module in modules
                    for retired in RETIRED_RUNTIME_MODULES
                ):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: retired runtime import"
                    )
    return sorted(violations)


def _instruction_budget(root: Path) -> tuple[int | None, int]:
    config = (root / ".codex" / "config.toml").read_text(encoding="utf-8-sig")
    match = re.search(r"(?m)^project_doc_max_bytes\s*=\s*(\d+)\s*$", config)
    maximum = int(match.group(1)) if match else None
    total = sum(
        len((root / relative).read_bytes())
        for relative in (Path("AGENTS.md"), Path(".codex/AGENTS.md"))
        if (root / relative).is_file()
    )
    return maximum, total


def check_repository(root: Path = REPO_ROOT) -> list[str]:
    root = root.resolve()
    violations: list[str] = []

    paper_root = root / CANONICAL_PAPER_ROOT
    retired_root = root / RETIRED_DOCUMENT_ROOT
    if not paper_root.is_dir():
        violations.append(f"missing canonical paper root: {CANONICAL_PAPER_ROOT}/")
    if retired_root.exists():
        violations.append(f"retired document root still exists: {RETIRED_DOCUMENT_ROOT}/")

    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            violations.append(f"missing harness entrypoint: {relative.as_posix()}")

    planner = root / ".codex" / "skills" / "ce-explanation-planner" / "SKILL.md"
    if planner.is_file():
        planner_text = planner.read_text(encoding="utf-8-sig")
        for marker in PLANNER_MARKERS:
            if marker not in planner_text:
                violations.append(
                    f"explanation planner missing invariant marker: {marker}"
                )

    for relative in PLANNER_ALIGNMENT_PATHS:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        for marker in PLANNER_ALIGNMENT_MARKERS:
            if marker not in text:
                violations.append(
                    f"{relative.as_posix()} missing goal-alignment marker: {marker}"
                )

    violations.extend(find_retired_path_references(root))
    violations.extend(find_retired_runtime_imports(root))

    agents_path = root / "AGENTS.md"
    if agents_path.is_file():
        agents = agents_path.read_text(encoding="utf-8-sig")
        for reference in REQUIRED_AGENT_REFERENCES:
            if reference not in agents:
                violations.append(f"AGENTS.md does not map required entrypoint: {reference}")

    maximum, total = _instruction_budget(root)
    if maximum is None:
        violations.append(".codex/config.toml has no project_doc_max_bytes")
    elif total > maximum:
        violations.append(
            f"AGENTS instruction chain exceeds budget: {total} > {maximum} bytes"
        )

    return sorted(violations)


def main() -> int:
    violations = check_repository()
    maximum, instruction_bytes = _instruction_budget(REPO_ROOT)
    report = {
        "active_text_files": len(iter_active_text_files(REPO_ROOT)),
        "canonical_root": f"{CANONICAL_PAPER_ROOT}/",
        "instruction_bytes": instruction_bytes,
        "instruction_limit": maximum,
        "status": "FAIL" if violations else "PASS",
    }
    if violations:
        report["violations"] = violations
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
