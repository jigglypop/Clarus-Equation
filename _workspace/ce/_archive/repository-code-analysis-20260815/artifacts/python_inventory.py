"""Reproducible static inventory for the repository-wide analysis run."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
INCLUDED = (
    "reality_stone",
    "clarus-agent-guard",
    "tests",
    "examples",
    "experiments",
    "benchmarks",
    "docs",
)
EXCLUDED_PARTS = {".tmp", "target", "__pycache__", ".pytest_cache", "raw"}


def python_files() -> list[Path]:
    result: list[Path] = []
    for name in INCLUDED:
        base = ROOT / name
        if not base.exists():
            continue
        result.extend(
            path
            for path in base.rglob("*.py")
            if not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
            and not any(part.startswith(".venv") for part in path.relative_to(ROOT).parts)
        )
    return sorted(set(result))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    files = python_files()
    totals = Counter()
    imports = Counter()
    large_modules: list[tuple[int, str]] = []
    large_functions: list[tuple[int, str, int, str]] = []
    syntax_errors: list[tuple[str, int | None, str]] = []
    risk_sites: list[tuple[str, int, str]] = []

    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        nonblank = sum(bool(line.strip()) for line in lines)
        large_modules.append((len(lines), rel(path)))
        totals["files"] += 1
        totals["lines"] += len(lines)
        totals["nonblank"] += nonblank
        totals["test_files"] += path.name.startswith("test_") or "tests" in path.parts
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append((rel(path), exc.lineno, exc.msg))
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                totals["functions"] += 1
                end = getattr(node, "end_lineno", node.lineno)
                span = end - node.lineno + 1
                large_functions.append((span, rel(path), node.lineno, node.name))
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        risk_sites.append((rel(path), node.lineno, "mutable-default"))
            elif isinstance(node, ast.ClassDef):
                totals["classes"] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name.split(".")[0]] += 1
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports[node.module.split(".")[0]] += 1
                if any(alias.name == "*" for alias in node.names):
                    risk_sites.append((rel(path), node.lineno, "wildcard-import"))
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    risk_sites.append((rel(path), node.lineno, "bare-except"))
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    risk_sites.append((rel(path), node.lineno, "except-Exception"))
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if isinstance(node.func, ast.Name) and name in {"eval", "exec"}:
                    risk_sites.append((rel(path), node.lineno, name))
                if name in {"run", "Popen", "call", "check_call", "check_output"}:
                    if any(
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    ):
                        risk_sites.append((rel(path), node.lineno, "subprocess-shell-true"))

    print("TOTALS", dict(sorted(totals.items())))
    print("SYNTAX_ERRORS", syntax_errors)
    print("TOP_IMPORT_ROOTS", imports.most_common(25))
    print("LARGEST_MODULES")
    for item in sorted(large_modules, reverse=True)[:30]:
        print(item)
    print("LARGEST_FUNCTIONS")
    for item in sorted(large_functions, reverse=True)[:35]:
        print(item)
    print("RISK_COUNTS", Counter(kind for _, _, kind in risk_sites))
    print("RISK_SITES_BY_KIND")
    for kind in sorted({item[2] for item in risk_sites}):
        print(kind)
        for item in (item for item in risk_sites if item[2] == kind):
            print(item)


if __name__ == "__main__":
    main()
