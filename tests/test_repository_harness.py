from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / ".codex" / "hooks" / "repository_harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("ce_repository_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_repository_satisfies_harness_contract() -> None:
    harness = _load_harness()
    assert harness.check_repository(ROOT) == []


def test_retired_document_path_is_detected(tmp_path: Path) -> None:
    harness = _load_harness()
    retired_reference = ("do" + "cs") + "/old-entry.md"
    (tmp_path / "README.md").write_text(retired_reference, encoding="utf-8")

    violations = harness.find_retired_path_references(
        tmp_path, relative_paths=(Path("README.md"),)
    )

    assert len(violations) == 1
    assert violations[0].startswith("README.md:1:")


def test_external_document_urls_do_not_hide_local_retired_paths(tmp_path: Path) -> None:
    harness = _load_harness()
    retired = "do" + "cs"
    external = f"https://example.org/{retired}/guide"
    content = (
        f"[공식 문서]({external})\n"
        f"<{external}>\n"
        f"{external} {retired}/old.md\n"
        f"[외부]({external})[내부]({retired}/old.md)\n"
        f"`{external}` `{retired}\\old.md`\n"
    )
    (tmp_path / "README.md").write_text(content, encoding="utf-8")

    violations = harness.find_retired_path_references(
        tmp_path, relative_paths=(Path("README.md"),)
    )

    assert len(violations) == 3
    assert [entry.split(":")[1] for entry in violations] == ["3", "4", "5"]


def test_retired_runtime_import_is_detected(tmp_path: Path) -> None:
    harness = _load_harness()
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_legacy.py").write_text(
        "from reality_stone.clarus import runtime\nfrom clarus import pre_eq\n",
        encoding="utf-8",
    )

    violations = harness.find_retired_runtime_imports(tmp_path)

    assert len(violations) == 2
    assert violations[0].replace("\\", "/").startswith("tests/test_legacy.py:1:")
    assert violations[1].endswith(":2: retired runtime import")


def _load_python_harness():
    source = ROOT / ".codex" / "hooks" / "python_harness.py"
    spec = importlib.util.spec_from_file_location("ce_python_harness_scope_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_source_scan_reaches_research_drivers_and_claude_hooks(tmp_path, monkeypatch):
    harness = _load_python_harness()
    monkeypatch.setattr(harness, "REPO_ROOT", tmp_path)
    for name in (".codex", ".claude", "scripts", "verify", "tests", "examples", "experiments", "paper"):
        (tmp_path / name).mkdir()
    for name in (".claude", "scripts", "verify"):
        invalid = tmp_path / name / "invalid.py"
        invalid.write_text("def unfinished(\n", encoding="utf-8")
        try:
            try:
                harness._run_source([])
            except SyntaxError as error:
                assert Path(error.filename) == invalid
            else:
                raise AssertionError(f"구문 오류가 검사에서 빠졌습니다: {name}")
        finally:
            invalid.unlink()


def test_explicit_source_scan_stays_focused(tmp_path, monkeypatch):
    harness = _load_python_harness()
    monkeypatch.setattr(harness, "REPO_ROOT", tmp_path)
    (tmp_path / "valid.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "verify").mkdir()
    (tmp_path / "verify" / "invalid.py").write_text("def unfinished(\n", encoding="utf-8")
    assert harness._run_source(["valid.py"]) == 0


def test_child_pipe_preserves_korean_and_math_symbols(monkeypatch):
    harness = _load_python_harness()
    monkeypatch.setenv("PYTHONIOENCODING", "cp949")
    message = "한국어 검증 ✓"
    child = harness.subprocess.run(
        [harness.sys.executable, "-B", "-c", f"print({message!r})"],
        env=harness._child_environment(), capture_output=True, check=True,
    )
    assert child.stdout.decode("utf-8").strip() == message
