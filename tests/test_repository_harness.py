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
