"""모듈 이동 뒤 끊어진 import를 실행 없이 찾아내는 회귀 검사."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "lib" / "examples_index.py"
spec = importlib.util.spec_from_file_location("examples_index_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("statement", [
    "from examples.physics.record.target import value",
    "import examples.physics.record.target as target",
])
@pytest.mark.parametrize("caller", ["examples/physics/causal/caller.py", "tests/test_caller.py"])
def test_deleted_import_target_is_reported(tmp_path, statement, caller):
    target = tmp_path / "examples" / "physics" / "record" / "target.py"
    target.parent.mkdir(parents=True)
    target.write_text('"""대상 모듈."""\nvalue = 1\n', encoding="utf-8")
    source = tmp_path / caller
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(statement + "\n", encoding="utf-8")
    assert module.local_import_violations(tmp_path) == []

    target.unlink()
    errors = module.local_import_violations(tmp_path)
    assert len(errors) == 1
    assert "examples.physics.record.target" in errors[0]
    assert f"{source.relative_to(tmp_path)}:1:" in errors[0]


def test_function_imports_and_unexecuted_strings_are_not_missing_modules(tmp_path):
    target = tmp_path / "examples" / "physics" / "record" / "target.py"
    target.parent.mkdir(parents=True)
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text(
        "from examples.physics.record.target import value\n"
        "from examples.physics import record\n"
        "text = 'from examples.physics.record.deleted import value'\n",
        encoding="utf-8",
    )
    assert module.local_import_violations(tmp_path) == []
