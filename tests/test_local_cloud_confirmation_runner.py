import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_RUNNER = Path(__file__).parents[1] / "examples" / "agi" / "local_cloud_confirmation_run.py"
_SPEC = importlib.util.spec_from_file_location("local_cloud_confirmation_run", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fixture(tmp_path):
    locked = tmp_path / "locked.py"
    locked.write_text("LOCKED = True\n", encoding="utf-8")
    development_registration = {
        "sha256": {"locked.py": _sha(locked)},
        "reserved_confirmation_seeds": [101, 102],
        "development_seeds": [1, 2],
        "burned_diagnostic_seeds": [3, 4],
        "config": {"train_episodes": 256},
        "bootstrap_samples": 5000,
        "bootstrap_seed": 77,
    }
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"registration": development_registration}), encoding="utf-8")
    confirmation = {
        "schema": "clarus.local_cloud.confirmation.v1",
        "development_result_path": "development.json",
        "development_result_sha256": _sha(development),
        "sha256": {"locked.py": _sha(locked)},
        "confirmation_seeds": [101, 102],
        "config": {"train_episodes": 256},
        "bootstrap_samples": 5000,
        "bootstrap_seed": 77,
    }
    return confirmation


def test_confirmation_requires_exact_predevelopment_seed_reservation(tmp_path) -> None:
    registration = _fixture(tmp_path)
    assert _MODULE._validate_registration(registration, tmp_path) == (101, 102)
    registration["confirmation_seeds"] = [101, 103]
    with pytest.raises(ValueError, match="reservation"):
        _MODULE._validate_registration(registration, tmp_path)


def test_confirmation_rejects_locked_file_mutation(tmp_path) -> None:
    registration = _fixture(tmp_path)
    (tmp_path / "locked.py").write_text("LOCKED = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _MODULE._validate_registration(registration, tmp_path)


def test_confirmation_rejects_config_or_bootstrap_change(tmp_path) -> None:
    registration = _fixture(tmp_path)
    registration["bootstrap_seed"] = 78
    with pytest.raises(ValueError, match="bootstrap seed"):
        _MODULE._validate_registration(registration, tmp_path)
