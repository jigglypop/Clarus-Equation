from pathlib import Path
import pytest
from reality_stone.clarus.real_robot_log import load_robot_log

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "real_robot_log_v1.json"


def test_public_log_integrity_when_present() -> None:
    import json
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = ROOT / config["source"]["relative_path"]
    if not path.exists():
        pytest.skip("public log is intentionally not committed; see manifest")
    features, labels = load_robot_log(path, config["source"])
    assert features.shape == (5456, 24)
    assert len(set(labels)) == 4
