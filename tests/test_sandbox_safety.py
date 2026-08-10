from pathlib import Path
from reality_stone.clarus.sandbox_safety import _advance, run_sandbox_safety_gate, State

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sandbox_safety_v1.json"


def test_actuation_sign_flip_reverses_acceleration() -> None:
    normal = _advance(State(), 1.0, 1.0, False, 0.05)
    flipped = _advance(State(), 1.0, 1.0, True, 0.05)
    assert normal.velocity == -flipped.velocity


def test_sandbox_is_offline_and_action_bounded() -> None:
    report = run_sandbox_safety_gate(CONFIG)
    assert report["resource_usage"]["external_download_bytes"] == 0
    assert report["metrics"]["max_abs_command"] <= 1.0
