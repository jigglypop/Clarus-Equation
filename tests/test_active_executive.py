from reality_stone.clarus.active_executive_benchmark import evaluate_active_executive
from reality_stone.clarus.executive_control import ActiveExecutiveController
from reality_stone.clarus.executive_switch_benchmark import ExecutiveBenchConfig


def test_active_choice_preserves_simplex() -> None:
    controller = ActiveExecutiveController()
    features = (0, 1, 2)
    action = controller.choose(features)
    controller.update(features, action, False)
    assert 0 <= action < 4
    assert controller.simplex_valid()


def test_small_active_benchmark_has_integrity_guards() -> None:
    result = evaluate_active_executive(ExecutiveBenchConfig(trials=72, seeds=4))
    assert result["schema"] == "clarus.active-executive.validation.v1"
    assert result["future_reads"] == 0
    assert result["id"]["simplex_valid"]
