from reality_stone.clarus.executive_control import ExecutiveRuleController
from reality_stone.clarus.executive_switch_benchmark import (
    ExecutiveBenchConfig,
    evaluate_executive_switch,
)


def test_surprising_feedback_releases_rule_belief() -> None:
    controller = ExecutiveRuleController()
    features = (0, 1, 2)
    for _ in range(5):
        action = controller.choose(features)
        controller.update(features, action, True)
    action = controller.choose(features)
    update = controller.update(features, action, False)
    assert update.switched_attention
    assert controller.simplex_valid()


def test_small_executive_benchmark_has_integrity_guards() -> None:
    result = evaluate_executive_switch(ExecutiveBenchConfig(trials=72, seeds=4))
    assert result["schema"] == "clarus.executive-switch.validation.v1"
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
    assert result["id"]["simplex_valid"]
