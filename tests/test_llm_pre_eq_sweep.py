from examples.pre_eq.llm_manifest_verifier_sweep import (
    run_sweep,
    synthetic_adversarial_cases,
    synthetic_noisy_cases,
)


def test_synthetic_adversarial_sweep_beats_prior_baseline() -> None:
    cases = synthetic_adversarial_cases(seed=1234, n_cases=200)
    best = run_sweep(cases)[0]

    assert best.baseline_accuracy < 0.05
    assert best.exact_accuracy > 0.90
    assert best.improvement > 0.85
    assert best.hallucination_rate_on_answered < 0.10


def test_synthetic_adversarial_cases_are_deterministic() -> None:
    first = synthetic_adversarial_cases(seed=99, n_cases=10)
    second = synthetic_adversarial_cases(seed=99, n_cases=10)

    assert first == second


def test_synthetic_noisy_sweep_still_improves_prior_baseline() -> None:
    cases = synthetic_noisy_cases(seed=2026, n_cases=300)
    best = run_sweep(cases)[0]

    assert best.exact_accuracy > best.baseline_accuracy + 0.50
    assert best.accuracy_on_answered > 0.60
    assert best.hallucination_rate_on_answered < best.baseline_hallucination_rate
