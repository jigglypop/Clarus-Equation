from examples.pre_eq.claim_residual_verifier_sweep import run_sweep, synthetic_cases


def test_claim_residual_sweep_cases_are_deterministic() -> None:
    first = synthetic_cases("noisy", seed=123, n_cases=10)
    second = synthetic_cases("noisy", seed=123, n_cases=10)

    assert first == second


def test_claim_residual_sweep_exposes_recoverable_noisy_mode() -> None:
    best = run_sweep("noisy", seed=1234, n_cases=80)[0]

    assert best.baseline_accuracy < 0.05
    assert best.exact_accuracy > 0.85
    assert best.hallucination_rate_on_answered < 0.15


def test_claim_residual_sweep_covers_partial_and_source_modes() -> None:
    partial = run_sweep("partial", seed=4321, n_cases=80)[0]
    source = run_sweep("source", seed=4321, n_cases=80)[0]

    assert partial.exact_accuracy > 0.85
    assert source.exact_accuracy > 0.85
