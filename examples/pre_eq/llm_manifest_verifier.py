"""Synthetic numeric check for the PreEq LLM manifest verifier.

This example does not call an LLM.  It fixes small candidate sets that mimic a
common inference-time situation: the model prior likes fluent but unsupported
answers, while evidence defects identify the manifest answer.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reality_stone.clarus.llm_pre_eq import (  # noqa: E402
    CandidateAnswer,
    LabeledCandidateSet,
    PreEqVerifier,
    PreEqVerifierConfig,
    evaluate_labeled_sets,
)


def benchmark_cases() -> tuple[LabeledCandidateSet, ...]:
    return (
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer(
                    "The system uses Redis for durable consensus.",
                    prior_weight=0.70,
                    supported_claims=1,
                    unsupported_claims=2,
                    contradicted_claims=1,
                ),
                CandidateAnswer(
                    "The system keeps consensus state in the local SQLite store.",
                    prior_weight=0.20,
                    supported_claims=3,
                ),
                CandidateAnswer(
                    "The document does not specify a storage layer.",
                    prior_weight=0.10,
                    supported_claims=1,
                    unsupported_claims=1,
                    uncertainty_flags=1,
                ),
            ),
            correct_index=1,
        ),
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer(
                    "The CE ratios exactly solve particle dark matter.",
                    prior_weight=0.62,
                    supported_claims=1,
                    contradicted_claims=1,
                    unsupported_claims=2,
                ),
                CandidateAnswer(
                    "The CE ratios are density boundary conditions; particle dark matter remains open.",
                    prior_weight=0.25,
                    supported_claims=4,
                ),
                CandidateAnswer(
                    "The CE ratio audit only covers baryons.",
                    prior_weight=0.13,
                    supported_claims=1,
                    unsupported_claims=1,
                ),
            ),
            correct_index=1,
        ),
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer(
                    "Riemann attention proves RH.",
                    prior_weight=0.55,
                    supported_claims=1,
                    contradicted_claims=1,
                    unsupported_claims=1,
                ),
                CandidateAnswer(
                    "Riemann zeros are used as an engineering axiom, not an RH proof.",
                    prior_weight=0.30,
                    supported_claims=3,
                ),
                CandidateAnswer(
                    "The repository has no Riemann-related code.",
                    prior_weight=0.15,
                    contradicted_claims=1,
                ),
            ),
            correct_index=1,
        ),
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer(
                    "The answer is definitely 42.",
                    prior_weight=0.50,
                    unsupported_claims=2,
                ),
                CandidateAnswer(
                    "The answer is definitely 43.",
                    prior_weight=0.30,
                    unsupported_claims=2,
                ),
                CandidateAnswer(
                    "The available evidence is insufficient to answer.",
                    prior_weight=0.20,
                    supported_claims=1,
                    uncertainty_flags=1,
                ),
            ),
            correct_index=2,
        ),
    )


def main() -> int:
    verifier = PreEqVerifier(
        PreEqVerifierConfig(beta=2.0, min_gap=0.4, max_energy=3.0, min_manifest_posterior=0.45)
    )
    cases = benchmark_cases()
    metrics = evaluate_labeled_sets(verifier, cases)

    print("# PreEq LLM Manifest Verifier synthetic check")
    print(f"total {metrics.total}")
    print(f"answered {metrics.answered}")
    print(f"abstained {metrics.abstained}")
    print(f"correct {metrics.correct}")
    print(f"answer_rate {metrics.answer_rate:.6f}")
    print(f"accuracy_on_answered {metrics.accuracy_on_answered:.6f}")
    print(f"exact_accuracy {metrics.exact_accuracy:.6f}")
    print(f"hallucination_rate_on_answered {metrics.hallucination_rate_on_answered:.6f}")
    print(f"baseline_accuracy {metrics.baseline_accuracy:.6f}")
    print(f"baseline_hallucination_rate {metrics.baseline_hallucination_rate:.6f}")
    print()
    for idx, case in enumerate(cases):
        decision = verifier.select(case.candidates)
        print(
            "case",
            idx,
            "selected",
            decision.selected_index,
            "correct",
            case.correct_index,
            "gap",
            f"{decision.energy_gap:.6f}",
            "confidence",
            f"{decision.confidence:.6f}",
            "reason",
            decision.reason,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
