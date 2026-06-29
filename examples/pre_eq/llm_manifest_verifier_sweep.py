"""Adversarial synthetic sweep for the PreEq LLM manifest verifier.

The benchmark stresses a specific LLM failure mode: fluent high-prior answers
are often unsupported or contradicted, while a lower-prior candidate is grounded.
It is not a SOTA claim; it is a reproducible gate before real QA benchmarks.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SweepResult:
    beta: float
    min_gap: float
    max_energy: float
    min_manifest_posterior: float
    exact_accuracy: float
    answer_rate: float
    accuracy_on_answered: float
    hallucination_rate_on_answered: float
    baseline_accuracy: float
    baseline_hallucination_rate: float
    improvement: float


def synthetic_adversarial_cases(seed: int, n_cases: int) -> tuple[LabeledCandidateSet, ...]:
    """Generate deterministic prior-trap cases with one grounded candidate."""
    rng = random.Random(seed)
    cases: list[LabeledCandidateSet] = []
    for case_idx in range(n_cases):
        n_candidates = rng.randint(3, 6)
        correct_index = rng.randrange(1, n_candidates)
        high_prior_wrong = 0
        candidates: list[CandidateAnswer] = []
        for idx in range(n_candidates):
            if idx == correct_index:
                candidates.append(
                    CandidateAnswer(
                        text=f"case {case_idx} grounded answer {idx}",
                        prior_weight=rng.uniform(0.08, 0.28),
                        supported_claims=rng.randint(3, 6),
                        unsupported_claims=0,
                        contradicted_claims=0,
                    )
                )
            elif idx == high_prior_wrong:
                candidates.append(
                    CandidateAnswer(
                        text=f"case {case_idx} fluent hallucination {idx}",
                        prior_weight=rng.uniform(0.45, 0.85),
                        supported_claims=rng.randint(0, 2),
                        unsupported_claims=rng.randint(1, 3),
                        contradicted_claims=rng.randint(0, 1),
                    )
                )
            else:
                candidates.append(
                    CandidateAnswer(
                        text=f"case {case_idx} weak distractor {idx}",
                        prior_weight=rng.uniform(0.03, 0.22),
                        supported_claims=rng.randint(0, 2),
                        unsupported_claims=rng.randint(1, 4),
                        contradicted_claims=rng.randint(0, 1),
                        uncertainty_flags=rng.randint(0, 1),
                    )
                )
        cases.append(LabeledCandidateSet(candidates=tuple(candidates), correct_index=correct_index))
    return tuple(cases)


def synthetic_noisy_cases(seed: int, n_cases: int) -> tuple[LabeledCandidateSet, ...]:
    """Generate harder cases where correct and wrong defect distributions overlap."""
    rng = random.Random(seed)
    cases: list[LabeledCandidateSet] = []
    for case_idx in range(n_cases):
        n_candidates = rng.randint(4, 7)
        correct_index = rng.randrange(n_candidates)
        prior_trap = (correct_index + rng.randrange(1, n_candidates)) % n_candidates
        candidates: list[CandidateAnswer] = []
        for idx in range(n_candidates):
            if idx == correct_index:
                candidates.append(
                    CandidateAnswer(
                        text=f"noisy case {case_idx} grounded answer {idx}",
                        prior_weight=rng.uniform(0.08, 0.35),
                        supported_claims=rng.randint(2, 5),
                        unsupported_claims=rng.randint(0, 1),
                        contradicted_claims=0,
                        uncertainty_flags=rng.randint(0, 1),
                    )
                )
            elif idx == prior_trap:
                candidates.append(
                    CandidateAnswer(
                        text=f"noisy case {case_idx} high-prior distractor {idx}",
                        prior_weight=rng.uniform(0.35, 0.75),
                        supported_claims=rng.randint(1, 4),
                        unsupported_claims=rng.randint(1, 3),
                        contradicted_claims=1 if rng.random() < 0.35 else 0,
                        uncertainty_flags=rng.randint(0, 1),
                    )
                )
            else:
                candidates.append(
                    CandidateAnswer(
                        text=f"noisy case {case_idx} mixed distractor {idx}",
                        prior_weight=rng.uniform(0.04, 0.35),
                        supported_claims=rng.randint(0, 4),
                        unsupported_claims=rng.randint(1, 4),
                        contradicted_claims=1 if rng.random() < 0.45 else 0,
                        instruction_violations=1 if rng.random() < 0.10 else 0,
                        self_contradictions=1 if rng.random() < 0.10 else 0,
                        uncertainty_flags=rng.randint(0, 2),
                    )
                )
        cases.append(LabeledCandidateSet(candidates=tuple(candidates), correct_index=correct_index))
    return tuple(cases)


def config_grid() -> tuple[PreEqVerifierConfig, ...]:
    configs: list[PreEqVerifierConfig] = []
    for beta in (0.75, 1.5, 2.0, 3.0):
        for min_gap in (0.0, 0.25, 0.5, 0.75):
            for max_energy in (1.5, 2.5, 3.5):
                for min_posterior in (0.35, 0.45, 0.60):
                    configs.append(
                        PreEqVerifierConfig(
                            beta=beta,
                            min_gap=min_gap,
                            max_energy=max_energy,
                            min_manifest_posterior=min_posterior,
                        )
                    )
    return tuple(configs)


def run_sweep(cases: tuple[LabeledCandidateSet, ...]) -> tuple[SweepResult, ...]:
    results: list[SweepResult] = []
    for config in config_grid():
        metrics = evaluate_labeled_sets(PreEqVerifier(config), cases)
        improvement = metrics.exact_accuracy - metrics.baseline_accuracy
        results.append(
            SweepResult(
                beta=config.beta,
                min_gap=config.min_gap,
                max_energy=config.max_energy,
                min_manifest_posterior=config.min_manifest_posterior,
                exact_accuracy=metrics.exact_accuracy,
                answer_rate=metrics.answer_rate,
                accuracy_on_answered=metrics.accuracy_on_answered,
                hallucination_rate_on_answered=metrics.hallucination_rate_on_answered,
                baseline_accuracy=metrics.baseline_accuracy,
                baseline_hallucination_rate=metrics.baseline_hallucination_rate,
                improvement=improvement,
            )
        )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.exact_accuracy,
                item.accuracy_on_answered,
                -item.hallucination_rate_on_answered,
                item.answer_rate,
            ),
            reverse=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", choices=["adversarial", "noisy"], default="adversarial")
    args = parser.parse_args()

    if args.mode == "adversarial":
        cases = synthetic_adversarial_cases(seed=args.seed, n_cases=args.cases)
    else:
        cases = synthetic_noisy_cases(seed=args.seed, n_cases=args.cases)
    results = run_sweep(cases)
    best = results[0]

    print("# PreEq LLM Manifest Verifier sweep")
    print(f"mode {args.mode}")
    print(f"seed {args.seed}")
    print(f"cases {args.cases}")
    print(f"configs {len(results)}")
    print(f"best_beta {best.beta:.6f}")
    print(f"best_min_gap {best.min_gap:.6f}")
    print(f"best_max_energy {best.max_energy:.6f}")
    print(f"best_min_manifest_posterior {best.min_manifest_posterior:.6f}")
    print(f"best_exact_accuracy {best.exact_accuracy:.6f}")
    print(f"best_answer_rate {best.answer_rate:.6f}")
    print(f"best_accuracy_on_answered {best.accuracy_on_answered:.6f}")
    print(f"best_hallucination_rate_on_answered {best.hallucination_rate_on_answered:.6f}")
    print(f"baseline_accuracy {best.baseline_accuracy:.6f}")
    print(f"baseline_hallucination_rate {best.baseline_hallucination_rate:.6f}")
    print(f"absolute_accuracy_gain {best.improvement:.6f}")
    print()
    print("rank,beta,min_gap,max_energy,min_posterior,exact_acc,answer_rate,acc_answered,halluc_answered,gain")
    for rank, result in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank},{result.beta:.3f},{result.min_gap:.3f},{result.max_energy:.3f},"
            f"{result.min_manifest_posterior:.3f},{result.exact_accuracy:.6f},"
            f"{result.answer_rate:.6f},{result.accuracy_on_answered:.6f},"
            f"{result.hallucination_rate_on_answered:.6f},{result.improvement:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
