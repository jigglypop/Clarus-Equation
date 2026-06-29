"""Diagnostic sweep for the CE Claim Residual Verifier.

This is a synthetic failure-mode harness.  It is not a SOTA benchmark; its job
is to expose where the v2 residual verifier loses to priors, partial claims,
single-source support, graph incoherence, or missing evidence.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from clarus.llm_pre_eq import (  # noqa: E402
    ClaimActionWeights,
    ClaimAxisEvidence,
    ClaimGraphEdge,
    ClaimResidualVerifier,
    ClaimResidualVerifierConfig,
    ResidualAnswerCandidate,
    ResidualClaim,
)


@dataclass(frozen=True)
class ResidualSweepResult:
    mode: str
    beta: float
    residual_weight: float
    accept_score: float
    max_action: float
    max_residual_norm: float
    exact_accuracy: float
    answer_rate: float
    accuracy_on_answered: float
    hallucination_rate_on_answered: float
    baseline_accuracy: float
    improvement: float


def axis(
    name: str,
    value: float,
    reference: float,
    family: str,
    *,
    sigma: float = 0.5,
    reliability: float = 1.0,
) -> ClaimAxisEvidence:
    return ClaimAxisEvidence(
        axis=name,
        value=value,
        reference=reference,
        sigma=sigma,
        source_reliability=reliability,
        source_family=family,
    )


def one_claim_candidate(
    text: str,
    prior: float,
    value: float,
    reference: float,
    *,
    sources: int = 2,
    tier_penalty: float = 0.0,
    required_slots: int = 1,
    covered_slots: int = 1,
) -> ResidualAnswerCandidate:
    families = tuple(chr(ord("a") + idx) for idx in range(sources))
    claim_axes = tuple(axis("support", value, reference, family) for family in families)
    return ResidualAnswerCandidate(
        text=text,
        prior_weight=prior,
        claims=(ResidualClaim(text, claim_axes),),
        required_slots=required_slots,
        covered_slots=covered_slots,
        tier_penalty=tier_penalty,
    )


def graph_candidate(
    text: str,
    prior: float,
    second_value: float,
    *,
    tier_penalty: float = 0.0,
) -> ResidualAnswerCandidate:
    return ResidualAnswerCandidate(
        text=text,
        prior_weight=prior,
        claims=(
            ResidualClaim(
                f"{text} claim A",
                (
                    axis("truth", 1.0, 1.0, "a"),
                    axis("truth", 1.0, 1.0, "b"),
                ),
            ),
            ResidualClaim(
                f"{text} claim B",
                (
                    axis("truth", second_value, 1.0, "a"),
                    axis("truth", second_value, 1.0, "b"),
                ),
            ),
        ),
        graph_edges=(ClaimGraphEdge(0, 1, relation=1),),
        required_slots=2,
        covered_slots=2,
        tier_penalty=tier_penalty,
    )


def synthetic_cases(
    mode: str,
    *,
    seed: int,
    n_cases: int,
) -> tuple[tuple[ResidualAnswerCandidate, ...], ...]:
    rng = random.Random(seed)
    cases: list[tuple[ResidualAnswerCandidate, ...]] = []
    for _ in range(n_cases):
        if mode == "adversarial":
            cases.append(
                (
                    one_claim_candidate(
                        "fluent false",
                        rng.uniform(0.55, 0.85),
                        1.0,
                        0.0,
                        tier_penalty=0.5,
                    ),
                    one_claim_candidate("grounded true", rng.uniform(0.08, 0.25), 1.0, 1.0),
                    one_claim_candidate(
                        "weak partial",
                        rng.uniform(0.05, 0.20),
                        rng.uniform(0.45, 0.75),
                        1.0,
                    ),
                )
            )
        elif mode == "noisy":
            cases.append(
                (
                    one_claim_candidate(
                        "fluent noisy false",
                        rng.uniform(0.45, 0.80),
                        rng.uniform(0.65, 1.0),
                        0.0 if rng.random() < 0.7 else 0.4,
                        tier_penalty=0.2,
                    ),
                    one_claim_candidate(
                        "grounded noisy true",
                        rng.uniform(0.08, 0.35),
                        rng.uniform(0.85, 1.0),
                        1.0,
                    ),
                    one_claim_candidate(
                        "single-source weak",
                        rng.uniform(0.05, 0.30),
                        rng.uniform(0.35, 0.75),
                        1.0,
                        sources=1,
                    ),
                )
            )
        elif mode == "partial":
            cases.append(
                (
                    one_claim_candidate(
                        "high-prior partial",
                        rng.uniform(0.45, 0.80),
                        rng.uniform(0.55, 0.78),
                        1.0,
                        covered_slots=1,
                        required_slots=2,
                    ),
                    one_claim_candidate(
                        "complete answer",
                        rng.uniform(0.08, 0.35),
                        rng.uniform(0.95, 1.0),
                        1.0,
                        covered_slots=2,
                        required_slots=2,
                    ),
                    one_claim_candidate(
                        "unsupported distractor",
                        rng.uniform(0.05, 0.25),
                        rng.uniform(0.1, 0.5),
                        1.0,
                        sources=1,
                    ),
                )
            )
        elif mode == "source":
            cases.append(
                (
                    one_claim_candidate(
                        "single-source high-prior",
                        rng.uniform(0.45, 0.80),
                        rng.uniform(0.90, 1.0),
                        1.0,
                        sources=1,
                    ),
                    one_claim_candidate(
                        "independent-source answer",
                        rng.uniform(0.08, 0.35),
                        rng.uniform(0.90, 1.0),
                        1.0,
                        sources=2,
                    ),
                    one_claim_candidate(
                        "bad source answer",
                        rng.uniform(0.05, 0.25),
                        rng.uniform(0.45, 0.75),
                        1.0,
                        sources=1,
                    ),
                )
            )
        elif mode == "graph":
            cases.append(
                (
                    graph_candidate(
                        "graph-incoherent high-prior",
                        rng.uniform(0.45, 0.80),
                        rng.uniform(0.0, 0.4),
                        tier_penalty=0.2,
                    ),
                    graph_candidate(
                        "graph-coherent answer",
                        rng.uniform(0.08, 0.35),
                        rng.uniform(0.90, 1.0),
                    ),
                    one_claim_candidate(
                        "single claim escape",
                        rng.uniform(0.05, 0.25),
                        rng.uniform(0.75, 1.0),
                        1.0,
                        required_slots=2,
                        covered_slots=1,
                    ),
                )
            )
        elif mode == "missing":
            cases.append(
                (
                    ResidualAnswerCandidate(
                        "empty high-prior answer",
                        claims=(),
                        prior_weight=rng.uniform(0.45, 0.80),
                        required_slots=1,
                    ),
                    one_claim_candidate(
                        "evidence-backed answer",
                        rng.uniform(0.08, 0.35),
                        rng.uniform(0.95, 1.0),
                        1.0,
                    ),
                    one_claim_candidate(
                        "weak answer",
                        rng.uniform(0.05, 0.25),
                        rng.uniform(0.45, 0.75),
                        1.0,
                        sources=1,
                    ),
                )
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
    return tuple(cases)


def config_grid() -> tuple[ClaimResidualVerifierConfig, ...]:
    configs: list[ClaimResidualVerifierConfig] = []
    for beta in (2.0, 4.0, 6.0):
        for residual_weight in (0.10, 0.20, 0.30):
            for accept_score in (0.35, 0.70, 0.85):
                for max_action in (1.0, 1.5, 3.0):
                    for max_residual_norm in (0.5, 1.0, 2.5):
                        configs.append(
                            ClaimResidualVerifierConfig(
                                beta=beta,
                                accept_score=accept_score,
                                max_action=max_action,
                                max_residual_norm=max_residual_norm,
                                min_manifest_posterior=0.1,
                                weights=ClaimActionWeights(residual=residual_weight),
                            )
                        )
    return tuple(configs)


def score_config(
    mode: str,
    config: ClaimResidualVerifierConfig,
    cases: tuple[tuple[ResidualAnswerCandidate, ...], ...],
) -> ResidualSweepResult:
    verifier = ClaimResidualVerifier(config)
    answered = 0
    correct = 0
    hallucinated = 0
    baseline_correct = 0
    for case in cases:
        if max(range(len(case)), key=lambda idx: case[idx].prior_weight) == 1:
            baseline_correct += 1
        decision = verifier.select(case)
        if decision.abstained:
            continue
        answered += 1
        if decision.selected_index == 1:
            correct += 1
        else:
            hallucinated += 1
    total = len(cases)
    answer_rate = answered / total if total else 0.0
    exact_accuracy = correct / total if total else 0.0
    accuracy_on_answered = correct / answered if answered else 0.0
    hallucination_rate = hallucinated / answered if answered else 0.0
    baseline_accuracy = baseline_correct / total if total else 0.0
    return ResidualSweepResult(
        mode=mode,
        beta=config.beta,
        residual_weight=config.weights.residual,
        accept_score=config.accept_score,
        max_action=config.max_action,
        max_residual_norm=config.max_residual_norm,
        exact_accuracy=exact_accuracy,
        answer_rate=answer_rate,
        accuracy_on_answered=accuracy_on_answered,
        hallucination_rate_on_answered=hallucination_rate,
        baseline_accuracy=baseline_accuracy,
        improvement=exact_accuracy - baseline_accuracy,
    )


def run_sweep(
    mode: str,
    *,
    seed: int,
    n_cases: int,
) -> tuple[ResidualSweepResult, ...]:
    cases = synthetic_cases(mode, seed=seed, n_cases=n_cases)
    results = [score_config(mode, config, cases) for config in config_grid()]
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
    parser.add_argument(
        "--mode",
        choices=("adversarial", "noisy", "partial", "source", "graph", "missing", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    modes = (
        ("adversarial", "noisy", "partial", "source", "graph", "missing")
        if args.mode == "all"
        else (args.mode,)
    )
    for mode in modes:
        results = run_sweep(mode, seed=args.seed, n_cases=args.cases)
        best = results[0]
        print("# CE Claim Residual Verifier sweep")
        print(f"mode {mode}")
        print(f"seed {args.seed}")
        print(f"cases {args.cases}")
        print(f"configs {len(results)}")
        print(f"best_beta {best.beta:.6f}")
        print(f"best_residual_weight {best.residual_weight:.6f}")
        print(f"best_accept_score {best.accept_score:.6f}")
        print(f"best_max_action {best.max_action:.6f}")
        print(f"best_max_residual_norm {best.max_residual_norm:.6f}")
        print(f"best_exact_accuracy {best.exact_accuracy:.6f}")
        print(f"best_answer_rate {best.answer_rate:.6f}")
        print(f"best_accuracy_on_answered {best.accuracy_on_answered:.6f}")
        print(f"best_hallucination_rate_on_answered {best.hallucination_rate_on_answered:.6f}")
        print(f"baseline_accuracy {best.baseline_accuracy:.6f}")
        print(f"absolute_accuracy_gain {best.improvement:.6f}")
        print(
            "rank,beta,residual_weight,accept_score,max_action,max_residual_norm,"
            "exact_acc,answer_rate,acc_answered,halluc_answered,gain"
        )
        for rank, result in enumerate(results[: args.top_k], start=1):
            print(
                f"{rank},{result.beta:.3f},{result.residual_weight:.3f},"
                f"{result.accept_score:.3f},{result.max_action:.3f},"
                f"{result.max_residual_norm:.3f},{result.exact_accuracy:.6f},"
                f"{result.answer_rate:.6f},{result.accuracy_on_answered:.6f},"
                f"{result.hallucination_rate_on_answered:.6f},{result.improvement:.6f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
