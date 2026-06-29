"""Synthetic check for the CE Claim Residual Verifier.

This example is a local numeric harness, not a SOTA benchmark.  It verifies the
v2 path from dimensionless claim evidence residuals to answer actions and the
PreEq Gibbs posterior.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from clarus.llm_pre_eq import (  # noqa: E402
    ClaimAxisEvidence,
    ClaimResidualVerifier,
    ClaimResidualVerifierConfig,
    ResidualAnswerCandidate,
    ResidualClaim,
)


def axis(name: str, value: float, reference: float, family: str) -> ClaimAxisEvidence:
    return ClaimAxisEvidence(
        axis=name,
        value=value,
        reference=reference,
        sigma=0.5,
        source_reliability=1.0,
        source_family=family,
    )


def candidates() -> tuple[ResidualAnswerCandidate, ...]:
    return (
        ResidualAnswerCandidate(
            "CE exactly proves RH and closes the Millennium Problem.",
            prior_weight=0.70,
            claims=(
                ResidualClaim(
                    "CE exactly proves RH.",
                    (
                        axis("support", 1.0, 0.0, "docs"),
                        axis("support", 1.0, 0.0, "math-audit"),
                    ),
                ),
            ),
            required_slots=1,
            covered_slots=1,
            tier_penalty=1.0,
        ),
        ResidualAnswerCandidate(
            "CE uses Riemann structure as an engineering axiom; RH remains open.",
            prior_weight=0.20,
            claims=(
                ResidualClaim(
                    "CE uses Riemann structure as an engineering axiom; RH remains open.",
                    (
                        axis("support", 1.0, 1.0, "docs"),
                        axis("support", 1.0, 1.0, "math-audit"),
                    ),
                ),
            ),
            required_slots=1,
            covered_slots=1,
        ),
        ResidualAnswerCandidate(
            "The evidence is insufficient to make a claim.",
            prior_weight=0.10,
            claims=(
                ResidualClaim(
                    "The evidence is insufficient to make a claim.",
                    (
                        axis("support", 0.6, 1.0, "docs"),
                        axis("support", 0.6, 1.0, "math-audit"),
                    ),
                ),
            ),
            required_slots=1,
            covered_slots=1,
        ),
    )


def main() -> int:
    verifier = ClaimResidualVerifier(
        ClaimResidualVerifierConfig(beta=2.0, min_gap=0.0, min_manifest_posterior=0.1)
    )
    sample = candidates()
    decision = verifier.select(sample)
    states = verifier.answer_states(sample)

    print("# CE Claim Residual Verifier synthetic check")
    print(f"backend {decision.backend}")
    print(f"selected {decision.selected_index}")
    print(f"confidence {decision.confidence:.6f}")
    print(f"reason {decision.reason}")
    print("idx,action,posterior,accepted_fraction")
    for idx, state in enumerate(states):
        print(
            f"{idx},{decision.actions[idx]:.6f},{decision.posterior[idx]:.6f},"
            f"{state.accepted_fraction:.6f}"
        )
    print("accepted_claims")
    for claim in decision.accepted_claims:
        print(f"- {claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
