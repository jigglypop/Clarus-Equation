"""Generate a falsifiable prediction ledger for the H0 source-role paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    target: str
    source_role_test: str
    expected_family: str
    decisive_check: str
    falsifier: str
    status: str


PREDICTIONS = [
    Prediction(
        "GW event-level posterior samples",
        "separate luminosity-distance information from host-redshift anchoring",
        "bridge/intermediate",
        "q_F remains between the endpoint families after source roles fixed",
        "event-level covariance collapses to a stable global/low or local/high endpoint",
        "future data-facing",
    ),
    Prediction(
        "TRGB/JAGB/CCHP local ladders",
        "map calibrator, anchor, and Hubble-flow labels before H0 comparison",
        "local/high or semi-local high",
        "local endpoint conductance dominates unless a population closure is added",
        "source-aware ladder map selects the global/low family without an added closure",
        "future data-facing",
    ),
    Prediction(
        "BAO+SN inverse-distance-ladder covariance",
        "test whether the joint covariance is still standard-ruler closure dominated",
        "global/low",
        "global closure conductance dominates even when SN covariance is attached",
        "joint BAO+SN covariance selects the local/high endpoint before refit",
        "future data-facing",
    ),
    Prediction(
        "CMB covariance variants",
        "compare lensing and non-lensing Planck-like covariance adapters",
        "global/low",
        "q_F remains near the global endpoint across acoustic-scale variants",
        "CMB source-role covariance becomes bridge or local/high under the same map",
        "current cross-check",
    ),
    Prediction(
        "TDCOSMO alternative notebooks and chains",
        "rebuild likelihood-factor roles from sampler composition rather than labels",
        "role-transition split",
        "SLACS population closure lowers q_F relative to time-delay-only chains",
        "static role maps explain the rows as well as the source-aware map",
        "current cross-check",
    ),
]


def main() -> int:
    print("# H0 Paper Prediction Ledger Gate")
    print()
    print("| target | source-role test | expected family | decisive check | falsifier | status |")
    print("|---|---|---|---|---|---|")
    for row in PREDICTIONS:
        print(
            f"| {row.target} | {row.source_role_test} | {row.expected_family} | "
            f"{row.decisive_check} | {row.falsifier} | {row.status} |"
        )

    statuses = {row.status for row in PREDICTIONS}
    families = {row.expected_family for row in PREDICTIONS}
    if "future data-facing" not in statuses:
        raise SystemExit("prediction ledger needs at least one future data-facing test")
    if "current cross-check" not in statuses:
        raise SystemExit("prediction ledger needs at least one current cross-check")
    if not all(row.falsifier.strip() for row in PREDICTIONS):
        raise SystemExit("prediction ledger needs explicit falsifiers")
    if not {"global/low", "bridge/intermediate"} <= families:
        raise SystemExit("prediction ledger should cover endpoint and bridge predictions")
    if len(PREDICTIONS) < 5:
        raise SystemExit("prediction ledger should contain at least five tests")

    print()
    print("Verdict: prediction ledger states future tests, expected families, and falsifiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
