"""Scout public Planck/CMB sources for the H0 readout extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceTarget:
    name: str
    role_in_readout: str
    source_url: str
    source_kind: str
    next_ingest_target: str
    status: str


TARGETS = [
    SourceTarget(
        name="Planck 2018 cosmological parameter chains",
        role_in_readout="parameter covariance for early global horizon closure",
        source_url="https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters",
        source_kind="Planck Legacy Archive parameter-chain index",
        next_ingest_target="base_plikHM_TTTEEE_lowl_lowE_lensing chain covariance",
        status="source-target",
    ),
    SourceTarget(
        name="Planck 2018 likelihood paper",
        role_in_readout="likelihood provenance for TT/TE/EE and low-ell closure roles",
        source_url="https://arxiv.org/abs/1907.12875",
        source_kind="primary likelihood-method paper",
        next_ingest_target="map high-ell TTTEEE, lowl, lowE, and lensing terms into source roles",
        status="provenance-reference",
    ),
    SourceTarget(
        name="Planck 2018 cosmological-parameters paper",
        role_in_readout="published low-H0 CMB branch reference under base-LambdaCDM",
        source_url="https://arxiv.org/abs/1807.06209",
        source_kind="primary parameter-results paper",
        next_ingest_target="compare source-role branch prediction to reported base-LambdaCDM H0 family",
        status="result-reference",
    ),
]


def main() -> int:
    print("# H0 CMB Source Scout Gate")
    print()
    print("| source | role in readout | source kind | next ingest target | status |")
    print("|---|---|---|---|---|")

    for target in TARGETS:
        print(
            f"| {target.name} | {target.role_in_readout} | {target.source_kind} | "
            f"{target.next_ingest_target} | {target.status} |"
        )

    required_statuses = {"source-target", "provenance-reference", "result-reference"}
    seen = {target.status for target in TARGETS}
    missing = sorted(required_statuses - seen)
    if missing:
        raise SystemExit(f"missing CMB source scout statuses: {', '.join(missing)}")

    print()
    print("Verdict: CMB source targets are identified for Planck covariance/likelihood ingestion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
