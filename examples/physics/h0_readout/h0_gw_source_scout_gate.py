"""Scout public gravitational-wave standard-siren sources for H0 readout."""

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
        name="GW170817 bright standard siren",
        role_in_readout="absolute GW luminosity distance with electromagnetic host-redshift anchor",
        source_url="https://www.nature.com/articles/s41550-019-0820-1",
        source_kind="published H0 posterior reference",
        next_ingest_target="event-level distance-inclination posterior and NGC 4993 velocity correction",
        status="bridge-reference",
    ),
    SourceTarget(
        name="GW170817 discovery and standard-siren provenance",
        role_in_readout="multi-messenger source provenance for bright siren role split",
        source_url="https://dcc.ligo.org/LIGO-P1700296/public",
        source_kind="LIGO/Virgo public document control center record",
        next_ingest_target="GW170817 posterior sample archive if available in a stable public package",
        status="provenance-reference",
    ),
    SourceTarget(
        name="O4a standard-siren extension",
        role_in_readout="dark/bright siren population bridge candidates",
        source_url="https://ligo.org/wp-content/uploads/2025/08/O4a_opendata.pdf",
        source_kind="LVK O4a public-data release reference",
        next_ingest_target="catalog event posteriors plus host/catalog redshift likelihoods",
        status="future-target",
    ),
]


def main() -> int:
    print("# H0 GW Source Scout Gate")
    print()
    print("| source | role in readout | source kind | next ingest target | status |")
    print("|---|---|---|---|---|")
    for target in TARGETS:
        print(
            f"| {target.name} | {target.role_in_readout} | {target.source_kind} | "
            f"{target.next_ingest_target} | {target.status} |"
        )

    required = {"bridge-reference", "provenance-reference", "future-target"}
    seen = {target.status for target in TARGETS}
    missing = sorted(required - seen)
    if missing:
        raise SystemExit(f"missing GW source scout statuses: {', '.join(missing)}")

    print()
    print("Verdict: GW standard-siren source targets are identified for bridge readout tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
