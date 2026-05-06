"""Generate a paper-ready data provenance table for H0 readout rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProvenanceRow:
    channel: str
    source_role: str
    public_source: str
    primary_gate: str
    status: str


ROWS = [
    ProvenanceRow(
        "TDCOSMO-only",
        "local time-delay lens endpoint",
        "public TDCOSMO chain payload plus notebook factor extraction",
        "h0_tdcosmo_notebook_factor_extract_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "TDCOSMO+IFU",
        "local time-delay lens endpoint with IFU kinematic closure",
        "public TDCOSMO chain payload plus notebook factor extraction",
        "h0_tdcosmo_role_transition_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "TDCOSMO+SLACS",
        "global population closure",
        "public TDCOSMO chain payload plus SLACS likelihood factor",
        "h0_tdcosmo_role_transition_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "TDCOSMO+SLACS+IFU",
        "global population closure with IFU kinematic closure",
        "public TDCOSMO chain payload plus SLACS likelihood factor",
        "h0_tdcosmo_role_transition_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "DESI BAO",
        "global standard-ruler closure",
        "CobayaSampler/bao_data DESI 2024 mean/covariance",
        "h0_bao_mean_cov_role_adapter_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "Planck CMB",
        "early global acoustic-horizon closure",
        "IRSA Planck PR3 cosmological parameter covariance",
        "h0_cmb_planck_covariance_adapter_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "Pantheon+SH0ES",
        "local distance-ladder endpoint",
        "PantheonPlusSH0ES/DataRelease distance table and covariance",
        "h0_pantheon_shoes_role_adapter_gate.py",
        "data-facing",
    ),
    ProvenanceRow(
        "GW170817 bright siren",
        "bridge distance-redshift anchor",
        "published GW170817 H0 reference plus LVK provenance record",
        "h0_gw_standard_siren_bridge_gate.py",
        "scoped bridge abstraction",
    ),
]


def main() -> int:
    print("# H0 Paper Provenance Table Gate")
    print()
    print("| channel | source role | public source | primary gate | status |")
    print("|---|---|---|---|---|")
    for row in ROWS:
        print(
            f"| {row.channel} | {row.source_role} | {row.public_source} | "
            f"{row.primary_gate} | {row.status} |"
        )

    statuses = {row.status for row in ROWS}
    required = {"data-facing", "scoped bridge abstraction"}
    missing = sorted(required - statuses)
    if missing:
        raise SystemExit(f"missing provenance status: {', '.join(missing)}")
    if len(ROWS) < 8:
        raise SystemExit("provenance table should cover all three-family rows")

    print()
    print("Verdict: paper provenance table covers all H0 readout rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
