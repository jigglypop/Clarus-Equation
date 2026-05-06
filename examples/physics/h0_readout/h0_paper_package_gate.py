"""Assemble the H0 readout result into a paper-ready package."""

from __future__ import annotations

from h0_cross_channel_branch_contrast_gate import rows as endpoint_rows
from h0_three_family_readout_table_gate import branch_label, rows as three_family_rows


FIGURES = [
    {
        "id": "Figure 1",
        "title": "Endpoint source-role split",
        "claim": "Hubble-tension channels split into local/high and global/low branches before a joint H0 refit.",
        "required_families": {"local", "global"},
    },
    {
        "id": "Figure 2",
        "title": "Three-family readout law",
        "claim": "The same readout law admits global/low, bridge/intermediate, and local/high source-role families.",
        "required_families": {"local", "global", "bridge"},
    },
]


LIMITATIONS = [
    "Full joint BAO/SN/TDCOSMO posterior refit remains future work.",
    "GW bridge gate uses a source-role covariance abstraction, not event-level posterior samples yet.",
    "CMB covariance gate uses Planck PR3 parameter covariance, not a fresh Planck likelihood optimization.",
]


def endpoint_family_counts() -> dict[str, int]:
    counts = {"local": 0, "global": 0}
    for item in endpoint_rows():
        expected = str(item["expected"])
        if expected in counts:
            counts[expected] += 1
    return counts


def three_family_counts() -> dict[str, int]:
    counts = {"global/low": 0, "bridge/intermediate": 0, "local/high": 0}
    for item in three_family_rows():
        counts[branch_label(float(item["q_f"]))] += 1
    return counts


def main() -> int:
    endpoint_counts = endpoint_family_counts()
    family_counts = three_family_counts()
    failed = 0

    print("# H0 Paper Package Gate")
    print()
    print("## Paper-ready figures")
    print()
    print("| figure | title | central claim | package status |")
    print("|---|---|---|---|")

    figure_status = {
        "Figure 1": "PASS" if min(endpoint_counts.values()) >= 2 else "FAIL",
        "Figure 2": "PASS" if min(family_counts.values()) >= 1 else "FAIL",
    }
    for figure in FIGURES:
        status = figure_status[figure["id"]]
        if status != "PASS":
            failed += 1
        print(f"| {figure['id']} | {figure['title']} | {figure['claim']} | {status} |")

    print()
    print("## Figure support counts")
    print()
    print(f"endpoint local rows = {endpoint_counts['local']}")
    print(f"endpoint global rows = {endpoint_counts['global']}")
    print(f"three-family global/low rows = {family_counts['global/low']}")
    print(f"three-family bridge/intermediate rows = {family_counts['bridge/intermediate']}")
    print(f"three-family local/high rows = {family_counts['local/high']}")

    print()
    print("## Required limitations")
    print()
    for item in LIMITATIONS:
        print(f"- {item}")

    if not LIMITATIONS:
        failed += 1
    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: paper package has reproducible figures and explicit limitations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
