"""Generate a paper-ready numeric H0 readout results table."""

from __future__ import annotations

from h0_bao_global_readout_gate import branch_payload as bao_payload
from h0_cmb_planck_covariance_adapter_gate import branch_payload as cmb_payload
from h0_cross_channel_branch_contrast_gate import TDCOSMO_FILES, tdcosmo_payload
from h0_fisher_matrix_io_gate import channel_from_payload, run_channel
from h0_gw_standard_siren_bridge_gate import branch_payload as gw_payload
from h0_pantheon_shoes_local_readout_gate import branch_payload as shoes_payload
from h0_three_family_readout_table_gate import branch_label


def channel_rows() -> list[tuple[str, str, str, dict[str, object]]]:
    rows: list[tuple[str, str, str, dict[str, object]]] = []
    for label, file_name, expected in TDCOSMO_FILES:
        rows.append((label, "time-delay lensing", expected, tdcosmo_payload(file_name)))
    rows.extend(
        [
            ("DESI BAO", "standard ruler", "global", bao_payload()),
            ("Planck CMB", "early acoustic horizon", "global", cmb_payload()),
            ("GW170817 bright siren", "standard siren", "bridge", gw_payload()),
            ("Pantheon+SH0ES", "distance ladder", "local", shoes_payload()),
        ]
    )
    return rows


def main() -> int:
    print("# H0 Paper Numeric Results Gate")
    print()
    print("| channel | family | source role | q_F | readout | H0 readout | H0 reference | pull |")
    print("|---|---|---|---:|---|---:|---:|---:|")

    failed = 0
    scored = 0
    for label, family, expected, payload in sorted(channel_rows(), key=lambda row: row[0]):
        channel = channel_from_payload(payload)
        result = run_channel(channel)
        q_f = float(result["q_f"])
        readout = branch_label(q_f)
        if channel.h0_obs is None or channel.h0_sigma is None:
            reference = "--"
            pull_text = "--"
        else:
            pull = (float(result["h0_pred"]) - channel.h0_obs) / channel.h0_sigma
            reference = f"{channel.h0_obs:.3f} +/- {channel.h0_sigma:.3f}"
            pull_text = f"{pull:+.3f}"
            scored += 1
            if abs(pull) > 1.5:
                failed += 1
        print(
            f"| {label} | {family} | {expected} | {q_f:.6f} | {readout} | "
            f"{float(result['h0_pred']):.6f} | {reference} | {pull_text} |"
        )

    print()
    print(f"rows = {len(channel_rows())}")
    print(f"scored rows = {scored}")

    if len(channel_rows()) != 8:
        failed += 1
    if scored < 6:
        failed += 1
    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: numeric H0 readout table is reproducible and scoped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
