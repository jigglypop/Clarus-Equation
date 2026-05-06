"""Run an H0 readout bridge check for GW170817-like standard sirens."""

from __future__ import annotations

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


def branch_payload() -> dict[str, object]:
    return {
        "name": "GW170817 bright standard-siren bridge branch check",
        "nodes": ["gw_strain_observable", "gw_luminosity_distance", "host_redshift_anchor"],
        "observable": "gw_strain_observable",
        "local_nodes": ["gw_luminosity_distance"],
        "global_nodes": ["host_redshift_anchor"],
        "matrix_type": "fisher",
        "matrix": [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.0],
            [0.2, 0.0, 1.0],
        ],
        "conductance_mode": "direct",
        "h0_obs": 70.3,
        "h0_sigma": 5.15,
        "source": {
            "event": "GW170817",
            "role_basis": "GW amplitude gives an absolute distance; host/counterpart information supplies redshift anchoring",
            "published_reference": "https://www.nature.com/articles/s41550-019-0820-1",
            "provenance_reference": "https://dcc.ligo.org/LIGO-P1700296/public",
        },
    }


def branch_label(q_f: float) -> str:
    if q_f >= 0.75:
        return "local/high"
    if q_f <= 0.25:
        return "global/low"
    return "bridge"


def main() -> int:
    payload = branch_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)
    pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma
    label = branch_label(result["q_f"])

    print("# H0 GW Standard-Siren Bridge Gate")
    print()
    print(f"channel = {channel.name}")
    print(f"nodes = {len(channel.nodes)}")
    print(f"local_nodes = {len(channel.local_nodes)}")
    print(f"global_nodes = {len(channel.global_nodes)}")
    print(f"C_local = {result['c_local']:.8f}")
    print(f"C_global = {result['c_global']:.8f}")
    print(f"q_F = {result['q_f']:.8f}")
    print(f"classified_readout = {label}")
    print(f"H0_branch_pred = {result['h0_pred']:.6f} km/s/Mpc")
    print(f"GW170817_H0 = {channel.h0_obs:.6f} +/- {channel.h0_sigma:.6f}")
    print(f"pull = {pull:+.3f}")
    print()

    if label != "bridge":
        raise SystemExit("GW standard-siren readout should be a bridge branch")
    if abs(result["q_f"] - 0.5) > 1e-12:
        raise SystemExit("GW bridge selector should be balanced at q_F=0.5 for this source-role check")
    if abs(pull) > 1.0:
        raise SystemExit("GW bridge prediction is not within 1 sigma of the reference H0")

    print("Verdict: GW170817-like standard sirens select the bridge/intermediate H0 readout branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
