"""Adapt the public Planck PR3 parameter covariance into an H0 readout channel."""

from __future__ import annotations

from pathlib import Path

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


REAL_DATA = Path(__file__).with_name("h0_real_data")
PLANCK = REAL_DATA / "Planck_PR3" / "extract" / "base" / "plikHM_TTTEEE_lowl_lowE_lensing"
PARAMNAMES = PLANCK / "base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames"
COVMAT = PLANCK / "dist" / "base_plikHM_TTTEEE_lowl_lowE_lensing.covmat"
MARGESTATS = PLANCK / "dist" / "base_plikHM_TTTEEE_lowl_lowE_lensing.margestats"


def read_covmat(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("#"):
        raise ValueError("Planck covmat is missing a parameter header")
    names = lines[0].lstrip("#").split()
    matrix = [[float(value) for value in line.split()] for line in lines[1:] if line.strip()]
    if len(matrix) != len(names):
        raise ValueError("Planck covmat row count does not match parameter header")
    if any(len(row) != len(names) for row in matrix):
        raise ValueError("Planck covmat is not square")
    return names, matrix


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        labels[parts[0]] = parts[1] if len(parts) > 1 else parts[0]
    return labels


def read_margestat(path: Path, parameter: str) -> tuple[float, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == parameter:
            return float(parts[1]), float(parts[2])
    raise ValueError(f"missing Planck margestat parameter: {parameter}")


def branch_payload() -> dict[str, object]:
    names, covariance = read_covmat(COVMAT)
    labels = read_labels(PARAMNAMES)
    h0_obs, h0_sigma = read_margestat(MARGESTATS, "H0*")

    observable = "theta"
    if observable not in names:
        raise ValueError("Planck baseline covariance is missing theta observable")

    global_nodes = [name for name in names if name != observable]
    role_notes = {
        "primary_cosmology": [name for name in names[:6] if name != observable],
        "calibration_and_foregrounds": [name for name in names[6:] if name != observable],
        "observable_label": labels.get(observable, observable),
    }

    return {
        "name": "Planck PR3 TTTEEE+lowl+lowE+lensing covariance branch check",
        "nodes": names,
        "observable": observable,
        "local_nodes": [],
        "global_nodes": global_nodes,
        "matrix_type": "covariance",
        "matrix": covariance,
        "conductance_mode": "direct",
        "h0_obs": h0_obs,
        "h0_sigma": h0_sigma,
        "source": {
            "archive": "IRSA Planck PR3 ancillary data",
            "package": "COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip",
            "root": "base_plikHM_TTTEEE_lowl_lowE_lensing",
            "role_basis": "CMB acoustic-scale covariance: theta_MC constrained by early global horizon and nuisance closure parameters",
            "role_notes": role_notes,
        },
    }


def main() -> int:
    payload = branch_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)
    pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma

    print("# H0 CMB Planck Covariance Adapter Gate")
    print()
    print(f"channel = {channel.name}")
    print(f"covmat = {COVMAT.name}")
    print(f"parameters = {len(channel.nodes)}")
    print(f"observable = {channel.observable}")
    print(f"local_nodes = {len(channel.local_nodes)}")
    print(f"global_nodes = {len(channel.global_nodes)}")
    print(f"C_local = {result['c_local']:.8f}")
    print(f"C_global = {result['c_global']:.8f}")
    print(f"q_F = {result['q_f']:.8f}")
    print(f"H0_branch_pred = {result['h0_pred']:.6f} km/s/Mpc")
    print(f"Planck_H0 = {channel.h0_obs:.6f} +/- {channel.h0_sigma:.6f}")
    print(f"pull = {pull:+.3f}")
    print()

    if result["c_global"] <= 0.0:
        raise SystemExit("Planck CMB global conductance should be positive")
    if result["c_local"] != 0.0:
        raise SystemExit("Planck CMB covariance check should not have local conductance")
    if result["q_f"] != 0.0:
        raise SystemExit("Planck CMB covariance check should select the global endpoint")
    if abs(pull) > 1.0:
        raise SystemExit("Planck CMB branch prediction is not within 1 sigma of Planck H0")

    print("Verdict: Planck covariance selects the global/low-side H0 readout branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
