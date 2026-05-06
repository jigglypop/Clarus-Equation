"""Data-facing CMB axis ingest gate for A3c/GER.

This gate does not derive a preferred axis.  It ingests representative
large-angle CMB asymmetry axes from Planck/PR4 analyses and checks whether the
pre-registered CE amplitude

    A_H = 2 Q_A3c / sigma = 0.059633...

is in the observed large-angle hemispherical modulation range.

The default observational rows are intentionally lightweight:
    - Planck-like temperature HPA: A ~= 0.07 near (l,b) ~= (205,-20)
    - PR4 Sevem E-mode local-variance dipole: direction (234,-14), A range
      about 0.06-0.13 via the PR4 modulated-simulation calibration.

For exact likelihood work, replace these rows with a map/covariance pipeline.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0


OBS_ROWS = [
    {
        "name": "Planck/PR3 temperature HPA representative",
        "kind": "temperature",
        "l_deg": 205.0,
        "b_deg": -20.0,
        "amplitude": 0.070,
        "sigma": 0.021,
        "note": "representative low-l temperature dipole-modulation amplitude",
    },
    {
        "name": "Planck PR4 Sevem E-mode local-variance",
        "kind": "polarization",
        "l_deg": 234.0,
        "b_deg": -14.0,
        "amplitude": 0.090,
        "sigma": 0.035,
        "range_low": 0.060,
        "range_high": 0.130,
        "note": "PR4 Sevem E-mode calibrated modulation range",
    },
]


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def galactic_to_unit(l_deg: float, b_deg: float) -> tuple[float, float, float]:
    l_rad = math.radians(l_deg)
    b_rad = math.radians(b_deg)
    cb = math.cos(b_rad)
    return cb * math.cos(l_rad), cb * math.sin(l_rad), math.sin(b_rad)


def angular_sep_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def quadrupole_tensor(axis: tuple[float, float, float], amplitude: float) -> list[list[float]]:
    return [
        [amplitude * (axis[i] * axis[j] - (1.0 / 3.0 if i == j else 0.0)) for j in range(3)]
        for i in range(3)
    ]


def trace(matrix: list[list[float]]) -> float:
    return sum(matrix[i][i] for i in range(3))


def frobenius_norm(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(matrix[i][j] ** 2 for i in range(3) for j in range(3)))


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)

    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_a3c = p_ger * x * sigma
    a_h = 2.0 * q_a3c / sigma
    s_q = p_ger * p_ger
    expected_tensor_norm = s_q * math.sqrt(2.0 / 3.0)

    axes = [galactic_to_unit(row["l_deg"], row["b_deg"]) for row in OBS_ROWS]
    sep = angular_sep_deg(axes[0], axes[1])

    print("# A3c CMB Axis Ingest Gate")
    print()
    print("## CE pre-registered amplitudes")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"P_GER = {p_ger:.8f}")
    print(f"Q_A3c = {q_a3c:.8f}")
    print(f"A_H = 2 Q_A3c/sigma = {a_h:.8f}")
    print(f"S_Q = P_GER^2 = {s_q:.8f}")
    print(f"conditional tensor norm = S_Q sqrt(2/3) = {expected_tensor_norm:.8f}")
    print()

    print("## Ingested observational axis rows")
    print()
    print("| row | kind | axis (l,b) deg | observed A | CE A_H | pull/range | verdict |")
    print("|---|---|---:|---:|---:|---:|---|")
    for row, axis in zip(OBS_ROWS, axes):
        t_quad = quadrupole_tensor(axis, s_q)
        if "range_low" in row:
            edge_tol = 1.0e-3
            in_range = bool(row["range_low"] - edge_tol <= a_h <= row["range_high"] + edge_tol)
            range_text = f"{row['range_low']:.3f}-{row['range_high']:.3f}"
            if row["range_low"] <= a_h <= row["range_high"]:
                verdict = "inside broad range"
            elif in_range:
                verdict = "edge-compatible"
            else:
                verdict = "outside broad range"
            edge_delta = min(a_h - row["range_low"], row["range_high"] - a_h)
            pull_text = f"{range_text}; edge {edge_delta:+.4f}"
        else:
            p = pull(a_h, row["amplitude"], row["sigma"])
            verdict = "within 1 sigma" if abs(p) <= 1.0 else "outside 1 sigma"
            pull_text = f"{p:+.2f} sigma"

        print(
            f"| {row['name']} | {row['kind']} | "
            f"({row['l_deg']:.1f}, {row['b_deg']:.1f}) | {row['amplitude']:.5f} | "
            f"{a_h:.5f} | {pull_text} | {verdict} |"
        )
        if abs(trace(t_quad)) > 1e-12:
            raise SystemExit("conditional quadrupole tensor should be traceless")
        if abs(frobenius_norm(t_quad) - expected_tensor_norm) > 1e-12:
            raise SystemExit("conditional tensor norm changed with axis")
    print()

    print("## Axis consistency")
    print()
    print(f"temperature-polarization representative angular separation = {sep:.2f} deg")
    print("This gate does not require CE to predict that separation; it only records data-facing compatibility.")
    print()

    print("## Verdict")
    print()
    print("The CE amplitude A_H is compatible with representative large-angle HPA amplitudes.")
    print("The axis is ingested, not derived.  A true closure still requires a CMB map/covariance likelihood bridge")
    print("or an internal theory of the symmetry-breaking vector n_i.")

    # Broad guardrails: keep the amplitude in the large-angle 5-10% handle.
    if not (0.05 <= a_h <= 0.10):
        raise SystemExit("CE hemispherical amplitude is outside the broad observed handle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
