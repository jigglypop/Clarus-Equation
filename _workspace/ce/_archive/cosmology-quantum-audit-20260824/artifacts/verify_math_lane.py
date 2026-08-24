"""Independent, stdlib-only checks for the cosmology--quantum math lane."""
from __future__ import annotations

import json
import math


ALPHA_S = 0.11789


def bisect_low_root(d: float) -> float:
    lo, hi = 0.0, 1.0 / d
    for _ in range(200):
        q = (lo + hi) / 2.0
        f = q - math.exp(-d * (1.0 - q))
        if f > 0.0:
            hi = q
        else:
            lo = q
    return (lo + hi) / 2.0


def main() -> None:
    sin2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2 * (1.0 - sin2)
    d = 3.0 + delta
    q = bisect_low_root(d)
    z = -d * math.exp(-d)
    q_lambert_principal = -float(math.nan)  # stdlib has no Lambert W
    # Newton solve W exp(W)=z on the principal branch, independently of q solve.
    w = -0.2
    for _ in range(100):
        ew = math.exp(w)
        w -= (w * ew - z) / (ew * (w + 1.0))
    q_lambert_principal = -w / d
    legacy_d = 3.17776
    q_legacy = bisect_low_root(legacy_d)
    r_lo = ALPHA_S * d
    omega_c = (1.0 - q) * r_lo / (1.0 + r_lo)
    omega_lambda = (1.0 - q) / (1.0 + r_lo)
    print(json.dumps({
        "alpha_s": ALPHA_S,
        "sin2": sin2,
        "delta": delta,
        "d_eff": d,
        "q_bisection": q,
        "q_lambert_w0": q_lambert_principal,
        "lambert_difference": q_lambert_principal - q,
        "fixed_point_residual": q - math.exp(-d * (1.0 - q)),
        "contraction_Dq": d * q,
        "identity_root": 1.0,
        "legacy_q": q_legacy,
        "legacy_minus_exact_q": q_legacy - q,
        "lo_ratio_alpha_s_D": r_lo,
        "lo_omega_b": q,
        "lo_omega_c": omega_c,
        "lo_omega_lambda": omega_lambda,
        "lo_partition_sum": q + omega_c + omega_lambda,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
