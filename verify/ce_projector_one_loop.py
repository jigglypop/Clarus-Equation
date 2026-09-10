"""CE-GN4: relative quadratic response of a fixed-rank mass projector.

Complex scalar, flat Euclidean R^4, positive squared masses.
No data fitting; this checks conditional mathematics, not physical unification.
Run: python -B verify/ce_projector_one_loop.py
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def coefficient(a: float, b: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 60
        aa, bb = Decimal(str(a)), Decimal(str(b))
        if aa == bb:
            return 0.0
        value = (aa + bb) / 2 - aa * bb / (bb - aa) * (bb / aa).ln()
        return float(value) / (16 * np.pi**2)


def parameter(a: float, b: float, t: float, n: int) -> tuple[float, float, float]:
    x, w = rule(n)
    ratio = x * (1 - x) / (x * a + (1 - x) * b)
    factor = (b - a)**2 / (16 * np.pi**2)
    kernel = factor * np.dot(w, np.log1p(t * ratio))
    c = factor * np.dot(w, ratio)
    remainder_bound = factor * t**2 / 2 * np.dot(w, ratio**2)
    return float(kernel), float(c), float(remainder_bound)


def radial(a: float, b: float, t: float, n: int) -> float:
    # Angular integral in R^4 performed exactly; rationalization avoids cancellation.
    x, w = rule(n)
    r = np.sqrt(b) * x / (1 - x)
    jac = np.sqrt(b) / (1 - x)**2
    rr = r * r
    root = np.sqrt((rr + b - t)**2 + 4 * b * t)
    bracket = 4 * b * t / ((rr + b) * (rr + b + t + root) * (root + rr + b - t))
    integral = np.dot(w, jac * r**3 / (rr + a) * bracket) / (8 * np.pi**2)
    return float((b - a)**2 * integral)


def relative_error(x: float, y: float) -> float:
    return abs(x - y) / max(abs(x), abs(y), 1e-300)


def main() -> None:
    rows = []
    for b in (1.0001, 2.0, 5.0, 17.0, 65.0):
        a = 1.0
        c = coefficient(a, b)
        assert c > 0
        for t in (0.0, 1e-4, 0.01, 1.0, 10.0):
            params = [parameter(a, b, t, n) for n in (64, 128, 256)]
            radials = [radial(a, b, t, n) for n in (256, 512)]
            kernel, c_quad, remainder = params[-1]
            errors = [relative_error(c, item[1]) for item in params]
            errors += [relative_error(kernel, item[0]) for item in params]
            errors += [relative_error(kernel, item) for item in radials]
            assert max(errors) < 1e-8, (a, b, t, errors)
            slack = 1e-12 * max(c * t, 1e-300)
            assert kernel >= 0
            assert kernel <= c * t + slack
            assert kernel >= c * t - remainder - slack
            rows.append({"a": a, "b": b, "k_squared": t, "coefficient": c,
                         "parameter_kernel_64_128_256": [p[0] for p in params],
                         "radial_kernel_256_512": radials,
                         "coefficient_quadrature": c_quad,
                         "max_relative_error": max(errors),
                         "low_momentum_upper": c * t,
                         "low_momentum_lower": c * t - remainder,
                         "gn3_scalar_Z": 54 * c})

    # Constant orientation cannot change eigenvalues or the regulated potential.
    # The seagull identity holds pointwise, before any divergent integral.
    orientation_errors, seagull_errors, omitted_seagull = [], [], []
    for b in (1.0001, 2.0, 5.0, 17.0, 65.0):
        for angle in (0.0, 0.2, 0.7):
            v = np.array([np.cos(angle), np.sin(angle)])
            p = np.outer(v, v)
            mass = b * np.eye(2) - (b - 1) * p
            orientation_errors.append(float(np.max(np.abs(np.linalg.eigvalsh(mass) - [1, b]))))
        for p2 in (0.0, 0.1, 1.0, 10.0):
            delta = b - 1
            seagull = delta * (1 / (p2 + 1) - 1 / (p2 + b))
            bubble = delta**2 / ((p2 + 1) * (p2 + b))
            seagull_errors.append(relative_error(seagull, bubble))
            omitted_seagull.append(-bubble)
    assert max(orientation_errors) < 1e-12
    assert max(seagull_errors) < 1e-8
    assert all(value < 0 for value in omitted_seagull)

    # Independent finite projector geometry check for Tr(dP^2)=2 Tr(dV^* Q dV).
    rng = np.random.default_rng(4104)
    tangent = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    d_p = np.block([[np.zeros((3, 3)), tangent.conj().T],
                    [tangent, np.zeros((7, 7))]])
    lhs = float(np.trace(d_p @ d_p).real / 2)
    rhs = float(np.trace(tangent.conj().T @ tangent).real)
    assert abs(lhs - rhs) < 1e-12

    here = Path(__file__).resolve()
    output = {"candidate": "CE-GN4", "status": "conditional quadratic one-loop response only",
              "python": platform.python_version(), "numpy": np.__version__,
              "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
              "fixed_plan": {"a": 1, "b": [1.0001, 2, 5, 17, 65],
                             "k_squared": [0, 1e-4, 0.01, 1, 10]},
              "cases": rows, "max_relative_error": max(r["max_relative_error"] for r in rows),
              "constant_orientation_max_error": max(orientation_errors),
              "seagull_identity_max_relative_error": max(seagull_errors),
              "omitted_seagull_negative_control": "fails constant-orientation invariance",
              "projector_metric_identity_error": abs(lhs - rhs),
              "limits": ["No absolute infinite-dimensional determinant established",
                         "No independent gauge vectors or Einstein dynamics derived",
                         "No observation selection or joint RMSE calculation"]}
    here.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("candidate", "max_relative_error",
          "constant_orientation_max_error", "seagull_identity_max_relative_error",
          "projector_metric_identity_error")}, indent=2))


if __name__ == "__main__":
    main()
