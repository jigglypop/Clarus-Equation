"""Independent NumPy reference scalars for the sealed NRM3-D Gate A fixtures.

This is deliberately not imported by Rust and opens no synthetic or PFC data.
It provides only analytic/spectral reference values for a code-audit comparison.
"""

import argparse
import hashlib
import json

import numpy as np


def sym_exp(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    return (vectors * np.exp(values)) @ vectors.T


def sym_log(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    if np.min(values) <= 0:
        raise ValueError("reference received a non-SPD matrix")
    return (vectors * np.log(values)) @ vectors.T


def payload() -> dict[str, object]:
    c = np.array(
        [[0.20, 0.10, -0.08], [0.10, -0.15, 0.07], [-0.08, 0.07, 0.12]],
        dtype=np.float64,
    )
    rebuilt = sym_log(sym_exp(c))
    scalar_origin = -0.56 * float(np.exp(-np.linalg.eigvalsh(c)).sum())
    return {
                "oracle": "numpy.linalg.eigh float64",
                "symmetric_exp_log_relative": float(
                    np.linalg.norm(rebuilt - c, ord="fro") / np.linalg.norm(c, ord="fro")
                ),
                "curved_origin_scalar": scalar_origin,
                "curved_origin_abs": abs(scalar_origin),
                "atol": 1e-11,
                "rtol": 1e-9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    encoded = json.dumps(payload(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
