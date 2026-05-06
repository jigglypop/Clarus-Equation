"""Negative validation tests for H0 Fisher/covariance JSON ingestion."""

from __future__ import annotations

from h0_fisher_io_validate_gate import validate_payload


BAD_PAYLOADS = [
    (
        "unknown local node",
        {
            "nodes": ["obs", "global"],
            "observable": "obs",
            "local_nodes": ["missing_local"],
            "global_nodes": ["global"],
            "matrix_type": "fisher",
            "matrix": [[1.0, 0.1], [0.1, 1.0]],
        },
        "local_nodes contains unknown nodes",
    ),
    (
        "nonsymmetric matrix",
        {
            "nodes": ["obs", "local"],
            "observable": "obs",
            "local_nodes": ["local"],
            "global_nodes": [],
            "matrix_type": "fisher",
            "matrix": [[1.0, 0.2], [0.1, 1.0]],
        },
        "matrix must be symmetric",
    ),
    (
        "nonpositive diagonal",
        {
            "nodes": ["obs", "local"],
            "observable": "obs",
            "local_nodes": ["local"],
            "global_nodes": [],
            "matrix_type": "fisher",
            "matrix": [[0.0, 0.0], [0.0, 1.0]],
        },
        "matrix diagonal must be positive",
    ),
    (
        "singular covariance",
        {
            "nodes": ["obs", "local"],
            "observable": "obs",
            "local_nodes": ["local"],
            "global_nodes": [],
            "matrix_type": "covariance",
            "matrix": [[1.0, 1.0], [1.0, 1.0]],
        },
        "pipeline ingestion failed",
    ),
]


def main() -> int:
    print("# H0 Fisher IO Negative Gate")
    print()
    print("| case | expected fragment | status | validator notes |")
    print("|---|---|---|---|")
    failed = 0
    for name, payload, expected in BAD_PAYLOADS:
        errors = validate_payload(payload)
        joined = "; ".join(errors)
        passed = bool(errors) and expected in joined
        if not passed:
            failed += 1
        print(f"| {name} | {expected} | {'PASS' if passed else 'FAIL'} | {joined or 'no error'} |")
    print()
    print(f"negative cases = {len(BAD_PAYLOADS)}")
    print(f"failed = {failed}")
    if failed:
        raise SystemExit(1)
    print("Verdict: validator rejects malformed Fisher/covariance inputs as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
