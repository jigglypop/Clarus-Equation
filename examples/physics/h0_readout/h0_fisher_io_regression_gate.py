"""Regression checks for the Fisher/covariance IO gate."""

from __future__ import annotations

import json
from pathlib import Path

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


EXAMPLE_DIR = Path(__file__).with_name("h0_fisher_io_examples")


def load_result(name: str) -> dict[str, float]:
    payload = json.loads((EXAMPLE_DIR / name).read_text(encoding="utf-8"))
    return run_channel(channel_from_payload(payload))


def main() -> int:
    fisher = load_result("gw_like_fisher.json")
    covariance = load_result("gw_like_covariance.json")

    print("# H0 Fisher IO Regression Gate")
    print()
    print("| input | C_local | C_global | q_F | H0_pred |")
    print("|---|---:|---:|---:|---:|")
    for label, result in [("fisher", fisher), ("covariance", covariance)]:
        print(
            f"| {label} | {result['c_local']:.8f} | {result['c_global']:.8f} | "
            f"{result['q_f']:.8f} | {result['h0_pred']:.6f} |"
        )

    dq = abs(fisher["q_f"] - covariance["q_f"])
    dh0 = abs(fisher["h0_pred"] - covariance["h0_pred"])
    print()
    print(f"Delta q_F = {dq:.3e}")
    print(f"Delta H0 = {dh0:.3e}")

    if dq > 1e-12 or dh0 > 1e-9:
        raise SystemExit("Fisher/covariance IO regression failed")

    print()
    print("Verdict: Fisher and covariance JSON inputs are equivalent for the smoke channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
