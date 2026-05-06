"""Check the CE role transition across public TDCOSMO covariance chains.

This gate tests the source-aware closure rule:

* lens-only/IFU chains keep MST conductance in the local endpoint role;
* SLACS population chains move MST conductance into global closure.

The thresholds are deliberately broad. They do not fit H0; they check that the
qualitative branch transition implied by the declared source roles is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


EXPECTED = {
    "tdcosmo_only_alpha_free_om_covariance.json": {
        "branch": "local",
        "q_min": 0.75,
        "q_max": 0.95,
        "pull_abs_max": 0.75,
    },
    "tdcosmo_ifu_covariance.json": {
        "branch": "local",
        "q_min": 0.75,
        "q_max": 0.95,
        "pull_abs_max": 0.75,
    },
    "tdcosmo_slacs_covariance.json": {
        "branch": "global",
        "q_min": 0.0,
        "q_max": 0.10,
        "pull_abs_max": 0.25,
    },
    "tdcosmo_slacs_ifu_covariance.json": {
        "branch": "global",
        "q_min": 0.0,
        "q_max": 0.10,
        "pull_abs_max": 0.25,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="directory containing generated TDCOSMO covariance JSON files",
    )
    args = parser.parse_args()
    root = Path(args.path)

    print("# H0 TDCOSMO Role Transition Gate")
    print()
    print("| file | branch | mode | local nodes | global nodes | q_F | H0_pred | pull | status |")
    print("|---|---|---|---|---|---:|---:|---:|---|")

    failed = 0
    q_by_branch: dict[str, list[float]] = {"local": [], "global": []}
    for name, expected in EXPECTED.items():
        file = root / name
        payload = json.loads(file.read_text(encoding="utf-8"))
        channel = channel_from_payload(payload)
        result = run_channel(channel)
        if channel.h0_obs is None or channel.h0_sigma is None:
            raise ValueError(f"{name} needs h0_obs and h0_sigma")
        pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma
        q_f = result["q_f"]
        q_by_branch[expected["branch"]].append(q_f)
        ok = (
            expected["q_min"] <= q_f <= expected["q_max"]
            and abs(pull) <= expected["pull_abs_max"]
        )
        if not ok:
            failed += 1
        print(
            f"| {name} | {expected['branch']} | {channel.conductance_mode} | "
            f"{', '.join(sorted(channel.local_nodes))} | {', '.join(sorted(channel.global_nodes))} | "
            f"{q_f:.6f} | {result['h0_pred']:.6f} | {pull:+.3f} | {'PASS' if ok else 'FAIL'} |"
        )

    local_mean = sum(q_by_branch["local"]) / len(q_by_branch["local"])
    global_mean = sum(q_by_branch["global"]) / len(q_by_branch["global"])
    separation = local_mean - global_mean
    transition_ok = separation > 0.70

    print()
    print(f"local mean q_F = {local_mean:.6f}")
    print(f"global mean q_F = {global_mean:.6f}")
    print(f"branch separation = {separation:.6f}")
    print(f"transition status = {'PASS' if transition_ok else 'FAIL'}")
    print()
    if failed or not transition_ok:
        raise SystemExit(1)
    print("Verdict: source-aware closure roles produce the expected TDCOSMO branch transition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
