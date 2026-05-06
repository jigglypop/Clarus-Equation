"""Ablate the TDCOSMO role map and verify that wrong closures fail.

This is a falsification-style companion to h0_tdcosmo_role_transition_gate.py.
It compares the declared source-aware role map with two ablations:

* all-MST-local: every lambda_mst family node is local for every chain;
* all-MST-global: only alpha_lambda is local for every chain.

The CE role-transition claim is useful only if the source-aware map improves
both local and SLACS chains at once. If a single static map performs equally
well, the role-transition interpretation is unnecessary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from h0_fisher_matrix_io_gate import MatrixChannel, channel_from_payload, run_channel


FILES = [
    "tdcosmo_only_alpha_free_om_covariance.json",
    "tdcosmo_ifu_covariance.json",
    "tdcosmo_slacs_covariance.json",
    "tdcosmo_slacs_ifu_covariance.json",
]


def with_roles(channel: MatrixChannel, model: str) -> MatrixChannel:
    nodes = set(channel.nodes) - {channel.observable}
    lambda_family = {node for node in nodes if node.startswith("lambda_mst")}
    alpha = {"alpha_lambda"} & nodes
    if model == "declared":
        local = set(channel.local_nodes)
        global_ = set(channel.global_nodes)
        mode = channel.conductance_mode
    elif model == "all_mst_local":
        local = lambda_family | alpha
        global_ = nodes - local
        mode = "direct"
    elif model == "all_mst_global":
        local = alpha
        global_ = nodes - local
        mode = "direct"
    elif model == "legacy_path":
        local = lambda_family | alpha
        global_ = nodes - local
        mode = "path"
    else:
        raise ValueError(f"unknown model: {model}")
    return MatrixChannel(
        name=channel.name,
        nodes=channel.nodes,
        observable=channel.observable,
        local_nodes=local,
        global_nodes=global_,
        fisher=channel.fisher,
        h0_obs=channel.h0_obs,
        h0_sigma=channel.h0_sigma,
        conductance_mode=mode,
    )


def score(root: Path, model: str) -> tuple[float, list[tuple[str, float, float, float]]]:
    chi2 = 0.0
    rows = []
    for name in FILES:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        channel = with_roles(channel_from_payload(payload), model)
        result = run_channel(channel)
        if channel.h0_obs is None or channel.h0_sigma is None:
            raise ValueError(f"{name} needs h0_obs and h0_sigma")
        pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma
        chi2 += pull * pull
        rows.append((name, result["q_f"], result["h0_pred"], pull))
    return chi2, rows


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

    models = ["declared", "all_mst_local", "all_mst_global", "legacy_path"]
    scores = {model: score(root, model) for model in models}

    print("# H0 TDCOSMO Role Ablation Gate")
    print()
    print("| model | chi2 | rms pull | verdict |")
    print("|---|---:|---:|---|")
    for model in models:
        chi2, _ = scores[model]
        rms = math.sqrt(chi2 / len(FILES))
        verdict = "PASS" if model == "declared" else "ablation"
        print(f"| {model} | {chi2:.6f} | {rms:.6f} | {verdict} |")

    print()
    print("## Per-chain ablation detail")
    print()
    print("| model | file | q_F | H0_pred | pull |")
    print("|---|---|---:|---:|---:|")
    for model in models:
        _, rows = scores[model]
        for name, q_f, h0_pred, pull in rows:
            print(f"| {model} | {name} | {q_f:.6f} | {h0_pred:.6f} | {pull:+.3f} |")

    declared_chi2 = scores["declared"][0]
    best_ablation = min(scores[model][0] for model in models if model != "declared")
    improvement = best_ablation / declared_chi2 if declared_chi2 else float("inf")

    print()
    print(f"best ablation / declared chi2 = {improvement:.3f}")
    print()
    if improvement < 5.0:
        raise SystemExit(1)
    print("Verdict: static role maps fail; source-aware role transition is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
