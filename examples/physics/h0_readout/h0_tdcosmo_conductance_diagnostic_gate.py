"""Diagnose direct-edge vs path conductance for public TDCOSMO chains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from h0_dataset_falsification_gate import log_s_from_h0
from h0_fisher_matrix_io_gate import (
    MatrixChannel,
    ce_scales,
    channel_from_payload,
    normalized_edge_graph,
    run_channel,
)


def conductance_direct(channel: MatrixChannel, targets: set[str]) -> float:
    graph = normalized_edge_graph(channel)
    return sum(reliability for node, reliability in graph[channel.observable] if node in targets)


def h0_from_q(q_f: float) -> float:
    from h0_dataset_falsification_gate import h0_from_log_s

    defect, log_s_global = ce_scales()
    return h0_from_log_s(log_s_global - q_f * defect)


def q_required(h0_obs: float) -> float:
    defect, log_s_global = ce_scales()
    return (log_s_global - log_s_from_h0(h0_obs)) / defect


def with_partition(
    channel: MatrixChannel,
    local_nodes: Iterable[str],
    global_nodes: Iterable[str],
) -> MatrixChannel:
    return MatrixChannel(
        name=channel.name,
        nodes=channel.nodes,
        observable=channel.observable,
        local_nodes=set(local_nodes),
        global_nodes=set(global_nodes),
        fisher=channel.fisher,
        h0_obs=channel.h0_obs,
        h0_sigma=channel.h0_sigma,
    )


def partitions(channel: MatrixChannel) -> list[tuple[str, set[str], set[str]]]:
    nodes = set(channel.nodes) - {channel.observable}
    lambda_mean = {node for node in nodes if node in {"lambda_mst", "lambda_mst_ifu"}}
    lambda_family = {node for node in nodes if node.startswith("lambda_mst")}
    alpha = {node for node in nodes if node == "alpha_lambda"}
    anisotropy = {node for node in nodes if node.startswith("a_ani")}

    candidates = [
        ("payload_declared_roles", set(channel.local_nodes)),
        ("mst_mean_only", lambda_mean),
        ("mst_family_only", lambda_family),
        ("alpha_only", alpha),
        ("mst_mean_plus_alpha", lambda_mean | alpha),
        ("mst_family_plus_anisotropy", lambda_family | anisotropy),
    ]

    out = []
    seen = set()
    for name, local in candidates:
        local = local & nodes
        key = tuple(sorted(local))
        if not local or key in seen:
            continue
        seen.add(key)
        out.append((name, local, nodes - local))
    return out


def score(channel: MatrixChannel, mode: str) -> tuple[float, float, float, float]:
    if mode == "path":
        result = run_channel(channel)
        c_local = result["c_local"]
        c_global = result["c_global"]
        q_f = result["q_f"]
        h0_pred = result["h0_pred"]
    elif mode == "direct":
        c_local = conductance_direct(channel, channel.local_nodes)
        c_global = conductance_direct(channel, channel.global_nodes)
        q_f = c_local / (c_local + c_global) if c_local + c_global else 0.0
        h0_pred = h0_from_q(q_f)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return c_local, c_global, q_f, h0_pred


def iter_tdcosmo_files(path: Path) -> list[Path]:
    return sorted(path.glob("tdcosmo*_covariance.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="directory containing TDCOSMO covariance JSON files",
    )
    args = parser.parse_args()

    files = iter_tdcosmo_files(Path(args.path))
    if not files:
        raise SystemExit("No tdcosmo*_covariance.json files found")

    print("# H0 TDCOSMO Conductance Diagnostic Gate")
    print()
    print("| file | partition | mode | q_req | q_F | H0_pred | pull | C_L | C_G |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")

    rows = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        base_channel = channel_from_payload(payload)
        if base_channel.h0_obs is None or base_channel.h0_sigma is None:
            raise ValueError(f"{file.name} needs h0_obs and h0_sigma")
        q_req = q_required(base_channel.h0_obs)
        for partition_name, local, global_ in partitions(base_channel):
            channel = with_partition(base_channel, local, global_)
            for mode in ["path", "direct"]:
                c_local, c_global, q_f, h0_pred = score(channel, mode)
                pull = (h0_pred - base_channel.h0_obs) / base_channel.h0_sigma
                rows.append((abs(pull), file.name, partition_name, mode, q_req, q_f, h0_pred, pull, c_local, c_global))
                print(
                    f"| {file.name} | {partition_name} | {mode} | {q_req:.6f} | {q_f:.6f} | "
                    f"{h0_pred:.6f} | {pull:+.3f} | {c_local:.6f} | {c_global:.6f} |"
                )

    best = min(rows, key=lambda row: row[0])
    print()
    print("## Best absolute pull")
    print()
    print(
        f"{best[1]} / {best[2]} / {best[3]}: q_req={best[4]:.6f}, "
        f"q_F={best[5]:.6f}, H0_pred={best[6]:.6f}, pull={best[7]:+.3f}"
    )
    print()
    print("Verdict: conductance and node-partition diagnostics completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
