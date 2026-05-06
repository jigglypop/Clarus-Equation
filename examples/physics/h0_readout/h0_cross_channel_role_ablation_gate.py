"""Ablate cross-channel source roles for the H0 readout contrast.

The contrast gate shows that source roles split channels into local/high and
global/low readouts. This gate asks whether that split is trivial: if all
channels are forced into one static role, or if local/global roles are flipped,
the classification should fail.
"""

from __future__ import annotations

from copy import deepcopy

from h0_bao_global_readout_gate import branch_payload as bao_payload
from h0_cmb_planck_covariance_adapter_gate import branch_payload as cmb_payload
from h0_cross_channel_branch_contrast_gate import TDCOSMO_FILES, classify, tdcosmo_payload
from h0_fisher_matrix_io_gate import channel_from_payload, run_channel
from h0_pantheon_shoes_local_readout_gate import branch_payload as shoes_payload


def payloads() -> list[tuple[str, str, dict[str, object]]]:
    out: list[tuple[str, str, dict[str, object]]] = []
    for label, file_name, expected in TDCOSMO_FILES:
        out.append((label, expected, tdcosmo_payload(file_name)))
    out.append(("DESI BAO", "global", bao_payload()))
    out.append(("Planck CMB", "global", cmb_payload()))
    out.append(("Pantheon+SH0ES", "local", shoes_payload()))
    return out


def mutate(payload: dict[str, object], mode: str) -> dict[str, object]:
    out = deepcopy(payload)
    nodes = [str(node) for node in out["nodes"]]  # type: ignore[index]
    observable = str(out["observable"])
    non_observable = [node for node in nodes if node != observable]
    local_nodes = [str(node) for node in out.get("local_nodes", [])]  # type: ignore[union-attr]
    global_nodes = [str(node) for node in out.get("global_nodes", [])]  # type: ignore[union-attr]

    if mode == "declared":
        return out
    if mode == "all_local":
        out["local_nodes"] = non_observable
        out["global_nodes"] = []
    elif mode == "all_global":
        out["local_nodes"] = []
        out["global_nodes"] = non_observable
    elif mode == "flipped":
        out["local_nodes"] = global_nodes
        out["global_nodes"] = local_nodes
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    return out


def score_mode(mode: str) -> tuple[int, list[tuple[str, str, float, str, bool]]]:
    rows: list[tuple[str, str, float, str, bool]] = []
    correct = 0
    for label, expected, payload in payloads():
        channel = channel_from_payload(mutate(payload, mode))
        result = run_channel(channel)
        classified = classify(result["q_f"])
        ok = (
            (expected == "local" and classified == "local/high")
            or (expected == "global" and classified == "global/low")
        )
        correct += 1 if ok else 0
        rows.append((label, expected, result["q_f"], classified, ok))
    return correct, rows


def main() -> int:
    modes = ["declared", "all_local", "all_global", "flipped"]

    print("# H0 Cross-Channel Role Ablation Gate")
    print()
    print("| model | correct / total | status |")
    print("|---|---:|---|")

    scores: dict[str, int] = {}
    details: dict[str, list[tuple[str, str, float, str, bool]]] = {}
    total = len(payloads())
    for mode in modes:
        correct, rows = score_mode(mode)
        scores[mode] = correct
        details[mode] = rows
        status = "PASS" if (mode == "declared" and correct == total) or (mode != "declared" and correct < total) else "FAIL"
        print(f"| {mode} | {correct}/{total} | {status} |")

    print()
    print("| model | channel | expected | q_F | classified | ok |")
    print("|---|---|---|---:|---|---|")
    for mode in modes:
        for label, expected, q_f, classified, ok in details[mode]:
            print(f"| {mode} | {label} | {expected} | {q_f:.6f} | {classified} | {'yes' if ok else 'no'} |")

    if scores["declared"] != total:
        raise SystemExit("declared source roles should classify every channel")
    if any(scores[mode] == total for mode in modes if mode != "declared"):
        raise SystemExit("a static/flipped ablation unexpectedly classified every channel")

    print()
    print("Verdict: cross-channel separation requires source-aware roles; static role maps fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
