"""C. elegans stimulus-to-behavior structural gate.

This is a first primitive nervous-system behavior proxy. It does not use
recorded behavior. Instead, it asks whether the weighted chemical connectome
preserves stimulus domain channels across the minimal hierarchy:

    L1 input -> L2 relay -> L3 premotor/integrative output

The matched criterion is:

    L1_domain -> L3_same_domain flow > L1_domain -> L3_wrong_domain flow

This checks whether the connectome contains domain-specific routes that could
support behavior classes such as avoidance and taxis.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from c_elegans_connectome_gate import (
    DEFAULT_XLSX,
    aggregate_module_matrices,
    ensure_xlsx,
    load_edges,
)


DOMAINS = ["Anterior", "Lateral", "Avoidance", "Taxis"]
MODULE_FOR = {
    ("L1", "Anterior"): "L1_Anterior",
    ("L1", "Lateral"): "L1_Lateral",
    ("L1", "Avoidance"): "L1_Avoidance",
    ("L1", "Taxis"): "L1_Taxis",
    ("L2", "Anterior"): "L2_Anterior",
    ("L2", "Lateral"): "L2_LatSub",
    ("L2", "Avoidance"): "L2_Avoidance",
    ("L2", "Taxis"): "L2_Taxis",
    ("L3", "Anterior"): "L3_Anterior",
    ("L3", "Lateral"): "L3_LatSub",
    ("L3", "Avoidance"): "L3_Avoidance",
    ("L3", "Taxis"): "L3_Taxis",
}
RNG_SEED = 1729


def row_stochastic(matrix: np.ndarray) -> np.ndarray:
    row_sum = np.sum(matrix, axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)


def domain_flow_matrix(modules: list[str], matrix: np.ndarray) -> dict[str, object]:
    index = {module: idx for idx, module in enumerate(modules)}
    transition = row_stochastic(matrix)
    two_step = transition @ transition
    combined = 0.5 * transition + 0.5 * two_step
    flows = {}
    direct = {}
    relay = {}
    for src_domain in DOMAINS:
        src = index[MODULE_FOR[("L1", src_domain)]]
        for dst_domain in DOMAINS:
            dst = index[MODULE_FOR[("L3", dst_domain)]]
            key = f"{src_domain}->{dst_domain}"
            direct[key] = float(transition[src, dst])
            relay[key] = float(two_step[src, dst])
            flows[key] = float(combined[src, dst])
    matched = [flows[f"{domain}->{domain}"] for domain in DOMAINS]
    wrong = [
        flows[f"{src}->{dst}"]
        for src in DOMAINS
        for dst in DOMAINS
        if src != dst
    ]
    return {
        "flows": flows,
        "direct_flows": direct,
        "two_step_flows": relay,
        "matched_mean": mean(matched),
        "wrong_mean": mean(wrong),
        "matched_over_wrong": mean(matched) / max(mean(wrong), 1e-12),
    }


def permuted_domain_score(modules: list[str], matrix: np.ndarray, labels: list[str]) -> float:
    index = {module: idx for idx, module in enumerate(modules)}
    transition = row_stochastic(matrix)
    two_step = transition @ transition
    combined = 0.5 * transition + 0.5 * two_step
    matched = []
    wrong = []
    for src_domain, dst_domain in zip(DOMAINS, labels):
        src = index[MODULE_FOR[("L1", src_domain)]]
        dst = index[MODULE_FOR[("L3", dst_domain)]]
        matched.append(float(combined[src, dst]))
    for src_domain in DOMAINS:
        src = index[MODULE_FOR[("L1", src_domain)]]
        for dst_domain in labels:
            if dst_domain == src_domain:
                continue
            dst = index[MODULE_FOR[("L3", dst_domain)]]
            wrong.append(float(combined[src, dst]))
    return mean(matched) / max(mean(wrong), 1e-12)


def permutation_test(modules: list[str], matrix: np.ndarray, permutations: int) -> dict[str, object]:
    observed = domain_flow_matrix(modules, matrix)["matched_over_wrong"]
    rng = random.Random(RNG_SEED)
    labels = list(DOMAINS)
    hits = 0
    scores = []
    for _ in range(permutations):
        rng.shuffle(labels)
        score = permuted_domain_score(modules, matrix, labels)
        scores.append(score)
        if score >= observed:
            hits += 1
    return {
        "statistic": "matched_over_wrong; higher is better",
        "observed": observed,
        "permutation_mean": mean(scores),
        "permutation_max": max(scores),
        "permutation_min": min(scores),
        "permutations": permutations,
        "p_value_ge_observed": (hits + 1) / (permutations + 1),
    }


def evaluate_matrix(modules: list[str], matrix: np.ndarray, permutations: int) -> dict[str, object]:
    flow = domain_flow_matrix(modules, matrix)
    perm = permutation_test(modules, matrix, permutations)
    return {
        "domain_flow": flow,
        "permutation_test": perm,
        "passed": flow["matched_mean"] > flow["wrong_mean"]
        and flow["matched_over_wrong"] > 1.5
        and perm["p_value_ge_observed"] < 0.05,
    }


def write_report(output: dict[str, object], path: Path) -> None:
    chem = output["chemical_weighted"]  # type: ignore[assignment]
    flow = chem["domain_flow"]
    perm = chem["permutation_test"]
    lines = [
        "# C. elegans 자극-행동 구조 게이트",
        "",
        "이 검증은 실제 행동 기록이 아니라 connectome 기반 proxy다. 질문은 L1 input domain이 L2 relay를 거쳐 같은 L3 premotor/integrative domain으로 더 강하게 흐르는가이다.",
        "",
        "## 검증식",
        "",
        "$$",
        "\\mathrm{Flow}(L1_d\\to L3_d)",
        ">",
        "\\mathrm{Flow}(L1_d\\to L3_{d'\\ne d})",
        "$$",
        "",
        "## 결과",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| matched mean | {flow['matched_mean']:.8f} |",
        f"| wrong mean | {flow['wrong_mean']:.8f} |",
        f"| matched / wrong | {flow['matched_over_wrong']:.6f} |",
        f"| permutation p | {perm['p_value_ge_observed']:.6f} |",
        "| effect threshold | 1.500000 |",
        f"| passed | {chem['passed']} |",
        "",
        "## domain flow",
        "",
        "| route | combined flow | direct | two-step |",
        "|---|---:|---:|---:|",
    ]
    for key, value in flow["flows"].items():
        lines.append(
            f"| {key} | {value:.8f} | "
            f"{flow['direct_flows'][key]:.8f} | {flow['two_step_flows'][key]:.8f} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 통과하면 C. elegans connectome 안에 자극 domain을 같은 행동-output domain으로 보존하는 구조 경로가 있다는 뜻이다.",
            "- 실패하면 L1/L2/L3 층화는 있어도 행동 domain channel은 아직 connectome proxy만으로 닫히지 않는다는 뜻이다.",
            "- 이 검증은 실제 행동 데이터가 아니므로 최종 행동 방정식 검증은 아니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    xlsx = ensure_xlsx(args.xlsx, force=args.force_download)
    edges = load_edges(xlsx)
    modules, matrices, edge_summary = aggregate_module_matrices(edges)
    output = {
        "dataset": "Witvliet adult dataset 8 / C. elegans",
        "xlsx": str(xlsx),
        "edge_summary": edge_summary,
        "domains": DOMAINS,
        "chemical_weighted": evaluate_matrix(
            modules, matrices["chemical_weighted"], args.permutations
        ),
        "all_weighted": evaluate_matrix(
            modules, matrices["all_weighted"], args.permutations
        ),
        "all_binary": evaluate_matrix(
            modules, matrices["all_binary"], args.permutations
        ),
        "criterion": (
            "matched stimulus-domain flow > wrong-domain flow, matched/wrong > 1.5, "
            "and permutation p < 0.05"
        ),
    }
    out_json = Path(__file__).with_name("c_elegans_stimulus_behavior_gate_results.json")
    out_md = Path(__file__).with_name("c_elegans_stimulus_behavior_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    chem = output["chemical_weighted"]
    flow = chem["domain_flow"]
    perm = chem["permutation_test"]
    print("C. elegans stimulus-behavior structural gate")
    print(f"  matched_mean       = {flow['matched_mean']:.8f}")
    print(f"  wrong_mean         = {flow['wrong_mean']:.8f}")
    print(f"  matched/wrong      = {flow['matched_over_wrong']:.6f}")
    print(f"  permutation_p      = {perm['p_value_ge_observed']:.6f}")
    print(f"  passed             = {chem['passed']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
