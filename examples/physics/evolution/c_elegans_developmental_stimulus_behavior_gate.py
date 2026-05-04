"""Developmental C. elegans stimulus-to-output channel gate.

This extends ``c_elegans_stimulus_behavior_gate.py`` from adult dataset 8 to
Witvliet datasets 1-8. It asks when the stimulus-domain -> same output-domain
structural channel appears during development.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from c_elegans_connectome_gate import (
    DEFAULT_DATA_ROOT,
    aggregate_module_matrices,
    load_edges,
)
from c_elegans_developmental_connectome_gate import ensure_stage_file
from c_elegans_stimulus_behavior_gate import (
    DOMAINS,
    domain_flow_matrix,
    row_stochastic,
)


RNG_SEED = 1729
STAGES = list(range(1, 9))


def permuted_score_from_flow(modules: list[str], matrix: np.ndarray, labels: list[str]) -> float:
    from c_elegans_stimulus_behavior_gate import MODULE_FOR

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


def permutation_test(modules: list[str], matrix: np.ndarray, observed: float, permutations: int) -> dict[str, object]:
    rng = random.Random(RNG_SEED)
    labels = list(DOMAINS)
    scores = []
    hits = 0
    for _ in range(permutations):
        rng.shuffle(labels)
        score = permuted_score_from_flow(modules, matrix, labels)
        scores.append(score)
        if score >= observed:
            hits += 1
    return {
        "observed": observed,
        "permutation_mean": mean(scores),
        "permutation_max": max(scores),
        "permutation_min": min(scores),
        "p_value_ge_observed": (hits + 1) / (permutations + 1),
    }


def evaluate_stage(path: Path, stage: int, permutations: int) -> dict[str, object]:
    edges = load_edges(path)
    modules, matrices, edge_summary = aggregate_module_matrices(edges)
    out = {"stage": stage, "path": str(path), "edge_summary": edge_summary}
    for matrix_name in ("chemical_weighted", "all_weighted", "all_binary"):
        flow = domain_flow_matrix(modules, matrices[matrix_name])
        perm = permutation_test(
            modules,
            matrices[matrix_name],
            float(flow["matched_over_wrong"]),
            permutations,
        )
        out[matrix_name] = {
            "domain_flow": flow,
            "permutation_test": perm,
            "passed": flow["matched_over_wrong"] > 1.5 and perm["p_value_ge_observed"] < 0.05,
        }
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda idx: values[idx])
        out = [0.0] * len(values)
        for rank, idx in enumerate(order, start=1):
            out[idx] = float(rank)
        return out

    rx = np.asarray(ranks(xs), dtype=np.float64)
    ry = np.asarray(ranks(ys), dtype=np.float64)
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / denom) if denom else 0.0


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    stages = [float(row["stage"]) for row in rows]
    ratios = [
        float(row["chemical_weighted"]["domain_flow"]["matched_over_wrong"])  # type: ignore[index]
        for row in rows
    ]
    p_values = [
        float(row["chemical_weighted"]["permutation_test"]["p_value_ge_observed"])  # type: ignore[index]
        for row in rows
    ]
    synapses = [float(row["edge_summary"]["used_synapses"]) for row in rows]  # type: ignore[index]
    return {
        "stage_count": len(rows),
        "passed_chemical_weighted_count": sum(
            bool(row["chemical_weighted"]["passed"]) for row in rows  # type: ignore[index]
        ),
        "mean_chemical_matched_over_wrong": mean(ratios),
        "min_chemical_matched_over_wrong": min(ratios),
        "max_chemical_matched_over_wrong": max(ratios),
        "mean_p_value": mean(p_values),
        "spearman_stage_vs_matched_over_wrong": spearman(stages, ratios),
        "spearman_stage_vs_synapses": spearman(stages, synapses),
    }


def write_report(output: dict[str, object], path: Path) -> None:
    summary = output["summary"]  # type: ignore[assignment]
    lines = [
        "# C. elegans 발달 자극-행동 구조 게이트",
        "",
        "Witvliet dataset 1-8 전체에서 L1 stimulus domain이 같은 L3 output domain으로 보존되는지 확인했다.",
        "",
        "## 기준",
        "",
        "$$",
        "\\mathrm{Flow}(L1_d\\to L3_d)",
        ">",
        "\\mathrm{Flow}(L1_d\\to L3_{d'\\ne d}),",
        "\\qquad",
        "\\mathrm{matched/wrong}>1.5",
        "$$",
        "",
        "## 단계별 결과",
        "",
        "| stage | synapses | chem matched/wrong | chem p | chem pass | binary matched/wrong | binary pass |",
        "|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in output["stages"]:  # type: ignore[index]
        chem = row["chemical_weighted"]
        binary = row["all_binary"]
        lines.append(
            f"| {row['stage']} | {float(row['edge_summary']['used_synapses']):.1f} | "
            f"{chem['domain_flow']['matched_over_wrong']:.6f} | "
            f"{chem['permutation_test']['p_value_ge_observed']:.6f} | "
            f"{chem['passed']} | "
            f"{binary['domain_flow']['matched_over_wrong']:.6f} | "
            f"{binary['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 요약",
            "",
            "| 항목 | 값 |",
            "|---|---:|",
            f"| stages | {summary['stage_count']} |",
            f"| passed chemical weighted stages | {summary['passed_chemical_weighted_count']} |",
            f"| mean chemical matched/wrong | {summary['mean_chemical_matched_over_wrong']:.6f} |",
            f"| min chemical matched/wrong | {summary['min_chemical_matched_over_wrong']:.6f} |",
            f"| max chemical matched/wrong | {summary['max_chemical_matched_over_wrong']:.6f} |",
            f"| mean p value | {summary['mean_p_value']:.6f} |",
            f"| Spearman stage vs matched/wrong | {summary['spearman_stage_vs_matched_over_wrong']:.6f} |",
            f"| Spearman stage vs synapses | {summary['spearman_stage_vs_synapses']:.6f} |",
            "",
            "## 해석",
            "",
            "- 통과 단계가 많으면 자극-domain에서 output-domain으로 이어지는 구조 channel이 발달 전반에서 보존된다는 뜻이다.",
            "- binary가 실패하고 weighted chemical이 통과하면, 행동 proxy도 단순 연결 유무가 아니라 synaptic weight에 실린다는 뜻이다.",
            "- 이 검증은 실제 행동 trial이 아니라 구조 proxy다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    rows = []
    for stage in STAGES:
        path = ensure_stage_file(args.data_root, stage, force=args.force_download)
        rows.append(evaluate_stage(path, stage, args.permutations))
    output = {
        "dataset": "Witvliet C. elegans datasets 1-8",
        "permutations": args.permutations,
        "stages": rows,
        "summary": summarize(rows),
    }
    out_json = Path(__file__).with_name("c_elegans_developmental_stimulus_behavior_results.json")
    out_md = Path(__file__).with_name("c_elegans_developmental_stimulus_behavior_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    print("C. elegans developmental stimulus-behavior gate")
    for row in rows:
        chem = row["chemical_weighted"]
        print(
            f"  stage {row['stage']}: "
            f"chem matched/wrong={chem['domain_flow']['matched_over_wrong']:.6f}, "
            f"p={chem['permutation_test']['p_value_ge_observed']:.6f}, "
            f"passed={chem['passed']}"
        )
    summary = output["summary"]
    print("  summary:")
    print(
        "    passed chemical weighted stages = "
        f"{summary['passed_chemical_weighted_count']}/{summary['stage_count']}"
    )
    print(
        "    Spearman stage vs matched/wrong = "
        f"{summary['spearman_stage_vs_matched_over_wrong']:.6f}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
