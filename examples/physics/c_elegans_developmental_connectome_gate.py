"""Developmental C. elegans connectome gate across Witvliet datasets 1-8.

The adult-only gate asks whether L1/L2/L3 layer structure is visible in one
connectome. This script asks the stronger evolutionary/developmental question:

    Is the weighted chemical layer structure already present across development,
    and does it strengthen or reorganize toward adulthood?

It reuses the module assignment and structural losses from
``c_elegans_connectome_gate.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from urllib.request import urlretrieve

import numpy as np

from c_elegans_connectome_gate import (
    DEFAULT_DATA_ROOT,
    aggregate_module_matrices,
    directionality,
    laplacian_stability,
    load_edges,
    module_layer,
    permutation_gate,
)


BASE_URL = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data"
STAGES = list(range(1, 9))


def stage_url(stage: int) -> str:
    return f"{BASE_URL}/witvliet_2020_{stage}.xlsx"


def stage_path(root: Path, stage: int) -> Path:
    return root / f"witvliet_2020_{stage}.xlsx"


def ensure_stage_file(root: Path, stage: int, *, force: bool) -> Path:
    path = stage_path(root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        urlretrieve(stage_url(stage), path)
    return path


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else 0.0


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx
        while j + 1 < len(order) and values[order[j + 1]] == values[order[idx]]:
            j += 1
        rank = (idx + j) / 2.0 + 1.0
        for k in range(idx, j + 1):
            ranks[order[k]] = rank
        idx = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def evaluate_stage(path: Path, stage: int, permutations: int) -> dict[str, object]:
    edges = load_edges(path)
    modules, matrices, edge_summary = aggregate_module_matrices(edges)
    labels = [module_layer(module) for module in modules]
    all_gate = permutation_gate(matrices["all_weighted"], labels, permutations)
    chemical_gate = permutation_gate(matrices["chemical_weighted"], labels, permutations)
    electrical_gate = permutation_gate(matrices["electrical_weighted"], labels, permutations)
    binary_gate = permutation_gate(matrices["all_binary"], labels, permutations)
    stability = laplacian_stability(matrices["all_weighted"])
    direction = directionality(matrices["all_weighted"], modules)
    return {
        "stage": stage,
        "path": str(path),
        "used_edges": edge_summary["used_edges"],
        "used_synapses": edge_summary["used_synapses"],
        "type_synapses": edge_summary["type_synapses"],
        "all_weighted": all_gate,
        "chemical_weighted": chemical_gate,
        "electrical_weighted": electrical_gate,
        "all_binary": binary_gate,
        "directionality": direction,
        "laplacian_stability": stability,
        "passed_weighted_chemical": chemical_gate["layer_block_loss"] < chemical_gate["flat_loss"]
        and chemical_gate["layer_block_loss"] < chemical_gate["random_layer_loss_mean"]
        and chemical_gate["p_value_loss_le_observed"] < 0.05,
    }


def summarize(stages: list[dict[str, object]]) -> dict[str, object]:
    stage_ids = [float(row["stage"]) for row in stages]
    chemical_ratios = [
        float(row["chemical_weighted"]["layer_over_flat"])  # type: ignore[index]
        for row in stages
    ]
    chemical_p = [
        float(row["chemical_weighted"]["p_value_loss_le_observed"])  # type: ignore[index]
        for row in stages
    ]
    synapses = [float(row["used_synapses"]) for row in stages]
    lambda_max = [
        float(row["laplacian_stability"]["laplacian_lambda_max"])  # type: ignore[index]
        for row in stages
    ]
    return {
        "stage_count": len(stages),
        "passed_stage_count": sum(bool(row["passed_weighted_chemical"]) for row in stages),
        "mean_chemical_block_over_flat": mean(chemical_ratios),
        "min_chemical_block_over_flat": min(chemical_ratios),
        "max_chemical_block_over_flat": max(chemical_ratios),
        "mean_chemical_permutation_p": mean(chemical_p),
        "spearman_stage_vs_chemical_block_over_flat": spearman(stage_ids, chemical_ratios),
        "spearman_stage_vs_synapses": spearman(stage_ids, synapses),
        "spearman_stage_vs_lambda_max": spearman(stage_ids, lambda_max),
    }


def write_report(output: dict[str, object], path: Path) -> None:
    summary = output["summary"]  # type: ignore[assignment]
    lines = [
        "# C. elegans 발달 connectome 게이트",
        "",
        "Witvliet dataset 1-8을 모두 읽어 L1/L2/L3 층화 구조가 발달 전반에서 유지되는지 점검했다.",
        "",
        "## 질문",
        "",
        "$$",
        "\\mathcal L_{\\mathrm{chemical\\ weighted\\ L1/L2/L3}}",
        "<",
        "\\mathcal L_{\\mathrm{flat/random}}",
        "$$",
        "",
        "## 단계별 결과",
        "",
        "| stage | synapses | chem block/flat | chem p | all block/flat | electrical block/flat | binary block/flat | lambda max | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["stages"]:  # type: ignore[index]
        chem = row["chemical_weighted"]
        all_gate = row["all_weighted"]
        electrical = row["electrical_weighted"]
        binary = row["all_binary"]
        stability = row["laplacian_stability"]
        lines.append(
            f"| {row['stage']} | {float(row['used_synapses']):.1f} | "
            f"{chem['layer_over_flat']:.6f} | {chem['p_value_loss_le_observed']:.6f} | "
            f"{all_gate['layer_over_flat']:.6f} | {electrical['layer_over_flat']:.6f} | "
            f"{binary['layer_over_flat']:.6f} | {stability['laplacian_lambda_max']:.6f} | "
            f"{row['passed_weighted_chemical']} |"
        )
    lines.extend(
        [
            "",
            "## 요약",
            "",
            "| 항목 | 값 |",
            "|---|---:|",
            f"| stages | {summary['stage_count']} |",
            f"| passed weighted chemical stages | {summary['passed_stage_count']} |",
            f"| mean chemical block/flat | {summary['mean_chemical_block_over_flat']:.6f} |",
            f"| min chemical block/flat | {summary['min_chemical_block_over_flat']:.6f} |",
            f"| max chemical block/flat | {summary['max_chemical_block_over_flat']:.6f} |",
            f"| mean chemical permutation p | {summary['mean_chemical_permutation_p']:.6f} |",
            f"| Spearman stage vs chemical block/flat | {summary['spearman_stage_vs_chemical_block_over_flat']:.6f} |",
            f"| Spearman stage vs synapses | {summary['spearman_stage_vs_synapses']:.6f} |",
            f"| Spearman stage vs lambda max | {summary['spearman_stage_vs_lambda_max']:.6f} |",
            "",
            "## 해석",
            "",
            "- weighted chemical 구조가 여러 발달 단계에서 유지되면, 층화 그래프 문법은 성체에서 갑자기 생긴 것이 아니다.",
            "- stage와 synapse/lambda max의 상관은 성장하면서 그래프 강도와 안정성 스케일이 어떻게 변하는지 보여준다.",
            "- 이 검증은 구조 connectome만 본 것이며, 행동 동역학 전이 검증은 아니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    stage_rows = []
    for stage in STAGES:
        path = ensure_stage_file(args.data_root, stage, force=args.force_download)
        stage_rows.append(evaluate_stage(path, stage, args.permutations))

    output = {
        "dataset": "Witvliet et al. 2021 C. elegans datasets 1-8",
        "base_url": BASE_URL,
        "permutations": args.permutations,
        "stages": stage_rows,
        "summary": summarize(stage_rows),
    }
    out_json = Path(__file__).with_name("c_elegans_developmental_gate_results.json")
    out_md = Path(__file__).with_name("c_elegans_developmental_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    print("C. elegans developmental connectome gate")
    for row in stage_rows:
        chem = row["chemical_weighted"]
        print(
            f"  stage {row['stage']}: synapses={float(row['used_synapses']):.1f}, "
            f"chem block/flat={chem['layer_over_flat']:.6f}, "
            f"p={chem['p_value_loss_le_observed']:.6f}, "
            f"passed={row['passed_weighted_chemical']}"
        )
    summary = output["summary"]
    print("  summary:")
    print(f"    passed stages = {summary['passed_stage_count']}/{summary['stage_count']}")
    print(
        "    Spearman stage vs chemical block/flat = "
        f"{summary['spearman_stage_vs_chemical_block_over_flat']:.6f}"
    )
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
