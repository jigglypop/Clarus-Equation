"""Drosophila larva next-step gate after primitive nervous systems.

This uses the public Netzschleuder copy of the Winding et al. larval
Drosophila brain connectome. The question is:

    What appears after the primitive sensory -> relay -> motor grammar?

The candidate answer tested here is the emergence of a separated
learning/memory/action-selection loop, represented by the mushroom-body
classes KC, MBIN, MBON, MB-FBN, and MB-FFN.

This is a structural connectome gate, not a behavioral dynamics proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from urllib.request import urlretrieve

import numpy as np


DATASET = "Drosophila larva brain connectome / Winding et al. 2023"
SOURCE_URL = "https://networks.skewed.de/net/fly_larva/files/fly_larva.csv.zip"
DEFAULT_ZIP = Path("data/evolution/drosophila_larva/fly_larva.csv.zip")
RNG_SEED = 1729


MEMORY_TYPES = {"KC", "MBIN", "MBON", "MB-FBN", "MB-FFN"}
DESCENDING_TYPES = {"DN-VNC", "DN-SEZ", "pre-DN-VNC", "pre-DN-SEZ"}
RELAY_TYPES = {"PN", "PN-somato", "LN", "ascending"}
INTEGRATION_TYPES = {"LHN", "CN", "RGN"}
INPUT_TYPES = {"sensory"}


def ensure_zip(path: Path, *, force: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        urlretrieve(SOURCE_URL, path)
    return path


def read_zip(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with zipfile.ZipFile(path) as zf:
        with zf.open("nodes.csv") as handle:
            node_lines = (line.decode("utf-8").lstrip("# ").strip() for line in handle)
            nodes = list(csv.DictReader(node_lines, skipinitialspace=True))
        with zf.open("edges.csv") as handle:
            edge_lines = (line.decode("utf-8").lstrip("# ").strip() for line in handle)
            edges = list(csv.DictReader(edge_lines, skipinitialspace=True))
    return nodes, edges


def clean_type(cell_type: str) -> str:
    return cell_type.strip() or "unknown"


def functional_class(cell_type: str) -> str:
    cell_type = clean_type(cell_type)
    if cell_type in INPUT_TYPES:
        return "sensory_input"
    if cell_type in RELAY_TYPES:
        return "projection_relay"
    if cell_type in MEMORY_TYPES:
        return "mushroom_body_memory"
    if cell_type in INTEGRATION_TYPES:
        return "lateral_integration"
    if cell_type in DESCENDING_TYPES:
        return "descending_action"
    return "unknown"


def primitive_class(cell_type: str) -> str:
    group = functional_class(cell_type)
    if group in {"sensory_input", "projection_relay"}:
        return "input_relay"
    if group in {"mushroom_body_memory", "lateral_integration", "unknown"}:
        return "integration"
    if group == "descending_action":
        return "action"
    return "integration"


def action_split_class(cell_type: str) -> str:
    cell_type = clean_type(cell_type)
    if cell_type in INPUT_TYPES:
        return "sensory_input"
    if cell_type in RELAY_TYPES:
        return "projection_relay"
    if cell_type in MEMORY_TYPES or cell_type in INTEGRATION_TYPES or cell_type == "unknown":
        return "integration"
    if cell_type in {"pre-DN-VNC", "pre-DN-SEZ"}:
        return "pre_descending"
    if cell_type in {"DN-VNC", "DN-SEZ"}:
        return "descending"
    return "integration"


def sensory_modality_class(cell_type: str, annotations: str) -> str:
    group = functional_class(cell_type)
    text = annotations.lower()
    if group != "sensory_input":
        return group
    if "gustatory" in text:
        return "sensory_gustatory"
    if "visual" in text:
        return "sensory_visual"
    if "noci" in text:
        return "sensory_nociceptive"
    if "mechano" in text or "proprio" in text:
        return "sensory_mechano"
    if "thermo" in text:
        return "sensory_thermo"
    if "olfactory" in text:
        return "sensory_olfactory"
    return "sensory_other"


def matrix_by_type(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    index_to_type = {row["index"].strip(): clean_type(row["cell_type"]) for row in nodes}
    types = sorted(set(index_to_type.values()))
    type_index = {name: idx for idx, name in enumerate(types)}
    matrix = np.zeros((len(types), len(types)), dtype=np.float64)
    used_edges = 0
    used_synapses = 0.0
    etype_counts: dict[str, float] = defaultdict(float)
    for edge in edges:
        source = edge["source"].strip()
        target = edge["target"].strip()
        if source not in index_to_type or target not in index_to_type:
            continue
        count = float(edge["count"])
        matrix[type_index[index_to_type[source]], type_index[index_to_type[target]]] += count
        used_edges += 1
        used_synapses += count
        etype_counts[edge["etype"].strip()] += count
    return types, matrix, {
        "used_edges": used_edges,
        "used_synapses": used_synapses,
        "etype_synapses": dict(etype_counts),
    }


def flat_loss(values: np.ndarray) -> float:
    mu = float(np.mean(values))
    return float(np.sum((values - mu) ** 2))


def block_loss(values: np.ndarray, labels: list[str]) -> tuple[float, dict[str, float]]:
    labels_arr = np.asarray(labels)
    losses = 0.0
    means = {}
    for pre in sorted(set(labels)):
        for post in sorted(set(labels)):
            mask = np.outer(labels_arr == pre, labels_arr == post)
            block = values[mask]
            mu = float(np.mean(block)) if len(block) else 0.0
            means[f"{pre}->{post}"] = mu
            losses += float(np.sum((block - mu) ** 2))
    return losses, means


def permutation_gate(values: np.ndarray, labels: list[str], permutations: int) -> dict[str, object]:
    observed, means = block_loss(values, labels)
    baseline = flat_loss(values)
    rng = random.Random(RNG_SEED)
    shuffled = list(labels)
    hits = 0
    losses = []
    for _ in range(permutations):
        rng.shuffle(shuffled)
        loss, _ = block_loss(values, shuffled)
        losses.append(loss)
        if loss <= observed:
            hits += 1
    return {
        "flat_loss": baseline,
        "block_loss": observed,
        "block_over_flat": observed / max(baseline, 1e-12),
        "block_means": means,
        "random_loss_mean": mean(losses),
        "block_over_random_mean": observed / max(mean(losses), 1e-12),
        "permutations": permutations,
        "p_value_loss_le_observed": (hits + 1) / (permutations + 1),
    }


def class_flow(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
    class_fn,
) -> dict[str, object]:
    index_to_class = {
        row["index"].strip(): class_fn(clean_type(row["cell_type"])) for row in nodes
    }
    flow: dict[str, float] = defaultdict(float)
    total = 0.0
    for edge in edges:
        source = edge["source"].strip()
        target = edge["target"].strip()
        if source not in index_to_class or target not in index_to_class:
            continue
        count = float(edge["count"])
        flow[f"{index_to_class[source]}->{index_to_class[target]}"] += count
        total += count
    return {
        "total_synapses": total,
        "flows": dict(sorted(flow.items())),
        "fractions": {
            key: value / max(total, 1e-12)
            for key, value in sorted(flow.items())
        },
    }


def labels_from_nodes(
    nodes: list[dict[str, str]],
    type_order: list[str],
    label_name: str,
) -> list[str]:
    by_type_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nodes:
        by_type_rows[clean_type(row["cell_type"])].append(row)
    labels = []
    for cell_type in type_order:
        rows = by_type_rows[cell_type]
        if label_name == "primitive":
            labels.append(primitive_class(cell_type))
        elif label_name == "extended_memory":
            labels.append(functional_class(cell_type))
        elif label_name == "action_split":
            labels.append(action_split_class(cell_type))
        elif label_name == "sensory_modality":
            text = "; ".join(row.get("annotations", "") for row in rows)
            labels.append(sensory_modality_class(cell_type, text))
        elif label_name == "cell_type":
            labels.append(cell_type)
        elif label_name == "all_one":
            labels.append("all")
        else:
            raise ValueError(f"unknown label set: {label_name}")
    return labels


def compare_label_models(
    nodes: list[dict[str, str]],
    types: list[str],
    matrix: np.ndarray,
    permutations: int,
) -> dict[str, object]:
    out = {}
    for label_name in (
        "all_one",
        "primitive",
        "extended_memory",
        "action_split",
        "sensory_modality",
        "cell_type",
    ):
        labels = labels_from_nodes(nodes, types, label_name)
        gate = permutation_gate(matrix, labels, permutations)
        label_count = len(set(labels))
        # A small BIC-like score using number of block means as parameter count.
        n = matrix.size
        k = label_count * label_count
        rss = max(float(gate["block_loss"]), 1e-12)
        bic = n * float(np.log(rss / max(n, 1))) + k * float(np.log(max(n, 1)))
        saturated = k >= n or float(gate["block_loss"]) <= 1e-12
        out[label_name] = {
            "label_count": label_count,
            "parameter_count": k,
            "saturated": saturated,
            "gate": gate,
            "bic_like": bic,
        }
    eligible = {
        name: row for name, row in out.items() if not bool(row["saturated"])
    }
    best = min(eligible.items(), key=lambda item: float(item[1]["bic_like"]))
    best_loss = min(eligible.items(), key=lambda item: float(item[1]["gate"]["block_loss"]))
    return {
        "models": out,
        "best_bic_like_model_excluding_saturated": best[0],
        "best_loss_model_excluding_saturated": best_loss[0],
        "warning": (
            "BIC-like scores compare coarse block reconstructions only; they are "
            "not final biological proof. Saturated cell-type blocks are excluded "
            "from the best-model call."
        ),
    }


def node_summary(nodes: list[dict[str, str]]) -> dict[str, object]:
    type_counts = Counter(clean_type(row["cell_type"]) for row in nodes)
    class_counts = Counter(functional_class(clean_type(row["cell_type"])) for row in nodes)
    memory_count = sum(type_counts[name] for name in MEMORY_TYPES)
    return {
        "node_count": len(nodes),
        "cell_type_count": len(type_counts),
        "cell_type_counts": dict(type_counts.most_common()),
        "functional_class_counts": dict(class_counts.most_common()),
        "memory_node_count": memory_count,
        "memory_node_fraction": memory_count / max(len(nodes), 1),
    }


def next_step_indices(flow: dict[str, object]) -> dict[str, float]:
    fractions = flow["fractions"]  # type: ignore[assignment]
    def frac(key: str) -> float:
        return float(fractions.get(key, 0.0))

    memory_internal = frac("mushroom_body_memory->mushroom_body_memory")
    projection_to_memory = frac("projection_relay->mushroom_body_memory")
    memory_to_action = frac("mushroom_body_memory->descending_action")
    memory_to_lateral = frac("mushroom_body_memory->lateral_integration")
    lateral_to_memory = frac("lateral_integration->mushroom_body_memory")
    descending_to_memory = frac("descending_action->mushroom_body_memory")
    memory_loop = (
        memory_internal
        + projection_to_memory
        + memory_to_action
        + memory_to_lateral
        + lateral_to_memory
        + descending_to_memory
    )
    sensory_to_projection = frac("sensory_input->projection_relay")
    sensory_to_action = frac("sensory_input->descending_action")
    return {
        "memory_internal_fraction": memory_internal,
        "projection_to_memory_fraction": projection_to_memory,
        "memory_to_action_fraction": memory_to_action,
        "memory_to_lateral_fraction": memory_to_lateral,
        "lateral_to_memory_fraction": lateral_to_memory,
        "descending_to_memory_fraction": descending_to_memory,
        "memory_loop_fraction": memory_loop,
        "sensory_to_projection_fraction": sensory_to_projection,
        "sensory_to_action_fraction": sensory_to_action,
        "sensory_projection_over_direct_action": sensory_to_projection
        / max(sensory_to_action, 1e-12),
    }


def type_strengths(
    types: list[str],
    matrix: np.ndarray,
    labels: list[str],
) -> list[dict[str, object]]:
    total_out = np.sum(matrix, axis=1)
    total_in = np.sum(matrix, axis=0)
    out = []
    for idx, cell_type in enumerate(types):
        out.append(
            {
                "cell_type": cell_type,
                "class": labels[idx],
                "out_strength": float(total_out[idx]),
                "in_strength": float(total_in[idx]),
                "total_strength": float(total_out[idx] + total_in[idx]),
                "self_strength": float(matrix[idx, idx]),
            }
        )
    return sorted(out, key=lambda row: float(row["total_strength"]), reverse=True)


def memory_subsystem_indices(
    types: list[str],
    matrix: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    labels_arr = np.asarray(labels)
    memory = labels_arr == "mushroom_body_memory"
    non_memory = ~memory
    total = float(np.sum(matrix))
    memory_internal = float(np.sum(matrix[np.outer(memory, memory)]))
    memory_out = float(np.sum(matrix[np.outer(memory, non_memory)]))
    memory_in = float(np.sum(matrix[np.outer(non_memory, memory)]))
    non_memory_total = float(np.sum(matrix[np.outer(non_memory, non_memory)]))
    type_rows = type_strengths(types, matrix, labels)
    top_memory = [row for row in type_rows if row["class"] == "mushroom_body_memory"]
    return {
        "memory_internal_synapses": memory_internal,
        "memory_out_synapses": memory_out,
        "memory_in_synapses": memory_in,
        "non_memory_synapses": non_memory_total,
        "memory_internal_fraction": memory_internal / max(total, 1e-12),
        "memory_boundary_fraction": (memory_in + memory_out) / max(total, 1e-12),
        "memory_touched_fraction_matrix": (
            memory_internal + memory_in + memory_out
        )
        / max(total, 1e-12),
        "memory_internal_over_boundary": memory_internal / max(memory_in + memory_out, 1e-12),
        "top_memory_types_by_strength": top_memory,
        "top_all_types_by_strength": type_rows,
    }


def write_report(output: dict[str, object], path: Path) -> None:
    node = output["node_summary"]  # type: ignore[assignment]
    primitive = output["primitive_gate"]  # type: ignore[assignment]
    extended = output["extended_gate"]  # type: ignore[assignment]
    indices = output["next_step_indices"]  # type: ignore[assignment]
    memory_subsystem = output["memory_subsystem"]  # type: ignore[assignment]
    model_check = output["model_check"]  # type: ignore[assignment]
    lines = [
        "# Drosophila larva: 원시 신경계 다음 단계",
        "",
        "C. elegans의 weighted chemical L1/L2/L3 구조 다음에 무엇이 추가되는지 보기 위해 Drosophila larva connectome을 분석했다.",
        "",
        "## 핵심 가설",
        "",
        "다음 단계에서 새로 뚜렷해지는 것은 단순한 양 증가가 아니라, mushroom body 기반의 학습/기억/action-selection loop다.",
        "",
        "$$",
        "\\mathrm{primitive\\ control}",
        "\\rightarrow",
        "\\mathrm{primitive\\ control + memory/action\\ selection\\ loop}",
        "$$",
        "",
        "## 데이터",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| nodes | {node['node_count']} |",
        f"| cell types | {node['cell_type_count']} |",
        f"| memory nodes | {node['memory_node_count']} |",
        f"| memory node fraction | {node['memory_node_fraction']:.6f} |",
        f"| used edges | {output['edge_summary']['used_edges']} |",  # type: ignore[index]
        f"| used synapses | {output['edge_summary']['used_synapses']:.1f} |",  # type: ignore[index]
        "",
        "## 기능 class",
        "",
        "| class | nodes |",
        "|---|---:|",
    ]
    for name, count in node["functional_class_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## 모델 비교",
            "",
            "| model | block/flat | block/random mean | permutation p |",
            "|---|---:|---:|---:|",
            f"| primitive 3-class | {primitive['block_over_flat']:.6f} | {primitive['block_over_random_mean']:.6f} | {primitive['p_value_loss_le_observed']:.6f} |",
            f"| extended 5-class with mushroom body | {extended['block_over_flat']:.6f} | {extended['block_over_random_mean']:.6f} | {extended['p_value_loss_le_observed']:.6f} |",
            "",
            "extended 5-class는 primitive 3-class보다 손실을 낮추지만, permutation p 기준은 통과하지 못한다. 따라서 이것은 최종 게이트 통과가 아니라 다음 진화 단계 후보의 정량적 신호로 둔다.",
            "",
            "## 반례 점검: competing models",
            "",
            "| model | labels | params | block/flat | p | BIC-like | saturated |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, row in model_check["models"].items():
        gate = row["gate"]
        lines.append(
            f"| {name} | {row['label_count']} | {row['parameter_count']} | "
            f"{gate['block_over_flat']:.6f} | {gate['p_value_loss_le_observed']:.6f} | "
            f"{row['bic_like']:.3f} | {row['saturated']} |"
        )
    lines.extend(
        [
            "",
            f"포화모델을 제외한 BIC-like 최저 모델은 `{model_check['best_bic_like_model_excluding_saturated']}`이다.",
            f"포화모델을 제외한 순수 손실 최저 모델은 `{model_check['best_loss_model_excluding_saturated']}`이다.",
            "이 값들은 생물학적 최종 증명이 아니라, 우리 예상식이 competing explanation보다 얼마나 경제적인지 보는 반례 점검이다.",
            "",
            "## 새 loop 지표",
            "",
            "| 지표 | 값 |",
            "|---|---:|",
            f"| memory internal fraction | {indices['memory_internal_fraction']:.6f} |",
            f"| projection -> memory fraction | {indices['projection_to_memory_fraction']:.6f} |",
            f"| memory -> action fraction | {indices['memory_to_action_fraction']:.6f} |",
            f"| memory -> lateral fraction | {indices['memory_to_lateral_fraction']:.6f} |",
            f"| lateral -> memory fraction | {indices['lateral_to_memory_fraction']:.6f} |",
            f"| descending -> memory fraction | {indices['descending_to_memory_fraction']:.6f} |",
            f"| total memory-loop touched fraction | {indices['memory_loop_fraction']:.6f} |",
            f"| sensory -> projection / sensory -> action | {indices['sensory_projection_over_direct_action']:.6f} |",
            f"| memory internal / boundary | {memory_subsystem['memory_internal_over_boundary']:.6f} |",
            f"| matrix memory touched fraction | {memory_subsystem['memory_touched_fraction_matrix']:.6f} |",
            "",
            "## memory 계열 cell type strength",
            "",
            "| cell type | in | out | total | self |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in memory_subsystem["top_memory_types_by_strength"]:
        lines.append(
            f"| {row['cell_type']} | {row['in_strength']:.1f} | "
            f"{row['out_strength']:.1f} | {row['total_strength']:.1f} | "
            f"{row['self_strength']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- C. elegans 이후 단계에서 단순 감각-중간-운동 구조 위에 mushroom body 계열 memory/action-selection 회로가 후보로 나타난다.",
            "- 그러나 현재 cell-type block permutation은 통과하지 못했으므로, 이 단계는 '검증 완료'가 아니라 '후보 발견'이다.",
            "- 가장 강한 정량 신호는 mushroom-body 내부 synapse가 전체 synapse의 큰 비중을 차지하고, projection/lateral/action class와 반복적으로 연결된다는 점이다.",
            "- 따라서 지능으로 가는 다음 스텝 후보는 양 증가 자체가 아니라 학습 가능한 내부 상태 loop의 출현이다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    zip_path = ensure_zip(args.zip, force=args.force_download)
    nodes, edges = read_zip(zip_path)
    types, matrix, edge_summary = matrix_by_type(nodes, edges)
    primitive_labels = labels_from_nodes(nodes, types, "primitive")
    extended_labels = labels_from_nodes(nodes, types, "extended_memory")
    primitive_gate = permutation_gate(matrix, primitive_labels, args.permutations)
    extended_gate = permutation_gate(matrix, extended_labels, args.permutations)
    model_check = compare_label_models(nodes, types, matrix, args.permutations)
    output = {
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "zip": str(zip_path),
        "node_summary": node_summary(nodes),
        "cell_types": types,
        "edge_summary": edge_summary,
        "primitive_labels": dict(zip(types, primitive_labels)),
        "extended_labels": dict(zip(types, extended_labels)),
        "primitive_gate": primitive_gate,
        "extended_gate": extended_gate,
        "model_check": model_check,
        "memory_subsystem": memory_subsystem_indices(types, matrix, extended_labels),
        "extended_improvement_over_primitive": (
            primitive_gate["block_loss"] - extended_gate["block_loss"]
        )
        / max(primitive_gate["block_loss"], 1e-12),
        "primitive_flow": class_flow(nodes, edges, primitive_class),
        "extended_flow": class_flow(nodes, edges, functional_class),
        "criterion": "extended 5-class model beats primitive 3-class model and random labels",
        "passed": extended_gate["block_loss"] < primitive_gate["block_loss"]
        and extended_gate["block_loss"] < extended_gate["random_loss_mean"]
        and extended_gate["p_value_loss_le_observed"] < 0.05,
    }
    output["next_step_indices"] = next_step_indices(output["extended_flow"])  # type: ignore[arg-type]
    out_json = Path(__file__).with_name("drosophila_larva_next_step_results.json")
    out_md = Path(__file__).with_name("drosophila_larva_next_step_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    print("Drosophila larva next-step gate")
    print(f"  nodes                 = {output['node_summary']['node_count']}")
    print(f"  used_edges            = {edge_summary['used_edges']}")
    print(f"  used_synapses         = {edge_summary['used_synapses']:.1f}")
    print(f"  memory_node_fraction  = {output['node_summary']['memory_node_fraction']:.6f}")
    print(f"  primitive block/flat  = {primitive_gate['block_over_flat']:.6f}")
    print(f"  primitive p           = {primitive_gate['p_value_loss_le_observed']:.6f}")
    print(f"  extended block/flat   = {extended_gate['block_over_flat']:.6f}")
    print(f"  extended p            = {extended_gate['p_value_loss_le_observed']:.6f}")
    print(f"  improvement           = {output['extended_improvement_over_primitive']:.6f}")
    print(
        "  best BIC-like model   = "
        f"{model_check['best_bic_like_model_excluding_saturated']}"
    )
    print(
        "  best loss model       = "
        f"{model_check['best_loss_model_excluding_saturated']}"
    )
    print(
        "  memory_loop_fraction  = "
        f"{output['next_step_indices']['memory_loop_fraction']:.6f}"
    )
    print(
        "  memory_touched_matrix = "
        f"{output['memory_subsystem']['memory_touched_fraction_matrix']:.6f}"
    )
    print(
        "  memory internal/bound = "
        f"{output['memory_subsystem']['memory_internal_over_boundary']:.6f}"
    )
    print(f"  passed                = {output['passed']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
