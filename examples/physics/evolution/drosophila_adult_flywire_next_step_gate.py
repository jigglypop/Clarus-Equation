"""Adult FlyWire next-step gate after primitive nervous systems.

This gate closes the weak larval Drosophila result with adult FlyWire-derived
connectivity. It asks whether the post-C. elegans equation needs a joint
celltype/action/memory term rather than only the primitive sensory-relay-action
grammar.

Data source:
    Betzel et al. Flywire Female Adult Fly Brain Drosophila Derivatives
    https://zenodo.org/records/18555170

The connectome matrix is neuron x neuron synapse count, stratified by
neurotransmitter. The gate uses W.TOT and the matching annotations.mat labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse
from scipy.io import loadmat


DATASET = "FlyWire FAFB adult female Drosophila derivatives / Betzel et al. 2026"
SOURCE_RECORD = "https://zenodo.org/records/18555170"
ANNOTATIONS_URL = "https://zenodo.org/records/18555170/files/annotations.mat?download=1"
CONNECTOME_URL = "https://zenodo.org/records/18555170/files/connectome.mat?download=1"
ANNOTATIONS_MD5 = "0d2be44229cd13bc2544e972397503ed"
CONNECTOME_MD5 = "a5f4bb8f12c12775a0806457e66cb148"
DEFAULT_DIR = Path("data/evolution/drosophila_adult_flywire")
RNG_SEED = 1729


SENSORY_CLASSES = {
    "gustatory",
    "hygrosensory",
    "mechanosensory",
    "ocellar",
    "olfactory",
    "thermosensory",
    "unknown_sensory",
}
PROJECTION_CLASSES = {"ALIN", "ALLN", "ALON", "ALPN", "TPN"}
CENTRAL_CLASSES = {
    "LHCENT",
    "LHLN",
    "bilateral",
    "clock",
    "mAL",
    "pars_intercerebralis",
    "pars_lateralis",
}
MEMORY_CLASSES = {"Kenyon_Cell", "MBIN", "MBON", "DAN"}
CENTRAL_COMPLEX_CLASSES = {"CX", "TuBu"}
VISUAL_CLASSES = {"visual", "optic_lobes"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(path: Path, url: str, expected_md5: str, *, force: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        urlretrieve(url, path)
    observed = md5(path)
    if observed != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {path}: {observed} != {expected_md5}")
    return path


def clean_name(value: object) -> str:
    return str(value).strip().strip('"')


def decode_label_field(labels, names, field: str) -> tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(getattr(labels, field), dtype=np.int64).ravel()
    table = [clean_name(value) for value in np.asarray(getattr(names, field)).ravel()]
    decoded = np.empty(codes.shape, dtype=object)
    for idx, code in enumerate(codes):
        decoded[idx] = table[code - 1] if 0 < code <= len(table) else ""
    return decoded, codes


def functional_class(super_class: str, cell_class: str) -> str:
    if cell_class in MEMORY_CLASSES:
        return "mushroom_body_memory"
    if cell_class in CENTRAL_COMPLEX_CLASSES:
        return "central_complex_action"
    if super_class in {"descending", "motor"}:
        return "descending_motor"
    if super_class == "ascending" or cell_class == "AN":
        return "ascending_body_input"
    if super_class == "sensory" or cell_class in SENSORY_CLASSES:
        return "sensory_input"
    if (
        super_class in {"optic", "visual_projection", "visual_centrifugal"}
        or cell_class in VISUAL_CLASSES
    ):
        return "visual_system"
    if cell_class in PROJECTION_CLASSES:
        return "projection_relay"
    if cell_class in CENTRAL_CLASSES or super_class == "central":
        return "central_integration"
    if super_class == "endocrine":
        return "homeostatic_modulatory"
    return "other"


def primitive_class(refined_class: str) -> str:
    if refined_class in {
        "sensory_input",
        "visual_system",
        "projection_relay",
        "ascending_body_input",
    }:
        return "input_relay"
    if refined_class == "descending_motor":
        return "action"
    return "integration"


def mode(values: np.ndarray) -> str:
    return Counter(values.tolist()).most_common(1)[0][0]


def load_annotations(path: Path) -> dict[str, object]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    labels = mat["labels"]
    names = mat["names"]
    super_names, _ = decode_label_field(labels, names, "super_class")
    class_names, _ = decode_label_field(labels, names, "class")
    hemibrain_names, hemibrain_codes = decode_label_field(labels, names, "hemibrain")
    neuron_refined = np.asarray(
        [functional_class(sc, cc) for sc, cc in zip(super_names, class_names)],
        dtype=object,
    )
    by_code: dict[int, list[int]] = defaultdict(list)
    for idx, code in enumerate(hemibrain_codes):
        if int(code) > 1:
            by_code[int(code)].append(idx)
    unit_codes = sorted(by_code)
    unit_names = []
    refined_labels = []
    primitive_labels = []
    super_labels = []
    class_labels = []
    neuron_counts = []
    for code in unit_codes:
        idxs = np.asarray(by_code[code], dtype=np.int64)
        refined = mode(neuron_refined[idxs])
        super_label = mode(super_names[idxs])
        class_label = mode(class_names[idxs])
        if not class_label.strip():
            class_label = refined
        unit_names.append(clean_name(hemibrain_names[idxs[0]]))
        refined_labels.append(refined)
        primitive_labels.append(primitive_class(refined))
        super_labels.append(super_label)
        class_labels.append(class_label)
        neuron_counts.append(int(len(idxs)))
    return {
        "neuron_count": int(len(hemibrain_codes)),
        "hemibrain_codes": hemibrain_codes,
        "unit_codes": np.asarray(unit_codes, dtype=np.int64),
        "unit_names": unit_names,
        "unit_neuron_counts": neuron_counts,
        "refined_labels": refined_labels,
        "primitive_labels": primitive_labels,
        "super_labels": super_labels,
        "class_labels": class_labels,
        "neuron_refined_counts": dict(Counter(neuron_refined.tolist()).most_common()),
        "unit_refined_counts": dict(Counter(refined_labels).most_common()),
    }


def load_total_connectome(path: Path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    total = mat["W"].TOT
    if not sparse.issparse(total):
        raise TypeError("Expected W.TOT to be a scipy sparse matrix")
    return total.tocoo()


def aggregate_to_units(connectome, hemibrain_codes: np.ndarray, unit_codes: np.ndarray):
    mapper = np.full(int(np.max(hemibrain_codes)) + 1, -1, dtype=np.int32)
    for unit_idx, code in enumerate(unit_codes):
        mapper[int(code)] = int(unit_idx)
    row = mapper[hemibrain_codes[connectome.row]]
    col = mapper[hemibrain_codes[connectome.col]]
    mask = (row >= 0) & (col >= 0)
    aggregated = sparse.coo_matrix(
        (connectome.data[mask], (row[mask], col[mask])),
        shape=(len(unit_codes), len(unit_codes)),
    ).tocsr()
    aggregated.sum_duplicates()
    return aggregated


def block_loss_sparse(matrix, labels: list[str]) -> tuple[float, float, np.ndarray, dict[int, str]]:
    labels_arr = np.asarray(labels, dtype=object)
    label_index = {value: idx for idx, value in enumerate(sorted(set(labels)))}
    label_ids = np.asarray([label_index[value] for value in labels_arr], dtype=np.int32)
    label_count = len(label_index)
    unit_count = len(labels)
    coo = matrix.tocoo()
    data = np.asarray(coo.data, dtype=np.float64)
    flat_sum = float(np.sum(data))
    flat_sumsq = float(np.sum(data * data))
    flat_loss = flat_sumsq - (flat_sum * flat_sum) / max(unit_count * unit_count, 1)
    block_ids = label_ids[coo.row] * label_count + label_ids[coo.col]
    block_sums = np.bincount(
        block_ids,
        weights=data,
        minlength=label_count * label_count,
    ).astype(np.float64)
    block_sumsq = np.bincount(
        block_ids,
        weights=data * data,
        minlength=label_count * label_count,
    ).astype(np.float64)
    label_sizes = np.bincount(label_ids, minlength=label_count).astype(np.float64)
    block_counts = np.outer(label_sizes, label_sizes).ravel()
    block_loss = float(
        np.sum(block_sumsq - block_sums * block_sums / np.maximum(block_counts, 1.0))
    )
    reverse = {idx: value for value, idx in label_index.items()}
    return block_loss, flat_loss, block_sums.reshape(label_count, label_count), reverse


def transformed_matrix(matrix, transform: str):
    out = matrix.copy()
    if transform == "raw":
        return out
    if transform == "log1p":
        out.data = np.log1p(out.data)
        return out
    raise ValueError(f"unknown transform: {transform}")


def permutation_gate(
    matrix,
    labels: list[str],
    *,
    permutations: int,
    transform: str,
) -> dict[str, object]:
    values = transformed_matrix(matrix, transform)
    observed, flat, _, _ = block_loss_sparse(values, labels)
    rng = random.Random(RNG_SEED)
    shuffled = list(labels)
    hits = 0
    losses = []
    for _ in range(permutations):
        rng.shuffle(shuffled)
        loss, _, _, _ = block_loss_sparse(values, shuffled)
        losses.append(loss)
        if loss <= observed:
            hits += 1
    random_mean = mean(losses)
    return {
        "label_count": len(set(labels)),
        "parameter_count": len(set(labels)) ** 2,
        "transform": transform,
        "flat_loss": flat,
        "block_loss": observed,
        "block_over_flat": observed / max(flat, 1e-12),
        "random_loss_mean": random_mean,
        "block_over_random_mean": observed / max(random_mean, 1e-12),
        "permutations": permutations,
        "p_value_loss_le_observed": (hits + 1) / (permutations + 1),
    }


def model_comparison(
    matrix,
    label_sets: dict[str, list[str]],
    *,
    permutations: int,
    transform: str,
) -> dict[str, object]:
    models = {}
    for name, labels in label_sets.items():
        gate = permutation_gate(
            matrix,
            labels,
            permutations=permutations,
            transform=transform,
        )
        n = matrix.shape[0] * matrix.shape[1]
        k = int(gate["parameter_count"])
        rss = max(float(gate["block_loss"]), 1e-12)
        bic = n * float(np.log(rss / max(n, 1))) + k * float(np.log(max(n, 1)))
        models[name] = {
            "gate": gate,
            "bic_like": bic,
            "saturated": k >= n or float(gate["block_loss"]) <= 1e-12,
        }
    eligible = {name: row for name, row in models.items() if not row["saturated"]}
    best_loss = min(
        eligible.items(),
        key=lambda item: float(item[1]["gate"]["block_loss"]),
    )[0]
    best_bic = min(eligible.items(), key=lambda item: float(item[1]["bic_like"]))[0]
    return {
        "models": models,
        "best_loss_model_excluding_saturated": best_loss,
        "best_bic_like_model_excluding_saturated": best_bic,
    }


def flow_fraction(flows: np.ndarray, reverse: dict[int, str], source: str, target: str) -> float:
    index = {value: key for key, value in reverse.items()}
    total = float(np.sum(flows))
    if source not in index or target not in index or total <= 0:
        return 0.0
    return float(flows[index[source], index[target]] / total)


def adult_loop_score(flows: np.ndarray, reverse: dict[int, str]) -> dict[str, float]:
    edges = [
        ("mushroom_body_memory", "mushroom_body_memory"),
        ("projection_relay", "mushroom_body_memory"),
        ("mushroom_body_memory", "projection_relay"),
        ("mushroom_body_memory", "central_complex_action"),
        ("central_complex_action", "mushroom_body_memory"),
        ("mushroom_body_memory", "descending_motor"),
        ("central_complex_action", "descending_motor"),
        ("descending_motor", "central_complex_action"),
    ]
    values = {f"{pre}->{post}": flow_fraction(flows, reverse, pre, post) for pre, post in edges}
    values["memory_action_loop_fraction"] = sum(values.values())
    values["memory_internal_fraction"] = values["mushroom_body_memory->mushroom_body_memory"]
    values["projection_to_memory_fraction"] = values["projection_relay->mushroom_body_memory"]
    values["memory_to_action_fraction"] = (
        values["mushroom_body_memory->central_complex_action"]
        + values["mushroom_body_memory->descending_motor"]
    )
    values["central_complex_to_descending_fraction"] = values[
        "central_complex_action->descending_motor"
    ]
    return values


def loop_permutation_gate(
    matrix,
    labels: list[str],
    *,
    permutations: int,
) -> dict[str, object]:
    _, _, flows, reverse = block_loss_sparse(matrix, labels)
    observed = adult_loop_score(flows, reverse)
    observed_score = observed["memory_action_loop_fraction"]
    rng = random.Random(RNG_SEED)
    shuffled = list(labels)
    hits = 0
    random_scores = []
    for _ in range(permutations):
        rng.shuffle(shuffled)
        _, _, perm_flows, perm_reverse = block_loss_sparse(matrix, shuffled)
        score = adult_loop_score(perm_flows, perm_reverse)["memory_action_loop_fraction"]
        random_scores.append(score)
        if score >= observed_score:
            hits += 1
    random_mean = mean(random_scores)
    return {
        "observed": observed,
        "random_mean_memory_action_loop_fraction": random_mean,
        "observed_over_random_mean": observed_score / max(random_mean, 1e-12),
        "permutations": permutations,
        "p_value_random_ge_observed": (hits + 1) / (permutations + 1),
    }


def top_unit_strengths(
    matrix,
    unit_names: list[str],
    labels: list[str],
    target_label: str,
    limit: int = 12,
) -> list[dict[str, object]]:
    out_strength = np.asarray(matrix.sum(axis=1)).ravel()
    in_strength = np.asarray(matrix.sum(axis=0)).ravel()
    rows = []
    for idx, label in enumerate(labels):
        if label != target_label:
            continue
        rows.append(
            {
                "unit": unit_names[idx],
                "label": label,
                "in_strength": float(in_strength[idx]),
                "out_strength": float(out_strength[idx]),
                "total_strength": float(in_strength[idx] + out_strength[idx]),
                "self_strength": float(matrix[idx, idx]),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_strength"]), reverse=True)[:limit]


def write_report(output: dict[str, object], path: Path) -> None:
    models = output["model_comparison"]["models"]  # type: ignore[index]
    loop = output["loop_gate"]  # type: ignore[assignment]
    lines = [
        "# Drosophila adult FlyWire: celltype/action/memory gate",
        "",
        "C. elegans의 weighted chemical routing 다음 단계가 무엇인지 adult FlyWire connectome으로 다시 점검했다.",
        "",
        "## 데이터",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| source neurons | {output['neuron_count']} |",
        f"| hemibrain-typed units | {output['unit_count']} |",
        f"| W.TOT nonzero edges | {output['connectome_nnz']} |",
        f"| W.TOT synapses | {output['connectome_synapses']:.1f} |",
        f"| kept unit-level synapses | {output['unit_synapses']:.1f} |",
        f"| kept fraction | {output['kept_synapse_fraction']:.6f} |",
        "",
        "자료는 FlyWire/Codex 파생 `connectome.mat`의 `W.TOT`와 같은 record의 `annotations.mat`를 사용했다.",
        "",
        "## refined functional class",
        "",
        "| class | units |",
        "|---|---:|",
    ]
    for label, count in output["unit_refined_counts"].items():  # type: ignore[index]
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## 모델 비교",
            "",
            "| model | labels | block/flat | block/random mean | p | BIC-like |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in models.items():
        gate = row["gate"]
        lines.append(
            f"| {name} | {gate['label_count']} | {gate['block_over_flat']:.6f} | "
            f"{gate['block_over_random_mean']:.6f} | "
            f"{gate['p_value_loss_le_observed']:.6f} | {row['bic_like']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"포화모델 제외 순수 손실 최저는 `{output['model_comparison']['best_loss_model_excluding_saturated']}`이다.",
            f"포화모델 제외 BIC-like 최저는 `{output['model_comparison']['best_bic_like_model_excluding_saturated']}`이다.",
            "",
            "## memory/action loop",
            "",
            "| 지표 | 값 |",
            "|---|---:|",
        ]
    )
    observed = loop["observed"]
    for key in (
        "memory_internal_fraction",
        "projection_to_memory_fraction",
        "memory_to_action_fraction",
        "central_complex_to_descending_fraction",
        "memory_action_loop_fraction",
    ):
        lines.append(f"| {key} | {observed[key]:.6f} |")
    lines.extend(
        [
            f"| random mean memory_action_loop_fraction | {loop['random_mean_memory_action_loop_fraction']:.6f} |",
            f"| observed/random mean | {loop['observed_over_random_mean']:.6f} |",
            f"| random >= observed p | {loop['p_value_random_ge_observed']:.6f} |",
            "",
            "## top memory/action units",
            "",
            "### mushroom body memory",
            "",
            "| unit | in | out | total | self |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in output["top_memory_units"]:  # type: ignore[index]
        lines.append(
            f"| {row['unit']} | {row['in_strength']:.1f} | {row['out_strength']:.1f} | "
            f"{row['total_strength']:.1f} | {row['self_strength']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### central complex/action",
            "",
            "| unit | in | out | total | self |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in output["top_cx_units"]:  # type: ignore[index]
        lines.append(
            f"| {row['unit']} | {row['in_strength']:.1f} | {row['out_strength']:.1f} | "
            f"{row['total_strength']:.1f} | {row['self_strength']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- adult refined model gate: `{output['adult_refined_gate_pass']}`",
            f"- memory/action loop gate: `{output['memory_action_loop_gate_pass']}`",
            f"- closed: `{output['closed']}`",
            "",
            "Drosophila larva에서는 memory/action/celltype 항이 후보였지만 strict block gate가 약했다.",
            "adult FlyWire에서는 refined functional block이 random label보다 유의하게 좋고, mushroom body와 central-complex/descending action loop가 random보다 강하다.",
            "따라서 이 단계는 memory 단독 추가가 아니라 celltype/action/memory 공동 분화 항으로 닫는다.",
            "",
            "$$",
            "P_{n+1}",
            "=",
            "\\Pi[",
            "\\rho P_n",
            "+\\gamma\\mathcal L(W_{\\mathrm{chem}})P_n",
            "+\\sum_d a_{d,n}U_d",
            "+D_{\\mathrm{celltype}}c_n",
            "+A_{\\mathrm{action}}b_n",
            "+M_{\\mathrm{MB/CX}}m_n",
            "+H(q_n-q_*)",
            "]",
            "$$",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--transform", choices=("log1p", "raw"), default="log1p")
    args = parser.parse_args()

    annotations_path = ensure_file(
        args.data_dir / "annotations.mat",
        ANNOTATIONS_URL,
        ANNOTATIONS_MD5,
        force=args.force_download,
    )
    connectome_path = ensure_file(
        args.data_dir / "connectome.mat",
        CONNECTOME_URL,
        CONNECTOME_MD5,
        force=args.force_download,
    )

    annotations = load_annotations(annotations_path)
    connectome = load_total_connectome(connectome_path)
    unit_matrix = aggregate_to_units(
        connectome,
        annotations["hemibrain_codes"],  # type: ignore[arg-type]
        annotations["unit_codes"],  # type: ignore[arg-type]
    )

    refined_labels = annotations["refined_labels"]  # type: ignore[assignment]
    label_sets = {
        "all_one": ["all"] * len(refined_labels),
        "primitive": annotations["primitive_labels"],
        "adult_refined_celltype_action_memory": refined_labels,
        "super_class": annotations["super_labels"],
        "class_or_refined": annotations["class_labels"],
    }
    comparison = model_comparison(
        unit_matrix,
        label_sets,  # type: ignore[arg-type]
        permutations=args.permutations,
        transform=args.transform,
    )
    loop_gate = loop_permutation_gate(
        unit_matrix,
        refined_labels,  # type: ignore[arg-type]
        permutations=args.permutations,
    )
    primitive_gate = comparison["models"]["primitive"]["gate"]  # type: ignore[index]
    adult_gate = comparison["models"]["adult_refined_celltype_action_memory"]["gate"]  # type: ignore[index]
    adult_refined_gate_pass = (
        float(adult_gate["block_over_flat"]) < 1.0
        and float(adult_gate["p_value_loss_le_observed"]) < 0.05
        and float(adult_gate["block_loss"]) < float(primitive_gate["block_loss"])
    )
    memory_action_loop_gate_pass = (
        float(loop_gate["observed_over_random_mean"]) > 2.0
        and float(loop_gate["p_value_random_ge_observed"]) < 0.05
    )
    output = {
        "dataset": DATASET,
        "source_record": SOURCE_RECORD,
        "annotations_url": ANNOTATIONS_URL,
        "connectome_url": CONNECTOME_URL,
        "annotations_path": str(annotations_path),
        "connectome_path": str(connectome_path),
        "permutations": args.permutations,
        "transform": args.transform,
        "neuron_count": annotations["neuron_count"],
        "unit_count": int(unit_matrix.shape[0]),
        "connectome_nnz": int(connectome.nnz),
        "connectome_synapses": float(connectome.data.sum()),
        "unit_nnz": int(unit_matrix.nnz),
        "unit_synapses": float(unit_matrix.data.sum()),
        "kept_synapse_fraction": float(unit_matrix.data.sum() / max(connectome.data.sum(), 1)),
        "neuron_refined_counts": annotations["neuron_refined_counts"],
        "unit_refined_counts": annotations["unit_refined_counts"],
        "model_comparison": comparison,
        "loop_gate": loop_gate,
        "top_memory_units": top_unit_strengths(
            unit_matrix,
            annotations["unit_names"],  # type: ignore[arg-type]
            refined_labels,  # type: ignore[arg-type]
            "mushroom_body_memory",
        ),
        "top_cx_units": top_unit_strengths(
            unit_matrix,
            annotations["unit_names"],  # type: ignore[arg-type]
            refined_labels,  # type: ignore[arg-type]
            "central_complex_action",
        ),
        "criterion": (
            "adult refined celltype/action/memory block beats primitive and random labels; "
            "memory/action loop fraction is >2x random and p<0.05"
        ),
        "adult_refined_gate_pass": adult_refined_gate_pass,
        "memory_action_loop_gate_pass": memory_action_loop_gate_pass,
        "closed": adult_refined_gate_pass and memory_action_loop_gate_pass,
    }

    out_json = Path(__file__).with_name("drosophila_adult_flywire_next_step_results.json")
    out_md = Path(__file__).with_name("drosophila_adult_flywire_next_step_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    print("Drosophila adult FlyWire next-step gate")
    print(f"  neurons                       = {output['neuron_count']}")
    print(f"  hemibrain-typed units          = {output['unit_count']}")
    print(f"  W.TOT nnz                      = {output['connectome_nnz']}")
    print(f"  W.TOT synapses                 = {output['connectome_synapses']:.1f}")
    print(f"  kept unit synapses             = {output['unit_synapses']:.1f}")
    print(f"  kept fraction                  = {output['kept_synapse_fraction']:.6f}")
    print(
        "  primitive block/flat           = "
        f"{primitive_gate['block_over_flat']:.6f}"
    )
    print(
        "  adult refined block/flat       = "
        f"{adult_gate['block_over_flat']:.6f}"
    )
    print(
        "  adult refined p                = "
        f"{adult_gate['p_value_loss_le_observed']:.6f}"
    )
    print(
        "  memory/action loop fraction    = "
        f"{loop_gate['observed']['memory_action_loop_fraction']:.6f}"
    )
    print(
        "  loop observed/random mean      = "
        f"{loop_gate['observed_over_random_mean']:.6f}"
    )
    print(
        "  loop p random>=observed        = "
        f"{loop_gate['p_value_random_ge_observed']:.6f}"
    )
    print(f"  closed                         = {output['closed']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()

