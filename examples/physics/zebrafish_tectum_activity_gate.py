"""Zebrafish tectum spontaneous activity gate.

This is the first activity-data step after connectome-only primitive systems.
It uses a small public figshare dataset of larval zebrafish optic tectum
calcium activity. There is no behavior in this dataset, so the gate asks a
narrower question:

    1. Do detected assemblies explain spontaneous neural correlation structure
       better than random assembly membership?
    2. Does a low-rank recurrent state predict P_{n+1} better than a mean-only
       baseline?

The assembly CSV files contain one row per assembly with one-based neuron
indices. Assemblies overlap, so this script evaluates co-assembly at the neuron
pair level instead of forcing each neuron into a single label.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path
from statistics import mean
from urllib.request import urlretrieve

import numpy as np


SOURCE_URL = "https://ndownloader.figshare.com/files/12732374"
DEFAULT_ZIP = Path("data/evolution/zebrafish/tectum_spontaneous/data.zip")
RNG_SEED = 1729


def ensure_zip(path: Path, *, force: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        urlretrieve(SOURCE_URL, path)
    return path


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[list[str]]:
    with zf.open(name) as handle:
        text = (line.decode("utf-8-sig").strip() for line in handle)
        return list(csv.reader(text))


def numeric_matrix(rows: list[list[str]]) -> np.ndarray:
    parsed = []
    for row in rows:
        if not row:
            continue
        try:
            parsed.append([float(value) for value in row if value != ""])
        except ValueError:
            continue
    return np.asarray(parsed, dtype=np.float64)


def numeric_row_vectors(rows: list[list[str]]) -> list[list[int]]:
    vectors = []
    for row in rows:
        values = []
        for item in row:
            try:
                values.append(int(float(item)))
            except ValueError:
                pass
        if values:
            vectors.append(values)
    return vectors


def load_fish(zf: zipfile.ZipFile, fish_dir: str) -> dict[str, object]:
    prefix = f"data/{fish_dir}/{fish_dir}"
    activity = numeric_matrix(read_csv_from_zip(zf, f"{prefix}_activity_matrix.csv"))
    assemblies = numeric_row_vectors(read_csv_from_zip(zf, f"{prefix}_assembly_assignments.csv"))
    coords = numeric_matrix(read_csv_from_zip(zf, f"{prefix}_cell_coordinates.csv"))
    # The public MATLAB plotting script treats activity_matrix as cells x time.
    if coords.size and activity.shape[0] != coords.shape[0] and activity.shape[1] == coords.shape[0]:
        activity = activity.T
    return {
        "fish_id": fish_dir,
        "activity": activity,
        "assemblies": assemblies,
        "coordinates": coords,
    }


def fish_dirs(zf: zipfile.ZipFile) -> list[str]:
    out = []
    for name in zf.namelist():
        if name.startswith("data/zf_") and name.endswith("_activity_matrix.csv"):
            out.append(name.split("/")[1])
    return sorted(set(out))


def corr_matrix(activity: np.ndarray) -> np.ndarray:
    centered = activity - np.mean(activity, axis=1, keepdims=True)
    std = np.std(centered, axis=1, keepdims=True)
    z = np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)
    corr = np.corrcoef(z)
    np.fill_diagonal(corr, np.nan)
    return corr


def assembly_membership(assemblies: list[list[int]], cell_count: int) -> np.ndarray:
    membership = np.zeros((cell_count, len(assemblies)), dtype=bool)
    for assembly_idx, assembly in enumerate(assemblies):
        for one_based_cell in assembly:
            cell_idx = one_based_cell - 1
            if 0 <= cell_idx < cell_count:
                membership[cell_idx, assembly_idx] = True
    return membership


def shared_assembly_matrix(membership: np.ndarray) -> np.ndarray:
    shared = (membership.astype(np.int16) @ membership.astype(np.int16).T) > 0
    np.fill_diagonal(shared, False)
    return shared


def flat_loss(values: np.ndarray) -> float:
    block = values[np.isfinite(values)]
    mu = float(np.nanmean(block))
    return float(np.nansum((block - mu) ** 2))


def coassembly_loss(values: np.ndarray, shared: np.ndarray) -> tuple[float, dict[str, float]]:
    finite = np.isfinite(values)
    losses = 0.0
    means = {}
    for name, mask in {
        "shared": shared & finite,
        "not_shared": ~shared & finite,
    }.items():
        block = values[mask]
        mu = float(np.nanmean(block)) if len(block) else 0.0
        means[name] = mu
        losses += float(np.nansum((block - mu) ** 2))
    return losses, means


def permutation_coassembly_gate(
    values: np.ndarray, membership: np.ndarray, permutations: int
) -> dict[str, object]:
    shared = shared_assembly_matrix(membership)
    observed, means = coassembly_loss(values, shared)
    baseline = flat_loss(values)
    rng = random.Random(RNG_SEED)
    hits = 0
    losses = []
    cell_count = membership.shape[0]
    for _ in range(permutations):
        order = list(range(cell_count))
        rng.shuffle(order)
        shuffled_shared = shared_assembly_matrix(membership[order, :])
        loss, _ = coassembly_loss(values, shuffled_shared)
        losses.append(loss)
        if loss <= observed:
            hits += 1
    offdiag = ~np.eye(shared.shape[0], dtype=bool)
    return {
        "flat_loss": baseline,
        "coassembly_block_loss": observed,
        "coassembly_over_flat": observed / max(baseline, 1e-12),
        "random_loss_mean": mean(losses),
        "coassembly_over_random_mean": observed / max(mean(losses), 1e-12),
        "p_value_loss_le_observed": (hits + 1) / (permutations + 1),
        "block_means": means,
        "shared_pair_fraction": float(np.mean(shared[offdiag])),
    }


def recurrent_prediction_gate(activity: np.ndarray, rank: int) -> dict[str, float]:
    y = activity[:, 1:].T
    x = activity[:, :-1].T
    mean_next = np.mean(y, axis=0, keepdims=True)
    baseline = float(np.mean((y - mean_next) ** 2))
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    k = min(rank, len(s))
    x_low = u[:, :k] * s[:k]
    design = np.column_stack([np.ones(x_low.shape[0]), x_low])
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ coeff
    model = float(np.mean((y - pred) ** 2))
    return {
        "rank": k,
        "baseline_mse": baseline,
        "lowrank_recurrent_mse": model,
        "model_over_baseline": model / max(baseline, 1e-12),
        "r2_vs_baseline": 1.0 - model / max(baseline, 1e-12),
    }


def evaluate_fish(fish: dict[str, object], permutations: int, rank: int) -> dict[str, object]:
    activity = fish["activity"]  # type: ignore[assignment]
    assemblies = fish["assemblies"]  # type: ignore[assignment]
    membership = assembly_membership(assemblies, activity.shape[0])
    corr = corr_matrix(activity)
    block_gate = permutation_coassembly_gate(corr, membership, permutations)
    pred_gate = recurrent_prediction_gate(activity, rank)
    return {
        "fish_id": fish["fish_id"],
        "cell_count": int(activity.shape[0]),
        "time_count": int(activity.shape[1]),
        "assembly_count": len(assemblies),
        "assembly_covered_cells": int(np.sum(np.any(membership, axis=1))),
        "assembly_memberships": int(np.sum(membership)),
        "block_gate": block_gate,
        "prediction_gate": pred_gate,
        "passed": block_gate["p_value_loss_le_observed"] < 0.05
        and pred_gate["model_over_baseline"] < 1.0,
    }


def write_report(output: dict[str, object], path: Path) -> None:
    lines = [
        "# Zebrafish tectum spontaneous activity gate",
        "",
        "공개 zebrafish optic tectum calcium activity 자료로 assembly 공유 구조와 저차원 recurrent 예측을 점검했다.",
        "",
        "이 자료에는 행동이 없으므로 whole-brain behavior gate가 아니라 activity-only pilot이다.",
        "",
        "## 결과",
        "",
        "| fish | cells | time | assemblies | covered | coassembly/flat | coassembly p | recurrent/baseline | R2 | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["fish"]:  # type: ignore[index]
        block = row["block_gate"]
        pred = row["prediction_gate"]
        lines.append(
            f"| {row['fish_id']} | {row['cell_count']} | {row['time_count']} | "
            f"{row['assembly_count']} | {row['assembly_covered_cells']} | "
            f"{block['coassembly_over_flat']:.6f} | "
            f"{block['p_value_loss_le_observed']:.6f} | "
            f"{pred['model_over_baseline']:.6f} | {pred['r2_vs_baseline']:.6f} | "
            f"{row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 통과하면 connectome-only 단계를 넘어 실제 calcium activity에서 assembly/recurrent state 구조가 보인다는 뜻이다.",
            "- assembly CSV는 셀별 단일 라벨이 아니라 assembly별 뉴런 인덱스 목록이며, 일부 뉴런은 여러 assembly에 겹친다.",
            "- 따라서 검증 단위는 단일 라벨 블록이 아니라 두 뉴런이 assembly를 공유하는지 여부다.",
            "- 행동 자료가 없으므로 stimulus-action 방정식은 아직 아니고, 척추동물 국소 회로의 폐쇄 동역학 후보만 검증한다.",
            "- 다음은 whole-brain + behavior zebrafish 자료로 넘어가야 한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    zip_path = ensure_zip(args.zip, force=args.force_download)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for fish_dir in fish_dirs(zf):
            rows.append(evaluate_fish(load_fish(zf, fish_dir), args.permutations, args.rank))
    output = {
        "dataset": "Calcium imaging of spontaneous activity in larval zebrafish tectum",
        "source_url": SOURCE_URL,
        "zip": str(zip_path),
        "fish": rows,
        "passed_all": all(row["passed"] for row in rows),
        "caveat": "activity-only optic tectum dataset; no behavior and not whole-brain",
    }
    out_json = Path(__file__).with_name("zebrafish_tectum_activity_gate_results.json")
    out_md = Path(__file__).with_name("zebrafish_tectum_activity_report.md")
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, out_md)

    print("Zebrafish tectum spontaneous activity gate")
    for row in rows:
        block = row["block_gate"]
        pred = row["prediction_gate"]
        print(
            f"  {row['fish_id']}: cells={row['cell_count']}, time={row['time_count']}, "
            f"coassembly/flat={block['coassembly_over_flat']:.6f}, "
            f"p={block['p_value_loss_le_observed']:.6f}, "
            f"recurrent/base={pred['model_over_baseline']:.6f}, "
            f"passed={row['passed']}"
        )
    print(f"  passed_all = {output['passed_all']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
