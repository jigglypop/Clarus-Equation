"""Zebrafish activity to behavior-frame association gate.

This uses the Figure 8/g chunk from the public Figshare dataset
"All-optical interrogation of brain-wide activity in freely swimming larval
zebrafish".

The file contains:
    e2        region x frame neural activity matrix
    FrameBout behavior bout frames
    t_s       non-laser baseline frames used by the original analysis

This is still not a full continuous tail/stage movement decoder. It asks the
next stricter question:

    Can low-rank neural activity distinguish behavior bout frames from matched
    baseline frames on held-out data?
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from zebrafish_freely_swimming_activity_gate import load_mat_numeric


DEFAULT_MAT = Path("data/evolution/zebrafish/freely_swimming/figure8/g/Corr_FrameActive3_230830.mat")
RNG_SEED = 1729
RESULT_JSON = Path(__file__).with_name("zebrafish_activity_behavior_frame_gate_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_activity_behavior_frame_report.md")


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / (len(pos) * len(neg))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    pred = scores >= 0.5
    pos = labels == 1
    neg = labels == 0
    tpr = float(np.mean(pred[pos] == 1)) if np.any(pos) else float("nan")
    tnr = float(np.mean(pred[neg] == 0)) if np.any(neg) else float("nan")
    return 0.5 * (tpr + tnr)


def fit_lowrank_ridge(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, rank: int, ridge: float
) -> np.ndarray:
    mu = np.mean(train_x, axis=0, keepdims=True)
    sd = np.std(train_x, axis=0, keepdims=True)
    train_z = np.divide(train_x - mu, sd, out=np.zeros_like(train_x), where=sd > 0)
    test_z = np.divide(test_x - mu, sd, out=np.zeros_like(test_x), where=sd > 0)

    u, s, vh = np.linalg.svd(train_z, full_matrices=False)
    k = min(rank, len(s))
    train_low = u[:, :k] * s[:k]
    test_low = test_z @ vh[:k, :].T

    design = np.column_stack([np.ones(train_low.shape[0]), train_low])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coeff = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    test_design = np.column_stack([np.ones(test_low.shape[0]), test_low])
    return test_design @ coeff


def split_indices(pos_n: int, neg_n: int, holdout_fraction: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pos = list(range(pos_n))
    neg = list(range(pos_n, pos_n + neg_n))
    rng.shuffle(pos)
    rng.shuffle(neg)
    hold_pos = max(1, int(round(pos_n * holdout_fraction)))
    hold_neg = max(1, int(round(neg_n * holdout_fraction)))
    test = np.asarray(pos[:hold_pos] + neg[:hold_neg], dtype=int)
    train = np.asarray(pos[hold_pos:] + neg[hold_neg:], dtype=int)
    return train, test


def evaluate(
    mat_path: Path,
    *,
    rank: int,
    repeats: int,
    permutations: int,
    holdout_fraction: float,
    ridge: float,
) -> dict[str, object]:
    data = load_mat_numeric(mat_path)
    e2 = data["e2"]
    frame_bout = data["FrameBout"].astype(int).ravel() - 1
    baseline_pool = data["t_s"].astype(int).ravel() - 1

    frame_bout = frame_bout[(frame_bout >= 0) & (frame_bout < e2.shape[1])]
    baseline_pool = baseline_pool[(baseline_pool >= 0) & (baseline_pool < e2.shape[1])]
    baseline_pool = np.setdiff1d(baseline_pool, frame_bout, assume_unique=False)

    rng = random.Random(RNG_SEED)
    repeat_rows = []
    perm_aucs = []
    for repeat in range(repeats):
        neg = np.asarray(rng.sample(list(baseline_pool), len(frame_bout)), dtype=int)
        frames = np.concatenate([frame_bout, neg])
        labels = np.concatenate([np.ones(len(frame_bout)), np.zeros(len(neg))])
        x = e2[:, frames].T.astype(float)

        train_idx, test_idx = split_indices(len(frame_bout), len(neg), holdout_fraction, rng)
        scores = fit_lowrank_ridge(x[train_idx], labels[train_idx], x[test_idx], rank, ridge)
        observed_auc = auc_score(labels[test_idx], scores)
        observed_bacc = balanced_accuracy(labels[test_idx], scores)
        repeat_rows.append(
            {
                "repeat": repeat,
                "auc": float(observed_auc),
                "balanced_accuracy": float(observed_bacc),
            }
        )

        # Use the same low-rank test scores and permute held-out labels. This
        # tests whether the neural scores retain behavior-label information.
        for _ in range(permutations):
            shuffled = labels[test_idx].copy()
            rng.shuffle(shuffled)
            perm_aucs.append(auc_score(shuffled, scores))

    observed_aucs = np.asarray([row["auc"] for row in repeat_rows], dtype=float)
    observed_bacc = np.asarray([row["balanced_accuracy"] for row in repeat_rows], dtype=float)
    perm_aucs_arr = np.asarray(perm_aucs, dtype=float)
    p_value = float((np.sum(perm_aucs_arr >= np.mean(observed_aucs)) + 1) / (len(perm_aucs_arr) + 1))

    return {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "chunk": "figure8/g/Corr_FrameActive3_230830.mat",
        "mat_path": str(mat_path),
        "activity_shape_region_by_frame": list(e2.shape),
        "bout_frame_count": int(len(frame_bout)),
        "baseline_pool_count": int(len(baseline_pool)),
        "rank": int(rank),
        "repeats": int(repeats),
        "holdout_fraction": float(holdout_fraction),
        "mean_auc": float(np.mean(observed_aucs)),
        "std_auc": float(np.std(observed_aucs)),
        "mean_balanced_accuracy": float(np.mean(observed_bacc)),
        "std_balanced_accuracy": float(np.std(observed_bacc)),
        "permutation_auc_mean": float(np.mean(perm_aucs_arr)),
        "p_value_auc_ge_observed_mean": p_value,
        "repeat_rows": repeat_rows,
        "passed": bool(np.mean(observed_aucs) > 0.75 and np.mean(observed_bacc) > 0.65 and p_value < 0.001),
        "caveat": "Behavior-frame association, not full continuous tail/stage movement decoding.",
    }


def write_report(output: dict[str, object], path: Path) -> None:
    lines = [
        "# Zebrafish activity to behavior-frame gate",
        "",
        "Figshare figure8/g chunk의 `e2` neural activity와 `FrameBout` 행동 bout frame을 사용했다.",
        "",
        "이 검증은 연속 tail/stage movement decoding은 아니고, neural activity가 행동 bout frame과 baseline frame을 구분하는지 보는 association gate다.",
        "",
        "## 결과",
        "",
        f"- activity shape region x frame: {output['activity_shape_region_by_frame']}",
        f"- bout frames: {output['bout_frame_count']}",
        f"- baseline pool frames: {output['baseline_pool_count']}",
        f"- mean AUC: {output['mean_auc']:.6f} ± {output['std_auc']:.6f}",
        f"- mean balanced accuracy: {output['mean_balanced_accuracy']:.6f} ± {output['std_balanced_accuracy']:.6f}",
        f"- permutation AUC mean: {output['permutation_auc_mean']:.6f}",
        f"- p: {output['p_value_auc_ge_observed_mean']:.6f}",
        f"- pass: {output['passed']}",
        "",
        "## 해석",
        "",
        "- 저차원 neural activity만으로 행동 bout frame과 baseline frame을 holdout에서 구분할 수 있다면 activity-behavior 결합이 보인다는 뜻이다.",
        "- 이 결과는 perturbation->behavior보다 한 단계 더 자연 activity 쪽에 가깝지만, 아직 연속적인 tail/stage movement 예측은 아니다.",
        "- 다음 최종 gate는 frame별 tail/stage speed, heading, turn angle을 neural trace와 시간 정렬해서 직접 예측하는 것이다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()

    output = evaluate(
        args.mat,
        rank=args.rank,
        repeats=args.repeats,
        permutations=args.permutations,
        holdout_fraction=args.holdout_fraction,
        ridge=args.ridge,
    )
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)

    print("Zebrafish activity to behavior-frame gate")
    print(f"  activity={output['activity_shape_region_by_frame']}")
    print(f"  bout_frames={output['bout_frame_count']}")
    print(f"  mean_auc={output['mean_auc']:.6f}")
    print(f"  mean_balanced_accuracy={output['mean_balanced_accuracy']:.6f}")
    print(f"  p={output['p_value_auc_ge_observed_mean']:.6f}")
    print(f"  passed={output['passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
