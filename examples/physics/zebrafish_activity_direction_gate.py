"""Zebrafish activity to left/right direction gate.

This uses the Figure 8/f chunk from the public Figshare dataset
"All-optical interrogation of brain-wide activity in freely swimming larval
zebrafish".

The previous gates showed:
    perturbation -> signed turning behavior
    neural activity -> behavior bout frame

This gate joins those two pieces more tightly by asking:

    Can neural activity windows distinguish left-laser trials from right-laser
    trials on leave-one-trial-out validation?

Because these are laser-evoked trials, this is directional perturbation/activity
closure, not spontaneous continuous movement decoding.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from zebrafish_activity_behavior_frame_gate import auc_score, balanced_accuracy, fit_lowrank_ridge
from zebrafish_freely_swimming_activity_gate import load_mat_numeric


DEFAULT_MAT = Path("data/evolution/zebrafish/freely_swimming/figure8/f/newPropa2.mat")
RNG_SEED = 1729
RESULT_JSON = Path(__file__).with_name("zebrafish_activity_direction_gate_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_activity_direction_report.md")


def trial_features(e2: np.ndarray, starts: np.ndarray, window: int, pre_baseline: int) -> np.ndarray:
    rows = []
    frame_count = e2.shape[1]
    for one_based_start in starts.astype(int).ravel():
        start = one_based_start - 1
        lo = max(0, start)
        hi = min(frame_count, start + window)
        if hi <= lo:
            continue
        response = np.mean(e2[:, lo:hi], axis=1)
        if pre_baseline > 0 and start - pre_baseline >= 0:
            baseline = np.mean(e2[:, start - pre_baseline : start], axis=1)
            response = response - baseline
        rows.append(response)
    return np.asarray(rows, dtype=float)


def loo_scores(x: np.ndarray, y: np.ndarray, rank: int, ridge: float) -> np.ndarray:
    scores = np.zeros(len(y), dtype=float)
    for idx in range(len(y)):
        train = np.arange(len(y)) != idx
        test = np.asarray([idx])
        scores[test] = fit_lowrank_ridge(x[train], y[train], x[test], rank, ridge)
    return scores


def evaluate(mat_path: Path, *, window: int, pre_baseline: int, rank: int, ridge: float, permutations: int) -> dict[str, object]:
    data = load_mat_numeric(mat_path)
    e2 = data["e2"]
    left = trial_features(e2, data["LeftLS"], window, pre_baseline)
    right = trial_features(e2, data["RightLS"], window, pre_baseline)
    x = np.vstack([left, right])
    y = np.concatenate([np.ones(len(left)), np.zeros(len(right))])

    observed_scores = loo_scores(x, y, rank, ridge)
    observed_auc = auc_score(y, observed_scores)
    observed_bacc = balanced_accuracy(y, observed_scores)

    rng = random.Random(RNG_SEED)
    perm_aucs = []
    labels = y.copy()
    for _ in range(permutations):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        scores = loo_scores(x, shuffled, rank, ridge)
        perm_aucs.append(auc_score(shuffled, scores))
    perm_arr = np.asarray(perm_aucs, dtype=float)
    p_value = float((np.sum(perm_arr >= observed_auc) + 1) / (len(perm_arr) + 1))

    return {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "chunk": "figure8/f/newPropa2.mat",
        "mat_path": str(mat_path),
        "activity_shape_region_by_frame": list(e2.shape),
        "left_trial_count": int(len(left)),
        "right_trial_count": int(len(right)),
        "window": int(window),
        "pre_baseline": int(pre_baseline),
        "rank": int(rank),
        "auc": float(observed_auc),
        "balanced_accuracy": float(observed_bacc),
        "permutation_auc_mean": float(np.mean(perm_arr)),
        "p_value_auc_ge_observed": p_value,
        "left_score_mean": float(np.mean(observed_scores[y == 1])),
        "right_score_mean": float(np.mean(observed_scores[y == 0])),
        "passed": bool(observed_auc > 0.75 and observed_bacc > 0.70 and p_value < 0.01),
        "caveat": "Laser-evoked left/right activity direction, not spontaneous continuous movement decoding.",
    }


def write_report(output: dict[str, object], path: Path) -> None:
    lines = [
        "# Zebrafish activity to direction gate",
        "",
        "Figshare figure8/f chunk의 `e2`, `LeftLS`, `RightLS`를 사용해 neural activity window가 left/right 조건을 구분하는지 봤다.",
        "",
        "이 검증은 laser-evoked 방향성 activity gate이며, 아직 spontaneous continuous movement decoding은 아니다.",
        "",
        "## 결과",
        "",
        f"- activity shape region x frame: {output['activity_shape_region_by_frame']}",
        f"- left trials: {output['left_trial_count']}",
        f"- right trials: {output['right_trial_count']}",
        f"- window frames: {output['window']}",
        f"- pre-baseline frames: {output['pre_baseline']}",
        f"- AUC: {output['auc']:.6f}",
        f"- balanced accuracy: {output['balanced_accuracy']:.6f}",
        f"- permutation AUC mean: {output['permutation_auc_mean']:.6f}",
        f"- p: {output['p_value_auc_ge_observed']:.6f}",
        f"- pass: {output['passed']}",
        "",
        "## 해석",
        "",
        "- leave-one-trial-out에서 left/right laser trial을 neural activity만으로 구분한다.",
        "- 이미 left/right laser가 반대 방향 회전 행동을 만든다는 gate가 통과했으므로, 이 결과는 activity-direction-output 결합을 보강한다.",
        "- 표본 수가 21 trial로 작으므로 최종 결론은 continuous movement decoding에서 다시 확인해야 한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--pre-baseline", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()

    output = evaluate(
        args.mat,
        window=args.window,
        pre_baseline=args.pre_baseline,
        rank=args.rank,
        ridge=args.ridge,
        permutations=args.permutations,
    )
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)

    print("Zebrafish activity to direction gate")
    print(f"  trials L/R={output['left_trial_count']}/{output['right_trial_count']}")
    print(f"  auc={output['auc']:.6f}")
    print(f"  balanced_accuracy={output['balanced_accuracy']:.6f}")
    print(f"  p={output['p_value_auc_ge_observed']:.6f}")
    print(f"  passed={output['passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
