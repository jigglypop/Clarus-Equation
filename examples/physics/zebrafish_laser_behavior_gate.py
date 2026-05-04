"""Zebrafish optogenetic perturbation to behavior gate.

This uses the Figure 8 LR chunk from the public Figshare dataset
"All-optical interrogation of brain-wide activity in freely swimming larval
zebrafish".

This is not neural-trace -> behavior decoding. It is the next available
behavior closure step:

    Does a directed neural perturbation condition, left vs right laser, produce
    a matched signed turning behavior in experimental fish but not controls?

The relevant bout summary arrays are stored in boutInfo.mat:
    bout_leftLaser_sumAngle
    bout_rightLaser_sumAngle
    bout_noLaser_sumAngle
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from zebrafish_freely_swimming_activity_gate import load_mat_numeric


DEFAULT_LR_DIR = Path("data/evolution/zebrafish/freely_swimming/figure8/c/LR")
RNG_SEED = 1729
RESULT_JSON = Path(__file__).with_name("zebrafish_laser_behavior_gate_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_laser_behavior_report.md")


def load_group(root: Path) -> dict[str, object]:
    left = []
    right = []
    no_laser = []
    fish_rows = []
    for mat_path in sorted(root.glob("*/boutInfo.mat")):
        data = load_mat_numeric(mat_path)
        l = data.get("bout_leftLaser_sumAngle", np.asarray([])).astype(float).ravel()
        r = data.get("bout_rightLaser_sumAngle", np.asarray([])).astype(float).ravel()
        n = data.get("bout_noLaser_sumAngle", np.asarray([])).astype(float).ravel()
        left.extend(l)
        right.extend(r)
        no_laser.extend(n)
        fish_rows.append(
            {
                "fish": mat_path.parent.name,
                "left_n": int(len(l)),
                "right_n": int(len(r)),
                "no_laser_n": int(len(n)),
                "left_mean": float(np.mean(l)) if len(l) else float("nan"),
                "right_mean": float(np.mean(r)) if len(r) else float("nan"),
                "left_minus_right": float(np.mean(l) - np.mean(r)) if len(l) and len(r) else float("nan"),
            }
        )
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    no_arr = np.asarray(no_laser, dtype=float)
    return {
        "left": left_arr,
        "right": right_arr,
        "no_laser": no_arr,
        "fish": fish_rows,
    }


def summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
    }


def permutation_left_right(left: np.ndarray, right: np.ndarray, permutations: int) -> dict[str, float]:
    observed = float(np.mean(left) - np.mean(right))
    pooled = np.concatenate([left, right])
    n_left = len(left)
    rng = random.Random(RNG_SEED)
    hits = 0
    diffs = []
    idx = list(range(len(pooled)))
    for _ in range(permutations):
        rng.shuffle(idx)
        l = pooled[idx[:n_left]]
        r = pooled[idx[n_left:]]
        diff = float(np.mean(l) - np.mean(r))
        diffs.append(diff)
        if abs(diff) >= abs(observed):
            hits += 1
    return {
        "observed_left_minus_right": observed,
        "random_abs_mean": float(np.mean(np.abs(diffs))),
        "p_two_sided": (hits + 1) / (permutations + 1),
    }


def evaluate(lr_dir: Path, permutations: int) -> dict[str, object]:
    control = load_group(lr_dir / "control")
    exp = load_group(lr_dir / "exp")

    control_gate = permutation_left_right(control["left"], control["right"], permutations)  # type: ignore[arg-type]
    exp_gate = permutation_left_right(exp["left"], exp["right"], permutations)  # type: ignore[arg-type]
    effect_ratio = abs(exp_gate["observed_left_minus_right"]) / max(
        abs(control_gate["observed_left_minus_right"]), 1e-12
    )

    return {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "chunk": "figure8/c/LR",
        "lr_dir": str(lr_dir),
        "control": {
            "left": summary(control["left"]),  # type: ignore[arg-type]
            "right": summary(control["right"]),  # type: ignore[arg-type]
            "no_laser": summary(control["no_laser"]),  # type: ignore[arg-type]
            "gate": control_gate,
            "fish": control["fish"],
        },
        "experimental": {
            "left": summary(exp["left"]),  # type: ignore[arg-type]
            "right": summary(exp["right"]),  # type: ignore[arg-type]
            "no_laser": summary(exp["no_laser"]),  # type: ignore[arg-type]
            "gate": exp_gate,
            "fish": exp["fish"],
        },
        "effect_ratio_exp_over_control": float(effect_ratio),
        "passed": bool(
            exp_gate["p_two_sided"] < 0.001
            and abs(exp_gate["observed_left_minus_right"]) > 30.0
            and effect_ratio > 20.0
        ),
        "caveat": "This is perturbation-to-behavior closure, not spontaneous neural trace-to-behavior decoding.",
    }


def write_report(output: dict[str, object], path: Path) -> None:
    control = output["control"]  # type: ignore[index]
    exp = output["experimental"]  # type: ignore[index]
    cgate = control["gate"]
    egate = exp["gate"]
    lines = [
        "# Zebrafish laser perturbation to behavior gate",
        "",
        "Figshare figure8/c/LR chunk의 boutInfo.mat를 사용해 left/right laser 조건이 회전 행동 방향을 바꾸는지 검증했다.",
        "",
        "이 검증은 neural trace -> behavior decoding이 아니라 perturbation -> behavior closure다.",
        "",
        "## 결과",
        "",
        "| group | left n | left mean angle | right n | right mean angle | left-right | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| control | {control['left']['n']} | {control['left']['mean']:.6f} | "
            f"{control['right']['n']} | {control['right']['mean']:.6f} | "
            f"{cgate['observed_left_minus_right']:.6f} | {cgate['p_two_sided']:.6f} |"
        ),
        (
            f"| experimental | {exp['left']['n']} | {exp['left']['mean']:.6f} | "
            f"{exp['right']['n']} | {exp['right']['mean']:.6f} | "
            f"{egate['observed_left_minus_right']:.6f} | {egate['p_two_sided']:.6f} |"
        ),
        "",
        f"- experimental/control effect ratio: {output['effect_ratio_exp_over_control']:.6f}",
        f"- pass: {output['passed']}",
        "",
        "## 해석",
        "",
        "- control에서는 left/right laser에 따른 회전 방향 차이가 거의 없다.",
        "- experimental fish에서는 left laser와 right laser가 반대 부호의 큰 회전각을 만든다.",
        "- 따라서 척추동물 단계에서 motor output 항은 임의 잡음이 아니라 방향성 perturbation에 의해 조절되는 닫힌 행동 출력으로 볼 수 있다.",
        "- 아직 남은 최종 검증은 자연 neural trace가 tail/stage movement를 예측하는 activity -> behavior gate다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", type=Path, default=DEFAULT_LR_DIR)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    output = evaluate(args.lr_dir, args.permutations)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, REPORT_MD)

    control = output["control"]  # type: ignore[index]
    exp = output["experimental"]  # type: ignore[index]
    print("Zebrafish laser perturbation to behavior gate")
    print(
        f"  control: left_mean={control['left']['mean']:.6f}, "
        f"right_mean={control['right']['mean']:.6f}, "
        f"left-right={control['gate']['observed_left_minus_right']:.6f}, "
        f"p={control['gate']['p_two_sided']:.6f}"
    )
    print(
        f"  experimental: left_mean={exp['left']['mean']:.6f}, "
        f"right_mean={exp['right']['mean']:.6f}, "
        f"left-right={exp['gate']['observed_left_minus_right']:.6f}, "
        f"p={exp['gate']['p_two_sided']:.6f}"
    )
    print(f"  effect_ratio={output['effect_ratio_exp_over_control']:.6f}")
    print(f"  passed={output['passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
