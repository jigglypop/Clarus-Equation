"""Threshold robustness for cross-channel H0 branch classification.

The cross-channel contrast uses a simple local/high vs global/low classifier.
This gate checks that the result is not an artifact of one hand-picked
threshold by sweeping increasingly strict classification thresholds.
"""

from __future__ import annotations

from h0_cross_channel_branch_contrast_gate import rows


THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def classify_with_threshold(q_f: float, threshold: float) -> str:
    if q_f >= threshold:
        return "local/high"
    if q_f <= 1.0 - threshold:
        return "global/low"
    return "bridge"


def main() -> int:
    items = rows()

    print("# H0 Cross-Channel Threshold Robustness Gate")
    print()
    print("| threshold | correct / total | bridge count | status |")
    print("|---:|---:|---:|---|")

    failed = 0
    total = len(items)
    for threshold in THRESHOLDS:
        correct = 0
        bridge = 0
        for item in items:
            expected = str(item["expected"])
            q_f = float(item["q_f"])
            classified = classify_with_threshold(q_f, threshold)
            if classified == "bridge":
                bridge += 1
            ok = (
                (expected == "local" and classified == "local/high")
                or (expected == "global" and classified == "global/low")
            )
            correct += 1 if ok else 0
        status = "PASS" if correct == total and bridge == 0 else "FAIL"
        if status != "PASS":
            failed += 1
        print(f"| {threshold:.2f} | {correct}/{total} | {bridge} | {status} |")

    q_values = [float(item["q_f"]) for item in items]
    local_q = [float(item["q_f"]) for item in items if item["expected"] == "local"]
    global_q = [float(item["q_f"]) for item in items if item["expected"] == "global"]
    margin_to_middle = min(abs(q - 0.5) for q in q_values)
    local_floor = min(local_q)
    global_ceiling = max(global_q)

    print()
    print(f"min local q_F = {local_floor:.6f}")
    print(f"max global q_F = {global_ceiling:.6f}")
    print(f"minimum distance from bridge midpoint = {margin_to_middle:.6f}")

    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: cross-channel branch split is robust to broad classification thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
