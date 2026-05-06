"""Generate the paper-ready branch table for H0 source-role readout."""

from __future__ import annotations

from h0_cross_channel_branch_contrast_gate import rows


def branch_label(q_f: float) -> str:
    if q_f >= 0.75:
        return "local/high"
    if q_f <= 0.25:
        return "global/low"
    return "bridge"


def main() -> int:
    items = rows()
    items = sorted(items, key=lambda item: (str(item["expected"]), str(item["channel"])))

    print("# H0 Paper Figure Table Gate")
    print()
    print("| channel | observational family | source role | selector | branch | branch readout |")
    print("|---|---|---|---:|---|---:|")

    failed = 0
    local_count = 0
    global_count = 0
    for item in items:
        q_f = float(item["q_f"])
        branch = branch_label(q_f)
        expected = str(item["expected"])
        if expected == "local":
            local_count += 1
        elif expected == "global":
            global_count += 1
        ok = (
            (expected == "local" and branch == "local/high")
            or (expected == "global" and branch == "global/low")
        )
        if not ok:
            failed += 1
        print(
            f"| {item['channel']} | {item['family']} | {expected} | "
            f"{q_f:.6f} | {branch} | {float(item['h0_branch']):.6f} |"
        )

    print()
    print(f"local rows = {local_count}")
    print(f"global rows = {global_count}")
    print(f"total rows = {len(items)}")

    if local_count < 2 or global_count < 2:
        failed += 1
    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: paper-ready branch table is reproducible from source-role gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
