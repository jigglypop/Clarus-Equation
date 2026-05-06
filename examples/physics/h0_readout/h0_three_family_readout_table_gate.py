"""Generate a three-family H0 readout table including the GW bridge branch."""

from __future__ import annotations

from h0_cross_channel_branch_contrast_gate import rows as endpoint_rows
from h0_fisher_matrix_io_gate import channel_from_payload, run_channel
from h0_gw_standard_siren_bridge_gate import branch_payload as gw_payload


def branch_label(q_f: float) -> str:
    if q_f >= 0.75:
        return "local/high"
    if q_f <= 0.25:
        return "global/low"
    return "bridge/intermediate"


def rows() -> list[dict[str, object]]:
    out = list(endpoint_rows())
    payload = gw_payload()
    channel = channel_from_payload(payload)
    result = run_channel(channel)
    out.append(
        {
            "channel": "GW170817 bright siren",
            "family": "standard siren",
            "expected": "bridge",
            "q_f": result["q_f"],
            "h0_branch": result["h0_pred"],
            "classification": branch_label(result["q_f"]),
        }
    )
    return out


def main() -> int:
    items = sorted(rows(), key=lambda item: (float(item["q_f"]), str(item["channel"])))
    counts = {"global/low": 0, "bridge/intermediate": 0, "local/high": 0}
    failed = 0

    print("# H0 Three-Family Readout Table Gate")
    print()
    print("| channel | observational family | source role | selector | readout family | branch readout |")
    print("|---|---|---|---:|---|---:|")

    for item in items:
        q_f = float(item["q_f"])
        label = branch_label(q_f)
        expected = str(item["expected"])
        counts[label] += 1
        ok = (
            (expected == "global" and label == "global/low")
            or (expected == "local" and label == "local/high")
            or (expected == "bridge" and label == "bridge/intermediate")
        )
        if not ok:
            failed += 1
        print(
            f"| {item['channel']} | {item['family']} | {expected} | "
            f"{q_f:.6f} | {label} | {float(item['h0_branch']):.6f} |"
        )

    print()
    print(f"global/low rows = {counts['global/low']}")
    print(f"bridge/intermediate rows = {counts['bridge/intermediate']}")
    print(f"local/high rows = {counts['local/high']}")
    print(f"total rows = {len(items)}")

    if min(counts.values()) < 1:
        failed += 1
    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: H0 readout table spans global, bridge, and local source-role families.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
