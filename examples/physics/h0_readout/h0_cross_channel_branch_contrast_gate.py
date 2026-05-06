"""Cross-channel branch contrast for H0 source-role readout.

This gate is the paper-figure companion to the individual source gates. It
collects TDCOSMO, DESI BAO, and Pantheon+SH0ES into one branch table and checks
that source roles split the channels into local/high-side and global/low-side
readouts before any joint H0 refit is attempted.
"""

from __future__ import annotations

import json
from pathlib import Path

from h0_bao_global_readout_gate import branch_payload as bao_payload
from h0_cmb_planck_covariance_adapter_gate import branch_payload as cmb_payload
from h0_fisher_matrix_io_gate import channel_from_payload, run_channel
from h0_pantheon_shoes_local_readout_gate import branch_payload as shoes_payload


TDCOSMO = Path(__file__).with_name("h0_fisher_io_examples")
TDCOSMO_FILES = [
    ("TDCOSMO-only", "tdcosmo_only_alpha_free_om_covariance.json", "local"),
    ("TDCOSMO+IFU", "tdcosmo_ifu_covariance.json", "local"),
    ("TDCOSMO+SLACS", "tdcosmo_slacs_covariance.json", "global"),
    ("TDCOSMO+SLACS+IFU", "tdcosmo_slacs_ifu_covariance.json", "global"),
]


def tdcosmo_payload(file_name: str) -> dict[str, object]:
    return json.loads((TDCOSMO / file_name).read_text(encoding="utf-8"))


def classify(q_f: float) -> str:
    if q_f >= 0.75:
        return "local/high"
    if q_f <= 0.25:
        return "global/low"
    return "bridge"


def rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for label, file_name, expected in TDCOSMO_FILES:
        payload = tdcosmo_payload(file_name)
        channel = channel_from_payload(payload)
        result = run_channel(channel)
        out.append(
            {
                "channel": label,
                "family": "time-delay lensing",
                "expected": expected,
                "q_f": result["q_f"],
                "h0_branch": result["h0_pred"],
                "classification": classify(result["q_f"]),
            }
        )

    for label, family, expected, payload in [
        ("DESI BAO", "standard ruler", "global", bao_payload()),
        ("Planck CMB", "early acoustic horizon", "global", cmb_payload()),
        ("Pantheon+SH0ES", "distance ladder", "local", shoes_payload()),
    ]:
        channel = channel_from_payload(payload)
        result = run_channel(channel)
        out.append(
            {
                "channel": label,
                "family": family,
                "expected": expected,
                "q_f": result["q_f"],
                "h0_branch": result["h0_pred"],
                "classification": classify(result["q_f"]),
            }
        )
    return out


def main() -> int:
    items = rows()

    print("# H0 Cross-Channel Branch Contrast Gate")
    print()
    print("| channel | family | expected source role | q_F | branch H0 | classified readout | status |")
    print("|---|---|---|---:|---:|---|---|")

    failed = 0
    q_by_expected = {"local": [], "global": []}
    for item in items:
        expected = str(item["expected"])
        q_f = float(item["q_f"])
        classified = str(item["classification"])
        q_by_expected[expected].append(q_f)
        ok = (
            (expected == "local" and classified == "local/high")
            or (expected == "global" and classified == "global/low")
        )
        if not ok:
            failed += 1
        print(
            f"| {item['channel']} | {item['family']} | {expected} | "
            f"{q_f:.6f} | {float(item['h0_branch']):.6f} | {classified} | {'PASS' if ok else 'FAIL'} |"
        )

    local_mean = sum(q_by_expected["local"]) / len(q_by_expected["local"])
    global_mean = sum(q_by_expected["global"]) / len(q_by_expected["global"])
    separation = local_mean - global_mean

    print()
    print(f"local family mean q_F = {local_mean:.6f}")
    print(f"global family mean q_F = {global_mean:.6f}")
    print(f"cross-channel separation = {separation:.6f}")

    if separation < 0.75:
        failed += 1
    if failed:
        raise SystemExit(1)

    print()
    print("Verdict: source roles split independent H0 channels into local/high and global/low readouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
