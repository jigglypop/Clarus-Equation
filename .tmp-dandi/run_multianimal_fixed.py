from __future__ import annotations

import json

import analyze_bridge_multianimal as m


def main() -> None:
    sessions = [m.session(asset) for asset in m.ASSETS]
    summary = m.summarize(sessions)
    result = {
        "status": "COMPLETE",
        "bin_s": m.BIN,
        "rank": m.RANK,
        "ridge": m.RIDGE,
        "lags_ms": [lag * m.BIN * 1000 for lag in m.LAGS],
        "sessions": sessions,
        "summary": summary,
    }
    (m.OUT / "bridge_multianimal_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        "# DANDI 001695 fixed multi-animal bridge test",
        "",
        f"Fixed rank `{m.RANK}`, ridge `{m.RIDGE}`, bin `{m.BIN}s`; one behavior session each from M01/M02/M03/M05.",
        "",
        "| path/contrast | lag ms | mean ΔNLPD | positive animals | exact one-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        if "contrast" in row:
            name = row["contrast"]
        else:
            name = f"{row['source']}→{row['target']}"
        lines.append(
            f"| {name} | {row['lag_ms']:.0f} | {row['mean_delta']:.5f} | "
            f"{row['n_positive']}/{len(row['animal_values'])} | "
            f"{row['exact_one_sided_signflip_p']:.4f} |"
        )
    text = "\n".join(lines) + "\n"
    (m.OUT / "bridge_multianimal_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
