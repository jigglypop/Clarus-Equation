from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from dandi.dandiapi import DandiAPIClient
from scipy.stats import binomtest

from analyze_bridge import DANDISET, VERSION, analyze_session, load_session

OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)
MIN_UNITS = 5
MAX_SUBJECTS = 6


def discover_paths() -> dict[str, list[str]]:
    by_subject: dict[str, list[str]] = defaultdict(list)
    with DandiAPIClient() as client:
        ds = client.get_dandiset(DANDISET, VERSION)
        for asset in ds.get_assets():
            path = str(asset.path)
            if not path.endswith("behavior+ecephys.nwb"):
                continue
            m = re.match(r"(sub-[^/]+)/", path)
            if m:
                by_subject[m.group(1)].append(path)
    return {s: sorted(paths) for s, paths in sorted(by_subject.items())}


def sign_summary(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "n_positive": 0, "mean": None, "median": None, "sign_p_one_sided": None}
    npos = int(np.sum(arr > 0))
    return {
        "n": int(len(arr)),
        "n_positive": npos,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sign_p_one_sided": float(binomtest(npos, len(arr), 0.5, alternative="greater").pvalue),
    }


def aggregate(sessions: list[dict]) -> dict:
    directions = ["CA3_to_CA1", "CA1_to_CA3", "CA1_to_RSC", "RSC_to_CA1"]
    lags = ["20ms", "40ms"]
    out = {}
    for direction in directions:
        out[direction] = {}
        for lag in lags:
            bridge = []
            random_diff = []
            shifted_diff = []
            full_diff = []
            for s in sessions:
                sm = s["directions"][direction][lag]["summary"]
                b = sm["bridge_rank3"]["nlpd_improvement_vs_baseline"]
                bridge.append(b)
                random_diff.append(b - sm["random_rank3"]["nlpd_improvement_vs_baseline"])
                shifted_diff.append(b - sm["shifted_rank3"]["nlpd_improvement_vs_baseline"])
                full_diff.append(b - sm["full_source"]["nlpd_improvement_vs_baseline"])
            out[direction][lag] = {
                "bridge_vs_baseline": sign_summary(bridge),
                "bridge_minus_random": sign_summary(random_diff),
                "bridge_minus_shifted": sign_summary(shifted_diff),
                "bridge_minus_full": sign_summary(full_diff),
                "per_subject": {
                    s["path"].split("/")[0]: {
                        "path": s["path"],
                        "bridge": bridge[i],
                        "bridge_minus_random": random_diff[i],
                        "bridge_minus_shifted": shifted_diff[i],
                        "bridge_minus_full": full_diff[i],
                    }
                    for i, s in enumerate(sessions)
                },
            }
    return out


def report(results: dict) -> str:
    lines = [
        "# DANDI 001695 multisubject conditional bridge audit",
        "",
        f"Deterministic selection: lexicographically first valid behavior+ecephys session per subject; minimum {MIN_UNITS} kept units in CA3, CA1 and RSC.",
        "",
        "| direction | lag | animals | bridge positive | median ΔNLPD | sign p | bridge>random | p | bridge>shifted | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, lags in results["aggregate"].items():
        for lag, row in lags.items():
            b = row["bridge_vs_baseline"]
            r = row["bridge_minus_random"]
            s = row["bridge_minus_shifted"]
            lines.append(
                f"| {direction} | {lag} | {b['n']} | {b['n_positive']} | {b['median']:.5f} | {b['sign_p_one_sided']:.5f} | "
                f"{r['n_positive']}/{r['n']} | {r['sign_p_one_sided']:.5f} | {s['n_positive']}/{s['n']} | {s['sign_p_one_sided']:.5f} |"
            )
    if results["skipped"]:
        lines += ["", "## Skipped assets", ""]
        for x in results["skipped"]:
            lines.append(f"- `{x['path']}`: {x['reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    candidates = discover_paths()
    sessions = []
    skipped = []
    selected_paths = []
    for subject, paths in list(candidates.items())[:MAX_SUBJECTS]:
        chosen = None
        for path in paths:
            try:
                print(f"try {subject} {path}", flush=True)
                raw = load_session(path)
                if min(raw["unit_counts_kept"].values()) < MIN_UNITS:
                    raise ValueError(f"insufficient kept units {raw['unit_counts_kept']}")
                chosen = analyze_session(raw)
                selected_paths.append(path)
                sessions.append(chosen)
                print(f"selected {path}", flush=True)
                break
            except Exception as exc:
                skipped.append({"subject": subject, "path": path, "reason": f"{type(exc).__name__}: {exc}"})
                print(f"skip {path}: {exc}", flush=True)
        if chosen is None:
            print(f"no valid session for {subject}", flush=True)

    results = {
        "status": "COMPLETE",
        "dandiset": DANDISET,
        "version": VERSION,
        "candidate_paths": candidates,
        "selected_paths": selected_paths,
        "sessions": sessions,
        "skipped": skipped,
        "aggregate": aggregate(sessions),
        "caveat": "One deterministically selected session per subject; fixed pilot hyperparameters; discovery, not preregistered confirmation.",
    }
    (OUT / "multisubject_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    text = report(results)
    (OUT / "multisubject_report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
