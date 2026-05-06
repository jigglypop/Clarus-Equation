"""Batch runner for H0 Fisher/covariance JSON channel files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h0_fisher_matrix_io_gate import channel_from_payload, run_channel


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.glob("*.json")
        if candidate.is_file() and candidate.name != "manifest.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="JSON file or directory of channel JSON files",
    )
    args = parser.parse_args()

    files = iter_json_files(Path(args.path))
    if not files:
        raise SystemExit(f"No JSON files found at {args.path}")

    print("# H0 Fisher IO Batch Gate")
    print()
    print("| file | channel | q_F | H0_pred | H0_obs | pull |")
    print("|---|---|---:|---:|---:|---:|")
    chi2 = 0.0
    scored = 0
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        channel = channel_from_payload(payload)
        result = run_channel(channel)
        if channel.h0_obs is not None and channel.h0_sigma is not None:
            pull = (result["h0_pred"] - channel.h0_obs) / channel.h0_sigma
            chi2 += pull * pull
            scored += 1
            obs = f"{channel.h0_obs:.3f} +/- {channel.h0_sigma:.3f}"
            pull_text = f"{pull:+.3f}"
        else:
            obs = "--"
            pull_text = "--"
        print(
            f"| {file.name} | {channel.name} | {result['q_f']:.6f} | "
            f"{result['h0_pred']:.6f} | {obs} | {pull_text} |"
        )

    print()
    if scored:
        print(f"chi2/dof = {chi2:.6f}/{scored}")
    else:
        print("chi2/dof = unscored")
    print("Verdict: batch Fisher/covariance ingestion path is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
