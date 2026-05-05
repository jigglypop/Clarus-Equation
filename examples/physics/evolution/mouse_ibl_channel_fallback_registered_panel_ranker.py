"""Rank channel-fallback IBL/OpenAlyx sessions for the next mouse panel.

The innovation-to-behavior gate passed on the seeded 5-candidate panel.  The
next risk is panel bias: lab, subject, session date, and probe placement.  This
ranker scans the larger OpenAlyx channel-fallback bridge in chunks and selects
sessions that have the files needed to rerun:

    H_t = A H_{t-lag} + B X_t + C R_t + eps_t
    y_t = g(X_t, R_t, Hhat_t, eps_t)

It is metadata-only: it lists dataset names but does not download spike arrays.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mouse_ibl_larger_registered_panel_audit import RegisteredCandidate, candidate_from_eid
from mouse_ibl_neuropixels_audit import (
    CHANNEL_REGION_DATASET_TYPES,
    CORE_DATASET_TYPES,
    OPENALYX_URL,
    PROJECT,
)


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_channel_fallback_registered_panel_ranker_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_channel_fallback_registered_panel_ranker_report.md"
)


def diversity_select(
    candidates: list[RegisteredCandidate],
    limit: int,
    max_per_lab: int,
    max_per_subject: int,
) -> list[RegisteredCandidate]:
    selected: list[RegisteredCandidate] = []
    lab_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    for candidate in candidates:
        if lab_counts[candidate.lab] >= max_per_lab:
            continue
        if subject_counts[candidate.subject] >= max_per_subject:
            continue
        selected.append(candidate)
        lab_counts[candidate.lab] += 1
        subject_counts[candidate.subject] += 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_eids = {candidate.eid for candidate in selected}
        for candidate in candidates:
            if candidate.eid in selected_eids:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx channel-fallback registered panel ranker",
        "",
        "## target equation",
        "",
        "$$",
        "H_t=A H_{t-\\ell}+B X_t+C R_t+\\epsilon_t,",
        "\\qquad",
        "y_t=g(X_t,R_t,\\hat H_t,\\epsilon_t).",
        "$$",
        "",
        "This ranker does not train the model. It selects sessions where this gate can be rerun with channel-level region fallback.",
        "",
        "## scan",
        "",
        f"- OpenAlyx: `{output['openalyx_url']}`",
        f"- project: `{output['project']}`",
        f"- searched channel-fallback sessions: {output['searched_session_count']}",
        f"- scanned offset: {output['offset']}",
        f"- scanned limit: {output['max_sessions']}",
        f"- scanned sessions: {output['scanned_session_count']}",
        f"- ready sessions in chunk: {output['ready_session_count']}",
        f"- selected sessions: {output['selected_session_count']}",
        f"- panel ranker passed: `{output['panel_ranker_passed']}`",
        "",
        "## selected panel",
        "",
        "| rank | eid | session | lab | subject | probes | strict | channel | metrics | score |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, candidate in enumerate(output["selected_candidates"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    f"`{candidate['lab']}`",
                    f"`{candidate['subject']}`",
                    str(candidate["probe_count"]),
                    str(candidate["strict_probe_count"]),
                    str(candidate["channel_probe_count"]),
                    str(candidate["metrics_probe_count"]),
                    str(candidate["panel_score"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## chunk top candidates",
            "",
            "| rank | eid | session | lab | subject | probes | score |",
            "|---:|---|---|---|---|---:|---:|",
        ]
    )
    for idx, candidate in enumerate(output["top_candidates"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{candidate['eid']}`",
                    f"`{candidate['session_ref']}`",
                    f"`{candidate['lab']}`",
                    f"`{candidate['subject']}`",
                    str(candidate["probe_count"]),
                    str(candidate["panel_score"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## spread",
            "",
            "| item | count |",
            "|---|---:|",
        ]
    )
    for lab, count in output["selected_lab_counts"]:
        lines.append(f"| selected lab `{lab}` | {count} |")
    for subject, count in output["selected_subject_counts"]:
        lines.append(f"| selected subject `{subject}` | {count} |")
    lines.extend(
        [
            "",
            "## next",
            "",
            "- If the selected panel reaches the threshold, rerun the innovation-to-behavior gate on these sessions.",
            "- If not, scan the next chunk with a larger `--offset` or increase `--max-sessions`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=48)
    parser.add_argument("--select", type=int, default=12)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--max-per-lab", type=int, default=4)
    parser.add_argument("--max-per-subject", type=int, default=2)
    args = parser.parse_args()

    from one.api import ONE

    ONE.setup(base_url=OPENALYX_URL, silent=True)
    one = ONE(password="international")
    search_types = CORE_DATASET_TYPES + CHANNEL_REGION_DATASET_TYPES
    eids = one.search(query_type="remote", project=PROJECT, dataset_types=search_types)
    chunk = eids[args.offset : args.offset + args.max_sessions]

    candidates = []
    errors = []
    for eid in chunk:
        try:
            candidate = candidate_from_eid(one, eid)
        except Exception as exc:  # Metadata endpoint can occasionally miss a revision.
            errors.append({"eid": str(eid), "error": repr(exc)})
            continue
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.panel_score,
            item.channel_probe_count,
            item.strict_probe_count,
            item.metrics_probe_count,
            item.session_ref,
        ),
        reverse=True,
    )
    selected = diversity_select(
        candidates,
        args.select,
        args.max_per_lab,
        args.max_per_subject,
    )
    output = {
        "openalyx_url": OPENALYX_URL,
        "project": PROJECT,
        "searched_session_count": len(eids),
        "offset": args.offset,
        "max_sessions": args.max_sessions,
        "scanned_session_count": len(chunk),
        "ready_session_count": len(candidates),
        "selected_session_count": len(selected),
        "panel_ranker_passed": len(selected) >= args.select,
        "selection_threshold": args.select,
        "errors": errors,
        "selected_candidates": [asdict(candidate) for candidate in selected],
        "top_candidates": [asdict(candidate) for candidate in candidates[: args.top]],
        "selected_lab_counts": Counter(candidate.lab for candidate in selected).most_common(),
        "selected_subject_counts": Counter(
            candidate.subject for candidate in selected
        ).most_common(),
        "chunk_lab_counts": Counter(candidate.lab for candidate in candidates).most_common(),
        "chunk_subject_counts": Counter(
            candidate.subject for candidate in candidates
        ).most_common(),
    }
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(make_report(output))

    print("Mouse IBL/OpenAlyx channel-fallback registered panel ranker")
    print(f"  searched sessions = {len(eids)}")
    print(f"  scanned sessions  = {len(chunk)}")
    print(f"  ready sessions    = {len(candidates)}")
    print(f"  selected sessions = {len(selected)}")
    print(f"  panel_passed      = {output['panel_ranker_passed']}")
    if selected:
        print(f"  top selected      = {selected[0].session_ref}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
