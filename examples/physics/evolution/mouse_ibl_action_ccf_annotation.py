"""Annotate action-subspace CCF ids with atlas acronyms and names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("mouse_ibl_action_subspace_mechanism_results.json")
RESULT_JSON = Path(__file__).with_name("mouse_ibl_action_ccf_annotation_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_action_ccf_annotation_report.md")


def parse_ccf_id(region: str) -> int | None:
    prefix = "ccf_id:"
    if not region.startswith(prefix):
        return None
    try:
        return int(region[len(prefix) :])
    except ValueError:
        return None


def collect_ccf_ids(results: dict[str, Any]) -> list[int]:
    ccf_ids = set()
    for key in ("target_region_summary", "target_feature_summary"):
        for row in results.get(key, []):
            ccf_id = parse_ccf_id(str(row.get("region", "")))
            if ccf_id is not None:
                ccf_ids.add(ccf_id)
    return sorted(ccf_ids)


def atlas_annotations(ccf_ids: list[int]) -> dict[str, dict[str, Any]]:
    try:
        import numpy as np
        from iblatlas.atlas import BrainRegions
    except ImportError as exc:
        raise SystemExit(
            "This annotation step requires iblatlas. Run with: "
            "uv run --no-project --with iblatlas python "
            "examples/physics/evolution/mouse_ibl_action_ccf_annotation.py"
        ) from exc

    brain_regions = BrainRegions()
    annotations: dict[str, dict[str, Any]] = {}
    for ccf_id in ccf_ids:
        indices = np.where(brain_regions.id == ccf_id)[0]
        if len(indices) == 0:
            annotations[str(ccf_id)] = {
                "ccf_id": ccf_id,
                "acronym": "UNKNOWN",
                "name": "UNKNOWN",
                "found": False,
            }
            continue
        index = int(indices[0])
        annotations[str(ccf_id)] = {
            "ccf_id": ccf_id,
            "acronym": str(brain_regions.acronym[index]),
            "name": str(brain_regions.name[index]),
            "found": True,
        }
    return annotations


def annotate_region_rows(
    rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        out = dict(row)
        ccf_id = parse_ccf_id(str(row.get("region", "")))
        if ccf_id is not None:
            ann = annotations.get(str(ccf_id), {})
            out["ccf_id"] = ccf_id
            out["acronym"] = ann.get("acronym", "UNKNOWN")
            out["name"] = ann.get("name", "UNKNOWN")
            out["annotation_found"] = bool(ann.get("found", False))
        else:
            out["ccf_id"] = None
            out["acronym"] = str(row.get("region", "UNKNOWN"))
            out["name"] = str(row.get("region", "UNKNOWN"))
            out["annotation_found"] = False
        annotated.append(out)
    return annotated


def top_shared_regions(annotated_regions: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in annotated_regions:
        by_target.setdefault(str(row["target"]), []).append(row)
    for rows in by_target.values():
        rows.sort(key=lambda row: float(row["mean_mass"]), reverse=True)

    ranks: dict[int, dict[str, Any]] = {}
    for target, rows in by_target.items():
        for rank, row in enumerate(rows[:top_n], start=1):
            ccf_id = row.get("ccf_id")
            if ccf_id is None:
                continue
            entry = ranks.setdefault(
                int(ccf_id),
                {
                    "ccf_id": int(ccf_id),
                    "acronym": row["acronym"],
                    "name": row["name"],
                    "targets": {},
                },
            )
            entry["targets"][target] = {
                "rank": rank,
                "mean_mass": float(row["mean_mass"]),
            }

    shared = []
    for entry in ranks.values():
        target_values = entry["targets"].values()
        entry["target_count"] = len(entry["targets"])
        entry["mean_top_mass"] = sum(v["mean_mass"] for v in target_values) / max(
            len(entry["targets"]),
            1,
        )
        shared.append(entry)
    shared.sort(key=lambda row: (-row["target_count"], -row["mean_top_mass"], row["ccf_id"]))
    return shared


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def write_report(
    path: Path,
    results: dict[str, Any],
    annotated_regions: list[dict[str, Any]],
    annotated_features: list[dict[str, Any]],
    shared_regions: list[dict[str, Any]],
) -> None:
    lines = [
        "# Mouse IBL/OpenAlyx action CCF annotation",
        "",
        "Annotation source: `iblatlas.atlas.BrainRegions`.",
        "",
        "## target summary",
        "",
        "| target | supported | mean dBA | median dBA |",
        "|---|---:|---:|---:|",
    ]
    for row in results.get("target_summary", []):
        lines.append(
            "| `{target}` | {supported_count}/{candidates} | {mean} | {median} |".format(
                target=row["target"],
                supported_count=row["supported_count"],
                candidates=row["candidates"],
                mean=fmt_float(float(row["mean_increment"])),
                median=fmt_float(float(row["median_increment"])),
            )
        )

    lines += [
        "",
        "## annotated top region loading mass",
        "",
        "| target | rank | CCF id | acronym | anatomical name | mean mass |",
        "|---|---:|---:|---|---|---:|",
    ]
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in annotated_regions:
        by_target.setdefault(str(row["target"]), []).append(row)
    for target in sorted(by_target):
        rows = sorted(by_target[target], key=lambda row: float(row["mean_mass"]), reverse=True)
        for rank, row in enumerate(rows, start=1):
            lines.append(
                "| `{target}` | {rank} | {ccf_id} | `{acronym}` | {name} | {mass} |".format(
                    target=target,
                    rank=rank,
                    ccf_id=row["ccf_id"],
                    acronym=row["acronym"],
                    name=row["name"],
                    mass=fmt_float(float(row["mean_mass"])),
                )
            )

    lines += [
        "",
        "## shared top-12 CCF ids",
        "",
        "| CCF id | acronym | anatomical name | target count | mean top mass |",
        "|---:|---|---|---:|---:|",
    ]
    for row in shared_regions:
        lines.append(
            "| {ccf_id} | `{acronym}` | {name} | {target_count} | {mass} |".format(
                ccf_id=row["ccf_id"],
                acronym=row["acronym"],
                name=row["name"],
                target_count=row["target_count"],
                mass=fmt_float(float(row["mean_top_mass"])),
            )
        )

    lines += [
        "",
        "## annotated top feature loading mass",
        "",
        "| target | feature | probe | CCF id | acronym | mean top-feature mass |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in annotated_features:
        lines.append(
            "| `{target}` | `{feature}` | `{probe}` | {ccf_id} | `{acronym}` | {mass} |".format(
                target=row["target"],
                feature=row["feature"],
                probe=row["probe"],
                ccf_id=row["ccf_id"] if row["ccf_id"] is not None else "",
                acronym=row["acronym"],
                mass=fmt_float(float(row["mean_top_feature_mass"])),
            )
        )

    lines += [
        "",
        "## verdict",
        "",
        "- The action subspace map is no longer just numeric CCF ids.",
        "- Speed and wheel share a thalamic/midbrain/pons-heavy top-loading pattern.",
        "- This remains descriptive localization; ablation is the next causal-style check.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--shared-top-n", type=int, default=12)
    args = parser.parse_args()

    results = json.loads(args.input_json.read_text(encoding="utf-8"))
    ccf_ids = collect_ccf_ids(results)
    annotations = atlas_annotations(ccf_ids)
    annotated_regions = annotate_region_rows(results.get("target_region_summary", []), annotations)
    annotated_features = annotate_region_rows(results.get("target_feature_summary", []), annotations)
    shared_regions = top_shared_regions(annotated_regions, args.shared_top_n)

    output = {
        "input_json": str(args.input_json),
        "annotation_source": "iblatlas.atlas.BrainRegions",
        "ccf_ids": ccf_ids,
        "annotations": annotations,
        "target_summary": results.get("target_summary", []),
        "annotated_region_summary": annotated_regions,
        "annotated_feature_summary": annotated_features,
        "shared_top_regions": shared_regions,
    }
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.report_md, results, annotated_regions, annotated_features, shared_regions)

    print("Mouse IBL/OpenAlyx action CCF annotation")
    print(f"  annotated CCF ids: {len(ccf_ids)}")
    print(f"  shared top regions: {len(shared_regions)}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
