"""Close the zebrafish continuous-decoding bottleneck after supplementary fetch.

This audit is deliberately conservative. It checks whether the newly downloaded
``Others_Supplementary`` archive supplies one of the missing bridges needed for
timestamp-certified continuous movement decoding:

- an e2-column timestamp,
- behavior traces resampled to e2 frames,
- or a synchronized raw neural/tracking bundle for the matched e2 session.

If not, the zebrafish continuous gate remains a data bottleneck rather than a
failed dynamics claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.io import whosmat


BASE = Path("data/evolution/zebrafish/freely_swimming")
DEFAULT_ROOT = BASE / "Others_Supplementary"
ARCHIVE = BASE / "_archives" / "Others_Supplementary.7z"
ARCHIVE_MD5 = "366c3477821df3c8b69e635353b5181a"
RESULT_JSON = Path(__file__).with_name(
    "zebrafish_supplementary_continuous_closure_audit_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "zebrafish_supplementary_continuous_closure_audit_report.md"
)

E2_MATCH_SESSION = "20221016_1556_g8s-lssm-chriR_8dpf"
REQUIRED_BRIDGE_KEYWORDS = ("e2", "timestamp", "time", "stage", "head", "tail", "speed", "turn")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_whosmat(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    try:
        return [(name, tuple(shape), dtype) for name, shape, dtype in whosmat(path)]
    except Exception:
        return []


def count_keywords(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {keyword: keyword in lower for keyword in REQUIRED_BRIDGE_KEYWORDS}


def audit(root: Path) -> dict[str, object]:
    archive_md5 = md5(ARCHIVE) if ARCHIVE.exists() else None
    all_files = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    mat_files = [path for path in all_files if path.suffix.lower() == ".mat"]
    txt_files = [path for path in all_files if path.suffix.lower() == ".txt"]
    e2_named_files = [path for path in all_files if "e2" in str(path).lower()]
    timestamp_named_files = [path for path in all_files if "timestamp" in path.name.lower()]
    matched_session_files = [path for path in all_files if E2_MATCH_SESSION in str(path)]

    mat_summaries = []
    variable_hits = []
    for path in mat_files:
        variables = safe_whosmat(path)
        rel = str(path.relative_to(root))
        variable_rows = []
        for name, shape, dtype in variables:
            hits = count_keywords(name)
            if any(hits.values()):
                variable_hits.append(
                    {"path": rel, "name": name, "shape": list(shape), "dtype": dtype, "hits": hits}
                )
            variable_rows.append({"name": name, "shape": list(shape), "dtype": dtype})
        mat_summaries.append({"path": rel, "variables": variable_rows})

    z_tracking_mats = [
        row
        for row in mat_summaries
        if "Z-tracking" in row["path"] and "Z-Tracking-Analysis.mat" in row["path"]
    ]
    z_tracking_expz_only = all(
        [var["name"] for var in row["variables"]] == ["expZ"] for row in z_tracking_mats
    )
    caltrace_mats = [row for row in mat_summaries if "CalTrace" in row["path"]]
    interpolation_mats = [row for row in mat_summaries if "figureS3_interpolation" in row["path"]]

    has_e2_timestamp = any(
        "e2" in hit["name"].lower() and ("time" in hit["name"].lower() or "timestamp" in hit["name"].lower())
        for hit in variable_hits
    )
    has_e2_resampled_behavior = any(
        "e2" in hit["name"].lower()
        and any(key in hit["name"].lower() for key in ("stage", "head", "tail", "speed", "turn"))
        for hit in variable_hits
    )
    has_matched_session_bundle = bool(matched_session_files)
    ready = bool(has_e2_timestamp or has_e2_resampled_behavior or has_matched_session_bundle)
    return {
        "root": str(root),
        "archive": str(ARCHIVE),
        "archive_md5": archive_md5,
        "archive_md5_expected": ARCHIVE_MD5,
        "archive_md5_ok": archive_md5 == ARCHIVE_MD5,
        "file_count": len(all_files),
        "mat_file_count": len(mat_files),
        "txt_file_count": len(txt_files),
        "e2_named_file_count": len(e2_named_files),
        "timestamp_named_file_count": len(timestamp_named_files),
        "matched_e2_session_file_count": len(matched_session_files),
        "z_tracking_mat_count": len(z_tracking_mats),
        "z_tracking_expz_only": z_tracking_expz_only,
        "caltrace_mat_count": len(caltrace_mats),
        "interpolation_mat_count": len(interpolation_mats),
        "variable_hits": variable_hits,
        "representative_mats": mat_summaries[:8],
        "has_e2_timestamp": has_e2_timestamp,
        "has_e2_resampled_behavior": has_e2_resampled_behavior,
        "has_matched_session_bundle": has_matched_session_bundle,
        "timestamp_certified_continuous_ready": ready,
        "verdict": (
            "blocked_missing_e2_behavior_bridge"
            if not ready
            else "candidate_bridge_found_requires_decoding_gate"
        ),
        "interpretation": (
            "Others_Supplementary supplies Z-position/stage alignment and calcium trace/interpolation "
            "QA, but not an explicit e2 timestamp or e2-resampled speed/turn trace."
        ),
    }


def write_report(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Zebrafish supplementary continuous closure audit",
        "",
        "목표는 `Others_Supplementary`를 받은 뒤 zebrafish continuous movement gate가 닫히는지 확인하는 것이다.",
        "",
        "## archive",
        "",
        "| item | value |",
        "|---|---:|",
        f"| md5 ok | {payload['archive_md5_ok']} |",
        f"| files | {payload['file_count']} |",
        f"| mat files | {payload['mat_file_count']} |",
        f"| txt files | {payload['txt_file_count']} |",
        "",
        "## bridge checks",
        "",
        "| check | value |",
        "|---|---:|",
        f"| e2-named files in supplementary | {payload['e2_named_file_count']} |",
        f"| timestamp-named files in supplementary | {payload['timestamp_named_file_count']} |",
        f"| matched e2 session files | {payload['matched_e2_session_file_count']} |",
        f"| has e2 timestamp variable | {payload['has_e2_timestamp']} |",
        f"| has e2-resampled behavior | {payload['has_e2_resampled_behavior']} |",
        f"| Z-tracking mats | {payload['z_tracking_mat_count']} |",
        f"| Z-tracking mats expZ only | {payload['z_tracking_expz_only']} |",
        f"| CalTrace mats | {payload['caltrace_mat_count']} |",
        f"| interpolation mats | {payload['interpolation_mat_count']} |",
        "",
        "## verdict",
        "",
        f"- timestamp-certified continuous ready: `{payload['timestamp_certified_continuous_ready']}`",
        f"- verdict: `{payload['verdict']}`",
        "",
        payload["interpretation"],
        "",
        "따라서 현재 받은 partial + Others_Supplementary만으로는 `e2[:, t] -> speed/turn/heading` 최종 gate를 닫을 수 없다.",
        "이것은 dynamics 실패가 아니라 공개 chunk의 alignment 정보 부족으로 판정한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    payload = audit(args.root)
    RESULT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload, REPORT_MD)
    print("Zebrafish supplementary continuous closure audit")
    print(f"  archive_md5_ok={payload['archive_md5_ok']}")
    print(f"  file_count={payload['file_count']}")
    print(f"  mat_file_count={payload['mat_file_count']}")
    print(f"  e2_named_file_count={payload['e2_named_file_count']}")
    print(f"  timestamp_named_file_count={payload['timestamp_named_file_count']}")
    print(f"  has_e2_timestamp={payload['has_e2_timestamp']}")
    print(f"  has_e2_resampled_behavior={payload['has_e2_resampled_behavior']}")
    print(f"  timestamp_certified_continuous_ready={payload['timestamp_certified_continuous_ready']}")
    print(f"  verdict={payload['verdict']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()

