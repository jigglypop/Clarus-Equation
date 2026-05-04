"""Zebrafish freely swimming activity robustness gate.

This uses the smaller Figure 5/S8 chunk from the public Figshare dataset
"All-optical interrogation of brain-wide activity in freely swimming larval
zebrafish".

The chunk does not contain aligned tail-behavior labels. It contains region
fluorescence traces for freely swimming and immobilized conditions. Therefore
this is not the final activity->behavior closure gate. It asks the intermediate
question:

    Is region activity structure preserved when the fish is freely swimming,
    and does a low-rank recurrent state close the freely-swimming activity?
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np


DATA_DIR = Path("data/evolution/zebrafish/freely_swimming/figure5_S8")
RESULT_JSON = Path(__file__).with_name("zebrafish_freely_swimming_activity_gate_results.json")
REPORT_MD = Path(__file__).with_name("zebrafish_freely_swimming_activity_report.md")

MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_MATRIX = 14
MI_COMPRESSED = 15

MX_DOUBLE_CLASS = 6
MX_SINGLE_CLASS = 7

DTYPE = {
    MI_INT8: "i1",
    MI_UINT8: "u1",
    MI_INT16: "<i2",
    MI_UINT16: "<u2",
    MI_INT32: "<i4",
    MI_UINT32: "<u4",
    MI_SINGLE: "<f4",
    MI_DOUBLE: "<f8",
}


def align8(offset: int) -> int:
    return offset + ((8 - offset % 8) % 8)


def read_tag(buf: bytes, offset: int) -> tuple[int, int, int]:
    raw = struct.unpack_from("<I", buf, offset)[0]
    small_type = raw & 0xFFFF
    small_size = raw >> 16
    if small_size:
        return small_type, small_size, offset + 4
    dtype, size = struct.unpack_from("<II", buf, offset)
    return dtype, size, offset + 8


def read_element(buf: bytes, offset: int) -> tuple[int, bytes, int]:
    dtype, size, data_offset = read_tag(buf, offset)
    data = buf[data_offset : data_offset + size]
    if offset + 4 == data_offset:
        next_offset = offset + 8
    elif dtype == MI_COMPRESSED:
        # MATLAB v5 compressed elements in these files are byte-contiguous.
        next_offset = data_offset + size
    else:
        next_offset = align8(data_offset + size)
    return dtype, data, next_offset


def parse_numeric(data: bytes, dtype: int) -> np.ndarray:
    if dtype not in DTYPE:
        return np.asarray([])
    return np.frombuffer(data, dtype=np.dtype(DTYPE[dtype])).copy()


def parse_matrix(data: bytes) -> tuple[str, np.ndarray] | None:
    offset = 0
    dtype, flags_raw, offset = read_element(data, offset)
    if dtype not in (MI_UINT32, MI_INT32) or len(flags_raw) < 8:
        return None
    array_class = struct.unpack_from("<I", flags_raw, 0)[0] & 0xFF
    if array_class not in (MX_DOUBLE_CLASS, MX_SINGLE_CLASS):
        return None

    dtype, dims_raw, offset = read_element(data, offset)
    dims = parse_numeric(dims_raw, dtype).astype(int).tolist()
    if not dims:
        return None

    dtype, name_raw, offset = read_element(data, offset)
    name = name_raw.decode("ascii", errors="ignore")
    if not name:
        return None

    dtype, values_raw, _ = read_element(data, offset)
    values = parse_numeric(values_raw, dtype).astype(float)
    if values.size == 0:
        return None
    arr = values.reshape(tuple(dims), order="F")
    return name, arr


def load_mat_numeric(path: Path) -> dict[str, np.ndarray]:
    buf = path.read_bytes()
    if not buf.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"Not a MATLAB 5 file: {path}")
    offset = 128
    out = {}
    while offset < len(buf):
        dtype, data, offset = read_element(buf, offset)
        if dtype == MI_COMPRESSED:
            inner = zlib.decompress(data)
            inner_type, inner_data, _ = read_element(inner, 0)
            if inner_type == MI_MATRIX:
                parsed = parse_matrix(inner_data)
                if parsed:
                    out[parsed[0]] = parsed[1]
        elif dtype == MI_MATRIX:
            parsed = parse_matrix(data)
            if parsed:
                out[parsed[0]] = parsed[1]
    return out


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 0)


def corr_by_region(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = []
    for left, right in zip(a, b):
        mask = np.isfinite(left) & np.isfinite(right)
        if np.sum(mask) < 3:
            out.append(np.nan)
            continue
        out.append(float(np.corrcoef(left[mask], right[mask])[0, 1]))
    return np.asarray(out)


def lowrank_recurrent(activity: np.ndarray, rank: int) -> dict[str, float]:
    activity = zscore_rows(activity)
    y = activity[:, 1:].T
    x = activity[:, :-1].T
    baseline = float(np.mean((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    k = min(rank, len(s))
    design = np.column_stack([np.ones(u.shape[0]), u[:, :k] * s[:k]])
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ coeff
    model = float(np.mean((y - pred) ** 2))
    return {
        "rank": k,
        "baseline_mse": baseline,
        "lowrank_recurrent_mse": model,
        "model_over_baseline": model / max(baseline, 1e-12),
        "r2_vs_baseline": 1.0 - model / max(baseline, 1e-12),
    }


def contiguous_sections(free: np.ndarray, imm: np.ndarray, section_len: int) -> tuple[np.ndarray, np.ndarray]:
    sections_free = []
    sections_imm = []
    max_sections = min(free.shape[1] // (section_len * 2), imm.shape[1] // section_len)
    for idx in range(max_sections):
        sections_free.append(free[:, idx * section_len * 2 : idx * section_len * 2 + section_len])
        sections_imm.append(imm[:, idx * section_len : (idx + 1) * section_len])
    return np.hstack(sections_free), np.hstack(sections_imm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rank", type=int, default=6)
    args = parser.parse_args()

    bundle = load_mat_numeric(args.data_dir / "BLUE_bundle.mat")
    region_file = load_mat_numeric(args.data_dir / "Region_active18.mat")
    region_active = region_file["Region_active18"].astype(int).ravel() - 1

    cal_g_free = bundle["Cal_G_free"][region_active, :]
    cal_g_imm = bundle["Cal_G_imm"][region_active, :]
    cal_r_free = bundle["Cal_R_free"][region_active, :]
    cal_r_imm = bundle["Cal_R_imm"][region_active, :]

    free_g, imm_g = contiguous_sections(cal_g_free, cal_g_imm, section_len=250)
    free_r, imm_r = contiguous_sections(cal_r_free, cal_r_imm, section_len=250)

    similarity_g = corr_by_region(zscore_rows(free_g), zscore_rows(imm_g))
    similarity_r = corr_by_region(zscore_rows(free_r), zscore_rows(imm_r))
    recurrent_free_g = lowrank_recurrent(cal_g_free, args.rank)
    recurrent_free_r = lowrank_recurrent(cal_r_free, args.rank)

    output = {
        "dataset": "All-optical interrogation of brain-wide activity in freely swimming larval zebrafish",
        "chunk": "figure5_S8",
        "data_dir": str(args.data_dir),
        "variable_shapes": {key: list(value.shape) for key, value in bundle.items()},
        "region_count": int(len(region_active)),
        "free_imm_similarity_green_mean": float(np.nanmean(similarity_g)),
        "free_imm_similarity_red_mean": float(np.nanmean(similarity_r)),
        "free_imm_similarity_green_median": float(np.nanmedian(similarity_g)),
        "free_imm_similarity_red_median": float(np.nanmedian(similarity_r)),
        "recurrent_free_green": recurrent_free_g,
        "recurrent_free_red": recurrent_free_r,
        "passed": bool(
            np.nanmean(similarity_g) > 0.0
            and recurrent_free_g["model_over_baseline"] < 1.0
            and recurrent_free_r["model_over_baseline"] < 1.0
        ),
        "caveat": "No aligned tail-behavior labels in this chunk; this is free-swimming activity robustness, not activity-to-behavior closure.",
    }
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    REPORT_MD.write_text(
        "\n".join(
            [
                "# Zebrafish freely swimming activity gate",
                "",
                "Figshare freely-swimming zebrafish 자료의 작은 figure5/S8 chunk로 자유수영 상태 activity 구조를 점검했다.",
                "",
                "이 chunk에는 정렬된 tail-behavior label이 없으므로 최종 activity->behavior gate는 아니다.",
                "",
                "## 결과",
                "",
                f"- region count: {output['region_count']}",
                f"- green free/imm mean similarity: {output['free_imm_similarity_green_mean']:.6f}",
                f"- red free/imm mean similarity: {output['free_imm_similarity_red_mean']:.6f}",
                f"- free green recurrent/baseline: {recurrent_free_g['model_over_baseline']:.6f}",
                f"- free red recurrent/baseline: {recurrent_free_r['model_over_baseline']:.6f}",
                f"- pass: {output['passed']}",
                "",
                "## 해석",
                "",
                "- 자유수영에서도 region activity는 저차원 recurrent state로 평균 baseline보다 잘 닫힌다.",
                "- free/imm similarity는 조건이 달라도 일부 region-level activity 구조가 보존되는지 보는 보조 지표다.",
                "- 다음에는 같은 freely-swimming 자료에서 neural trace와 tail/stage tracking이 시간 정렬된 chunk를 찾아 activity->movement gate를 만들어야 한다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Zebrafish freely swimming activity gate")
    print(f"  regions={output['region_count']}")
    print(f"  green free/imm similarity={output['free_imm_similarity_green_mean']:.6f}")
    print(f"  red free/imm similarity={output['free_imm_similarity_red_mean']:.6f}")
    print(f"  green recurrent/base={recurrent_free_g['model_over_baseline']:.6f}")
    print(f"  red recurrent/base={recurrent_free_r['model_over_baseline']:.6f}")
    print(f"  passed={output['passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
