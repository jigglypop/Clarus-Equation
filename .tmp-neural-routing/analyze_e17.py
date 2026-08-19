from __future__ import annotations

import json
import math
import pickle
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import io, stats

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
OUT = Path("e17_analysis_results")
OUT.mkdir(exist_ok=True)

TARGETS = [
    "Figure1/NDNFActivationExperiments/Relearning/Data/Figure1Relearning_DataSummary.mat",
    "Figure2/Data/NeuroPixels/extracted_data/extracted_neural_data.csv",
    "Figure2/Data/NeuroPixels/extracted_data/extracted_neural_data.pkl",
    "Figure3/FunctionalClustering/Data/SpatialCorrInstructionVSChoiceStats.mat",
    "Figure3/FunctionalClustering/Data/SpatialCorr_RuleARuleB.mat",
    "Figure3/Selectivity/Data/SelectivityStatsGluSNFR.mat",
    "Figure4/Data/DataRepDrift_CaImagingDendrites.mat",
    "Figure4/Data/DataSummary_CaImagingDendrites.mat",
    "Figure5/Data/Figure5Data_Selectivity_RepDrift.mat",
    "Figure5/Data/Figure5Data_TransitionError.mat",
    "Figure1/NDNFActivationExperiments/Relearning/README_Figure1_Relearning.md",
    "Figure2/README_Figure2.md",
    "Figure3/FunctionalClustering/README_Figure3.md",
    "Figure3/Selectivity/README_SelectivityStats.md",
    "Figure4/README_Figure4_Data.md",
    "Figure5/README_Figure5_Data.md",
    "Figure2/Code/NeuroPixels/analyze_and_plot_ephys.py",
    "Figure1/NDNFActivationExperiments/Relearning/Code/Figure1_RelearningCode.m",
    "Figure3/FunctionalClustering/Code/FunctionalClustering.m",
    "Figure3/Selectivity/Code/GluSNFR_SelectivityStats.m",
    "Figure4/Code/Figure4_Code.m",
    "Figure5/Code/Figure5_TransitionErrorCode.m",
]


def describe(obj: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return {"type": type(obj).__name__, "truncated": True}
    if isinstance(obj, dict):
        return {"type": "dict", "keys": {str(k): describe(v, depth + 1, max_depth) for k, v in list(obj.items())[:80]}}
    if isinstance(obj, np.ndarray):
        out: dict[str, Any] = {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if obj.dtype == object and obj.size and depth < max_depth:
            vals = list(obj.flat[: min(3, obj.size)])
            out["examples"] = [describe(v, depth + 1, max_depth) for v in vals]
        elif np.issubdtype(obj.dtype, np.number) and obj.size:
            finite = obj[np.isfinite(obj)] if np.issubdtype(obj.dtype, np.floating) else obj.ravel()
            if finite.size:
                out.update({"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite))})
        return out
    if isinstance(obj, (list, tuple)):
        return {"type": type(obj).__name__, "len": len(obj), "examples": [describe(v, depth + 1, max_depth) for v in list(obj)[:3]]}
    if hasattr(obj, "_fieldnames"):
        return {"type": "mat_struct", "fields": {f: describe(getattr(obj, f), depth + 1, max_depth) for f in obj._fieldnames}}
    if isinstance(obj, (str, int, float, bool, type(None), np.generic)):
        try:
            return obj.item() if isinstance(obj, np.generic) else obj
        except Exception:
            return str(obj)
    return {"type": type(obj).__name__, "repr": repr(obj)[:300]}


def numeric_matrix(x: Any) -> np.ndarray | None:
    try:
        a = np.asarray(x, dtype=float)
    except Exception:
        return None
    a = np.squeeze(a)
    if a.ndim not in (1, 2):
        return None
    return a


def get_field(x: Any, name: str) -> Any:
    if isinstance(x, dict):
        return x.get(name)
    if hasattr(x, name):
        return getattr(x, name)
    return None


def paired_summary(control: np.ndarray, opto: np.ndarray, label: str) -> dict[str, Any] | None:
    c = np.asarray(control, dtype=float)
    o = np.asarray(opto, dtype=float)
    if c.shape != o.shape:
        return None
    if c.ndim == 1:
        c = c[:, None]; o = o[:, None]
    # Expected trials x animals. If rows are fewer than columns, transpose.
    if c.shape[0] < c.shape[1]:
        c = c.T; o = o.T
    cm = np.nanmean(c, axis=0)
    om = np.nanmean(o, axis=0)
    mask = np.isfinite(cm) & np.isfinite(om)
    cm, om = cm[mask], om[mask]
    if len(cm) < 3:
        return None
    d = om - cm
    try:
        wil = stats.wilcoxon(om, cm, alternative="two-sided")
        wp = float(wil.pvalue)
    except Exception:
        wp = float("nan")
    tt = stats.ttest_rel(om, cm, nan_policy="omit")
    return {
        "label": label,
        "n_animals": int(len(cm)),
        "control_mean": float(np.mean(cm)),
        "opto_mean": float(np.mean(om)),
        "opto_minus_control": float(np.mean(d)),
        "paired_t": float(tt.statistic),
        "paired_t_p_two_sided": float(tt.pvalue),
        "wilcoxon_p_two_sided": wp,
        "per_animal_control": cm.tolist(),
        "per_animal_opto": om.tolist(),
    }


def sliding_post_switch(field: Any, switch: int = 125, width: int = 50) -> np.ndarray | None:
    a = numeric_matrix(field)
    if a is None:
        return None
    if a.ndim == 1:
        return a
    if a.shape[0] < a.shape[1]:
        a = a.T
    start = min(switch, a.shape[0])
    stop = min(start + width, a.shape[0])
    return a[start:stop]


def analyze_relearning(mat: dict[str, Any]) -> dict[str, Any]:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    result: dict[str, Any] = {"keys": keys, "comparisons": []}
    pairs = [("ABctr_cont", "ABopto_cont", "A_to_B"), ("BActr_cont", "BAopto_cont", "B_to_A")]
    fields = ["RDirOut", "LDirOut", "DirOut", "Outcomes", "RRT", "LRT", "RLicks", "LLicks"]
    for ck, ok, transition in pairs:
        c = mat.get(ck); o = mat.get(ok)
        if c is None or o is None:
            continue
        for field in fields:
            cf = get_field(c, field); of = get_field(o, field)
            if cf is None or of is None:
                continue
            cpost = sliding_post_switch(cf); opost = sliding_post_switch(of)
            if cpost is None or opost is None:
                continue
            s = paired_summary(cpost, opost, f"{transition}:{field}:trials126_175")
            if s is not None:
                result["comparisons"].append(s)
    return result


def analyze_neuropixels(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "head": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "numeric_summary": {},
        "candidate_tests": [],
    }
    for col in df.select_dtypes(include=[np.number]).columns:
        x = df[col].to_numpy(dtype=float)
        out["numeric_summary"][col] = {
            "n": int(np.isfinite(x).sum()),
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
        }
    lower = {c.lower(): c for c in df.columns}
    # Generic two-condition tests when columns expose treatment/condition and rate/burst values.
    cond_cols = [c for c in df.columns if any(k in c.lower() for k in ["condition", "group", "treatment", "dcz", "saline", "opto", "drug"])]
    value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if any(k in c.lower() for k in ["rate", "burst", "firing", "ratio", "speed", "distance"])]
    for cc in cond_cols:
        vals = [v for v in df[cc].dropna().unique().tolist()]
        if len(vals) != 2:
            continue
        for vc in value_cols:
            a = df.loc[df[cc] == vals[0], vc].astype(float).dropna().to_numpy()
            b = df.loc[df[cc] == vals[1], vc].astype(float).dropna().to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            t = stats.ttest_ind(a, b, equal_var=False)
            out["candidate_tests"].append({
                "condition_column": cc, "value_column": vc,
                "group_a": str(vals[0]), "group_b": str(vals[1]),
                "n_a": len(a), "n_b": len(b),
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "welch_t": float(t.statistic), "p_two_sided": float(t.pvalue),
            })
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        archive = td / "e17.zip"
        print("Downloading E17", flush=True)
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for target in TARGETS:
                if target in names:
                    zf.extract(target, td)
        schemas: dict[str, Any] = {}
        texts: dict[str, str] = {}
        for target in TARGETS:
            path = td / target
            if not path.exists():
                schemas[target] = {"missing": True}
                continue
            suffix = path.suffix.lower()
            if suffix == ".mat":
                try:
                    mat = io.loadmat(path, simplify_cells=True)
                    schemas[target] = describe({k: v for k, v in mat.items() if not k.startswith("__")})
                except Exception as exc:
                    schemas[target] = {"load_error": repr(exc)}
            elif suffix == ".csv":
                df = pd.read_csv(path)
                schemas[target] = analyze_neuropixels(df)
            elif suffix == ".pkl":
                try:
                    with path.open("rb") as f:
                        obj = pickle.load(f)
                    schemas[target] = describe(obj)
                except Exception as exc:
                    schemas[target] = {"load_error": repr(exc)}
            else:
                try:
                    texts[target] = path.read_text(encoding="utf-8", errors="replace")[:50000]
                except Exception as exc:
                    texts[target] = repr(exc)

        relearn_path = td / TARGETS[0]
        relearn = None
        if relearn_path.exists():
            mat = io.loadmat(relearn_path, simplify_cells=True)
            relearn = analyze_relearning(mat)

        result = {
            "status": "COMPLETE",
            "source": URL,
            "schemas": schemas,
            "relearning_analysis": relearn,
            "text_previews": texts,
        }
        (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        lines = ["# E17 direct analysis", "", "## Relearning paired comparisons", ""]
        if relearn:
            for x in relearn["comparisons"]:
                lines.append(f"- `{x['label']}`: n={x['n_animals']}, control={x['control_mean']:.4f}, opto={x['opto_mean']:.4f}, diff={x['opto_minus_control']:.4f}, paired-t p={x['paired_t_p_two_sided']:.6g}, Wilcoxon p={x['wilcoxon_p_two_sided']:.6g}")
        npdata = schemas.get(TARGETS[1], {})
        lines += ["", "## Neuropixels CSV", "", f"- shape: `{npdata.get('shape')}`", f"- columns: `{npdata.get('columns')}`"]
        for x in npdata.get("candidate_tests", []):
            lines.append(f"- `{x['condition_column']}` / `{x['value_column']}`: {x['group_a']}={x['mean_a']:.4f} (n={x['n_a']}), {x['group_b']}={x['mean_b']:.4f} (n={x['n_b']}), p={x['p_two_sided']:.6g}")
        (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
