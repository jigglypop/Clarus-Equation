from __future__ import annotations

import itertools
import json
import math
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
ZIP = Path("/tmp/e17-etlk5k.zip")
OUT = Path("e17_direct_results")
OUT.mkdir(exist_ok=True)
SEED = 20260819
RNG = np.random.default_rng(SEED)
DRAWS = 20000
SWITCH_INDEX = 125  # trials 1-125 pre-switch, trial 126+ post-switch


def ensure_zip() -> None:
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)


def normalize_sessions(x: Any) -> list[dict]:
    if isinstance(x, dict):
        return [x]
    if isinstance(x, np.ndarray):
        vals = list(x.reshape(-1))
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        vals = [x]
    return [v for v in vals if isinstance(v, dict)]


def binary_dirout(session: dict) -> np.ndarray:
    x = np.asarray(session.get("DirOut", []), dtype=float).reshape(-1)
    # Preserve order, remove only nonfinite entries; DirOut is documented as binary.
    x = x[np.isfinite(x)]
    return x[np.isin(x, [0.0, 1.0])]


def window_mean(x: np.ndarray, start: int, stop: int) -> float:
    y = x[max(0, start):min(len(x), stop)]
    return float(np.mean(y)) if len(y) else float("nan")


def rolling_criterion(x: np.ndarray, start: int, width: int = 20, criterion: float = .75) -> float:
    for i in range(start, max(start, len(x) - width + 1)):
        if float(np.mean(x[i:i + width])) >= criterion:
            return float(i - start + 1)
    return float("nan")


def session_metrics(session: dict) -> dict:
    x = binary_dirout(session)
    return {
        "n_trials": int(len(x)),
        "pre50": window_mean(x, SWITCH_INDEX - 50, SWITCH_INDEX),
        "post50": window_mean(x, SWITCH_INDEX, SWITCH_INDEX + 50),
        "post100": window_mean(x, SWITCH_INDEX, SWITCH_INDEX + 100),
        "post51_100": window_mean(x, SWITCH_INDEX + 50, SWITCH_INDEX + 100),
        "drop_post50_minus_pre50": window_mean(x, SWITCH_INDEX, SWITCH_INDEX + 50) - window_mean(x, SWITCH_INDEX - 50, SWITCH_INDEX),
        "trials_to_75pct_rolling20": rolling_criterion(x, SWITCH_INDEX, width=20, criterion=.75),
    }


def collect() -> tuple[dict[str, dict[str, list[dict]]], list[dict]]:
    ensure_zip()
    root = Path("/tmp/e17-fig1-analysis")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    out: dict[str, dict[str, list[dict]]] = {"Control": {}, "Opto": {}}
    audit = []
    with zipfile.ZipFile(ZIP) as z:
        for condition in ("Control", "Opto"):
            prefix = f"Figure1/NDNFActivationExperiments/Relearning/{condition}/"
            names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".mat"))
            for name in names:
                animal = Path(name).stem
                dst = root / f"{condition}_{animal}.mat"
                with z.open(name) as src, dst.open("wb") as f:
                    shutil.copyfileobj(src, f)
                data = loadmat(dst, simplify_cells=True)
                sessions = normalize_sessions(data.get("cont_data"))
                out[condition][animal] = sessions
                audit.append({
                    "condition": condition,
                    "animal": animal,
                    "n_sessions": len(sessions),
                    "dirout_lengths": [int(len(binary_dirout(s))) for s in sessions],
                    "fields": [sorted(s.keys()) for s in sessions[:2]],
                })
    return out, audit


def exact_signflip_p(d: np.ndarray, alternative: str) -> float:
    d = d[np.isfinite(d)]
    obs = float(np.mean(d))
    n = len(d)
    vals = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        vals.append(float(np.mean(d * np.asarray(signs))))
    vals = np.asarray(vals)
    if alternative == "less":
        return float(np.mean(vals <= obs + 1e-15))
    if alternative == "greater":
        return float(np.mean(vals >= obs - 1e-15))
    return float(np.mean(np.abs(vals) >= abs(obs) - 1e-15))


def paired_bootstrap(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    vals = np.empty(DRAWS)
    for b in range(DRAWS):
        vals[b] = float(np.mean(RNG.choice(d, size=len(d), replace=True)))
    return {
        "mean": float(np.mean(vals)),
        "q025": float(np.quantile(vals, .025)),
        "q975": float(np.quantile(vals, .975)),
        "p_nonnegative": float((np.sum(vals >= 0) + 1) / (DRAWS + 1)),
        "p_nonpositive": float((np.sum(vals <= 0) + 1) / (DRAWS + 1)),
    }


def analyze(data: dict[str, dict[str, list[dict]]]) -> dict:
    animals = sorted(set(data["Control"]) & set(data["Opto"]))
    records = []
    # README documents transition sessions as entries 2 and 4 (1-based).
    transition_indices = [1, 3]
    for animal in animals:
        for transition_number, idx in enumerate(transition_indices, start=1):
            if idx >= len(data["Control"][animal]) or idx >= len(data["Opto"][animal]):
                continue
            control = session_metrics(data["Control"][animal][idx])
            opto = session_metrics(data["Opto"][animal][idx])
            records.append({
                "animal": animal,
                "transition": transition_number,
                "control": control,
                "opto": opto,
            })
    metrics = ["pre50", "post50", "post100", "post51_100", "drop_post50_minus_pre50", "trials_to_75pct_rolling20"]
    comparisons = {}
    for metric in metrics:
        by_animal = defaultdict(list)
        for r in records:
            c = r["control"][metric]
            o = r["opto"][metric]
            if np.isfinite(c) and np.isfinite(o):
                by_animal[r["animal"]].append(o - c)
        animal_diff = {a: float(np.mean(v)) for a, v in by_animal.items()}
        d = np.asarray(list(animal_diff.values()), dtype=float)
        # Accuracy expected lower under opto; trials-to-criterion expected higher.
        alternative = "greater" if metric == "trials_to_75pct_rolling20" else "less"
        comparisons[metric] = {
            "animal_differences_opto_minus_control": animal_diff,
            "mean_difference": float(np.mean(d)) if len(d) else float("nan"),
            "median_difference": float(np.median(d)) if len(d) else float("nan"),
            "exact_one_sided_signflip_p": exact_signflip_p(d, alternative) if len(d) else float("nan"),
            "exact_two_sided_signflip_p": exact_signflip_p(d, "two-sided") if len(d) else float("nan"),
            "bootstrap": paired_bootstrap(d) if len(d) else {},
            "n_animals": int(len(d)),
        }
    return {"animals": animals, "records": records, "comparisons": comparisons}


def report(result: dict) -> str:
    lines = [
        "# E17 Figure 1 direct relearning test",
        "",
        f"Switch index `{SWITCH_INDEX}`; seed `{SEED}`; paired bootstrap draws `{DRAWS}`.",
        "",
        "| metric | n animals | mean Opto-Control | 95% bootstrap CI | exact one-sided p |",
        "|---|---:|---:|---|---:|",
    ]
    for metric, r in result["analysis"]["comparisons"].items():
        b = r["bootstrap"]
        lines.append(f"| {metric} | {r['n_animals']} | {r['mean_difference']:.4f} | [{b.get('q025', float('nan')):.4f}, {b.get('q975', float('nan')):.4f}] | {r['exact_one_sided_signflip_p']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    data, audit = collect()
    analysis = analyze(data)
    result = {"status": "COMPLETE", "source": URL, "audit": audit, "analysis": analysis}
    (OUT / "figure1_relearning_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "figure1_relearning_report.md").write_text(report(result), encoding="utf-8")
    print(report(result))


if __name__ == "__main__":
    main()
