from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import requests
from pynwb import NWBHDF5IO

OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)
SEED = 20260819
RNG = np.random.default_rng(SEED)
BIN = 0.05
RANK = 5
RIDGE = 1.0
LAGS = (1, 2, 4)
MAX_DURATION = 1200.0
BLOCK = 100
BOOT = 3000
ASSETS = [
    {"animal": "M01", "asset_id": "6d733831-afbf-44c2-8c46-7b3550f5e672"},
    {"animal": "M02", "asset_id": "1e4d5403-a8cc-4814-a904-7aff57f8cc4d"},
    {"animal": "M03", "asset_id": "605ae4d4-454b-435b-97ef-84518ce63932"},
    {"animal": "M05", "asset_id": "091bc936-b149-4598-b89a-e9db45499a69"},
]
REGIONS = ("CA3", "CA1", "RSC")


def download(asset: dict) -> Path:
    path = Path(f"/tmp/dandi001695_{asset['animal']}.nwb")
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
    with requests.get(url, stream=True, timeout=180, allow_redirects=True) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path


def interval(nwb) -> tuple[float, float, dict]:
    if "behavior" in nwb.processing:
        module = nwb.processing["behavior"]
        for name, obj in module.data_interfaces.items():
            if hasattr(obj, "spatial_series"):
                for sname, ss in obj.spatial_series.items():
                    try:
                        if ss.timestamps is not None:
                            ts = np.asarray(ss.timestamps[:], dtype=float)
                        else:
                            ts = float(ss.starting_time) + np.arange(len(ss.data)) / float(ss.rate)
                        ts = ts[np.isfinite(ts)]
                        if len(ts) > 100:
                            start = float(ts.min()); stop = min(float(ts.max()), start + MAX_DURATION)
                            return start, stop, {"interface": name, "series": sname, "n": len(ts)}
                    except Exception:
                        pass
    df = nwb.units.to_dataframe()
    starts, stops = [], []
    for sp in df["spike_times"]:
        a = np.asarray(sp, dtype=float)
        if len(a): starts.append(float(a.min())); stops.append(float(a.max()))
    start = max(starts); stop = min(min(stops), start + MAX_DURATION)
    return start, stop, {"fallback": "common spike support"}


def binned(df, region: str, start: float, stop: float) -> np.ndarray:
    mask = df["cell_area"].astype(str).str.upper() == region
    sub = df.loc[mask]
    edges = np.arange(start, stop + BIN, BIN)
    x = np.zeros((len(edges) - 1, len(sub)), dtype=float)
    for j, sp in enumerate(sub["spike_times"]):
        x[:, j] = np.histogram(np.asarray(sp, dtype=float), bins=edges)[0]
    return x


def pca_scores(counts: np.ndarray, train_idx: np.ndarray, rank: int) -> np.ndarray:
    x = np.sqrt(counts)
    mu = x[train_idx].mean(axis=0); sd = x[train_idx].std(axis=0); sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    _, _, vt = np.linalg.svd(z[train_idx], full_matrices=False)
    r = min(rank, vt.shape[0], vt.shape[1])
    return z @ vt[:r].T


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    xm = x.mean(0); ym = y.mean(0)
    xc = x - xm; yc = y - ym
    coef = np.linalg.solve(xc.T @ xc + RIDGE * np.eye(x.shape[1]), xc.T @ yc)
    pred = xc @ coef + ym
    e = y - pred
    cov = np.cov(e, rowvar=False, ddof=1)
    if np.ndim(cov) == 0: cov = np.array([[float(cov)]])
    floor = max(1e-6, float(np.trace(cov)) / cov.shape[0] * 1e-4)
    cov = (cov + cov.T) / 2 + floor * np.eye(cov.shape[0])
    return {"xm": xm, "ym": ym, "coef": coef, "cov": cov}


def nll(model: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    yp = (x - model["xm"]) @ model["coef"] + model["ym"]
    e = y - yp
    inv = np.linalg.inv(model["cov"])
    _, logdet = np.linalg.slogdet(model["cov"])
    q = np.einsum("ni,ij,nj->n", e, inv, e)
    return .5 * (q + logdet + y.shape[1] * math.log(2 * math.pi))


def split(n: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    usable = n - lag
    boundary = int(usable * .80)
    gap = max(20, int(1 / BIN))
    train = np.arange(0, max(1, boundary - gap))
    test = np.arange(min(usable, boundary + gap), usable)
    return train, test


def feature(scores: dict[str, np.ndarray], target: str, source: str, third: str, idx: np.ndarray, lag: int, include_source: bool, shift: bool = False):
    parts = [scores[target][idx], scores[third][idx]]
    if include_source:
        src = scores[source]
        if shift: src = np.roll(src, max(100, len(src) // 5), axis=0)
        parts.append(src[idx])
    return np.column_stack(parts), scores[target][idx + lag]


def boot(delta: np.ndarray) -> dict:
    blocks = [delta[i:i + BLOCK] for i in range(0, len(delta), BLOCK) if len(delta[i:i + BLOCK])]
    vals = np.empty(BOOT)
    for b in range(BOOT):
        ids = RNG.integers(0, len(blocks), size=len(blocks))
        vals[b] = np.mean(np.concatenate([blocks[i] for i in ids]))
    return {"mean": float(np.mean(delta)), "q025": float(np.quantile(vals,.025)), "q975": float(np.quantile(vals,.975)), "p_nonpositive": float((np.sum(vals <= 0)+1)/(BOOT+1))}


def one_pair(scores: dict[str,np.ndarray], source: str, target: str, third: str, lag: int, train: np.ndarray, test: np.ndarray) -> dict:
    xbtr, ytr = feature(scores,target,source,third,train,lag,False)
    xbte, yte = feature(scores,target,source,third,test,lag,False)
    xftr, _ = feature(scores,target,source,third,train,lag,True)
    xfte, _ = feature(scores,target,source,third,test,lag,True)
    xstr, _ = feature(scores,target,source,third,train,lag,True,True)
    xste, _ = feature(scores,target,source,third,test,lag,True,True)
    mb, mf, ms = fit(xbtr,ytr), fit(xftr,ytr), fit(xstr,ytr)
    nb, nf, ns = nll(mb,xbte,yte), nll(mf,xfte,yte), nll(ms,xste,yte)
    return {"delta_base_minus_full": float(np.mean(nb-nf)), "delta_shift_minus_full": float(np.mean(ns-nf)), "block_bootstrap": boot(nb-nf), "n_test": len(test)}


def session(asset: dict) -> dict:
    path = download(asset)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read(); df = nwb.units.to_dataframe(); start, stop, imeta = interval(nwb)
        counts = {r: binned(df,r,start,stop) for r in REGIONS}
        region_counts = {r: counts[r].shape[1] for r in REGIONS}
        if min(region_counts.values()) < 2:
            return {"animal":asset["animal"],"status":"INSUFFICIENT_REGIONS","region_counts":region_counts}
        n = min(x.shape[0] for x in counts.values()); counts={k:v[:n] for k,v in counts.items()}
        results=[]
        for lag in LAGS:
            tr,te=split(n,lag)
            rank=min(RANK,*(counts[r].shape[1] for r in REGIONS))
            scores={r:pca_scores(counts[r],tr,rank) for r in REGIONS}
            for target in REGIONS:
                others=[r for r in REGIONS if r!=target]
                for source in others:
                    third=next(r for r in others if r!=source)
                    results.append({"source":source,"target":target,"third":third,"lag_ms":lag*BIN*1000,**one_pair(scores,source,target,third,lag,tr,te)})
        out={"animal":asset["animal"],"asset_id":asset["asset_id"],"status":"COMPLETE","session_description":nwb.session_description,"interval":{"start":start,"stop":stop,"duration":stop-start,"meta":imeta},"region_counts":region_counts,"rank":rank,"ridge":RIDGE,"results":results}
    path.unlink(missing_ok=True)
    return out


def exact_sign(values: list[float], alternative: str="greater") -> float:
    x=np.asarray(values,float); obs=x.mean(); allv=[]
    for s in itertools.product((-1.,1.),repeat=len(x)): allv.append(np.mean(x*np.asarray(s)))
    allv=np.asarray(allv)
    return float(np.mean(allv>=obs-1e-15)) if alternative=="greater" else float(np.mean(np.abs(allv)>=abs(obs)-1e-15))


def summarize(sessions: list[dict]) -> list[dict]:
    rows=[]
    for lag in [x*BIN*1000 for x in LAGS]:
        for source in REGIONS:
            for target in REGIONS:
                if source==target: continue
                vals=[]; shifts=[]
                for s in sessions:
                    if s.get("status")!="COMPLETE": continue
                    r=next((r for r in s["results"] if r["source"]==source and r["target"]==target and r["lag_ms"]==lag),None)
                    if r: vals.append(r["delta_base_minus_full"]); shifts.append(r["delta_shift_minus_full"])
                if vals:
                    rows.append({"source":source,"target":target,"lag_ms":lag,"animal_values":vals,"mean_delta":float(np.mean(vals)),"n_positive":int(np.sum(np.asarray(vals)>0)),"exact_one_sided_signflip_p":exact_sign(vals),"mean_shift_minus_full":float(np.mean(shifts))})
    # directional contrasts at equal lag
    for lag in [x*BIN*1000 for x in LAGS]:
        for a,b in (("CA3","CA1"),("CA1","RSC")):
            diffs=[]
            for s in sessions:
                if s.get("status")!="COMPLETE": continue
                f=next(r for r in s["results"] if r["source"]==a and r["target"]==b and r["lag_ms"]==lag)
                rev=next(r for r in s["results"] if r["source"]==b and r["target"]==a and r["lag_ms"]==lag)
                diffs.append(f["delta_base_minus_full"]-rev["delta_base_minus_full"])
            rows.append({"contrast":f"{a}->{b} minus {b}->{a}","lag_ms":lag,"animal_values":diffs,"mean_delta":float(np.mean(diffs)),"n_positive":int(np.sum(np.asarray(diffs)>0)),"exact_one_sided_signflip_p":exact_sign(diffs)})
    return rows


def main():
    sessions=[session(a) for a in ASSETS]
    summary=summarize(sessions)
    out={"status":"COMPLETE","bin_s":BIN,"rank":RANK,"ridge":RIDGE,"lags_ms":[x*BIN*1000 for x in LAGS],"sessions":sessions,"summary":summary}
    (OUT/"bridge_multianimal_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    lines=["# DANDI 001695 fixed multi-animal bridge test","",f"Fixed rank `{RANK}`, ridge `{RIDGE}`, bin `{BIN}s`; one behavior session each from M01/M02/M03/M05.","","| path/contrast | lag ms | mean ΔNLPD | positive animals | exact one-sided p |","|---|---:|---:|---:|---:|"]
    for r in summary:
        name=r.get("contrast",f"{r['source']}→{r['target']}")
        lines.append(f"| {name} | {r['lag_ms']:.0f} | {r['mean_delta']:.5f} | {r['n_positive']}/{len(r['animal_values'])} | {r['exact_one_sided_signflip_p']:.4f} |")
    (OUT/"bridge_multianimal_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
