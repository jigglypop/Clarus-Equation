"""Run-local descriptive ledger for every non-eligible frozen route.

No calculation here is a neural Riemannian-metric test.  Rows, windows and
spines are preserved as nested source summaries and never become subjects.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
try:
 from scipy.io import loadmat
 from scipy.stats import spearmanr
except ModuleNotFoundError:
 loadmat = None
 spearmanr = None

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
CE_ROOT = RUN.parent
OUT = RUN / "artifacts" / "route_dispositions.json"
E17 = CE_ROOT / "neural-riemannian-metric-validation-20260818" / "artifacts" / "realdata" / "NRM-E17-extracted"
SLEEP = CE_ROOT / "_archive" / "sleep-replay-routing-realdata-20260818" / "artifacts" / "realdata"
CE = CE_ROOT / "connectome-graph-replay-20260818" / "artifacts" / "herm_full_edgelist.csv"
BCI = RUN / "artifacts" / "input" / "bci_busch" / "avatarRT_analysis" / "results" / "final_results" / "behavioral_lm_results.csv"

ROUTES = {
 "R-E17-F3": ("PARTIAL_DESCRIPTIVE", "released aggregate spine arrays lack independent animal/same-unit endpoint"),
 "R-E17-F4": ("PARTIAL_DESCRIPTIVE", "released drift arrays lack verified earlier-to-later unit/animal split"),
 "R-E17-F5": ("PARTIAL_DESCRIPTIVE", "transition-error table lacks frozen earlier geometry and independent future target"),
 "R-E17-F2": ("INELIGIBLE_DEPENDENT", "locked predecessor reference; rerun is not independent evidence"),
 "R-SLEEP-E19": ("PARTIAL_DESCRIPTIVE", "source-selected clusters and no verified predictor-to-later-target chronology"),
 "R-SLEEP-E15": ("PARTIAL_DESCRIPTIVE", "processed windows/session labels do not establish independent animal endpoint"),
 "R-CELEGANS": ("PARTIAL_DESCRIPTIVE", "local structural fixture has unresolved source-object license and no biological future endpoint"),
 "R-GRID-TORUS": ("ELIGIBLE", "restricted topology-metric dissociation only; no structural W"),
 "R-DANDI-37": ("UNTESTABLE_MISSING_INPUT", "official 000037 is Openscope visual-cortex data, not frozen longitudinal M1 route"),
 "R-ALLOPTICAL": ("UNTESTABLE_MISSING_INPUT", "no released trial-level stimulation plus separate unperturbed path payload"),
 "R-BCI": ("PARTIAL_DESCRIPTIVE", "official derived subject tables only; raw Dryad archive unavailable behind WAF"),
 "R-MICRONS": ("PARTIAL_DESCRIPTIVE", "coregistration/nucleus inventory lacks same-root functional plus structural graph join"),
 "R-SYNTH": ("ELIGIBLE", "estimator validation only; excluded from biological confirmatory family"),
}

def sha(p: Path) -> str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def one(p: Path) -> dict: return {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
def serial(x):
 if isinstance(x, np.generic): return x.item()
 if isinstance(x, np.ndarray): return x.tolist()
 raise TypeError(type(x).__name__)
def write_once(p: Path, x: dict, overwrite: bool):
 if p.exists() and not overwrite: raise FileExistsError(f"refusing overwrite: {p}")
 p.write_text(json.dumps(x, indent=2, sort_keys=True, default=serial)+'\n',encoding='utf-8')

def e17_f3() -> dict:
 p=E17/'Figure3'/'FunctionalClustering'/'Data'/'SpatialCorr_RuleARuleB.mat'; d=loadmat(p)
 result={"inputs":[one(p)],"nested_unit":"released pair rows","source_reproduction":"Plot_Stats_spatialCorr.m: distance_um=3*col1; retain distance_um<=20; Spearman"}
 for name in ('branch_noise_RuleA','branch_noise_RuleB','branch_signal_RuleA','branch_signal_RuleB'):
  a=np.asarray(d[name]); distance=3*a[:,0]; keep=distance<=20; rho,pv=spearmanr(distance[keep],a[keep,1]); result[name]={"rows_before_filter":int(len(a)),"rows_after_filter":int(keep.sum()),"spearman_distance_correlation":float(rho),"pvalue_row_level_descriptive_only":float(pv)}
 return result
def e17_mat(relative: tuple[str,...]) -> dict:
 p=E17.joinpath(*relative)
 if loadmat is None: return {"inputs":[one(p)],"execution":"BLOCKED_MISSING_SCIPY","nested_unit":"source summary rows; no population inference"}
 d=loadmat(p); keys=sorted(k for k in d if not k.startswith('__'))
 return {"inputs":[one(p)],"mat_variables":keys,"nested_unit":"source summary rows; no population inference"}
def e19() -> dict:
 ps=[SLEEP/'e19_data'/'suball_sleep_staging.mat',SLEEP/'e19_data'/'suball_neural_sleep_param.mat',SLEEP/'e19_data'/'encode_corr_conds_final.mat']
 expected={"item_pixels":{"n_participants":34,"cluster":119,"spearman_rho":-0.5532467532,"pvalue":0.000689572,"slope":-0.02440725},"category":{"cluster":346,"spearman_rho":0.4695187166,"pvalue":0.005087157,"slope":0.01466378}}
 base={"inputs":[one(p) for p in ps],"nested_unit":"participant rows, source-selected clusters","frozen_predecessor_reference_not_compared":expected,"predecessor_path":"_workspace/ce/_archive/sleep-replay-routing-realdata-20260818/artifacts/realdata-results.json","label":"descriptive_only_not_a_confirmatory_pvalue"}
 if loadmat is None: return {**base,"execution":"BLOCKED_MISSING_SCIPY"}
 return {**base,"mat_variables":{p.name:sorted(k for k in loadmat(p) if not k.startswith('__')) for p in ps}}
def e15() -> dict:
 p=SLEEP/'e15_repo'/'ProcessedData'/'continuous_replay_number_1h_blocks.npy'; a=np.load(p,allow_pickle=True)
 root=a.item() if a.shape == () else a
 if not isinstance(root,dict) or not isinstance(root.get('data'),dict): raise ValueError('unexpected E15 processed-object schema')
 fields=root['data']; required=('grp','session','zt','is_cont')
 if any(not isinstance(fields.get(k),dict) for k in required): raise ValueError('missing E15 processed fields')
 indices=[set(fields[k]) for k in required]
 if any(s != indices[0] for s in indices[1:]): raise ValueError('unaligned E15 processed row indices')
 sessions=sorted(set(fields['session'].values())); groups={g:sorted({fields['session'][i] for i in indices[0] if fields['grp'][i] == g}) for g in sorted(set(fields['grp'].values()))}
 return {"inputs":[one(p)],"container_shape":list(a.shape),"dtype":str(a.dtype),"processed_schema":{"fields":list(required),"rows":len(indices[0]),"session_labels":len(sessions),"sessions_by_group":{g:len(v) for g,v in groups.items()}},"execution":"processed object schema inventoried; session-to-animal independence unresolved","nested_unit":"processed rows; session independence unresolved","frozen_predecessor_reference_not_compared":{"0_to_1":{"delta":-125.2857,"pvalue":0.0161057},"4_to_5":{"delta":-45.0238,"pvalue":0.12915196},"5_to_6":{"delta":-87.8333,"pvalue":0.00133728}},"predecessor_path":"_workspace/ce/_archive/sleep-replay-routing-realdata-20260818/artifacts/realdata-results.json","label":"descriptive_only_not_a_confirmatory_pvalue"}
def celegans() -> dict:
 with CE.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
 endpoints={(r['Source'].strip(),r['Target'].strip()) for r in rows}; nodes=sorted({x for pair in endpoints for x in pair})
 held=sorted(endpoints)[::10]; train=endpoints-set(held); retained=sum((a,b) in train for a,b in held)
 return {"inputs":[one(CE)],"rows":len(rows),"nodes":len(nodes),"chemical_rows":sum(r['Type'].lower().startswith('chem') for r in rows),"electrical_rows":sum(r['Type'].lower().startswith('elect') for r in rows),"synthetic_holdout":{"rule":"lexicographic_every_10th_edge","heldout_edges":len(held),"retained_in_train":retained},"claim_boundary":"structural_synthetic_fixture_only"}
def bci() -> dict:
 with BCI.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 return {"inputs":[one(BCI)],"columns":list(rows[0]) if rows else [],"rows":len(rows),"reproduction":"official derived directional accessibility table inventory only","protocol_deviation":"schema was inspected pre-gate; excluded from Holm"}
def microns() -> dict:
 base=RUN/'artifacts'/'input'/'microns'; ps=[base/'func_unit_em_match_release.csv',base/'nucleus_detection_v0.csv']
 return {"inputs":[one(p) for p in ps],"inventory":{"coreg_rows":sum(1 for _ in ps[0].open(encoding='utf-8')),"nucleus_rows":sum(1 for _ in ps[1].open(encoding='utf-8'))},"claim_boundary":"join_feasibility_only_no_W_change"}
def e17_f2() -> dict:
 p=E17.parent/'e17-acquisition-manifest.json'
 return {"inputs":[one(p)],"execution":"hash_reference_only_no_rerun","claim_boundary":"ineligible_dependent"}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--overwrite',action='store_true');args=ap.parse_args()
 details={"R-E17-F3": e17_f3() if loadmat is not None else {"inputs":[one(E17/'Figure3'/'FunctionalClustering'/'Data'/'SpatialCorr_RuleARuleB.mat')],"execution":"BLOCKED_MISSING_SCIPY"},"R-E17-F4":e17_mat(('Figure4','Data','DataSummary_CaImagingDendrites.mat')),
  "R-E17-F5":e17_mat(('Figure5','Data','Figure5Data_TransitionError.mat')),"R-E17-F2":e17_f2(),"R-SLEEP-E19":e19(),"R-SLEEP-E15":e15(),"R-CELEGANS":celegans(),"R-BCI":bci(),"R-MICRONS":microns()}
 routes=[]
 for route,(status,reason) in ROUTES.items(): routes.append({"route_id":route,"status":status,"reason":reason,"details":details.get(route,{})})
 payload={"schema":"nrm-route-dispositions-v1","routes":routes,"confirmatory_family":[],"assertions":{"bci_excluded":True,"celegans_excluded":True,"synth_excluded":True,"e17_f2_excluded":True,"forbidden":"no route establishes Delta_W_to_Delta_g_to_Delta_p"}}
 write_once(OUT,payload,args.overwrite);print(OUT)
if __name__=='__main__': main()
