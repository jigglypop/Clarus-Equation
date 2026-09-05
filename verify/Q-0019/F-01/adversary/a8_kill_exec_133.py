"""A8: kill 실행가능성(경로 존재 + 창 사전등록) 확인 및 13.3 정합.

kill 은 실행하지 않는다(6·7단의 몫). 코드 경로가 존재하고 요구 키를 내는지, 창이 카드에 박혀있는지만 본다.
13.3: 공통 궤도 O(B0)={alpha_a R_a B0} 에서 alpha_a 가 cell 마다 다르면 Regge 접착이 되는가?
"""
import ast, json, math, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import predict_lambda as PL

src=(ROOT/"predict_lambda.py").read_text(encoding="utf-8")
tree=ast.parse(src)
funcs={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
out={"functions_present":sorted(f for f in funcs if f.startswith("run_"))}
# 각 kill 이 요구하는 키가 실제로 반환 dict 에 literal 로 있는가
need={"two_level":["lambda_2_over_lambda_1_squared","lambda_2"],
      "irregular":["ratio_iso","ratio_max","sign_flip_iso","lambda_max_irregular"],
      "delta":["max_rel_change_0p01_to_0p005"]}
keys={}
for n in ast.walk(tree):
    if isinstance(n,ast.FunctionDef) and n.name.startswith("run_"):
        ks=[]
        for sub in ast.walk(n):
            if isinstance(sub,ast.Dict):
                ks += [k.value for k in sub.keys if isinstance(k,ast.Constant) and isinstance(k.value,str)]
        keys[n.name]=ks
out["kill_keys_resolvable"]={m:{k:(k in keys.get("run_"+m,[])) for k in ks} for m,ks in need.items()}
out["all_kill_keys_present"]=all(all(d.values()) for d in out["kill_keys_resolvable"].values())
# 창이 코드에도 박혀 있는가 (카드와 대조)
out["windows_hardcoded_in_code"]={
  "two_level":"kill_window_lambda2_over_lambda1_squared" in keys.get("run_two_level",[]),
  "irregular":"kill_window_ratio" in keys.get("run_irregular",[]),
  "delta":"kill_threshold" in keys.get("run_delta",[])}
# irregular 모드가 Cholesky 로 실패하지 않는지 (실행 아님: 경계 길이만 사전검사)
rng=np.random.default_rng(20260902)
signs=rng.choice([-1.0,1.0],size=10)
sq=2.0*(1.0+0.1*signs)
try:
    V=PL.vertices_from_squared_lengths(sq)
    ok=True; err=None
    # 그리고 linear_map 이 특이하지 않은지 (Gram solve 조건수)
    from itertools import combinations
    A=np.array([[ (V[i]-V[j])@b@(V[i]-V[j]) for b in PL.BASIS] for i,j in combinations(range(5),2)])
    cond=float(np.linalg.cond(A))
except Exception as e:
    ok=False; err=str(e); cond=None
out["irregular_geometry_realizable"]={"cholesky_ok":ok,"error":err,"boundary_gram_cond":cond,
                                      "squared_lengths":sq.tolist()}
# two_level: sub_cell 이 비정규 -> linear_map 이 도는지 (조건수만)
conds=[]
for cell in PL.sub_cells(PL.regular_simplex_vertices()):
    from itertools import combinations
    A=np.array([[ (cell[i]-cell[j])@b@(cell[i]-cell[j]) for b in PL.BASIS] for i,j in combinations(range(5),2)])
    conds.append(float(np.linalg.cond(A)))
out["two_level_subcell_gram_cond"]=conds

# --- 13.3 정합: cell 별 alpha_a (공통 아님) 는 Regge 접착되는가?
parent=PL.regular_simplex_vertices()
alphas=[1.0,1.1,0.9,1.05,0.95]
tet=[a*np.eye(4) for a in alphas]
blk=PL.nonlinear_block(parent,tet)
out["per_cell_alpha_mismatch"]={
  "alphas":alphas,
  "metric_fine_rms":blk["metric_fine_rms"],
  "metric_coarse":blk["metric_coarse"],
  "eps12_fine_rms":blk["eps12_fine_rms"],
  "eps12_coarse":blk["eps12_coarse"],
  "note":"12.4 simplicity 잔차가 0 이면 13.3 의미로는 여전히 궤도 안(각 cell 이 simple). Regge 계량 접착만 깨진다."}
# 공통 alpha 는?
tet2=[1.1*np.eye(4) for _ in range(5)]
blk2=PL.nonlinear_block(parent,tet2)
out["common_alpha"]={"metric_coarse_dev_from_identity":float(np.linalg.norm(blk2["coarse_metric"]-np.eye(4))),
                     "metric_coarse":blk2["metric_coarse"],"eps12_coarse":blk2["eps12_coarse"],
                     "note":"공통 alpha 는 coarse 계량을 alpha^2 I 로 정확히 옮긴다(중심화가 지움). 카드 recovers[0] 은 '중심화 뒤 0'을 말할 뿐."}
print(json.dumps(out,indent=2,default=float))
(HERE/"a8_kill_exec_133.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
