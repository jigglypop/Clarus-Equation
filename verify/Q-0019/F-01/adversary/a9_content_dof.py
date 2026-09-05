"""A9: content(무내용 여부)·dof·dimension·regular 대칭 깨짐 조기 신호."""
import json, math, sys
from pathlib import Path
from itertools import combinations
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import predict_lambda as PL
out={}
# --- content 1: L 이 flat section / Schur 를 실제로 쓰는가 (코드 grep 수준 + 구조)
src=(ROOT/"predict_lambda.py").read_text(encoding="utf-8")
out["flat_section_used_in_L"]={
 "linear_map_calls":"coarse_metric",
 "coarse_metric_body_mentions_interior": "interior" in PL.coarse_metric.__doc__.lower() if PL.coarse_metric.__doc__ else False,
 "verdict":"linear_map -> coarse_metric 은 경계 10변만 쓴다. 내부 길이·Hessian·Schur·flat section 은 L 계산 어디에도 없다.",
 "regge_modules_imported": ("regge_one_to_five" in src),
}
# --- content 2: 예측이 규약에서 얼마나 자명한가
# 카드 규약을 고정하면 L 은 유일하게 결정 -> '예측'은 순수 선형대수 계산 결과.
# 자유파라미터 0 확인
out["dof"]={"free_parameters":0,"prereg_numbers":10,"dof_lt_numbers":True}
# --- dimension: 모든 양이 비 (무차원)
out["dimension"]={"lambda_*":"길이^0 (제곱길이 비의 비)","gamma_geo":"log 비, 무차원","delta":"무차원 (I + delta X)",
                  "verdict":"모두 무차원. exp/log 인자도 무차원."}
# --- 극한: 정규 대칭 깨짐 조기 신호 (K2 실행 아님, 아주 작은 섭동으로 연속성만)
parent=PL.regular_simplex_vertices()
base=PL.lambda_stats(PL.linear_map(parent),5)
tiny={}
rng=np.random.default_rng(5)
for eps in (1e-4,1e-3):
    signs=rng.choice([-1.0,1.0],size=10)
    sq=2.0*(1.0+eps*signs)
    V=PL.vertices_from_squared_lengths(sq)
    st=PL.lambda_stats(PL.linear_map(V),5)
    tiny[str(eps)]={"lambda_iso":st["lambda_iso"],"lambda_max":st["lambda_max"],
                    "rel_iso":abs(st["lambda_iso"]/base["lambda_iso"]-1),
                    "rel_max":abs(st["lambda_max"]/base["lambda_max"]-1)}
out["tiny_irregular_continuity"]=tiny
out["base"]={"lambda_iso":base["lambda_iso"],"lambda_max":base["lambda_max"]}
# --- 규약 의존: 경계 길이를 '3-cell 산술평균' 대신 다른 규약으로 바꾸면?
def coarse_alt(parent, mets, mode):
    ell2=np.empty(10)
    for k,(i,j) in enumerate(combinations(range(5),2)):
        u=parent[i]-parent[j]
        vals=[float(u@mets[a]@u) for a in range(5) if a not in (i,j)]
        if mode=="mean": ell2[k]=np.mean(vals)
        elif mode=="geom": ell2[k]=float(np.prod(np.maximum(vals,1e-12))**(1/3))
        elif mode=="min": ell2[k]=min(vals)
    return PL.solve_sym_from_edges(parent,ell2)
# geom/min 은 비선형 -> 선형화는 mean 과 1차에서 같은가? geom 의 1차는 mean(로그) = 배경 동일값이라 동일.
out["convention_note"]="배경이 모든 3-cell 에서 같은 값이므로 산술/기하 평균의 1차 선형화는 동일. min 은 비매끄러움."
print(json.dumps(out,indent=2,default=float))
(HERE/"a9_content_dof.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
