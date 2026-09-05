"""A6: 제곱거리 좌표 사상의 정확 항등식 증명 (수치 아님, 조합론).

N_a := diag(w_a),  w_a[e] = 1/3 if a notin e else 0.
중심화 입력 D_a (sum_a D_a = 0) 에 대해 y = sum_a N_a D_a.
lambda^2 = ||y||^2 / ((1/5) sum ||D_a||^2)  의 최대/평균을 정확히 구한다.

edge e 성분끼리 분리된다: y[e] = (1/3) sum_{a notin e} D_a[e] = -(1/3) sum_{a in e} D_a[e]
(중심화 sum_a D_a[e]=0 사용!). e 는 두 원소이므로 y[e] = -(1/3)(D_i[e]+D_j[e]).
분모 (1/5) sum_a ||D_a||^2 = (1/5) sum_e sum_a D_a[e]^2.
edge 별로: 최대화 max (1/9)(x_i+x_j)^2 / ((1/5) sum_a x_a^2) subject to sum_a x_a = 0.
"""
import json, math
from fractions import Fraction as Fr
import numpy as np
HERE=__file__.replace("\\","/").rsplit("/",1)[0]
# 정확: sum_a x_a =0, 목적 (1/9)(x_i+x_j)^2, 제약 (1/5)sum x_a^2 = 1
# (x_i+x_j) 를 sum=0 초평면에서 최대화: 벡터 c = e_i+e_j 의 사영 c - (2/5)1, 노름^2 = 2 - 4/5 = 6/5
# max (x_i+x_j)^2 subject to sum x^2 = 5  ->  = 5 * 6/5 = 6
# 따라서 lambda^2 = (1/9)*6 = 2/3.  모든 edge 에서 동일 -> 스펙트럼 전부 sqrt(2/3).
lam2 = Fr(1,9)*Fr(6,1)
# 평균(등방 rms): 각 edge 마다 방향의 (1/9)*|proj c|^2 * (평균) 
# 40차원 등방 평균: E[(1/9)(x_i+x_j)^2] over sum_a x_a=0 균등... 정확 trace 계산:
# 연산자 T: (D_a) -> y, T T^* 의 trace. 각 edge 독립, 사영 노름^2 = 6/5 배 (1/9) 스케일
# rms lambda^2 = 5 * trace(T T^*) / 40, trace = sum_e (1/9)*(6/5) = 10*(6/45)=4/3
tr = Fr(10,1)*Fr(1,9)*Fr(6,5)
iso2 = Fr(5,1)*tr/Fr(40,1)
out={
 "edge_coord_lambda_max_squared_exact": [lam2.numerator, lam2.denominator],
 "edge_coord_lambda_max_exact_float": float(lam2)**0.5,
 "edge_coord_trace_exact": [tr.numerator, tr.denominator],
 "edge_coord_lambda_iso_squared_exact": [iso2.numerator, iso2.denominator],
 "edge_coord_lambda_iso_exact_float": float(iso2)**0.5,
 "all_ten_singular_values_equal": True,
 "proof": "중심화 sum_a D_a[e]=0 에서 y[e]=-(1/3)(D_i[e]+D_j[e]); edge별 분리; sum=0 초평면 위 |e_i+e_j| 사영 노름^2=6/5; lambda^2=(1/9)(6/5)*5=2/3 모든 방향.",
 "card_metric_coord_values": {"lambda_max_sq": 11/9, "lambda_iso_sq": 13/60},
 "contrast": "길이 좌표: 완전 축퇴 2/3(스펙트럼 갈림 없음, 전부 <1). 계량 좌표: 1/3,5/9,11/9 세 갈래.",
}
print(json.dumps(out,indent=2))
open(HERE+"/a6_edge_exact.json","w").write(json.dumps(out,indent=2))
