# 30-implementation — 재현 검산 artifact

Status: COMPLETE

production code와 canonical 문서는 수정하지 않았다. 연구 run 내부에 표준 라이브러리
전용 검산기

`artifacts/verify_opportunity_cost.py`

를 추가했다. 이 검산기는 다음만 확인한다.

1. 두 outcome entropy, weighted excluded information과 KL 수치,
2. $C_I=p_U[-\ln p_U+H(q)]$,
3. $F_T(\rho)-F_T(\gamma_T)=k_BT D(\rho\|\gamma_T)$의 유한 고전 대각 예,
4. information, energy, action과 energy-density 차원 벡터.

검산기 성공은 기회비용의 중력 ontology 또는 암흑부문 동일성을 검증하지 않는다.
