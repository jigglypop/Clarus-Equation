# Canonical integration

Status: COMPLETE

## Scope

이 run은 runtime이나 estimator를 구현하지 않았다. 감사에서 안정화한 식을 두 정본 문서에 반영했다.

- `docs/6_뇌/11_리만계량_라우팅_논문.md`: 기존 원고를 output-relative conditional Fisher $G$, held-out predictive routing $R$, ordered pair $\mathcal B/\Xi$, 비식별 정리, PFC 결과의 좁은 재해석 중심으로 전면 개정했다.
- `docs/6_뇌/00_읽기지도.md`: 실제 뇌 기록의 현재 우선식을 $\mathcal B=(G,R)$로 올리고, 기존 $p/q$/graph 전역식은 `[공리: 모델 선택]`인 synthetic global-state candidate로 격리했다.

## Preserved evidence

PFC Exp1/Exp2, row-fold NLL, decoder와 middle-link 수치는 기존 동결 결과 그대로 보존했다. 재현 링크 네 개는 현재 실제 위치인 `_workspace/ce/_archive/`로 갱신하고 존재를 확인했다.

## Removed promotions

- $C^{-1}$을 일반 nonlinear chart의 local brain metric으로 부르는 서술.
- SPD geometry가 routing을 열거나 $W\to G\to x$를 매개한다는 확정 서술.
- 한 점/상수 SPD에서 curvature 또는 geodesic dynamics를 읽는 서술.
- SCC, BrainRuntime memory/control 결과를 생물학적 기억·의식 알고리즘으로 올리는 서술.

## Non-implementation boundary

새 dataset, seed, fit, benchmark, dependency는 추가하지 않았다. 다음 코드 작업은 별도 actual-data contract가 입력 적격성을 통과한 뒤에만 허용한다.
