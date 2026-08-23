# BA-TR30 최종 보고 — 독립 후보 스트림 하 다항 차수·잡음 모델 식별

Status: COMPLETE

기계 상태: `SYNTHETIC_DEGREE_NOISE_MODEL_IDENTIFICATION / DEVELOPMENT_GO / CONFIRMATION_SEALED` (기계 판정이며 이론 지위가 아니다)

## 초록

BA-TR25–29는 사전 선언된 함수족 안에서의 조건부 내용 전이만 확립했고, "지배식이 데이터에서 나오는가"라는 질문은 열려 있었다. 본 run은 이 질문을 선언된 다항 차수 집합 $D=\{1,2,3\}$과 잡음 수준 $H=\{0,10^{-3},10^{-2}\}$ 위의 모델-클래스 식별 문제로 형식화하고, held-out 진리·예측 어느 쪽에서도 유도되지 않는 후보 스트림 아래에서 동결 학습 연산자를 시험했다. 개발 seed 16개의 주 fold 144개 전부에서 연산자는 생성기 차수를 정확히 선택하고($\hat d=d^\*$ 144/144) 직접 벡터 예측 게이트를 통과했으며, 선언 밖(차수 4) witness 16개 전부에서 fail-closed 기권했다. 초기 계약(R0)의 기권 상수는 검산 레인이 완전 반례로 무너뜨렸고, 수리(studentized PRESS·파시모니 floor·$N{=}24$)는 seed 일체 미개봉 상태에서 채택된 뒤 신선한 calibration으로 통과했다. 형식 지위는 [경험식: 동결 시뮬레이터 비교]이며, 자연 후보 생성·의미 인자 발견·생물학·AGI는 확립하지 않는다.

## 1. 서론

선행 경로가 남긴 정확한 상태는 다음과 같다. BA-TR28은 선언된 2차 조건에서 정확 예측을 보였으나 무작위 packet endpoint에서 STOP했고, 선언 밖 query-only 완성 $y_q+\delta$의 비식별 no-go를 확정했다. BA-TR29는 동결 2차 학습자가 hard negative를 구별함을 보였으나, 후보 은행이 환경 중심이고 decoy가 기준선에서 유도되었다는 한계를 스스로 기록했다. 따라서 살아 있는 질문은 "법칙을 아는 학습자가 잘 맞히는가"가 아니라 "법칙의 형태(차수)를 데이터가 고르게 할 수 있는가, 그리고 선언 밖에서는 정직하게 기권하는가"였다. 원장 우선순위 1(BA-TR30)이 요구한 조건 — 독립 후보 스트림, 예측 유도 decoy 금지, 다중 차수·잡음에서의 직접 벡터 예측 주 endpoint — 을 본 run의 계약이 그대로 동결했다.

## 2. 정의·표기

- cue $z\in\mathbb R^2$ (무차원 opaque), content $y\in\mathbb R^6$.
- $\phi_d(z)$: 총차수 $\le d$ 단항 특징. 열 수는 $\binom{d+2}{2}$로 $d=1,2,3$에서 각각 $3,6,10$ [정리, 11-math].
- 훈련 행 $N=24$: $y_i=\phi_{d^\*}(z_i)^\top C^\*+\eta\varepsilon_i$, $z_i\overset{iid}\sim\mathcal N(0,I_2)$, $C^\*$ 성분 $\overset{iid}\sim\mathcal U(-1,1)$, $\varepsilon_i\overset{iid}\sim\mathcal N(0,I_6)$ — 전부 seeded, query와 독립.
- leave-one-out 잔차 $e_i^{\rm loo}=y_i-\phi_d(z_i)^\top C_d^{(-i)}$, hat 대각 $h_{ii}$.

## 3. 공리

1. [공리] 학습 연산자는 fold의 참 잡음 수준 $\eta$를 부가 정보로 공급받는다. $\eta$ 추정은 주장 범위 밖이다.
2. [공리] 계수 스케일 $C^\*\sim\mathcal U(-1,1)$. 기권 상수 $\tau_{\rm class}=8\eta$와 예측 게이트의 유효성은 이 $O(1)$ 스케일에 조건부다.
3. [공리] 선행 승계(계약 §2): TR20 수리 runtime(SHA `5dc1ba…`), TR28 비식별 no-go, TR29 독립성 요구.

## 4. 동결 연산자와 정리

$$
C_d=\Phi_d^{+}Y,\qquad
s'_d=\frac1N\sum_{i=1}^N\lVert e_i^{\rm loo}\rVert_2\sqrt{1-h_{ii}},
$$

$$
\hat d=\min\{d\in D:\ s'_d\le1.5\min_{d'}s'_{d'}+10^{-8}\},\qquad
\hat y_q=\phi_{\hat d}(z_q)^\top C_{\hat d},
$$

기권: $\min_d s'_d>\max(10^{-8},8\eta)$이면 `CLASS_EXTERNAL_ABSTAIN`. cue 퇴화: $\operatorname{cond}(\Phi_3)>10^6$이면 `CUE_DEGENERATE`.

검산 레인이 정리로 확인한 것 [11-math]: (a) 특징 수 공식, (b) $N-1=13\ge10$이므로 LOO가 잘 정의됨($N=24$에서는 여유 확대), (c) hat-matrix 항등식과 재적합 LOO의 동치(최대 상대차 $7.5\times10^{-8}$은 부동소수 층위). studentized 보정 $\sqrt{1-h_{ii}}$는 LOO 잔차 성분의 표준편차를 정확히 $\eta$로 만들어 레버리지 꼬리를 제거한다 — 이것이 R0 반례를 죽인 기전이다(§7).

## 5. 산출: 개발 결과

calibration `117001` 1차 통과(수정 루프 미발동) 후 development `117101..117116`을 개봉했다. seed당 주 fold 9($D\times H$)+witness 1, 총 주 144·witness 16.

| endpoint | 결과 | 게이트 | 최박 여유 |
|---|---|---|---|
| 차수 식별 $\hat d=d^\*$ | 144/144 | 필수 | — |
| 예측 $e$, $\eta=0$ | max $4.50\times10^{-15}$ | $10^{-10}$ | $2.2\times10^4$배 |
| 예측 $e$, $\eta=10^{-3}$ | max $1.86\times10^{-3}$ | $2\times10^{-2}$ | 10.7배 |
| 예측 $e$, $\eta=10^{-2}$ | max $1.50\times10^{-2}$ | $2\times10^{-1}$ | 13.3배 |
| witness(차수 4) 기권 | 16/16 | 필수 | 169배 |
| 은행 진리 선택 (부) | 144/144 | — | — |
| 은행 순서 영수증 | 위반 0/160 | fail-closed | — |

대조군: association shuffle 기각 160/160, wrong-cue 성공 0/144, 강제 $d=1$ on $d^\*\in\{2,3\}$ 게이트 실패 96/96 (필수 결과 충족), `CUE_DEGENERATE` 0건(최대 cond 104.3). 파시모니 ablation($\rho=0$) 진단: 일치 109/144 — 불일치 35건 전부 저차 잡음 셀의 과대 차수 선택으로, 파시모니 규칙이 식별을 실질 담당함을 보여 준다(게이트 아님, 기록만).

후보 은행은 fold마다 진리 원소 1(생성기 산출, 모델 비경유), 타 셀 clean content 4, 비query 셀 평균 norm에 맞춘 seeded distractor 3으로 구성되었고, 은행 SHA-256과 순서 카운터가 $\hat y_q$ 계산 전에 직렬화되었다. 예측·기준선에서 유도된 원소는 없다 — TR29의 한계 두 가지(환경 중심 은행, 기준선 유도 decoy)가 이 설계에서 제거되었다.

## 6. 관측 비교

해당 없음 — 전 fold가 seeded 합성 apparatus이며 외부 관측값을 인용하지 않았다(10-sources SKIPPED). 시뮬레이터 결과를 생물학적 주장으로 승격하지 않는다.

## 7. 수정 이력 (정직 기록)

- **R0→R1 (P0-1)**: 초기 계약의 $\tau_{\rm class}=8\eta$ + mean-L2 LOO 결합은 $N=14,p=10$ 레버리지 꼬리($h_{ii}\to0.9999$) 때문에 $d^\*=3,\eta>0$ fold의 79–80%를 false abstain시켰다 — 완전 반례가 `artifacts/math-verify/counterexample.json`에 있다. 12-routes가 우회 후보 4개를 수치 평가했고, R-B(studentized PRESS)+R-C(floor)+R-D($N{=}24$) 결합이 채택되었다. $N{=}24$ 확인 시뮬레이션은 전 9셀 실패 0/6000, 전량 통과 보수 하한 0.931이었다.
- **R1→R2 (감사 Gate: REVISE, revise contract 1/2)**: 숨은 공리($C^\*$ 스케일)를 명시 공리로 승격하고, 은행 독립성을 기계 검증 가능한 영수증 규격으로 강화했다.
- look-elsewhere 규율: 두 수리 모두 seed 일체 미개봉 상태에서 채택되었고, 레인 수치는 설계 근거로만 사용되었으며, 신선한 calibration이 통과를 제공했다.

## 8. 미완성 과제와 한계

- [경험식] 본 결과 전체는 동결 시뮬레이터 비교다. 합격 주장은 "선언된 유한 다항족·잡음 수준·$O(1)$ 계수 스케일에서, 독립 후보 스트림 하 합성 모델-클래스 식별과 직접 내용 예측"까지다.
- witness 기권과 TR28 no-go의 대응(감사 P2-e 이월): 차수-4 witness의 기권은 "선언 밖 완성은 시도조차 하지 않는다"는 fail-closed 성질이며, TR28의 $y_q+\delta$ 비식별 no-go를 침해하지 않고 오히려 운영화한다. 선언 밖 법칙의 **식별**은 여전히 불가능 영역으로 남는다 — 유한 관측은 선언된 클래스 밖의 완성을 결정하지 못한다.
- P1-3 잔존: $\eta=10^{-2}$ 게이트의 사전 여유 추정이 2.24배로 최박이었다(실측 여유는 13.3배). 게이트는 완화되지 않았다.
- $\eta$ 공급 공리: 잡음 수준을 스스로 추정하는 학습자는 후속 과제다.
- 차수 선택은 $D=\{1,2,3\}$ 유한 메뉴에서의 선택이다. 메뉴 밖 구조(비다항, 유리식, 조각별)는 witness처럼 기권 대상일 뿐 식별 대상이 아니다.
- confirmation `117201..117232` 봉인 유지 — 본 보고는 development 증거만 주장한다.

## 9. 재현성

- run: `_workspace/ce/brainruntime-independent-stream-degree-id-20260822`
- 모듈/벤치/테스트: `reality_stone/python/reality_stone/clarus/runtime_independent_stream_degree_id.py`, `runtime_independent_stream_degree_id_benchmark.py`, `tests/test_runtime_independent_stream_degree_id.py`
- 동결 영수증: `artifacts/source-freeze.json` (git HEAD `baaf20c`, 16파일 SHA-256, runtime `5dc1ba5722ffa430f5c4dd4502defd49130dbe36c7f5522f28a7df36338fed26` 대조 일치)
- 기계 결과: `artifacts/calibration-results.json`, `artifacts/development-results.json` (fold별 $e,\hat d,s'$, 기권, 은행 SHA·카운터, 대조군 전부)
- focused 검증: `.claude/hooks/python.cmd pytest tests/test_runtime_independent_stream_degree_id.py -p no:cacheprovider -q` → `2 passed`. 전체 pytest·전체 bench는 실행하지 않았다.
- 검산 재현: `artifacts/math-verify/` (P0 반례, R-D 확인 6000-draw 로그 포함)

## 10. 참조

- 선행 원장: `_workspace/ce/brain-algorithm-route-ledger.md` (BA-TR28, BA-TR29 행; 2026-08-22 접근)
- 선행 run: `brainruntime-rotating-low-degree-content-transfer-20260822`, `brainruntime-low-degree-hard-negative-transfer-20260822` (40-final; 2026-08-22 접근)
- 본 run 레인: `00-contract.md` (R2), `11-math.md`, `12-routes.md`, `20-audit.md` (Gate: PASS), `30-implementation.md`, `31-validation.md`
