# 학습된 계산 기하와 수면 재정렬 구현 기록

Status: COMPLETE

## 범위

`20-audit.md`가 허용한 문서 교정과 작은 수학 검증 fixture만 반영했다. 제품 runtime, 학습 알고리즘, public API와 benchmark 판정은 변경하지 않았다.

## 변경

1. `docs/7_AGI/1_AGI.md`에서 WAKE/NREM/REM 순환을 생물학적 알고리즘이 아니라 CE-AGI의 `[공리: 모델 선택]`으로 한정했다. 국소 유지, replay, 동기화와 표현 변화의 개별 근거를 전체 $\Delta W\to\Delta g\to\Delta x(t)$ 사슬의 증거로 합치지 않았다.
2. `docs/7_AGI/3_Sleep.md`에서 스칼라 수축률과 사람의 수면박탈 결과를 분리했다. NREM 곡률 평탄화와 REM 재조합을 software label 아래의 구현 가설로 내리고, `4.87/26.2/68.9` 반복은 선언된 toy map의 조건부 산출로 제한했다. 파괴적 망각 완화는 동일 계산량 baseline과 phase ablation을 가진 `[예측]`으로 바꿨다.
3. `docs/7_AGI/15_Equations.md`에 raw 연결 $W$, 문맥별 유향 비용, shortest-path quasi-distance와 SPD/Riemannian distance의 typed-object 경계를 추가했다. $W$에서 비용을 얻으려면 $\Phi_q(W,A,\tau,q)$와 측정 protocol을 선언하도록 했다.
4. `artifacts/verify_lgs_math.py`와 로그는 양의 비용 유향 multigraph의 one-edge APSP 정리, 영향 집합, many-pair/untouched-pair fixture를 exact `Fraction` 연산으로 재현한다.

## 불변식

- 강한 부모 주장 `LGS-N1`--`N3`을 활성 정본에 되살리지 않았다.
- 경험 가설 `LGS-H1`--`H6`을 정리나 산출로 승격하지 않았다.
- 기존 5계층 runtime, 코드 경로, 사전등록 seed와 결과 artifact를 수정하지 않았다.
- 수학 fixture는 run의 `artifacts/`에만 두고 제품 구현으로 승격하지 않았다.

## 검증

검증 명령과 원문 결과는 `31-validation.md`에 기록했다. 전체 pytest와 benchmark는 공용 코드가 바뀌지 않았고 사용자가 FULL 검증을 요청하지 않았으므로 실행하지 않았다.
