# Final report

Status: COMPLETE

## 비전공자 요약

기존 성공을 버리지 않고 체크포인트로 쌓는 방식으로 다음 알고리즘을 찾았다.
V5의 안정적인 sparse H20 경로는 유지하고, 불안정했던 adaptive dense
전문가는 제거했다.  대신 sparse 예측이 마지막 관측값에서 얼마나 멀리
움직일지를 기존 훈련 데이터에서 한 개의 전역 gain으로 보정했다.

새 식은

\[
\widehat Y=P+0.7868543064870357(S-P)
\]

이다.  즉 sparse parent의 예측 이동량을 78.69% 유지하고 21.31%를
persistence 쪽으로 수축한다.

## 형식 지위

- `[정리]`: 이 gain은 고정한 훈련창의 정규화 제곱오차를 `[0,1]` 구간에서
  유일하게 최소화하며 훈련 목적에서는 두 endpoint보다 나쁘지 않다.
- `[산출]`: 기존 train 176개 창에서 gain은 `0.7868543064870357`이다.
- `[경험식]`: 공개 V7 개발 데이터와 새 256-seed 개발 블록에서 V5 및
  persistence 대비 paired 하한이 양수였다.
- `[예측]`: 같은 결과가 사전 커밋된 미래 confirmatory 블록에서도
  재현될지는 아직 검사하지 않았다.
- `[미완성]`: sparse 고유 우위, open-world causal discovery, AGI 연결.

## 새 체크포인트

256개의 새 OOD seed에서 candidate RMSE는 `0.548433`이었다.  V5 parent는
`0.554139`, persistence는 `0.584029`, zero-bridge shrinkage는 `0.558918`,
기존 V7 consensus는 `0.560359`였다.  paired 95% 하한은 각각
`+0.001106`, `+0.017897`, `+0.002196`, `+0.003579`로 모두 양수였다.

남은 sparse component radius는 `0.781420`, 미래 read는 0, 비유한 출력은
0이었다.  반면 제외한 adaptive comparator는 새 블록에서도 최대 radius
`1.311462`와 17.19%의 상한 위반을 보였다.

## 정확한 해석

이번 돌파는 **희소 인과 모델이 dense보다 우월하다는 결과가 아니다**.
동일 절차의 dense shrinkage가 `0.548594`로 사실상 동률이었다.  입증된 새
개발 체크포인트는 "훈련 전용 covariance-aware convex shrinkage가 안정적인
부모 예측을 보존하면서 H20 오차를 줄였다"는 것이다.

## 다음 단계

R1 하나만 동결하고, 최소 256개의 또 다른 fresh seed를 쓰는 정식 등록을
코드와 결과보다 먼저 Git에 커밋해야 한다.  그 뒤 같은 결합 게이트를 한 번
실행한다.  R2 contractive consensus를 같은 블록에서 함께 돌려 좋은 쪽을
고르면 그 블록은 다시 개발 데이터가 되므로 금지한다.  V7 test split은 계속
봉인한다.

재현 파일은 `fresh_parent_anchor_pilot.py`, 수학 검산은 `11-math.md`, 경로
비교는 `12-routes.md`, fresh 결과는 `31-validation.md`에 있다.

CE_RUN=_workspace/ce/agi-v8-breakthrough-20260811

