# BA-SRM3 pre-support audit

Date: 2026-08-23

Gate: `AUTHORIZED_TO_RUN_SUPPORT_ONLY / MODEL_NOT_AUTHORIZED`

## 확인된 경계

- BA-SRM2 STOP은 보존했다.
- BA-SRM3는 별도 candidate와 별도 directory를 사용한다.
- train manifest를 재생성하거나 재표집하지 않는다.
- source correction은 official dynamics의 sign-matched response QC로 한정했다.
- 이미 본 train 집계는 contract에 공개했다.
- response-QC∩complete-target support, target MAD와 모델 score는 아직 계산하지 않았다.
- development와 confirmation은 봉인돼 있다.

## P0/P1

P0: 0.

P1은 모두 향후 empirical gate다: support 수, target scale, input rank, covariance condition,
KRR selection, local rank stability, gauge, missingness sensitivity와 controls. 이 항목이 아직
미측정이라는 사실은 실행을 허용하지만 성공 주장을 허용하지 않는다.
