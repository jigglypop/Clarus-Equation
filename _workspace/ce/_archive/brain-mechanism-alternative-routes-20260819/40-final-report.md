# Alternative brain-mechanism routes: final report

Status: COMPLETE

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 실험, 계산, 감사를 새로 수행하지
않았으며, 기존 stage 파일(`20-audit.md`, `31-validation.md`)과
`_workspace/ce/brain-algorithm-route-ledger.md`의 판정을 요약만 한다.
**이 보고서는 새 증거를 만들지 않았다.**

## 계약 질문

선행 run이 확정한 분리(native causal-STDP는 binding/factor transfer 실패, bounded
supervised projection은 capacity ceiling) 위에서, hyperparameter가 아니라 *방법*을
바꿨을 때 어떤 결론이 살아남는지를 동결된 후보 우주(M0--S1)로 경로별 독립 판정한다.

## 감사 판정 (20-audit.md 그대로)

- 1차 감사는 `Gate: REVISE` (P0 없음, P1 4건). 계약·수학 레인 수정 후 2차 독립
  읽기 전용 감사가 `Gate: PASS`.
- 인가 범위는 M0 후 M1뿐. M2, M3, G1--G3, C1, S1은 동결 상태로 보류(pending).
- 어떤 결과도 biological consolidation, real-brain geometry, metacognition,
  selfhood, consciousness로 기술할 수 없다는 제한이 감사에 명문화되어 있다.

## 검증 결과 (31-validation.md 그대로)

- M0 (supervised rank-4 write): development에서 rank 4가 첫 통과 rank (16/16
  complete GO), confirmation seeds `99201..99232`에서 rank 4 = 32/32. 이것은
  supervised structural capacity threshold이며 획득(acquisition) 증거가 아니다.
- M1 (fixed-clock delayed three-factor): development 16/16, confirmation 32/32
  complete binding GO. 여섯 adverse control(zero clock, sign-flip, time reversal,
  eligibility reset, no replay, target shuffle)은 모든 circuit에서 clean accuracy
  `0.0`. attractor cosine gain 최소 `0.653667`, deleted-cue 최대 `0.179505`로
  동결 `0.20` abstention 경계 유지.
- M1의 factorized held-out transfer는 이 run에서 실행되지 않았고 판정에 포함되지
  않는다 (31-validation 명문).

## 원장 인용

`brain-algorithm-route-ledger.md` 실행 증거 원장:

- BA-M0: `CONFIRMED_SIMULATOR` 32/32, **[산출]** 고정 simulator 계산. capacity
  ceiling으로만 사용, brain-algorithm 후보 재실행 금지.
- BA-M1: `CONFIRMED_SIMULATOR` 32/32, **[산출]**. factor transfer, prediction,
  biological consolidation은 별도 입증 필요.
- 이 run이 보류한 경로들은 후속 run이 인계했다: M1 transfer → BA-T1 (`STOP`
  11/16), M2/M3 → BA-M2/BA-M3 (`STOP`), G1--G3 → BA-G1/BA-G2/BA-G3D (`STOP` 계열),
  C1 → BA-C1 (`STOP`), S1 → BA-S1 (`DEFERRED_SYNTHETIC`).

## 미완 항목

- 이 run 자체의 31-validation은 IN_PROGRESS로 남아 있었다. M2--S1 경로의 판정은
  이 run 안에서는 미완성이며, 위 원장 행들이 후속 run 증거로 이를 대체한다.
- 재개 조건: 없음 — M0/M1 confirmation은 봉인 완료되었고, 원장 규칙상 M0/M1의
  brain-algorithm 후보 재실행은 금지된다.
