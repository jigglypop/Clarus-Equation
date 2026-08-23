# BrainRuntime weight--metric--dynamics intervention: final report

Status: COMPLETE

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 실험, 계산, 감사를 새로 수행하지
않았으며, 기존 stage 파일(`20-audit.md`, `21-g2-audit.md`, `22-g3-audit.md`,
`31-validation.md`)과 `_workspace/ce/brain-algorithm-route-ledger.md`의 판정을
요약만 한다. **이 보고서는 새 증거를 만들지 않았다.**

## 계약 질문

simulator 전용으로, 이전에 결여된 사슬 $\operatorname{do}(W)\to\Delta g$,
$\operatorname{do}(W)\to\Delta x$, $g$ 기반 $x$ 예측을 세 질문으로 분리해 판정한다.
G1은 joint effect만 식별 가능하며 metric mediation은 식별하지 못한다.

## 감사 판정 (그대로)

- G1 (`20-audit.md`): 1차 수학 감사 REVISE (P0 2건) → 계약 수리 후 최종 수학 감사
  PASS (P0 없음); 구현 전 smoke 감사에서 disconnected apparatus 결함 발견 → 수리 후
  재감사 PASS, 16-seed development 인가. focused tests 5 passed.
- G2 (`21-g2-audit.md`): 최초 "metric sufficiency" 문구는 구현 전 기각; v4 계약의
  incremental utility 시험만 `Gate: PASS` (결과 함의 없음).
- G3-D (`22-g3-audit.md`): 학습된 mediation은 영구 차단 (treatment가 $W$를 바꾸고
  response/recall 모두 post-treatment 함수). 별도의 falsification 지향 진단만 PASS.

## 검증 결과 (31-validation.md 그대로)

- G1: circuits 16, circuit-level GO **0/16**, verdict `STOP`. 유일한 보편 실패
  게이트는 사전등록 per-circuit endpoint advantage $\ge 0.05$ — 관측 범위
  `[0.0066936463, 0.0455462234]`, 평균 `0.0311778448`, bootstrap 하한
  `0.0247662745`. sub-threshold simulator 효과만 지지.
- G2: circuit-level GO **0/16**, verdict `STOP`. 모든 circuit에서 horizon-matched
  직접 `B_h`가 `D+g`를 이겼고 (mean Gaussian loss `15.78889148` vs `-2.48452940`),
  permuted metric이 named metric을 8/16에서 이겨 orientation 특이성도 불안정.
- G3-D: 첫 artifact는 `APPARATUS_INVALID` (reconstruction residual gate 위반,
  seeds `97701..97716` 퇴역, 격리 보존). 대체 실행(`97801..97816`)은 recall
  advantage 동시 95% LCB `0.66444402`로 강했으나 SPD-change advantage LCB
  `-0.04261143`, same-arm correlation LCB `-0.80389991`로 verdict `STOP`;
  `mediation_status = BLOCKED_NOT_IDENTIFIED`.
- confirmation seeds `99401..99432`, `99601..99632`, `99701..99732`는 모두 미개봉.

## 원장 인용

`brain-algorithm-route-ledger.md` 실행 증거 원장:

- BA-G1: `STOP` 0/16, **[산출]** 부정 계산. 효과크기 문턱·noise control의 사후
  변경 재시도 금지.
- BA-G2: `STOP` 0/16, **[산출]** 부정 계산. "metric sufficiency/고유 정보" 경로
  퇴역; 새 독립 mediator 개입 없이 재개 금지.
- BA-G3D-v1: `APPARATUS_INVALID`, **[미완성]** — 과학 결과로 사용하지 않음, seed
  `97701..97716` 영구 퇴역.
- BA-G3D: `STOP`; mediation `BLOCKED_NOT_IDENTIFIED`, **[산출]** 부정 계산 +
  **[미완성]** mediation.

## 미완 항목

- BLOCKED: metric mediation 식별 — 재개 조건 (원장 그대로): independently
  manipulable mediator 또는 충분한 causal state model.
- confirmation 미개봉은 미완이 아니라 규율이다: 세 경로 모두 development에서
  STOP이므로 confirmation 개봉 근거가 없다.
