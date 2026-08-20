# 상태-조건부 라우팅 경로

Status: COMPLETE

## 결과 전 경로 판정

| 순위 | 경로 | 상태 | 통과 조건 | STOP |
|---:|---|---|---|---|
| 1 | R1 명시 reference/code source join | `OPEN` | target ROI에서 정본 NeuroPAL source까지 immutable mapping 확보 | response outcome이나 geometric result로 tie-break |
| 2 | R2 outcome-blind geometric join | `OPEN_CONDITIONAL` | 논문 방법에 고정된 frame·unit·radius·ambiguity rule 재현 | 규칙 사후 선택 또는 복수 후보 잔존 |
| 3 | R3 publication-native `Fconn` event object | `OPEN_CONDITIONAL` | 공식 bytes/checksum에서 identity·assignment·failure·order 복원 | code semantics만 있고 source receipt 없음 |
| 4 | R4 processed Funatlas pair table | `PROCESSED_ONLY` | pair mean/q-value 재현 | event pre-state·assignment·carryover가 없어 state-conditioned test에는 STOP |

## R1 판정

**[산출]** frozen exemplar의 event target은 `TargetPlaneSegmentation`으로만 연결된다.
target table에는 explicit NeuroPAL identity나 외부 identity-table reference가 없다.
response-side `neuropal_ids`는 존재한다. 따라서 R1은
`BLOCKED_EXPLICIT_SOURCE_JOIN`이며 effect 분석으로 넘어가지 않는다.

## R2 판정

**[미완성]** target pixel/depth와 tracked-neuron centroid가 함께 존재할 수 있으나,
coordinate frame, z calibration, registration transform, matching radius와 ambiguity
처리가 동결되지 않았다. 이 경로는 새 matching algorithm이므로 현재 run에서 열지
않는다. 결과를 보지 않는 독립 validation 자료가 생기기 전까지 `OPEN_CONDITIONAL`을
유지한다.

## R3 판정

**[공리: 외부 입력]** 논문은 대부분 무작위 source 선택과 일부 수동 표적 예외를 함께
기술한다. NWB exemplar에는 이를 event별로 구분할 assignment field가 없다.
publication-native `Fconn` object가 이를 보존할 가능성은 있으나, 이번 run에는 공식
compact bytes와 checksum이 없다. 판정은 `BLOCKED_ASSIGNMENT_RECEIPT`다.

## 후속 control 설계

**[예측]** R1 또는 R3가 통과하면 다음 별도 계약에서만 상태-조건부 효과를 시험한다.

- $g$와 $r$ encoder는 train fold pre-state에서만 동결한다.
- primary null은 animal/session/randomization stratum과 history matching을 보존한
  event–state permutation이다.
- autocorrelation을 고려한 lag/block-circular shift를 결과 전에 고정하고 매
  permutation마다 fitting과 cross-validation을 다시 수행한다.
- source-label permutation은 실제 randomization stratum 안에서만 시행한다.
- global-gain-only $M_1$, additive full-state $M_{2,\mathrm{add}}$, optical/ROI adverse
  control을 함께 보고한다.

## 결론

**[미완성]** 현재 활성 경로는 R1/R3의 input 복원이다. schema-only probe의 판정은
`BLOCKED_EXPLICIT_SOURCE_JOIN / BLOCKED_ASSIGNMENT_RECEIPT`다. 이는 가설의 반증이
아니라 apparatus 입장 실패다. threshold·latent dimension·decoder·seed를 바꿔
response 효과를 먼저 계산하는 우회는 금지한다.
