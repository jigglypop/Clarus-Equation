# 실데이터 검증 결과

Status: COMPLETE_FAIL_GATE

## A0

`BLOCKED_PARENT_RECEIPT`. CloudCell에는 edge별 parent/delay/STP receipt와 무작위 source intervention이 없다. 인과식은 계산하지 않았다.

## A1 validation

- positive recordings: `0/5`
- mean delta log score: `-0.2764339264`
- mean circuit-shuffle delta: `-0.2592270095`
- GFP delta: `-0.2163492882`
- decision: `A1_FAIL`, `A2_ACTIVATED`

실패 영수증은 Frobenius norm ratio `1.0864–1.3666`, spectral-radius ratio `1.2745–1.4195`였다. 사후 positive gain이 predictor scale을 키웠다.

## A2 confirmation

- positive recordings: `4/5`
- mean delta log score: `4.0093442e-05`
- median delta log score: `2.2691343e-05`
- one-sided exact sign-flip: `p=0.0625`
- mean circuit-shuffle delta: `5.8820413e-05`
- GFP delta: `5.8156545e-06`
- decision: `PASS_PREDICTIVE_FEATURE=false`

A2는 A1의 큰 왜곡은 제거했으나 shuffle을 이기지 못했고 작은 표본의 confirmatory gate도 통과하지 못했다. 추가 threshold, graph, cycle, ridge, horizon 조정은 계약상 금지한다.

검증 산출물:

- `artifacts/validation-result.json`
- `artifacts/confirmation-result.json`
- `artifacts/attempt-1-apparatus-failure.json`
- `artifacts/attempt-2-apparatus-failure.json`

