# Zebrafish candidate continuous decoding gate

이 gate는 timestamp-certified 최종 검증이 아니라 inferred alignment 위에서 신호가 있는지 보는 정찰 검증이다.

## Alignment

- raw offset e2 -> raw: 9339
- raw frames: 18923
- stage rows: 94619
- stage/raw ratio: 5.000211
- status: candidate_inferred_not_timestamp_certified

## Best Decoding

| target | best lag e2 frames | R2 | mse/base | shift p | candidate |
|---|---:|---:|---:|---:|---|
| speed | 10 | 0.123460 | 0.876540 | 0.066667 | True |
| turn | 150 | 0.010998 | 0.989002 | 0.066667 | False |

- final continuous gate pass: False

## Interpretation

- speed는 inferred alignment에서 약한 후보 신호가 있다.
- turn/heading은 닫히지 않았다.
- 따라서 현재 partial로는 최종 continuous movement gate를 통과했다고 보지 않는다.
- 다음에는 explicit e2 timestamp 또는 e2-resampled behavior trace가 필요하다.
