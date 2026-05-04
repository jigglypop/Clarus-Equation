# Zebrafish supplementary continuous closure audit

목표는 `Others_Supplementary`를 받은 뒤 zebrafish continuous movement gate가 닫히는지 확인하는 것이다.

## archive

| item | value |
|---|---:|
| md5 ok | True |
| files | 109 |
| mat files | 19 |
| txt files | 35 |

## bridge checks

| check | value |
|---|---:|
| e2-named files in supplementary | 0 |
| timestamp-named files in supplementary | 0 |
| matched e2 session files | 0 |
| has e2 timestamp variable | False |
| has e2-resampled behavior | False |
| Z-tracking mats | 15 |
| Z-tracking mats expZ only | True |
| CalTrace mats | 1 |
| interpolation mats | 2 |

## verdict

- timestamp-certified continuous ready: `False`
- verdict: `blocked_missing_e2_behavior_bridge`

Others_Supplementary supplies Z-position/stage alignment and calcium trace/interpolation QA, but not an explicit e2 timestamp or e2-resampled speed/turn trace.

따라서 현재 받은 partial + Others_Supplementary만으로는 `e2[:, t] -> speed/turn/heading` 최종 gate를 닫을 수 없다.
이것은 dynamics 실패가 아니라 공개 chunk의 alignment 정보 부족으로 판정한다.
