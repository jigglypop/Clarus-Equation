# Zebrafish e2-LR alignment probe

목표는 `e2[:, t]` neural frame을 같은 시각의 stage/head/yolk movement에 붙일 수 있는지 확인하는 것이다.

## e2 event sequence

- e2 shape: [3987, 3360]
- event count: 21
- sorted events: [80, 212, 331, 441, 589, 732, 863, 1016, 1192, 1324, 1614, 1873, 2100, 2293, 2475, 2588, 2695, 2850, 3004, 3151, 3262]
- event gaps: [132, 119, 110, 148, 143, 131, 153, 176, 132, 290, 259, 227, 193, 182, 113, 107, 155, 154, 147, 111]

## LR session match summary

| session | raw frames | raw laser onsets | stage/raw | timestamp.mat | best RMSE e2 frames | laser match | candidate |
|---|---:|---:|---:|---|---:|---|---|
| control/20221018_1027_g8s-lssm-none_10dpf | 14238 | 60 | 5.000 | True | 90.050 | False | False |
| control/20221116_1027_g8s-lssm-huc-none_8dpf | 14244 | 60 | 5.000 | False | 89.998 | False | False |
| control/20221116_1108_g8s-lssm-huc-none_8dpf | 13340 | 60 | 5.001 | False | 89.957 | False | False |
| control/20221116_1543_g8s-lssm-huc-none_8dpf | 13943 | 60 | 5.001 | False | 89.998 | False | False |
| control/20221116_1627_g8s-lssm-huc-none_8dpf | 14078 | 63 | 5.000 | False | 89.998 | False | False |
| exp/20221016_1556_g8s-lssm-chriR_8dpf | 18923 | 100 | 5.000 | False | 0.000 | True | True |
| exp/20221017_1453_g8s-lssm-chriR_9dpf | 20715 | 85 | 0.000 | True | 89.868 | False | False |
| exp/20221018_1626_g8s-lssm-chriR_10dpf | 14192 | 60 | 5.000 | False | 90.050 | False | False |
| exp/20221019_1053_g8s-lssm-chriR_11dpf | 17959 | 60 | 4.868 | False | 44.800 | False | False |
| exp/20221019_1609_g8s-lssm-chriR_11dpf | 17380 | 60 | 5.000 | False | 48.381 | False | False |

## Best attempted match

- session: exp/20221016_1556_g8s-lssm-chriR_8dpf
- RMSE e2 frames: 0.000
- MAE e2 frames: 0.000
- max abs error e2 frames: 0.000
- slope raw->e2: 1.000000
- laser schedule match: True
- stage/raw frame ratio: 5.000211
- timestamp certified: False
- candidate inferred alignment: True

## Verdict

- laser-schedule matches: 1 / 10
- timestamp-certified alignments: 0 / 10
- candidate inferred alignments: 1 / 10
- certified continuous decoding ready: False
- candidate inferred decoding ready: True

`exp/20221016_1556...`은 `e2` event sequence와 raw laser schedule이 정확히 맞고 stage/raw frame count도 5배에 가깝다.
하지만 그 session에는 `timestamp.mat`가 없으므로 이것은 candidate inferred alignment이지 final timestamp-certified alignment는 아니다.
최종 결론에는 explicit e2 timestamp, e2-resampled tracking, 또는 raw synchronized bundle verification이 필요하다.
