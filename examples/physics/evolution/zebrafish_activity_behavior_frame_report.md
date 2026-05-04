# Zebrafish activity to behavior-frame gate

Figshare figure8/g chunk의 `e2` neural activity와 `FrameBout` 행동 bout frame을 사용했다.

이 검증은 연속 tail/stage movement decoding은 아니고, neural activity가 행동 bout frame과 baseline frame을 구분하는지 보는 association gate다.

## 결과

- activity shape region x frame: [3987, 3360]
- bout frames: 147
- baseline pool frames: 2270
- mean AUC: 0.887293 ± 0.033238
- mean balanced accuracy: 0.812500 ± 0.036968
- permutation AUC mean: 0.500352
- p: 0.000167
- pass: True

## 해석

- 저차원 neural activity만으로 행동 bout frame과 baseline frame을 holdout에서 구분할 수 있다면 activity-behavior 결합이 보인다는 뜻이다.
- 이 결과는 perturbation->behavior보다 한 단계 더 자연 activity 쪽에 가깝지만, 아직 연속적인 tail/stage movement 예측은 아니다.
- 다음 최종 gate는 frame별 tail/stage speed, heading, turn angle을 neural trace와 시간 정렬해서 직접 예측하는 것이다.
