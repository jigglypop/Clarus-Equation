# Zebrafish activity to direction gate

Figshare figure8/f chunk의 `e2`, `LeftLS`, `RightLS`를 사용해 neural activity window가 left/right 조건을 구분하는지 봤다.

이 검증은 laser-evoked 방향성 activity gate이며, 아직 spontaneous continuous movement decoding은 아니다.

## 결과

- activity shape region x frame: [3987, 3360]
- left trials: 11
- right trials: 10
- window frames: 5
- pre-baseline frames: 5
- AUC: 1.000000
- balanced accuracy: 1.000000
- permutation AUC mean: 0.355527
- p: 0.001996
- pass: True

## 해석

- leave-one-trial-out에서 left/right laser trial을 neural activity만으로 구분한다.
- 이미 left/right laser가 반대 방향 회전 행동을 만든다는 gate가 통과했으므로, 이 결과는 activity-direction-output 결합을 보강한다.
- 표본 수가 21 trial로 작으므로 최종 결론은 continuous movement decoding에서 다시 확인해야 한다.
