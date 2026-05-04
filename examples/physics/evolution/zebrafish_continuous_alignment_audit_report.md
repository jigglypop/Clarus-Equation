# Zebrafish continuous decoding alignment audit

목표는 neural activity frame으로 speed, heading, turn angle을 직접 예측하는 continuous decoding gate다.

현재 받은 partial chunk가 그 목표를 지원하는지 점검했다.

## 현재 가능한 것

- e2 neural matrix 있음: True
- behavior bout frame label 있음: True
- left/right laser frame label 있음: True
- stage/head/yolk tracking txt 있음: True
- LR 일부 폴더 timestamp.mat 있음: True

## 빠진 것

- neural mat 안에 stage/head/tail 좌표 있음: False
- neural mat 안에 e2 column별 absolute timestamp 있음: False

## 판정

- activity -> behavior-frame gate 가능: True
- activity -> direction gate 가능: True
- 현재 partial만으로 continuous movement decoding 가능: False

## 해석

- 지금 partial 자료는 neural activity와 discrete behavior labels는 연결한다.
- 하지만 e2의 각 column이 stage tracking의 어느 시간/프레임에 해당하는지 알려주는 per-frame alignment가 없다.
- 그래서 현재 partial만으로 speed, heading, turn angle을 직접 예측하면 임의 정렬이 되어 검증이 무효가 된다.
- 다음에는 더 큰 raw chunk나 e2-frame timestamp가 포함된 파일을 받아야 continuous decoding으로 넘어갈 수 있다.
