# Zebrafish laser perturbation to behavior gate

Figshare figure8/c/LR chunk의 boutInfo.mat를 사용해 left/right laser 조건이 회전 행동 방향을 바꾸는지 검증했다.

이 검증은 neural trace -> behavior decoding이 아니라 perturbation -> behavior closure다.

## 결과

| group | left n | left mean angle | right n | right mean angle | left-right | p |
|---|---:|---:|---:|---:|---:|---:|
| control | 264 | -0.753407 | 219 | -1.333287 | 0.579880 | 0.871826 |
| experimental | 308 | 48.403138 | 306 | -42.963969 | 91.367107 | 0.000200 |

- experimental/control effect ratio: 157.562119
- pass: True

## 해석

- control에서는 left/right laser에 따른 회전 방향 차이가 거의 없다.
- experimental fish에서는 left laser와 right laser가 반대 부호의 큰 회전각을 만든다.
- 따라서 척추동물 단계에서 motor output 항은 임의 잡음이 아니라 방향성 perturbation에 의해 조절되는 닫힌 행동 출력으로 볼 수 있다.
- 아직 남은 최종 검증은 자연 neural trace가 tail/stage movement를 예측하는 activity -> behavior gate다.
