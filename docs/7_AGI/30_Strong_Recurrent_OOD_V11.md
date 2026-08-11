# Strong Recurrent and OOD Audit V11

## 질문

V10의 확인된 local–cloud 메커니즘이 동일한 raw sequence를 학습하는 강한 recurrent model과
OOD horizon/noise 변화에서도 경쟁력이 있는가를 검사했다.

## 등록 비교

[정의] 네 모델은 V10, hidden 20 Elman RNN, hidden 20 GRU, hidden 3 Elman RNN이다. Elman-3는
tick당 추정 multiply가 `72`로 V10의 `76`과 가장 가깝다. Elman-20과 GRU-20은 각각 `820`,
`2460`으로 의도적으로 더 강한 계산 예산을 허용했다.

[경험식] 16개 fresh seed에서 ID, noise, horizon, combined 네 panel을 한 번 실행했다.

| panel | V10 | Elman-3 | Elman-20 | GRU-20 |
|---|---:|---:|---:|---:|
| ID | 0.660156 | 0.755859 | 0.998779 | 0.998535 |
| noise | 0.593018 | 0.736084 | 0.997559 | 0.998291 |
| horizon | 0.540283 | 0.622803 | 0.787109 | 0.998047 |
| combined | 0.520264 | 0.604980 | 0.781982 | 0.997803 |

[경험식] V10과 seed별 강한 Elman-20/GRU-20 중 최댓값의 차이는 ID `-0.338623`, noise
`-0.405762`, horizon `-0.457764`, combined `-0.477539`였다. 네 95% interval은 모두 0보다
작었다.

[경험식] V10은 계산량을 맞춘 Elman-3에도 모든 panel에서 패했다. Horizon과 combined에서
V10 accuracy는 사전등록 하한 `0.55` 아래로 내려갔다. 전체 판정은 `STOP`이며 14개 gate 중
10개가 실패했다.

## 형식 지위

[정리] V10 bounded transition의 contraction theorem은 그대로 유효하다. V11은 그 정리를
반박하지 않는다.

[경험식] V10 local/shared interaction이 자체 factorial control보다 낫다는 개발·confirmation
결과도 그대로 유효하다.

[미완성] 그러나 안정적이고 인과적으로 필요한 메커니즘이라는 사실은 강한 학습형 sequence
model과 경쟁할 충분조건이 아니다.

[삭제된 예측] V10이 강한 learned recurrent comparator와 noise/horizon OOD에서 우위를
유지한다는 예측은 V11의 완전한 음성 비교로 제거한다. 같은 V10 구조를 AGI 진전이나 일반
recurrent 우위로 서술해서는 안 된다.

## 남는 설계 교훈

V10은 hand-designed stable feature generator이고, GRU는 과제에 맞게 recurrent operator
자체를 학습했다. 다음 유효 경로는 V10을 반복 실행하거나 threshold를 낮추는 것이 아니라,
small-gain 제약 아래에서도 interaction operator를 학습 가능하게 만드는 것이다. 그 새
모델은 V11의 Elman-3와 GRU-20을 고정 대조군으로 다시 만나야 한다.

## 재현

- evaluator: `reality_stone/python/reality_stone/clarus/local_cloud_ood_benchmark.py`
- one-shot runner: `examples/agi/local_cloud_ood_run.py`
- run: `_workspace/ce/agi-v11-strong-ood-20260812/`
- result SHA-256:
  `456E95F5E0DC7BE89E86924F721C1A01696BF2D6DA71AD73C8414AC2B6167181`
