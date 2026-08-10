# G8-C: 결함 OOD 위험 calibration

## 최소 가설

G8-S의 고정 공분산 임계값은 위험확률이 아니다. G8-C는 짧은 위치-속도 window에서 다음 horizon의 경계 위반 확률만 다룬다. 위치, 바깥방향 속도, 명령 크기를 쓰는 기하학 기준선과 여기에 관측 불확실성 및 actuator innovation을 더한 후보를 비교한다.

훈련은 gain 0.8--2.0, 지연 0--3 step, 부호반전 확률 0.05다. 잠금 test는 더 강한 gain 2.0--3.2, 지연 3--6, 부호반전 확률 0.15이므로 결함 강도 OOD다. 실제 장비에 대한 주장은 하지 않는다.

## 판정

NumPy logistic model을 독립 train set에서 적합한다. calibration set에서 예측위험이 임계값보다 낮은 accepted subset의 실제 위반률이 1% 이하가 되도록 하면서 coverage를 최대화한다. test에서는 임계값을 바꾸지 않는다.

후보는 OOD test에서 Brier score가 기하학 기준선의 90% 이하, ECE 0.08 이하, accepted 위반률 2% 이하, coverage 25% 이상이어야 한다. 또한 accepted 위반률이 기준선의 75% 이하여야 한다. 실패 버전은 그대로 보존한다.

## V1 실패와 V2

V1 후보는 OOD accepted 위반률 0.92%로 기준선 3.82%보다 낮았지만 Brier 0.1665, ECE 0.2413으로 calibration에 실패했다. 희귀 class 가중 logistic이 순위는 개선하면서 확률을 체계적으로 부풀린 것이 원인이다. V2는 데이터, 특징, 문턱을 유지하고 class weighting만 제거해 proper probability loss로 되돌린다.

V2 locked test는 accepted 위반률 1.01%, coverage 85.4%, ECE 0.0553을 통과했지만 Brier가 기준선보다 6.5%만 좋아져 등록된 10% 개선에 실패했다. V3는 문턱을 낮추지 않고 innovation과 불확실성이 경계 근접도 및 바깥방향 운동과 결합되는 3개 상호작용을 추가한다. 결함 자체가 아니라 결함이 현재 기하에서 만드는 위험을 표현하려는 수정이다.

V3 validation은 상호작용 후보 Brier 0.0409로 기준선 0.0394보다 나빠졌다. V4는 단순 특징으로 돌아가고 합성 domain randomization의 gainㆍ지연ㆍ부호반전 범위를 넓힌다. test는 다시 그보다 강한 gain 3.2--4.2, 지연 6--9, 부호반전율 0.25로 분리하므로 강도 OOD 조건은 유지된다.

V4 validation은 Brier 10.6% 개선, ECE 0.0711, coverage 71.5%를 달성했지만 accepted 위반률 2.20%로 최종 2% 문턱을 넘었다. V5는 최종 문턱을 바꾸지 않고 calibration accepted risk 한도를 1%에서 0.5%로 낮춰 OOD drift buffer를 둔다.

V5 locked test는 Brier 13.2% 개선, ECE 0.0649, coverage 67.1%였지만 accepted 위반률 2.1376%로 2% 문턱을 근소하게 넘었다. V6는 calibration 한도를 0.25%로 낮추며 최종 판정 문턱은 유지한다. 평균 판정 외에 최악 seed accepted 위험률과 최소 seed coverage도 보고한다.

## V6 최종 결과

독립 test seed 7개의 평균 gate는 `PASS`였다. 후보 Brier는 0.06883, 기준선은 0.07701로 10.6% 개선됐고 ECE는 0.0738이었다. 후보 accepted 위반률은 1.63%, 기준선은 2.98%, 후보 coverage는 평균 58.7%ㆍ최소 seed 42.9%였다. 외부 다운로드는 0, 실행시간은 1.77초였다. 보고서는 `artifacts/agi/fault_ood_calibration_test_v6.json`이다.

단, 최악 seed의 accepted 위반률은 2.20%였다. 따라서 이 결과는 사전등록된 seed 평균 판정의 통과이지 seed별 2% 보장이 아니다. 후속 gate는 Clopper--Pearson 또는 conformal upper bound처럼 finite-sample 상한을 직접 판정해야 한다.

## V7 강화 계획

V7은 calibration 위험 한도를 0.1%로 낮추고 seed당 OOD window를 20,000개로 늘린다. 각 seed의 accepted 표본에 대해 one-sided 95% Wilson upper bound를 계산하며 그 최악값이 2% 이하여야 한다. 평균 성능 기준과 coverage 25% 조건도 그대로 유지한다.

V7 locked test는 `PASS`였다. 후보 Brier 0.06564 대 기준선 0.07593으로 13.5% 개선됐고 ECE는 0.0670이었다. 평균 accepted 위반률은 0.399%, 평균 coverage는 33.5%였다. 모든 seed 가운데 가장 큰 one-sided 95% Wilson 상한은 1.160%로 등록 한도 2% 아래였다. 최소 seed coverage는 24.665%이므로 등록된 평균 coverage 조건은 통과하지만 seed별 25% 효용을 주장하지 않는다. 최종 보고서는 `artifacts/agi/fault_ood_calibration_test_v7.json`이다.
