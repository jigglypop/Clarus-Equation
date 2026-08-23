# 실제 뇌 식 기반 발견 루프 하네스

Status: `ACTIVE / HIGHEST_PRIORITY_FOR_BRAIN_AGI_RESEARCH`

Date: 2026-08-23

뇌·기억·의식·AGI 연구의 최우선 경로는 실제 뇌에서 확립된 방정식·기전과
검증된 측정 과정을 출발점으로 삼고, CE 가설을 명시적 추가항으로 세운 뒤,
실제 뇌 데이터의 잔차와 반증 결과에 따라 판본을 바꾸어 가며 살아남는 식을
찾는 것이다. 여기서 "실제 뇌 식"은 뇌 전체를 설명하는 단일 완성식을
뜻하지 않는다. 1차 문헌과 데이터 문서가 지지하는 정의역별 기전식과
측정모형을 뜻한다.

이 하네스는 `brain_evidence_ladder.md`의 생물학적 증거 상한과
`empirical_calibration_loop.md`의 불일치 분류·식 개정 규약을 함께 적용한다.
두 규약과 충돌하면 더 엄격한 조건을 따른다.

## 1. 출발점

모든 뇌/AGI 연구 계약은 다음 세 층을 분리한다.

1. **확립된 생물학적 기전식**: 스파이크, 전도 지연, STDP, 3-인자
   가소성, 억제 회로, Dale 법칙, 항상성, 다중 시간척도, 수면 scaling 등
   원시 연산의 식·정의역·단위·시간척도·1차 출처.
2. **CE 가설**: 출발식에 더하거나 바꾸는 항, 새 상태, 결합 또는 경계조건.
   각 변경은 `[공리: 모델 선택]` 또는 `[경험식]`으로 표시하고 생물학적
   대응과 대안 모델을 함께 적는다.
3. **측정모형**: 숨은 신경 상태에서 실제 기록값으로 가는 관측 연산자,
   sampling, indicator kinetics, preprocessing, noise와 covariance. 측정모형을
   신경 동역학의 자유 파라미터에 숨기지 않는다.

권장 표기는 다음과 같다.

$$
dx_t=F_{\mathrm{bio}}(x_t,u_t;\theta_{\mathrm{bio}})\,dt
     +\Delta F_{\mathrm{CE}}(x_t,h_t;\phi)\,dt
     +G(x_t)\,dW_t,
$$

$$
y_k=\mathcal H(x_{t_k};\psi)+\varepsilon_k,
\qquad \varepsilon\sim\mathcal D(0,\Sigma).
$$

$F_{\mathrm{bio}}$는 출처가 고정된 생물 기준식, $\Delta F_{\mathrm{CE}}$는
시험할 CE 추가항, $\mathcal H$는 측정모형이다. 실제 과제가 이 형태와 다르면
억지로 맞추지 말고 동등한 상태·관측 분해를 계약에 정의한다.

## 2. 계약 필수 필드

`00-contract.md`에는 결과를 보기 전에 다음을 고정한다.

| 필드 | 내용 |
|---|---|
| `BIO_STARTING_MECHANISM` | 출발 기전식, 정의역, 단위·시간척도, 1차 출처 |
| `CE_DELTA` | CE 추가항·새 상태·경계조건과 추가 공리 |
| `MEASUREMENT_MODEL` | acquisition부터 분석 입력까지의 관측·전처리·noise 모형 |
| `DATA_PROVENANCE` | DOI/공식 저장소, 판본, 대상·세션·포함/제외 기준 |
| `DATA_SPLIT` | calibration/development/confirmation 또는 train/validation/held-out 분리 |
| `OBSERVABLES` | 단위, 분모, 불확도·공분산이 있는 사전 고정 관측량 |
| `RESIDUAL_RULE` | likelihood·잔차·적합도와 tension/실패 판정 |
| `FALSIFIER` | CE 추가항을 죽이는 독립 시험과 adverse control |
| `MATCHED_CONTROLS` | 생물 기준식, 단순 대안, 같은 정보·자원 대조군 |
| `MODEL_SELECTION` | 복잡도 벌점, 식별 가능성, 자유도와 비교 규칙 |
| `REVISION_TRIGGER` | 어떤 잔차가 어느 식 항의 새 판본을 허용하는지 |
| `CLAIM_CEILING` | 현재 evidence ladder 단계와 금지되는 상위 주장 |

자유 파라미터 수는 독립 관측 제약 수보다 작아야 한다. 구조적·실용적
식별 가능성을 각각 확인하고, 식별되지 않는 파라미터 조합은 개별 생물량으로
해석하지 않는다.

## 3. 발견 루프

한 판본은 다음 순서로만 진행한다.

1. **출처 잠금** — sourcer가 기전식, 수치, 측정모형, 데이터 판본과
   intervention 여부를 1차 출처에서 검증한다. `UNVERIFIED` 핵심 입력은
   채점 gate에 넣지 않는다.
2. **생물 기준식 재현** — CE 항을 넣기 전에 $F_{\mathrm{bio}}+\mathcal H$가
   지정 데이터와 baseline artifact를 재현하는지 확인한다. 실패하면 CE 항을
   더하지 않고 측정·구현·기준선 문제부터 분류한다.
3. **CE 식 사전 고정** — $\Delta F_{\mathrm{CE}}$, 초기조건, prior, 자유도,
   observables, residual, falsifier와 controls를 동결한다.
4. **분리 적합** — calibration/train에서만 파라미터를 적합한다. 구조 선택은
   development/validation까지만 허용하고 confirmation/held-out/개입 자료는
   마지막 한 번의 평가 전까지 봉인한다.
5. **잔차 진단** — 예측값·관측값·오차·잔차와 첫 분기 중간값을 기록하고
   `empirical_calibration_loop.md`의 D→I→P→C→B→T 순서로 원인을 분류한다.
6. **모델 비교** — CE 식이 생물 기준식과 단순 대안보다 held-out 예측 또는
   개입 판별에서 우수한지, 추가항 ablation에서 이득이 사라지는지 확인한다.
7. **판본 판정** — 통과, tension, STOP, 반례 또는 BLOCKED를 좁은 정의역으로
   기록한다. 음의 결과도 다음 식을 제한하는 발견으로 보존한다.

## 4. 식 개정 규칙

데이터에 맞게 식을 고치는 행위는 허용하지만, 다음 경계를 지킨다.

- 한 판본·한 사이클에는 구조 변경 한 건만 허용한다. 여러 항을 동시에
  바꾸어 무엇이 잔차를 줄였는지 잃지 않는다.
- T(이론) 잔차가 확정되기 전에 식을 바꾸지 않는다. 차원, 구현, 정밀도,
  convention, 기준선과 측정모형 문제를 먼저 기각한다.
- 구 식, 실패 데이터 범위, 반례, residual과 대조군 결과를 원장에 보존한다.
  새 식은 새 계약·새 판본·새 동결 fixture를 사용한다.
- 같은 confirmation/held-out 데이터를 식 제안과 확인에 함께 쓰지 않는다.
  확인 데이터를 본 뒤의 수정은 `[경험식]` 개발 판본이며 새 독립 확인이
  오기 전에는 `[예측]`으로 승격하지 않는다.
- threshold, tolerance, endpoint, seed, 제외 기준 또는 데이터 창을 결과에
  맞추어 바꾸지 않는다. 필요한 변경은 새 가설과 독립 falsifier를 가진 새
  계약이어야 한다.
- 관측량별 개별 파라미터 조정과 per-dataset retune을 금지한다. 같은 동결
  파라미터가 여러 비율·세션·개체·조건을 동시에 설명해야 한다.

## 5. 증거와 서술 상한

- L0 simulator 통과는 알고리즘 성립만 뜻한다.
- L1·L2는 관측 비율·창발 통계와의 정합이다.
- L3는 공개 실기록의 held-out 예측 정합이다.
- L4 개입 자료에서 대안 기전을 판별하기 전에는 "뇌가 이렇게 동작한다"고
  쓰지 않는다.
- 실제 데이터에 잘 맞는 새 식도 자동으로 정리나 생물학적 동일성이 되지
  않는다. 데이터에서 선택한 구조는 `[경험식]`, 독립 확인을 사전 고정한
  관측량은 `[예측]`이다.
- simulator와 실제 데이터가 충돌하면 생물학적 주장의 우선권은 실제 데이터
  쪽에 있다. simulator를 고쳐 실제 데이터 실패를 숨기지 않는다.

## 6. 후보 선택 우선순위

다음 연구 후보는 순서대로 우선한다.

1. 확립 기전식과 고품질 실제 데이터가 같은 변수·시간척도로 연결되는 경로.
2. 실제 데이터에서 baseline과 CE 추가항을 식별할 독립 개입 또는 held-out
   예측이 있는 경로.
3. 측정모형과 preprocessing artifact를 matched control로 제거할 수 있는 경로.
4. 그 다음에만 L0 simulator로 장치·식별 가능성을 진단하는 경로.

실제 데이터 입력이 막혔으면 필요한 DOI·schema·identity·intervention receipt를
재개 조건으로 남긴다. 이를 새 합성 seed 탐색으로 대체하지 않는다.

## 7. 적용 범위

이 우선순위는 뇌·기억·의식·AGI 연구 후보와 그 식·데이터 판본에 적용한다.
guard 벤치, 일반 코드 수정, 물리의 다른 분야나 단순 문서 작업을 뇌 연구로
임의 전환하지 않는다. 각 작업은 사용자가 지정한 범위를 유지한다.
