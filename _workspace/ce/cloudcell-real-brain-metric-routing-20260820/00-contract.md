# CloudCell 실제 신경 기록 적격성 계약

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brain-metric-routing-equation-rebuild-20260820`

## 목적

새로 정비한 두 객체

$$
\mathcal B_{j,c}^{A\to B}(z)
=\left(G_{j,c}^{o\leftarrow A}(z),R_{j,c}^{A\to B}\right)
$$

를 실제 신경 기록에 적용하기 전에, 로컬 CloudCell/Hallinen 자료가 각 객체의 입력 조건을 만족하는지 판정한다. 이 run은 입력·출처·식별 가능성 감사다. 모델 적합, 결과 선택, seed 실행, 새 데이터 다운로드는 하지 않는다.

## 고정 자료 범위

- 로컬 원자료: `data/external/cloudcell/{AML18_moving,AML32_moving,AML310_moving}.tar.gz`
- 로컬 추출본: `data/external/cloudcell/extracted/`
- 공식 분석 코드: `data/external/cloudcell/PredictionCode/`
- 1차 출처: Hallinen et al. (2021), eLife, DOI `10.7554/eLife.66135`, OSF `DPR3H`
- 후속 인과 라우팅 후보: Randi et al. (2023), Nature, DOI `10.1038/s41586-023-06683-4`, OSF `E2SYT`

AML18은 GFP control이며 GCaMP 신경 신호 패널로 승격하지 않는다. AML32 기록과
AML310-named archive 내부의 `AKS297.51_moving` 기록만 GCaMP 후보로 본다. 이
AKS 내부 root 불일치는 AML310 archive에만 해당한다. archive filename과 내부
root 이름의 차이는 명시적 path adapter 없이는 숨기지 않는다.

## 판정할 질문

### Q1 — 출력-상대 계량

같은 recording 안의 신경 형광 상태와 동기화된 미래 행동 출력으로 calibration-only conditional Fisher tensor

$$
G_{ab}^{o\leftarrow A}(z,c)
=\mathbb E_{h\mid z,c}\mathbb E_{o\mid z,h,c}
\left[
\partial_a\log p(o\mid z,h,c)
\partial_b\log p(o\mid z,h,c)
\right]
$$

를 정의할 수 있는가?

필수 조건은 같은 recording의 neural/behavior 길이 정렬, 고정 선두 guard 뒤의
단조 시간축, gap을 가로지르지 않는 window, GCaMP 신호, train-only chart,
held-out future output, full-rank Fisher 추정 가능성이다. 통과해도 지위는
**관측 형광–행동 output geometry**뿐이다. 물리 피질 계량, 구조 계량, 곡률 또는
시냅스 메커니즘으로 부르지 않는다.

### Q2 — 조건부 예측 라우팅

검증된 source population $A$와 target population $B$에 대해

$$
R_{j,c}^{A\to B}(\ell,\delta)
=\frac1{N_{\rm test}}
\sum_{t\in\mathcal T_{\rm test}}
\left[
\log p_1(x^B_{t+\delta}\mid H_t^B,z^A_{t-\ell:t},c)
-\log p_0(x^B_{t+\delta}\mid H_t^B,c)
\right]
$$

를 해부학적으로 해석할 수 있는가?

필수 조건은 canonical neuron identity 또는 사전 검증된 population label, 서로 겹치지 않는 $A/B$, 동일 clock, 미래 누출 없는 history, held-out time block이다. row index, XYZ 위치, correlation ordering, outcome-selected cluster를 해부학 label의 대용으로 쓰지 않는다.

### Q3 — 인과 라우팅

source intervention, sham, 반대 방향, non-target control이 있는가? 없으면 $R>0$이어도 lagged predictive transfer일 뿐이며 causal routing은 결과와 무관하게 `BLOCKED`다.

## 입력 게이트

각 자료군에 대해 다음을 기록한다.

1. archive/source SHA-256 및 공식 source 연결;
2. recording 수와 signal class;
3. neural/behavior timestamp의 단조성·길이 정렬;
4. 행동 output과 누락률;
5. recording 내 unit 추적과 recording 간 identity의 구분;
6. anatomical label, context/trial, intervention 유무;
7. local extraction과 official loader 경로의 일치 여부;
8. raw-data 비수정 및 synthetic result 비혼입.

## 사전 판정 언어

- `PASS_INPUT`: 다음 별도 empirical preregistration을 작성할 수 있다.
- `APPARATUS_PATH_BLOCKED`: 자료는 있으나 재현 loader가 현재 경로에서 실행되지 않는다.
- `BLOCKED_SOURCE_TARGET_DEFINITION`: $A/B$의 생물학적 정의가 없다.
- `BLOCKED_SIGNAL_CLASS`: primary neural signal로 사용할 수 없다.
- `BLOCKED_INTERVENTION`: causal routing을 식별할 개입이 없다.
- `DIAGNOSTIC_ONLY`: 임의의 고정 row partition으로 수치 계산은 가능하지만 뇌 라우터 후보로 승격하지 않는다.

## 결정 규칙

- Q1은 GCaMP recording의 same-record neural/behavior 정렬이 통과하면 별도 metric-only empirical run으로 입장시킨다.
- Q2는 canonical $A/B$가 없으면 해부학적 route를 중단한다. 임의 row split은 기계적 diagnostic으로만 남기며 “뇌의 알고리즘” 후보 선택에 쓰지 않는다.
- Q3은 randomized source perturbation 자료가 없으면 중단한다.
- AML18 GFP에서 같은 lag signature가 재현되면 그 signature를 calcium memory 또는 neural routing의 증거로 사용하지 않는다.
- 다음 실제-뇌 알고리즘 경로는 결과를 보고 threshold를 조정해 고르지 않는다. 입력 조건상 가장 직접적인 개입 자료를 우선한다.

## 산출물

- `10-sources.md`: 1차 출처와 로컬 자료 provenance
- `11-math.md`: 새 $G/R$ 식과 CloudCell 관측량의 정합성
- `12-routes.md`: metric-only, diagnostic predictive, causal-intervention 경로 비교
- `artifacts/cloudcell-input-audit.md`: recording/schema 수준 입력 감사
- `20-audit.md`: 안정 snapshot 판정
- `40-final-report.md`: 다음 실제 자료 경로 선택

## 금지 주장

이 run만으로 다음을 주장하지 않는다.

- 뇌의 물리적 Riemannian metric 또는 곡률;
- 구조 연결 $W$에서 $G$로의 인과 사슬;
- $G\to R$ 또는 metric mediation;
- CloudCell row partition의 해부학적 route;
- SCC가 기억·의식의 원인이라는 주장;
- C. elegans simulator/recording 결과의 인간 뇌 일반화.
