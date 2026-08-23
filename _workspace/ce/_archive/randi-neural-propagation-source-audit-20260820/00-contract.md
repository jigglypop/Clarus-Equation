# Randi E2SYT 개입 라우팅 자료 입장 계약

Status: COMPLETE

PREDECESSOR: `_workspace/ce/cloudcell-real-brain-metric-routing-20260820`

## 목적

Randi et al. (2023)의 *C. elegans* single-neuron optogenetic stimulation/whole-brain
calcium 자료가 실제-뇌 라우팅 검증에 필요한 source, intervention, target,
identity, timebase와 control을 한 분석 단위로 제공하는지 판정한다. 이번 run은
공개 1차 출처와 machine-readable manifest의 acquisition/source audit다. 큰
neural payload 다운로드, 효과 추정, 모델 선택, threshold 조정은 하지 않는다.

## PREDECESSOR_EVIDENCE

| 선행 결과 | 상태 | 증거와 SHA-256 | 보존 주장 | 재시도 금지 조건 |
|---|---|---|---|---|
| CloudCell route map | `PASS_INPUT / BLOCKED` | `cloudcell-real-brain-metric-routing-20260820/12-routes.md`, `8cecd2d06544615f2328c2a88c5e62e5f097234f82c00605f4254e2a5d80c19f` | output-Fisher 입력은 존재; 임의 row routing은 diagnostic뿐 | row/XYZ/correlation partition을 anatomy로 승격하지 않음 |
| CloudCell machine validation | `PASS_INPUT_SCHEMA` | `cloudcell-real-brain-metric-routing-20260820/31-validation.md`, `51e03925682fc58b4a5ea9dd8f4d1f3a5a181b720df45d4bb451a9126f4bfd1f` | 22 recording schema/clock apparatus | lag score를 neural memory로 재해석하지 않음 |
| CloudCell final decision | `BLOCKED_SOURCE_TARGET_DEFINITION / BLOCKED_INTERVENTION` | `cloudcell-real-brain-metric-routing-20260820/40-final-report.md`, `8b316b609915d8a350464c972b5bb941a128667c2047474f65890b0c2f739890` | canonical identity와 source-specific intervention 자료가 다음 dependency | CloudCell threshold·horizon·row split retune 금지 |

## 결과 전 후보 순위

| 순위 | 후보 | 선택 이유 | 독립 falsifier / STOP |
|---:|---|---|---|
| 1 | Randi E2SYT | canonical neuron identity와 targeted optogenetic intervention이 같은 whole-brain recording에 존재한다고 1차 논문이 보고 | event-level source/target/time/session/control join이 공개 파일에서 재현되지 않으면 STOP |
| 2 | IBL Brain-wide Map | mammalian simultaneous multi-area spikes와 task context | intervention-defined causal edge가 아니며 공식 simultaneous-session cache가 없으면 입장 보류 |
| 3 | CloudCell metric-only | 로컬 same-record neural/behavior clock | anatomical/causal routing 재개 금지; output geometry 보조 경로로만 유지 |

E2SYT는 양성 수치가 기대되어서가 아니라 현재 dependency인 canonical identity와
source intervention을 가장 직접적으로 제공할 가능성이 있어 선택한다.

## 고정 질문

### Q1 — source provenance와 acquisition

다음을 공식 paper, repository, OSF provider에서 확인한다.

1. DOI, OSF project ID, release/object 식별자와 접근일;
2. 파일 tree, byte size, provider checksum 또는 immutable version;
3. dataset/code licence와 재배포 경계;
4. raw/processed/derived layer의 구분;
5. 최소 재현 subset을 payload 결과를 열지 않고 선택할 수 있는가.

### Q2 — event-level causal apparatus

하나의 trial/event 행에서 다음 join key를 모두 구성할 수 있는가?

$$
(animal,session,event,t_{stim},A_{id},u_{stim},B_{id},x^B_{pre},x^B_{post},condition).
$$

필수 필드는 animal/session ID, targeted source neuron ID와 identification confidence,
stimulation onset/duration/power, synchronized target-neuron traces, target identity,
pre/post timebase, WT/control background, failed/no-response/missingness 표지다.

### Q3 — 허용되는 개입 효과

관측 predictive score $R^{A\to B}$와 개입 효과를 혼동하지 않는다. 최소 개입
estimand 후보는

$$
\tau_{A\to B}(\Delta)
=\mathbb E\left[
y^B_{t+\Delta}(do(u_A=1))-y^B_{t+\Delta}(do(u_A=0))
\right]
$$

이다. `do(u_A=0)`의 식별에는 같은 trial의 pre-stimulation baseline만으로 충분한지,
별도 no-stim/sham/failed-stim/control event가 필요한지를 source lane과 math lane이
독립 판정한다. source target 선택이 무작위가 아니면 그 사실을 보존하고 causal
claim ceiling을 낮춘다.

### Q4 — 새 계량과의 관계

자극 전 chart와 고정 output likelihood로 $G^{o\leftarrow A}$를 정의할 수 있는지
검사하되, $G$가 $\tau$를 매개한다는 주장은 금지한다. 동일 source/target/event가
있어도 $G$와 intervention effect는 ordered pair로 보고한다.

## acquisition gate

- `PASS_SOURCE`: primary source와 public object/version이 결합됨.
- `PASS_EVENT_SCHEMA`: Q2의 join이 공개 machine-readable 파일로 재현 가능함.
- `PASS_INTERVENTION_INPUT`: source intervention과 적절한 comparator를 고정할 수 있음.
- `CONDITIONAL_PROCESSED_ONLY`: event-level derived data는 있으나 raw/preprocessing
  독립 감사가 불가능함.
- `BLOCKED_CONTROL`: no-stim/sham/failed-stim comparator가 없어 do-effect가 식별되지 않음.
- `BLOCKED_IDENTITY`: source/target canonical identity 또는 confidence가 없음.
- `BLOCKED_ACQUISITION`: stable object/manifest/checksum 또는 필요한 payload에 접근할 수 없음.

## 필수 adverse controls

후속 empirical contract가 열릴 경우 다음을 결과 전에 고정한다.

1. pre-stimulation baseline 및 time-shifted pseudo-onset;
2. non-target neurons와 distance/expression-matched target controls;
3. reverse-direction events가 존재할 때 $B\to A$;
4. stimulation power/duration, global-state, photostimulation artifact controls;
5. WT 대 unc-31 또는 논문이 제공하는 독립 signalling control;
6. failed/no-response events를 outcome에 따라 삭제하지 않는 missingness policy.

## 판정 단위

독립 단위는 event나 neuron pair가 아니라 animal/session이다. neuron pair와 time bin을
독립 표본으로 bootstrap하지 않는다. 같은 animal의 반복 stimulation은 계층적 또는
clustered 단위로 처리한다.

## 산출물

- `10-sources.md`: official paper/code/OSF provenance와 licence
- `11-math.md`: observational $R$, intervention $\tau$, output-Fisher $G$의 식별 경계
- `12-routes.md`: event-level subset, processed atlas, alternative dataset 경로
- `artifacts/e2syt-public-manifest.json`: 공개 metadata만 담은 machine receipt
- `20-audit.md`: 안정 source/math gate
- `30-implementation.md`, `31-validation.md`: manifest fetch/verification이 구현될 때만
- `40-final-report.md`: acquisition GO/STOP과 다음 재개 조건

## 금지

- source-target pair 수를 animal 수로 취급하지 않는다.
- stimulation response를 synaptic monosynaptic edge로 자동 해석하지 않는다.
- $R>0$를 causal effect로, $\tau\ne0$를 $G$ mediation으로 부르지 않는다.
- 실제 *C. elegans* 결과를 mammalian PFC 또는 인간 의식으로 일반화하지 않는다.
- manifest 판정 전에 큰 payload를 다운로드하거나 결과 파일을 열어 endpoint를
  선택하지 않는다.
