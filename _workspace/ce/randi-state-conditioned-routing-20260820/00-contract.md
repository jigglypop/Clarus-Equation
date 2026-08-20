# Randi 상태-조건부 유효 라우팅 입장 계약

Status: COMPLETE

Scope: 위 상태는 계약 동결 단계의 완료만 뜻한다. apparatus와 empirical route의 최종
판정은 `20-audit.md`와 `40-final-report.md`에 기록한다.

PREDECESSOR: `_workspace/ce/randi-neural-propagation-source-audit-20260820`

## 목적

붙임 메모의 최신 우선순위인 다음 가설을 실제 *C. elegans* 개입 자료에서 검정할 수
있는지 판정한다.

$$
Y^{\mathrm{post}}=f(A,q,X^{\mathrm{pre}})
$$

여기서 $A$는 자극된 정본 뉴런, $q$는 자극량·기하·세션 조건,
$X^{\mathrm{pre}}$는 자극 직전 전뇌 상태다. 이번 run은 효과를 계산하지 않는다.
먼저 Randi/DANDI 자료가 결과를 보지 않고도 $A$, 배정 층, active-source control,
시계, 결측·carryover 규칙을 고정하게 하는지 검사한다. 통과할 때만 별도의 확인
run에서 $M_0=f(A,q)$, $M_1=f(A,q,\overline X^{\mathrm{pre}})$,
$M_2=f(A,q,X^{\mathrm{pre}})$와 time/session shuffle control을 비교한다.

## PREDECESSOR_EVIDENCE

| 선행 증거 | 판정 | SHA-256 | 보존하는 가장 좁은 주장 | 재시도 금지 조건 |
|---|---|---|---|---|
| route ledger | `BA-EMP-RANDI-ACTIVE / CONDITIONAL_INPUT` | `ad17d09591ceb5e614219532e78ecc0199cdfec67384b149296717114cfa636f` | Randi가 현재 1순위 실제-자료 후보다 | 합성 threshold나 seed로 대체하지 않음 |
| predecessor routes | `R1 RECOMMENDED, SCHEMA REQUIRED` | `cf5581bd08ecc8c00afb92500c796ca195fd230477820c729bf46dbef6a9b704` | compact segmentation NWB 경로를 먼저 검사한다 | 4.07-TB raw layer 선취 금지 |
| predecessor validation | `BLOCKED_EXPLICIT_JOIN / BLOCKED_CONTROL` | `31830d6ddf7440520f7836cf6cec593f80f48efcf944dc8be59dc9af3ebc763f` | exemplar에는 event·trace·response-side NeuroPAL schema가 있다 | outcome 기반 spatial matching 금지 |
| predecessor final | `SCHEMA_AUDIT_COMPLETE / EMPIRICAL_ROUTE_BLOCKED_CONDITIONAL` | `7ec326830b7b3e698c7b9a7878bd5bb7bac3294f65d21f36d7a5793dd31932aa` | source join과 comparator를 별도 입장시켜야 한다 | response matrix를 읽고 규칙 선택 금지 |

## 고정 질문

1. 논문·동결 코드·NWB 값 경로 중 하나가 stimulation target을 사후 NeuroPAL 정본
   identity에 연결하는 결과 독립 규칙을 제공하는가?
2. 자동 무작위 자극과 수동 표적 예외를 event 또는 session 수준에서 구분할 수 있는가?
3. 같은 배정 층의 다른 source stimulation을 active control로 삼을 positivity가 있는가?
4. $X^{\mathrm{pre}}$가 post-outcome, autoresponse 또는 target response를 사용하지 않고
   자극 직전 공통 시계에서 구성 가능한가?
5. animal/session split, 30초 반복 자극의 carryover, 결측·identity confidence를 결과
   전에 고정할 수 있는가?

## 결과 전 후보와 순서

| 순위 | 후보 | 입장 판정 | 독립 falsifier / STOP |
|---:|---|---|---|
| 1 | 명시적 NWB reference 또는 동결 변환 코드 join | `OPEN` | source identity가 post-response/autoresponse에 의존하면 STOP |
| 2 | 논문 방법에 고정된 geometric target-to-tracked-neuron join | `OPEN_CONDITIONAL` | 반경·tie-break·registration frame을 outcome 없이 고정할 수 없으면 STOP |
| 3 | pumpprobe/Fconn publication-native event objects | `OPEN_CONDITIONAL` | 공개 bytes와 checksum/동결 release를 확보하지 못하면 BLOCKED |
| 4 | processed Funatlas pair table | `PROCESSED_ONLY` | event pre-state와 assignment가 없으면 state-conditioned test에는 STOP |

후보 순서는 결과를 읽기 전에 고정한다. threshold, source subset, response window,
latent dimension, decoder 또는 seed만 바꾼 반복은 새 mechanism이 아니며 금지한다.

## 허용되는 schema/value probe

- DANDI `001075`, version `0.240920.1434`의 선행 동결 manifest만 사용한다.
- 먼저 결정론적 최소 segmentation exemplar 하나만 다시 받아 SHA-256을 검증한다.
- response fluorescence `data` 값은 읽지 않는다.
- 허용 값은 event target reference, ROI reference/index, identity label/confidence,
  assignment·condition·session metadata, timestamps shape/range뿐이다.
- geometric join을 검사할 경우 좌표계, registration frame, 거리 단위와 tie-break를
  source/math 감사가 먼저 고정해야 한다.
- exemplar만으로 이질성 문제가 남으면 결과 비열람 규칙으로 genotype와 크기 층을
  나눈 최소 추가 subset을 새 감사 후 열며, 이번 계약에서 임의 확대하지 않는다.

## 후속 확인 실험의 사전 성공 조건

이번 run이 `PASS_APPARATUS`를 낸 뒤에만 별도 계약에서 아래를 검정한다.

1. animal/session 완전 holdout에서 $M_2$가 $M_1$과 $M_0$를 모두 이긴다.
2. 같은 $X^{\mathrm{pre}}$를 잘못된 시간 또는 다른 session에 붙인 shuffle 이득은
   사라진다.
3. global gain, source identity, dose, genotype, event order를 모두 포함한 대조보다
   population configuration이 추가 예측 정보를 가진다.
4. metric은 고정 output likelihood의 predictive readout으로만 부르며 물리적
   connectome·기억·의식의 증거로 승격하지 않는다.

## 판정

- `PASS_APPARATUS`: source join, 자동/수동 배정 층, active controls, pre-state clock,
  결측·carryover가 outcome 없이 고정된다.
- `PASS_OBSERVATIONAL_ONLY`: source와 pre-state는 있으나 배정/positivity가 부족하다.
- `BLOCKED_SOURCE_JOIN`: 정본 source identity가 고정되지 않는다.
- `BLOCKED_ASSIGNMENT`: 자동 무작위와 수동 표적 또는 active control 층을 복원하지 못한다.
- `APPARATUS_INVALID`: 허용 probe 자체가 response outcome을 읽거나 post-treatment
  conditioning을 사용한다.

## 주장 상한

양성 apparatus 판정도 상태-조건부 라우팅 효과를 증명하지 않는다. 후속 양성 결과는
고정된 Randi 장치와 calcium 시간척도에서 자극 직전 population state가 held-out
response prediction에 추가 정보를 준다는 관측-개입 결합 주장까지다. 이는 synaptic
edge, 기억 결합, 포유류 AGI 또는 의식 메커니즘이 아니다.

## 실행기 예외

`$ce-research`가 지정한 `C:\Users\dongh\.codex\hooks\run.ps1`가 현재 환경에 없어
run 초기화·gate 명령을 실행할 수 없다. 동일한 8개 stage 이름과 수동 gate 검사를
사용하고 이 재현성 결함을 최종 보고에 보존한다.
