# CloudCell 입력 적격성 최종 보고

Status: COMPLETE

## 결론

새 식으로 실제 자료를 다시 열어 본 결과, CloudCell은 두 객체를 같은 수준으로
지원하지 않는다.

$$
\mathcal B^{A\to B}=(G^{o\leftarrow A},R^{A\to B})
$$

- $G^{o\leftarrow A}$: **PASS_INPUT**. 11개 GCaMP recording에는 같은 clock의
  population fluorescence와 미래 locomotion output이 있어, 별도 사전등록 후
  관측적 output-Fisher geometry를 추정할 수 있다.
- 임의 row-group $R$: **DIAGNOSTIC_ONLY**. 수치는 계산할 수 있지만 해부학적
  brain router 후보가 아니다.
- anatomical $R^{A\to B}$: **BLOCKED_SOURCE_TARGET_DEFINITION**. canonical
  neuron identity와 검증된 target-B/connectome join이 없다.
- causal routing: **BLOCKED_INTERVENTION**. source-specific randomized
  perturbation과 sham/reverse/non-target controls가 없다.

## 실제로 새로 확인한 것

세 archive와 22 recording을 기계적으로 다시 읽었다. GCaMP 11/GFP control 11,
공통 neural/behavior schema, archive hash가 확인됐다. 수기 inventory가 놓친
timestamp 결함도 발견했다. 10개 recording은 첫 timestamp가 한 번 중복되고,
한 recording은 큰 gap 3개를 갖는다. 선두 12-volume guard와 gap-crossing window
제거, 60/20/20 chronological split, 경계별 12-volume embargo를 exact anchor
목록으로 봉인했고 모든 recording에서 train/validation/test가 남았다.

## 뇌에 대해 허용되는 말

아직 새 생물학적 결과는 없다. 확인된 것은 “이 자료에서 미래 행동 출력에
상대적인 관측 형광 geometry를 정직하게 시험할 입력이 있다”는 것뿐이다.
기존 AML18 GFP 반례 때문에 단순 lag predictability는 calcium memory나 neural
routing의 증거가 될 수 없다.

## 다음 경로

“뇌의 알고리즘”이라는 원래 목표에는 CloudCell row split보다 Randi et al.
E2SYT가 우선이다. 이 자료는 canonical neuron identity와 single-neuron
optogenetic intervention을 함께 제공하는 후보이므로, 다음 단계는 분석이 아니라
먼저 OSF object/version, licence, manifest, event/identity fields와 control 조건을
고정하는 acquisition/source audit다. E2SYT는 아직 로컬에 없으므로 다운로드나
결과 실행은 이 run에서 하지 않았다.

CloudCell metric-only 분석은 별도 보조 경로로 열 수 있다. 그 경우 primary는
apparatus가 균질한 AML32 7 recordings, AML310/AKS 4 recordings은 path/BFP
condition이 다른 sensitivity panel, AML18 11 recordings은 GFP nuisance
falsifier로 사전 고정한다.

## 영구 경계

어느 다음 결과도 독립 개입 없이 $G\to R$, $W\to G\to x$, metric mediation,
curvature, SCC=기억/의식 또는 인간 뇌 일반화를 뜻하지 않는다.
