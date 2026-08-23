# 20-audit — BA-SRM2 독립 Gate 감사

Status: COMPLETE

Gate: `BLOCKED_OUTCOME / PASS_SCHEMA_ONLY_IMPLEMENTATION`

Scientific state: `FUNCTION_SPACE_HYPOTHESIS_VALID / SMALL_DATA_ROUTE_INVALID`

## P0

공식 producer의 `collect_pulse_amps = [[]] * 12`가 모든 pulse slot을 같은 aggregate로
만든다. 최초 36D→36D measurement route는 실제 관측 차원이 아니므로 source-level
반례로 기각했다. train blob은 slot-equality 구조 확인에만 접촉했고 amplitude
크기를 보고ㆍ적합ㆍ채점하지 않았다. confirmation blob 접근은 승인하지 않는다.

## P1 successor requirements

- target complete-case가 input selection에 들어가지 않는 missingness rule;
- kernel distance까지 transport하는 gauge-compatible estimator;
- covariance와 regularization의 완전한 사전 고정;
- pointwise rank가 아닌 constant-rank quotient condition;
- Fréchet differentiability, bounded derivative와 covariance precision domain;
- best-control 선택 규칙과 nested group CV.

active contract의 `revisions/01-medium-event-preaccess-prereg.md`는 이 조건과 strict
future-only $H_8\mapsto Y_{9:12}\in\mathbb R^{16}$를 outcome 접근 전에 고정했다.
독립 재감사는 byte/hash/integrity/FK/provenance만 읽는 schema-only auditor 구현을
`PASS`로 허용했지만 train outcome model, development와 confirmation은 계속 차단했다.

## 보존되는 수학

유한 관측에서 $\operatorname{rank}G\le m$이고 전체 무한차원 공간에는 SPD metric을
식별할 수 없다는 no-go는 유지된다. 각 점의 관측가능 quotient와 전역 constant-rank
조건도 유효하다.

## 입력 상태

small DB는 검증됐지만 event table이 0행이다. medium official object는
11,125,997,568 bytes이며 acquisition이 진행 중이다. 완료 파일의 checksum, integrity와
support는 아직 미검증이다. 따라서 schema-only implementation gate만 열렸고 scientific
outcome gate는 열리지 않는다.

Counts: P0 1 closed by route rejection; open input blocker 1; train structural-contact 1;
amplitude fit/score 0; confirmation contact 0; schema implementation authorization true;
outcome implementation authorization false.
