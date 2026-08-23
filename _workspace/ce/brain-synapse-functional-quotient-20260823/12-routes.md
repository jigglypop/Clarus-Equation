# 12-routes — BA-SRM2 차원 경로 선택

Status: COMPLETE

## R0 — BA-SRM1 fixed 4D

판정: `PRESERVED_PREDECESSOR_DIAGNOSTIC / NOT_BRAIN_DIMENSION`.

네 좌표는 strict sieve였고 실제 rank/예측 gate에서 STOP됐다. 이는 뇌 상태의 차원이
4라는 주장도, 모든 고차원 geometry가 실패했다는 주장도 아니다.

## R1 — 전체 무한차원 SPD metric

판정: `REJECTED_BY_FINITE_OBSERVATION_NO_GO`.

유한 관측의 pullback에는 무한차원 kernel이 남는다. 전체 함수공간이 아니라 finite
observable quotient만 식별할 수 있다.

## R2 — small `pulse_amplitudes` grid

판정: `REJECTED_SOURCE_ALIAS_BUG`.

12 slot이 같은 shared list aggregate의 반복이다. 12D/36D/48D trajectory로 쓰면
인공적인 중복 좌표와 singular geometry를 만든다. 이 경로는 결과를 열기 전에
source-level 반례로 폐기했다.

## R3 — small protocol STP summaries

판정: `DIAGNOSTIC_ONLY / NOT_SUBSTITUTE_FOR_HISTORY`.

독립적으로 계산된 네 STP summary의 protocol surface는 유한 다변량 기술에는 쓸 수
있다. 그러나 raw pulse order와 event noise가 없고 canonical delay duplicate가 있어
사용자 지시의 causal high-dimensional state를 대신하지 않는다. 이번 run에서
outcome fitting으로 우회하지 않는다.

## R4 — medium event-level functional sieve

판정: `SELECTED_SUCCESSOR / ACQUIRING_INPUT_10.36_GiB / PREACCESS_PREREG_FROZEN`.

medium은 pulse identity, stimulus/recording relation과 event fit을 보존할 것으로
source rule이 예고하지만 완료 파일의 schema audit 전에는 확정하지 않는다. 사용자 승인
뒤 acquisition을 시작했고 `revisions/01-medium-event-preaccess-prereg.md`에 strict
$H_8\mapsto Y_{9:12}\in\mathbb R^{16}$, split, model grid, rank/gauge/ELPD gate를
outcome 전에 고정했다.

## R5 — full/NWB waveform operator

판정: `DEFERRED_LARGE_INPUT`.

full/NWB는 $L^2$ waveform operator에 가장 가깝지만 수백 GB 규모와 sampling/QC,
clamp/access-resistance 계약이 필요하다.

## R6 — conductance/release/morphology product state

판정: `BLOCKED_JOINT_IDENTITY`.

conductance, $Npq$, contact/PSD와 longitudinal plasticity를 추가하려면 같은
synapse/event identity frame의 직접 측정이 필요하다. 별도 자료를 임의 join하지 않는다.

## 최종 경로

수학적으로 선택한 대상은 finite observable quotient이고 실제 자료 경로는 R4다.
현재 R4는 acquisition/schema gate까지만 진행 중이며 outcome model은 계속
`BLOCKED_INPUT`이다. medium 입력 없이 small summary의 차원을 사후 늘리지 않는다.
