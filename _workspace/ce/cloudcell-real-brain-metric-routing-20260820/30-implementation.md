# Input-audit implementation

Status: COMPLETE

## 구현 범위

실제 신경 모델을 적합하거나 경험 결과를 계산하지 않았다. 구현은 로컬 입력의
재현 가능한 read-only 영수증 하나로 제한했다.

- `artifacts/inspect_cloudcell_inputs.py`
  - 세 archive의 SHA-256을 다시 계산한다.
  - official dataset log의 22 recording을 정확히 열거한다.
  - 각 `heatDataMS.mat`에서 neural matrix, behavior struct, clock, XYZ와 derived
    correlation schema를 검사한다.
  - GCaMP 11개와 GFP control 11개를 분리한다.
  - timestamp 중복과 큰 acquisition gap을 기록한다.
  - 생물학적 결과를 계산하지 않고 JSON input receipt만 쓴다.
- `artifacts/cloudcell-input-audit.json`
  - schema `clarus.cloudcell.input-audit.v1`
  - 3 dataset, 22 recording, GCaMP 11, GFP control 11
  - `all_recording_checks_pass=true`
  - claim boundary를 `PASS_INPUT` / `BLOCKED_SOURCE_TARGET_DEFINITION` /
    `BLOCKED_INTERVENTION`으로 분리한다.

원 archive와 extracted MAT 파일은 수정·이동·재추출하지 않았다.

## 실행 환경

Windows App Control이 uv-managed interpreter를 차단하므로, 이미 설치되어 있고
`numpy`/`scipy`가 import되는 system CPython 3.11을 사용했다. 이 선택은 입력
schema 감사에만 해당한다. 후속 sealed empirical fit은 interpreter와 package
version을 별도 freeze해야 한다.

## 발견에 따른 apparatus 수정

첫 기계 실행은 `BrainScanner20200130_110803`의 첫 timestamp 중복을 발견하여
중단했다. 전체 22 recording을 재검사한 결과 10개에서 index 0의 단일 중복이,
`BrainScanner20200310_141211`에서 큰 gap 3개가 확인됐다. 결과를 버리거나
timestamp를 보간하지 않고 다음 outcome-blind 규칙을 입력 계약에 고정했다.

1. 모든 recording의 첫 12 volume을 분석 대상에서 제외한다.
2. 고정 horizon/history window가 큰 gap을 가로지르면 그 window를 제외한다.
3. guard 뒤에도 non-increasing timestamp가 있으면 해당 recording을 STOP한다.

이 규칙 아래 22/22 recording의 schema/clock apparatus가 통과했다. 이는 데이터
적격성 결과이지 output-Fisher 또는 routing 결과가 아니다.
