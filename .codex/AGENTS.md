# CE Codex 하네스 규칙

## 현재 저장 정책 (최우선)

- 하네스 작업은 direct로 수행하고 새 `CE_RUN`·`_workspace/`를 만들지 않는다. 기존 workspace는 과거 증거로만 읽는다.
- 최신 연구·검증 정본은 `paper/`에 둔다. `reality_stone`이나 은퇴한 `run.*`를 활성 경로로 전제하지 않는다.
- 임시 계산은 OS 임시 디렉터리를 사용하고 이번 실행이 만든 정확한 경로만 정리한다.

루트 `../AGENTS.md`가 기본 지침의 정본이다. 이 파일은 `.codex/` 자체를 고칠 때만 추가로 적용한다. 세부 구조와 알려진 부채는 `README.md`에서 필요할 때 읽는다.

## 라우팅

하네스 수정은 direct다. 연구·완결성·무차원성·검증·원장·논문 요청은 trigger가 맞는 CE 스킬 하나부터 읽고, 실제 독립성이 있는 레인만 병렬화한다. 원장은 먼저 안정화하고 논문 작성자는 이를 읽기 전용으로 사용한다.

## 과거 run 호환 경계

- `run.*`와 `skills/ce-research/core/`는 명시적인 과거 run 판독·수리에만 사용한다. 현재 상태는 `paper/`, 코드, 회귀검사와 작업 계획에 남긴다.

## 연구 판정

- 형식 지위·반례·대안 경로·부정 결과 보존은 `skills/ce-research/SKILL.md`가 소유한다. 기계 PASS와 관측 근접을 증명으로 쓰지 않는다.

## 검증

- 루트 AGENTS의 FAST→STANDARD 경계를 따른다. 하네스 변경은 `.codex/hooks/python.cmd harness`를 먼저 실행하고, 바뀐 표면의 가장 좁은 source/test만 추가한다.

## 전문 계약

- `harnesses/`의 분야별 계약은 그 분야를 실제로 검증할 때만 읽는다.

## Git과 발행

- 루트 AGENTS의 main-only Git 규율과 명시적 발행 preflight를 그대로 따른다. `.codex/` 지침은 그 권한을 넓히지 않는다.

## 데이터 반출

- 발행 시 루트 AGENTS가 지정한 `check-large-data.cmd`를 사용한다. 연구 원본·95MB 초과 blob·차단 확장자는 우회하지 않고 추적 대상에서 제외한다.
