# Implementation

Status: COMPLETE

## 승인 범위

감사에서 확인한 문서 유형 충돌을 하네스 역할 분리로 해소하고, 핵심 유도 문서의 열린 P1 두 건을 수학 내용을 추가하지 않는 최소 범위에서 닫았다. 기존 사용자 변경은 되돌리지 않았다.

## 변경

- `.codex/skills/ce-doc-write/SKILL.md`: 공통 형식 지위 규약과 문서 유형 라우터로 한정했다.
- `.codex/skills/ce-ledger-write/`: 주장·상수·판본·진리값 원장 전담 스킬을 추가했다.
- `.codex/skills/ce-paper-write/`: 강의·유도·독자 가이드·논문 원고 전담 스킬을 추가했다.
- `.codex/agents/ce-ledger-writer.md`, `.toml`: 원장 소유 역할을 추가했다.
- `.codex/agents/ce-paper-writer.md`, `.toml`: 논문형 문서 소유 역할을 추가했다.
- `AGENTS.md`, `.codex/AGENTS.md`, `.codex/README.md`: 원장을 먼저 안정화하고 논문 작성자가 읽기 전용으로 받는 순차 인계를 명시했다.
- `.claude/`: 기존 정책 미러와 같은 구조로 새 스킬·역할 카드를 동기화했다.
- `docs/5_유도/00_선택과_접힘.md`: Banach 수렴 초기값을 증명된 구간 $[0,1/D]$로 제한하고, 비정규화 대상을 잔류 측도라고 썼다.
- `docs/2_경로적분과_응용/normalize_markdown_math.py`: 코드 펜스·인라인 코드·LaTeX 행간 지정을 보존하면서 문서 수식 구분자를 `$...$`, `$$...$$`로 정규화하는 도구를 추가했다.
- `docs/**/*.md`: 구형 수식 구분자가 있던 120개 문서를 정규화했다.
- `docs/7_AGI/28_Nested_Infinite_SCC_V9.md`: 보관본이 있는 연구 산출물 링크 16개를 `_archive` 경로로 복구하고, 없는 산출물 1개는 재현 불가라고 명시했다.
- `tests/test_canonical_document_policy.py`: 역할 소유권 분리, 논문형 문서의 산문 시작, 별도 결론 절 금지, 고정점 정의역, 잔류 측도 용어, 수식 구분자와 상대 링크를 회귀 검사에 추가했다.
- `tests/test_bootstrap_math.py`: `torch` 패키지 facade를 거치지 않는 표준 라이브러리 bootstrap 수학 회귀를 추가했다.
- `tests/test_bootstrap_solver.py`: ML 통합 의존성인 `torch`가 없으면 수집 오류 대신 명시적으로 건너뛰도록 경계를 표시했다.

## 불변식

- 원장 작성자는 논문형 문서를 수정하지 않는다.
- 논문 작성자는 원장을 수정하지 않는다.
- 두 역할이 함께 필요하면 원장 동결 뒤 논문 작성자가 읽기 전용으로 받는다.
- 형식 지위 일곱 종류와 완전 반례 처리 규칙은 바꾸지 않았다.
- 수치 사슬과 물리적 readout 지위는 승격하지 않았다.
