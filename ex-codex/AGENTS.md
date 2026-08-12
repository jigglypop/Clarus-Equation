# CE Codex 하네스

Clarus-Equation 연구·구현을 위한 Codex 전역 지침. 프로젝트별 AGENTS.md보다 우선순위가 낮다.

## 라우팅

요청을 받으면 먼저 규모를 판단한다. **작은 요청에 run을 파지 않는 것이 속도 규율의 핵심이다.**

| 요청 | 행동 |
|---|---|
| 단순 질문·계산·파일 위치·한 줄 수정 | run 없이 바로 답한다 (spot). 스킬을 로드하지 않는다 |
| 특정 주장 지위·증명 판정 | $ce-closure-gate |
| 식의 차원·무차원 검사 | $ce-dimensionless |
| 테스트·회귀·수치 검증 | $ce-validate |
| 정본 문서 작성·수정 | $ce-doc-write |
| guard 벤치·ASR·회귀 | $clarus-guard-bench |
| 신규 주장 검증·승격 판단·논문화·병렬 연구 | $ce-research (full) |
| 완결 run의 후속·반복 (v8→v9 등) | $ce-research (light, PREDECESSOR 지정) |

선택한 스킬 하나만 로드한다. 역할 카드(agents/*.md)는 해당 레인을 실행하는 순간에만 읽는다.

## 코어 명령

    <codex-home>/hooks/run.sh <cmd> ...     # POSIX
    <codex-home>/hooks/run.cmd <cmd> ...    # Windows (프리빌드 바이너리 직행)

- `init <run-dir>` — run 생성, `.active-run` 포인터 설정. `REUSE?` 목록이 나오면 새로 파지 말고 이어받는다.
- `status <run-dir>` — 단계별 상태·Gate·수정 카운트를 한 화면에 출력. **run 현황 파악에 stage 파일을 다시 읽지 말고 이것을 쓴다.**
- `check <run-dir> <contract|lanes|gate|build|final>` — 해당 단계까지 전체 체인 검사.
- `revise <run-dir> <role>` — 수정 루프 기록. 역할당 2회 상한.
- `gc <workspace>` — 완결 run을 `_archive/`로 이동, 미완성 run 나열. 세션 종료 시 실행.

## 심각도 어휘 (모든 역할 공통)

- **P0**: 주장을 무너뜨리는 결함 — 완전한 반례, 증명 오류, 수치 불일치가 결론을 바꿈. 해소 전 Gate PASS 불가.
- **P1**: 결론은 유지되나 방치 불가 — 증명 공백, 숨은 공리, 기준선 불일치. revise로 처리하거나 BLOCKED로 기록.
- **P2**: 표기·분량·정리 문제. 최종 보고서에 목록만 남기고 수정 루프를 소모하지 않는다.

## 속도 규율

- 카드·스킬을 선로딩하지 않는다. 필요한 시점에 필요한 것만 읽는다.
- run 현황은 `status`로 본다. stage 파일 재독은 내용을 인용할 때만.
- PREDECESSOR가 있으면 선행 run이 검증한 결론을 재유도·재대조하지 않고 경로만 인용한다.
- 상세 계산·로그·발췌는 artifacts/에 쓰고 stage 파일에는 판정·표·경로만 남긴다.
- 레인 파일은 3–4만 자 재생성이 아니라 판정 중심의 짧은 문서다.

## 정직성 (협상 불가)

- 실패를 통과로 표현하지 않는다. 실행하지 않은 검증을 실행했다고 쓰지 않는다.
- 기계 검사 상태 문자열(PASS 등)을 이론 지위처럼 문서에 복사하지 않는다.
- ce-closure-gate 기록 없이 "유도됨·제1원리·닫힘"을 쓰지 않는다.
- 관측 근접은 증명이 아니다. 형식 지위는 정의·정리·공리·산출·경험식·미완성·예측 7종뿐이다.
