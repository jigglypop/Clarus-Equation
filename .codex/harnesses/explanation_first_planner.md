# 설명 우선 플래너 하네스

Status: `ACTIVE / POLICY`

이 계약은 수학·물리 판단을 사용자에게 전달할 때 계획과 설명이 함께
나오도록 하는 `.codex/skills/ce-explanation-planner`의 구조적 불변조건을
정한다. 문장 품질을 자동 판정하지 않고, 진입점·역할 카드·스킬·프롬프트가
모두 존재하며 핵심 규칙을 공유하는지만 검사한다.

## 필수 불변조건

- 스킬은 `SKILL.md` frontmatter의 `name`과 설명을 가진다.
- 스킬 본문에는 LaTeX 유도, 비유, 공리/정리 지위, 반례 뒤의 다음 경로가
  모두 명시된다.
- 에이전트 카드와 TOML은 읽기 전용 권한과 원장·논문 소유권 경계를 명시한다.
- 프롬프트는 목표량·정의역·단계별 유도·경계·다음 의무를 요구한다.
- 하네스 검사는 위 네 진입점이 빠졌을 때 실패한다.

## 사용

`.codex/hooks/python.cmd harness`가 저장소 계약과 함께 이 네 경로를 확인한다.
문법 검사는 `.codex/hooks/python.cmd source .codex`로, 실제 설명은
`$ce-explanation-planner` 또는 `.codex/prompts/ce-explain-plan.md`로 실행한다.
이 하네스의 구조 검사나 실행 결과는 수학적 증명이 아니다.
