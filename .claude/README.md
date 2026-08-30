# CE Claude 하네스

Claude Code용 Clarus-Equation 연구 하네스. `.codex` 정본과 같은 연구 파이프라인·게이트 코어를 쓰는 provider 미러다.

## 구조

```
CLAUDE.md            프로젝트 지침: 라우팅 표, 코어 명령, P0/P1/P2, 속도·정직성 규율
settings.json        훅: native payload gate + UserPromptSubmit + Stop
commands/            슬래시 커맨드 8종 (/ce-research, /ce-status, /ce-gc 등)
agents/              연구 역할 4종 + 원장/논문 작성 역할 2종
skills/              공통 연구·검증·작성 스킬 8종
harnesses/           수치·증거 하네스 계약 4종 미러 + 인덱스 README
hooks/               canonical run 위임 + Windows Python/payload 위임 래퍼
```

Rust 코어와 Windows 실행기의 단일 출처는 각각 `.codex/skills/ce-research/core`와 `.codex/hooks/`다. Claude의 `run.*`, `python.cmd`, `check-large-data.cmd`는 이 정본으로 위임한다. 역할 카드·공통 스킬·`harnesses/` 계약의 과학 규율은 provider 경로를 제외하고 함께 갱신한다.

## 설치

1. 프로젝트 루트에서 Claude Code를 시작한다.
2. Windows Python 확인: `.claude\hooks\python.cmd doctor`.
3. 연구 코어 확인: `.claude\hooks\run.cmd status <run-dir>`. 프리빌드 바이너리가 없을 때 수동 명령만 canonical Codex wrapper의 on-demand build를 사용할 수 있으며, hook 이벤트는 빌드하지 않는다.

## 속도 설계

- 훅은 절대 빌드하지 않는다. 바이너리 직행, 없으면 조용히 no-op한다.
- 오케스트레이터는 역할 카드를 읽지 않는다 — subagent 위임으로 문맥을 얇게 유지.
- run 현황은 `status` 한 번. 작은 요청은 spot(스킬·run 없음).
- bounded concurrency와 역할 소유권은 `CLAUDE.md` 및 공통 역할 카드가 정한다.
- 완결 규율은 끈질김 절(CLAUDE.md)이 담당: BLOCKED 최후 수단, 부정 결과도 40-final까지.

## 주의

- 산출물 규율을 바꾸면 역할 카드·SKILL.md를 `.codex` 정본과 같은 변경에서 고친다.
- 레포 안에 `target/`·Python cache를 만들지 않는다. focused 검증은 `.claude\hooks\python.cmd pytest <test>`를 사용한다.
- Windows에서 차단된 `.venv`나 uv Python, WSL `bash` 프롬프트를 기다리지 않는다. native `.cmd` 위임을 사용하고 Application Control을 끄지 않는다.
