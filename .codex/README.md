# ex-codex — CE Codex 하네스 (배포판)

Codex CLI용 Clarus-Equation 연구 하네스. `.codex/`의 개발 트랙을 코덱스 네이티브형으로 확장한 배포 사본이다.

## 구조

```
../AGENTS.md         Codex가 실제로 읽는 짧은 프로젝트 기본 지침
AGENTS.md            `.codex` 내부 작업용 상세 보조 지침
config.toml          프로젝트 기본 medium effort + bounded multi-agent 설정
prompts/             커스텀 슬래시 프롬프트 (/ce-research, /ce-status, /ce-gc 등 8종)
agents/              연구 역할 4개 + 원장/논문 작성 역할 2개 (TOML + 상세 Markdown 카드)
skills/              공통 6종 + 원장/논문 작성 분리 스킬 2종 (SKILL.md + agents/openai.yaml)
skills/ce-research/core/   Rust 게이트 코어 (init|status|check|revise|gc + hook)
hooks.json           프로젝트 로컬 UserPromptSubmit 라우팅(Stop 전역 차단 없음)
hooks/               run.* 코어 래퍼 + native Windows Python/payload 실행기
```

## 설치

1. 프로젝트 루트에서 Codex를 시작한다. 루트 `AGENTS.md`와 `.codex/config.toml`이 자동 로드된다.
2. 바이너리 빌드 (1회, OneDrive 밖 `%LOCALAPPDATA%\ce-research-core`에 생성):

   ```powershell
   .\hooks\run.ps1 gc _workspace\ce    # 아무 수동 명령이나 최초 실행 시 빌드됨
   ```

3. 확인: `.codex\hooks\run.cmd status <run-dir>` 이 단계 표를 출력하면 정상.
4. Windows Python 확인: `.codex\hooks\python.cmd doctor`. focused pytest는 `.codex\hooks\python.cmd pytest <test-path> -q`로 실행한다.

## 속도 설계

- **훅은 절대 빌드하지 않는다.** `hook` 이벤트는 프리빌드 바이너리 직행(run.cmd, PowerShell 기동 없음), 바이너리가 없으면 조용히 no-op. cargo 빌드는 수동 명령에서만 일어난다.
- 훅 timeout은 route 5s로 캡하며 Stop 훅은 사용하지 않는다.
- run 현황은 `status` 한 번으로 본다 — stage 파일 재독 금지.
- 작은 요청은 direct(스킬·run 없음), standard는 관련 통합 검사까지, full은 신규 주장·release에만 쓴다.
- 상세 계산·로그는 artifacts/로 위임하고 stage 파일은 판정·표만 유지한다.

## Codex ↔ Claude 동기화 경계

- 공통 정본: `.codex/agents/*.md`의 역할 내용, 공통 `skills/*/SKILL.md`의 과학·검증 규율, `.codex/skills/ce-research/core/`, `.codex/hooks/python_harness.py`, `.codex/hooks/check_large_data.py`.
- Claude 미러: `.claude/agents/*.md`; provider 경로만 다른 `ce-research`/`ce-validate` 문구; versioned `.claude/commands/*.md`; `.claude/hooks/run.*`, `python.cmd`, `check-large-data.cmd` 위임 래퍼.
- provider 전용으로 유지: Codex `config.toml`, `hooks.json`, agent TOML, prompt와 skill `openai.yaml`; Claude `CLAUDE.md`, `settings.json`, `commands/`.
- 공통 역할 카드의 의미를 바꾸면 두 트리를 같은 변경에서 갱신한다. provider 전용 파일을 다른 트리에 복사해 지원되지 않는 설정을 만들지 않는다.

## 주의 (Windows)

- PowerShell 5.1이 파일·stdin에 UTF-8 BOM을 붙인다. 코어는 양쪽 모두 BOM을 제거하므로, 코어를 수정할 때 이 처리를 유지할 것.
- 레포 안에 `target/`을 만들지 않는다(.gitignore로 차단). 빌드 캐시는 `%LOCALAPPDATA%`.
- 산출물 규율을 바꾸면 역할 카드와 SKILL.md 양쪽을 같이 고친다.
- agent 실행은 stdin이 비대화형일 수 있으므로 uv/Python/보안 프롬프트를 기다리지 않는다. 이 PC에서는 uv cache ACL과 Windows Code Integrity의 enterprise signing policy가 각각 uv cache와 `.venv\Scripts\python.exe`를 차단했다. `python.cmd doctor`는 PowerShell 실행정책을 변경하지 않고 정책 허용된 기존 system Python만 선택하며 dependency를 설치하지 않는다.
- 근본 조치는 관리자가 서명·allowlist된 Python 경로와 uv cache ACL을 복구하는 것이다. 하네스는 Application Control을 끄거나 차단된 interpreter를 실행하지 않으며 그 전까지 focused 검사만 system Python으로 수행한다.
- `.claude/hooks/run.*`와 `.claude/hooks/python.*`은 이 디렉터리의 정본 래퍼로 위임한다. 공통 역할 카드는 `.codex/agents/*.md`를 정본으로 미러링하고, provider별 설정 파일만 따로 유지한다.
