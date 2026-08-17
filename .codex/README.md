# ex-codex — CE Codex 하네스 (배포판)

Codex CLI용 Clarus-Equation 연구 하네스. `.codex/`의 개발 트랙을 코덱스 네이티브형으로 확장한 배포 사본이다.

## 구조

```
../AGENTS.md         Codex가 실제로 읽는 짧은 프로젝트 기본 지침
AGENTS.md            `.codex` 내부 작업용 상세 보조 지침
config.toml          프로젝트 기본 medium effort + bounded multi-agent 설정
prompts/             커스텀 슬래시 프롬프트 (/ce-research, /ce-status, /ce-gc 등 8종)
agents/              네이티브 TOML 역할 4개 + explicit CE run용 상세 Markdown 카드
skills/              스킬 6종 (SKILL.md + agents/openai.yaml)
skills/ce-research/core/   Rust 게이트 코어 (init|status|check|revise|gc + hook)
hooks.json           프로젝트 로컬 UserPromptSubmit 라우팅(Stop 전역 차단 없음)
hooks/               run.cmd(고속) / run.ps1 / run.sh 래퍼
```

## 설치

1. 프로젝트 루트에서 Codex를 시작한다. 루트 `AGENTS.md`와 `.codex/config.toml`이 자동 로드된다.
2. 바이너리 빌드 (1회, OneDrive 밖 `%LOCALAPPDATA%\ce-research-core`에 생성):

   ```powershell
   .\hooks\run.ps1 gc _workspace\ce    # 아무 수동 명령이나 최초 실행 시 빌드됨
   ```

3. 확인: `.codex\hooks\run.cmd status <run-dir>` 이 단계 표를 출력하면 정상.

## 속도 설계

- **훅은 절대 빌드하지 않는다.** `hook` 이벤트는 프리빌드 바이너리 직행(run.cmd, PowerShell 기동 없음), 바이너리가 없으면 조용히 no-op. cargo 빌드는 수동 명령에서만 일어난다.
- 훅 timeout은 route 5s로 캡하며 Stop 훅은 사용하지 않는다.
- run 현황은 `status` 한 번으로 본다 — stage 파일 재독 금지.
- 작은 요청은 direct(스킬·run 없음), standard는 관련 통합 검사까지, full은 신규 주장·release에만 쓴다.
- 상세 계산·로그는 artifacts/로 위임하고 stage 파일은 판정·표만 유지한다.

## 주의 (Windows)

- PowerShell 5.1이 파일·stdin에 UTF-8 BOM을 붙인다. 코어는 양쪽 모두 BOM을 제거하므로, 코어를 수정할 때 이 처리를 유지할 것.
- 레포 안에 `target/`을 만들지 않는다(.gitignore로 차단). 빌드 캐시는 `%LOCALAPPDATA%`.
- 산출물 규율을 바꾸면 역할 카드와 SKILL.md 양쪽을 같이 고친다.
