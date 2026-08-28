# CE Codex 하네스

현재 하네스는 단일 저장소 direct 모드다. 제거된 `reality_stone`을 요구하지 않고, 자동 CE_RUN 라우팅이나 새 `_workspace/` 생성을 하지 않는다. 검증은 `.codex/hooks/python.cmd doctor|source|python|pytest`로 실행하며 최신 결과는 루트 `docs/` 정본을 직접 갱신한다. 아래 run-core 설명은 과거 호환 정보다.

Clarus-Equation 연구용 프로젝트 로컬 Codex 설정이다. 일반 작업은 빠른 direct 경로를 쓰고, 명시적인 연구 요청만 역할 에이전트와 run 하네스를 사용한다.

## 구조

```
../AGENTS.md         프로젝트 공통 지침(있는 경우)
AGENTS.md            `.codex/` 작업과 CE run 라우팅 규칙
config.toml          low effort 기본값과 최대 3개 병렬 레인
prompts/             CE 명령 프롬프트 7종
agents/              독립 연구 역할 4개(TOML + 상세 Markdown 카드)
skills/              단일 책임 CE 스킬 6개
skills/ce-research/core/   Rust 게이트 코어 (init|status|check|revise|gc + hook)
harnesses/           수치·증거 하네스 계약 — backend parity(Rust/CUDA), 실측 교정 루프, 뇌 증거 사다리
hooks.json           프로젝트 로컬 UserPromptSubmit 라우팅(Stop 전역 차단 없음)
hooks/               run.* 코어 래퍼 + native Windows Python/payload 실행기
```

## 설치

1. 프로젝트 루트에서 Codex를 시작한다. 신뢰된 프로젝트에서는 `.codex/config.toml`, hooks, rules가 로드되고, 신뢰되지 않은 프로젝트에서는 프로젝트 로컬 계층이 건너뛰어진다.
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

## 주의 (Windows)

- PowerShell 5.1이 파일·stdin에 UTF-8 BOM을 붙인다. 코어는 양쪽 모두 BOM을 제거하므로, 코어를 수정할 때 이 처리를 유지할 것.
- 레포 안에 `target/`을 만들지 않는다(.gitignore로 차단). 빌드 캐시는 `%LOCALAPPDATA%`.
- 역할 에이전트는 독립 레인만 담당한다. 원장과 논문 작성 규율은 각각 `ce-ledger-write`, `ce-paper-write` 스킬이 단독 소유한다.
- agent 실행은 stdin이 비대화형일 수 있으므로 uv/Python/보안 프롬프트를 기다리지 않는다. 이 PC에서는 uv cache ACL과 Windows Code Integrity의 enterprise signing policy가 각각 uv cache와 `.venv\Scripts\python.exe`를 차단했다. `python.cmd doctor`는 PowerShell 실행정책을 변경하지 않고 정책 허용된 기존 system Python만 선택하며 dependency를 설치하지 않는다.
- 근본 조치는 관리자가 서명·allowlist된 Python 경로와 uv cache ACL을 복구하는 것이다. 하네스는 Application Control을 끄거나 차단된 interpreter를 실행하지 않으며 그 전까지 focused 검사만 system Python으로 수행한다.
