# CE Codex 하네스

현재 하네스는 단일 저장소 direct 모드다. 제거된 `reality_stone`을 요구하지 않고, 자동 CE_RUN 라우팅이나 새 `_workspace/` 생성을 하지 않는다. 검증은 `.codex/hooks/python.cmd doctor|harness|source|python|pytest`로 실행하며 최신 결과는 루트 `paper/` 정본을 직접 갱신한다. run core는 기존 실행 증거 판독을 위한 과거 호환 표면이다.

Clarus-Equation 연구용 프로젝트 로컬 Codex 설정이다. 일반 작업은 빠른 direct 경로를 쓰고, 연구 요청은 필요한 source·math·audit 레인만 독립화한다.

## 구조

```
../AGENTS.md         프로젝트 공통 지침(있는 경우)
AGENTS.md            `.codex/` 작업에만 추가되는 짧은 라우팅 규칙
config.toml          low effort 기본값과 최대 3개 병렬 레인
prompts/             CE 명령 프롬프트 7종
agents/              독립 연구·집필 역할 카드
skills/              단일 책임 CE 스킬
skills/ce-explanation-planner/  목표 정렬을 감사하고 수학·물리 판단을 채팅으로 설명하는 증명 계획 스킬
skills/ce-research/core/   기존 run 판독용 Rust 호환 코어
harnesses/           수치·증거 하네스 계약 — backend parity(Rust/CUDA), 실측 교정 루프, 뇌 증거 사다리
hooks.json           빈 자동 lifecycle hook 등록(의도된 상태)
hooks/               native Windows Python·저장소 계약·payload 실행기
```

## 진입과 검증

프로젝트 루트에서 Codex를 시작한다. 신뢰된 프로젝트만 `.codex/config.toml`과 프로젝트 로컬 지침을 로드한다.

```powershell
.codex\hooks\python.cmd doctor
.codex\hooks\python.cmd harness
.codex\hooks\python.cmd source <changed-python-paths>
.codex\hooks\python.cmd pytest <focused-test-paths> -q
```

- `doctor`는 허용된 interpreter와 핵심 dependency를 보고한다.
- `harness`는 `paper/` 전환, 이전 경로 잔존, 제거된 런타임 import, 필수 진입점과 AGENTS context budget을 검사한다.
- `source`는 실행 없이 AST를 파싱한다.
- `pytest`는 cache를 끄고 저장소 밖의 고유 임시 디렉터리를 사용한다.

수학·물리 판단은 `ce-explanation-planner`의 목표 계약·계획 설명·목표 이탈
감사·LaTeX·비유·지위·다음 증명 의무 계약을 따른다. 구조 검사는
`.codex/harnesses/explanation_first_planner.md`에 둔다.

## 점진적 공개와 피드백

- 루트 `AGENTS.md`는 저장소 지도·안전 경계·검증 등급만 담는다. 이 파일은 하네스 구조와 부채, 스킬은 반복 workflow, 코드는 정확히 판정할 수 있는 불변조건을 맡는다.
- 작은 요청은 direct로 처리한다. 관련 스킬과 역할 카드는 trigger가 실제로 맞을 때만 읽고, 독립성이 있는 레인만 병렬화한다.
- 같은 실패가 반복되면 프롬프트를 늘리기보다 누락된 지도·도구·기계 guard 중 하나를 가장 가까운 소유 표면에 보강한다.
- 자동 훅은 현재 없다. 저장소 계약은 빠른 `harness` 명령과 집중 테스트가 기계적으로 강제한다.
- FAST가 green이면 멈추고, 공용 경계나 인접 subsystem이 바뀐 경우에만 STANDARD로 넓힌다.

## 주의 (Windows)

- PowerShell 5.1이 파일·stdin에 UTF-8 BOM을 붙인다. 코어는 양쪽 모두 BOM을 제거하므로, 코어를 수정할 때 이 처리를 유지할 것.
- 레포 안에 `target/`을 만들지 않는다(.gitignore로 차단). 빌드 캐시는 `%LOCALAPPDATA%`.
- 역할 에이전트는 독립 레인만 담당한다. 원장과 논문 작성 규율은 각각 `ce-ledger-write`, `ce-paper-write` 스킬이 단독 소유한다.
- agent 실행은 stdin이 비대화형일 수 있으므로 uv/Python/보안 프롬프트를 기다리지 않는다. 이 PC에서는 uv cache ACL과 Windows Code Integrity의 enterprise signing policy가 각각 uv cache와 `.venv\Scripts\python.exe`를 차단했다. `python.cmd doctor`는 PowerShell 실행정책을 변경하지 않고 정책 허용된 기존 system Python만 선택하며 dependency를 설치하지 않는다.
- 근본 조치는 관리자가 서명·allowlist된 Python 경로와 uv cache ACL을 복구하는 것이다. 하네스는 Application Control을 끄거나 차단된 interpreter를 실행하지 않으며 그 전까지 focused 검사만 system Python으로 수행한다.

## 알려진 부채

| 항목 | 현재 경계 | 해소 조건 |
|---|---|---|
| `.claude/` lifecycle 설정 | Codex의 빈 자동 훅 정책과 별도 표면 | 두 provider 하네스를 함께 정비하는 요청 |
| retired CE run core | 기존 `_workspace/` 판독 호환만 유지 | 과거 run 호환 폐기 또는 마이그레이션 요청 |
