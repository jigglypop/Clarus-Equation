# CE Codex 하네스

현재 하네스는 단일 저장소 direct 모드다. 제거된 `reality_stone`을 요구하지 않고, 자동 CE_RUN 라우팅을 하지 않는다. 검증은 `.codex/hooks/python.cmd doctor|harness|source|python|pytest|links|lint`로 실행하며 최신 결과는 루트 `paper/` 정본을 직접 갱신한다. run core는 기존 실행 증거 판독을 위한 과거 호환 표면이다.

`paper/`에는 결과만 쓴다. 시행착오·중간 계산·아이디어 주차장은 `_workspace/` 노트(파일 20개·48KB·640KB·21일 상한, `harness`가 강제)에 둔다. 연구 현황판이자 유일한 표적은 `paper/진전_원장.md` §2다. 진전 정의는 `harnesses/closure_budget.md`, 매몰 방지는 `harnesses/goal_pursuit.md`가 소유한다. `.claude/`는 2026-09-02 v2부터 독립 설계(연구 루프 5 에이전트, `ledger/` yaml 원장)이며 v3부터 추측 우선(예측식·예산식 카드를 먼저 세우고 사다리로 증명, `harnesses/closure_budget.md` §1·§2)이다. `.codex`와 미러 관계가 아니다. 두 하네스가 공유하는 것은 `hooks/python.cmd`·`paper_links.py`·`paper_lint.py`·`goal_reminder.py`와 `harnesses/` 계약뿐이다.

## 구조

```
../AGENTS.md         프로젝트 공통 지침
AGENTS.md            `.codex/` 작업에만 추가되는 짧은 라우팅 규칙
config.toml          low effort 기본값과 최대 3개 병렬 레인
prompts/             CE 명령 프롬프트 8종 (ce-research, ce-light, ce-status, ce-gc, ce-audit, ce-dim, ce-validate, ce-explain-plan)
agents/              독립 연구·집필 역할 카드 7종
skills/              단일 책임 CE 스킬 8종
skills/ce-explanation-planner/  목표 정렬을 감사하고 수학·물리 판단을 채팅으로 설명하는 증명 계획 스킬
skills/ce-research/core/   기존 run 판독용 Rust 호환 코어 (은퇴)
harnesses/           수치·증거 하네스 계약 — closure_budget(닫힘 예산), goal_pursuit(매몰 방지), explanation_first_planner, empirical_calibration_loop, curvature_backend_parity, brain_evidence_ladder, real_brain_equation_discovery_loop
hooks.json           빈 자동 lifecycle hook 등록(의도된 상태)
hooks/               native Windows Python·저장소 계약·payload 실행기·paper 링크 검사기(paper_links)·독자 lint(paper_lint)·표적 주입(goal_reminder)
../_workspace/       시행착오 노트. README가 상한을 정하고 harness가 강제
```

## 진입과 검증

프로젝트 루트에서 Codex를 시작한다. 신뢰된 프로젝트만 `.codex/config.toml`과 프로젝트 로컬 지침을 로드한다.

```powershell
.codex\hooks\python.cmd doctor
.codex\hooks\python.cmd harness
.codex\hooks\python.cmd source <changed-python-paths>
.codex\hooks\python.cmd pytest <focused-test-paths> -q
.codex\hooks\python.cmd links [--strict]
```

- `doctor`는 허용된 interpreter와 핵심 dependency를 보고한다.
- `harness`는 `paper/` 전환, 이전 경로 잔존, 제거된 런타임 import, 필수 진입점과 AGENTS context budget을 검사한다.
- `source`는 실행 없이 AST를 파싱한다.
- `pytest`는 cache를 끄고 저장소 밖의 고유 임시 디렉터리를 사용한다.
- `links`는 `paper/`의 상대 링크·헤딩/HTML 앵커·고아 문서를 검사한다. 깨진 파일 링크만 실패이고, `--strict`면 깨진 앵커도 실패다. 문서 무결성 검사일 뿐 주장의 지위와 무관하다.
- `lint`는 `paper/`의 과정 서술 표현과 영어 명사 비율(상한 0.25)을 파일별로 보고한다. 권고용이며 `--strict`에서만 실패한다.
- `harness`는 `_workspace/` 상한(파일 수·크기·무활동 21일·`.md`만)도 검사한다.

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
| retired CE run core | 기존 `_workspace/` 판독 호환만 유지. 저장소에 `_workspace/`가 없으므로 실사용 없음 | 과거 run 호환 폐기 요청 |
| `paper/` 구조 부채 | 6_최신_연구 26–35장이 본선에서 벗어난 부록이며 제목이 영어 명사열, 양자 원장이 414KB 단일 파일, 고아 문서 23건 | `REDESIGN_2026-09.md` Phase 2의 사용자 확인 |
| `.codex` 안의 `ce-*` 역할 카드·스킬 | Codex 전용. Claude 쪽 미러는 2026-09-02에 삭제됨 | Codex 하네스도 5역할 구조로 줄일지 사용자 결정 |
