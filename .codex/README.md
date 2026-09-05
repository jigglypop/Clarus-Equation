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

## 모델 분담과 토큰 절약

2026-09-05 사용자 지시에 따라 보조 작업을 Luna·Terra로 분담한다. 메인 모델은 사용자가 앱에서 선택하며, 이 설정이 실행 중인 메인이나 기존 자식의 모델을 소급 변경하지는 않는다.

| 작업 | 모델·추론 | 담당 범위 |
|---|---|---|
| 좁은 코드 탐색, 원문·수치 추출, 로그·해시·지위 형식 확인 | `gpt-5.6-luna`, `low` | 근거 위치와 불일치를 짧게 반환 |
| 코드·문서 초안, 여러 파일의 논리 추적, 독립 수학 검산 | `gpt-5.6-terra`, `medium` | 명시한 대상과 완료 조건 안에서 분석 |
| 새로운 핵심 증명, 치명적 반례 판정, 레인 간 상충 해소 | 메인 `gpt-6-astra` | 최종 판단과 연구 목표 유지 |

- 독립 작업을 병렬화하면 시간이나 품질이 좋아질 때 위임한다. 단순 명령 실행·이미 확정된 결과 확인은 로컬 도구로 마친다. 구현 소유자 한 명과 메인 단독 Git 소유권을 유지한다.
- 일반 `explorer`도 기본 상속에 맡기지 않는다. 현재 협업 도구에서는 `fork_turns="none"`, `model="gpt-5.6-luna"`, `reasoning_effort="low"`처럼 명시한다. 복잡한 분석은 같은 방식으로 Terra를 선택한다. 역할에 모델이 고정된 `agent_type`에는 모델 재정의를 넣지 않는다.
- 전체 이력 복제는 그 이력이 결과에 꼭 필요한 경우에만 쓴다. 현재 도구의 전체 복제는 부모 모델·추론을 상속하므로 저비용 모델을 보장하지 않는다. 새 자식에는 질문, 정확한 파일·절, 확정 전제, 완료 조건, 읽기 전용 범위만 전달한다. 관련 작업의 기존 저비용 자식은 재사용한다.
- 자식 결과는 기본 5개 항목 이내로 결론·파일 위치·판별 수치·불확실성·다음 의무만 받는다. 증명 전문이나 편집할 초안이 산출물일 때만 길이를 늘린다. 원문·JSON·전체 도구 목록을 메인 대화에 복사하지 않는다.
- 같은 증거의 반복 확인을 위해 새 레인을 열지 않는다. 새 반례, 근거 불일치, 변경된 소스·판본이 있을 때 해당 범위만 다시 검사한다. Luna의 추출·형식 판단으로 새 수학·물리 주장을 승격하지 않는다. 불확실성이 남는 논리 검산은 Terra, 핵심 상충은 Astra로 올린다.
- `rg` 범위를 먼저 좁히고 필요한 줄만 읽는다. 독립 도구 호출은 함께 실행하되 출력은 요약한다. 개별 도구 기록은 `tool_output_token_limit=4000`으로 제한하고, 잘린 증거가 판단에 필요하면 그 부분을 다시 읽는다.

이 분담은 작업 비용을 낮추기 위한 로컬 운영 규칙이다. 실제 절감률은 측정하지 않았으며, 과학적 완료 조건·반례 기준·독립 검산은 그대로 유지한다. 공식 [Astra 지침](https://developers.openai.com/api/docs/guides/latest-model)의 명시적 위임·과잉 재검사 방지와 [하위 에이전트 지침](https://learn.chatgpt.com/docs/agent-configuration/subagents)의 모델별 분담·요약 반환을 참고했다. 설정 키는 [공식 설정 참조](https://learn.chatgpt.com/docs/config-file/config-reference)를 따른다.

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
