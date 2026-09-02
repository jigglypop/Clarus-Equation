# CE 하네스·paper 재설계 (2026-09-02)

Status: `Phase 1 적용 완료 / Phase 2–3 사용자 확인 대기`

사용자 지적: "paper가 논문 진행과정만 잔뜩 적혀 있고, 연구 진전이 생각보다 안 나오고, 한글 설명이 부족하고, 미싱링크가 많다." 이 문서는 그 네 지적의 원인을 저장소 실측으로 확정하고, 이미 적용한 수정(Phase 1)과 확인이 필요한 구조 변경(Phase 2–3)을 나눈다.

## 1. 진단 (실측)

### 1.1 하네스 두 벌이 서로 다른 세계를 가리켰다

| 항목 | `.codex` (정본) | `.claude` (재설계 전) |
|---|---|---|
| 실행 모드 | direct. run·`_workspace/` 금지 | run 파이프라인(`init/status/check/revise/gc`), `CE_RUN=` 보고 요구 |
| `_workspace/ce/` | 과거 근거로만 읽음 | 산출물 저장 위치로 지정. **저장소에 존재하지 않음** |
| 하네스 계약 경로 | `.codex/harnesses/*.md` 5종 | `.claude/harnesses/*.md` 3종을 참조. **디렉터리 없음** |
| 설명 우선 스킬 | `ce-explanation-planner` 있음 | 없음 |
| `ce-paper-writer.md` | 30줄 | 966줄 — 46줄 본문이 21번 중복 복제된 손상 파일 |
| 역할 카드 6종 | 최신 | 전부 갈라짐(impl-engineer는 제거된 clarus-agent-guard·reality_stone을 기술) |
| Gate 어휘 | PASS·REVISE·PIVOT·BLOCKED | PASS·REVISE·BLOCKED |
| UserPromptSubmit/Stop 훅 | 없음 | `run.sh hook route/stop` — 정본 스크립트가 `exit 0`이라 **매 턴 무의미한 프로세스 2개** |

결과: Claude 세션은 존재하지 않는 파이프라인을 따르거나 무시했고, Codex 세션과 다른 규율로 같은 `paper/`를 고쳤다.

### 1.2 진전이 안 보이는 구조적 이유

- **표적이 셋으로 갈라져 있다.** 05장은 "blockade record → 공변 $Q^\mu$", 암흑에너지 원장 §5는 "비조건화 $U$ → 공변 source map", 36장은 "QFT-next M2 Hadamard 양성"을 각각 "다음 하나"로 적는다. 셋은 다른 트랙이고 서로 함의하지 않는다. 어느 하네스도 표적 하나를 강제하지 않았다.
- **진전의 정의가 없었다.** 형식 지위 규약(7종)은 과장을 막지만 "이번 세션에 무엇이 닫혔는가"를 묻지 않는다. 그 결과 양자 보존 원장은 E1→E78로 자랐고, E61–E78 열여덟 항목이 E59 하나의 하위 게이트다. E78의 결론은 다음 산출물 다섯 개 추가다. 게이트가 게이트를 낳는다.
- **원장이 현황판을 대신했다.** 양자 원장 414KB, 우주론 원장 151KB, 차원 분류 원장 111KB. 새 세션은 여기서 문맥을 재구성해야 했고, "닫힌 것·죽은 것·다음 하나"를 한 화면에서 보는 문서가 없었다.
- **커밋 로그가 비었다.** 7월 이후 167커밋 중 108개의 메시지가 "new". 진전을 git에서 추적할 수 없다.
- **본선 이탈이 문서로 정당화됐다.** 26–35장(중력 브리지 부록)은 읽기 지도 스스로 "지금은 암흑에너지 본선에서 벗어난다"고 적는다. 8월 28–30일 커밋 53건의 대부분이 이 부록이다.

### 1.3 한글 설명이 부족한 이유

- 26–35장의 문장은 한국어지만 명사는 거의 전부 영어다(incidence normal, proper projector, sorted Cholesky chart, coset lift). 풀이가 어디에도 없고, `9_등호이전`에는 용어사전이 있으나 `6_최신_연구`에는 없다.
- 장 제목이 영어 명사열이다(`31_exact_oriented_bivector와_full_shape_반례`).
- `ce-explanation-planner`(1학년 LaTeX + 문과 비유 + 정렬 감사)가 `.codex`에만 있어 Claude 세션에는 적용되지 않았다.
- `paper/README.md`는 "무엇을 주장하지 않는가"의 벽이다. 처음 읽는 독자가 "CE가 실제로 확립한 것"을 한 화면에서 볼 수 없다.

### 1.4 미싱링크 실측

`python.cmd links` 신설 후 첫 실행: 내부 링크 891건 중 깨진 파일 링크 0, 깨진 앵커 162(대부분 HTML id 인식 문제였고 실제 미스는 아래), 고아 문서 24.

- 읽기 지도의 2번 진입점 `#qnb-e38-a`가 양자 원장에 없었다(앵커는 E54 이후만 존재). 연구 목표 계약의 `#qnb-e28-a`도 없었다.
- README와 읽기 지도의 10장 앵커가 제목("…J2c까지")과 불일치.
- 암흑에너지 원장의 `#qd-m5-k/l/m1/m2/m3`, 14–17장의 `#c4-*`, 양자 원장의 `#clg-conformal-nogo-04`가 대상 원장에 앵커 없음.
- 고아 문서 23건: `4_공학적_활용` 5건, `5_유도` 4건(Navier–Stokes, 단백질, 신경, 블랙홀), `8_리만` 2건, `검증_원장` 8건(코어·flavor·GR·우주론GR 정리 증명 등), `참조` 3건.

## 2. Phase 1 — 적용 완료 (내용 수정, 파일 이동·삭제 없음)

### 하네스
- `.claude/CLAUDE.md` 전면 재작성: direct 모드, 라우팅 표에 `/ce-explanation-planner` 추가, 코어 명령을 `python.cmd`로 통일, **닫힘 예산** 절, **설명 우선(한국어)** 절, Gate에 PIVOT, 커밋 메시지 형식 `<종류>: <항목> — <한 줄>`, 하네스 계약 경로를 `.codex/harnesses/`로 교정. 11.4KB → 8KB.
- `.claude/settings.json`: no-op 훅 2개 제거, payload gate만 유지.
- `.claude/commands/`: ce-research·ce-light·ce-status·ce-gc를 direct 모드로 재작성, `ce-explain-plan` 신설.
- `.claude/agents/` 7종, `.claude/skills/` 8종을 `.codex`에서 재복사(경로·`$ce-`만 치환). 손상된 paper-writer 카드 복구, `ce-explanation-planner` 미러 추가.
- `.codex/harnesses/closure_budget.md` 신설: 진전 = 닫힘·기각·축소, 새 항목·새 장의 개설 조건(표적 연결 한 줄, 하위 게이트 3개 상한), 세션 종료 의무, 장 머리 3줄 상자, 영어 용어 병기.
- `.codex/skills/ce-research/SKILL.md`: 진전 원장 시작·종료 의무, 닫힘 예산, 용어 등록, Gate PIVOT 반영.
- `.codex/skills/ce-paper-write/SKILL.md`: 장 머리 상자, 영어 명사 나열 금지, 용어사전 등록, 탈고 게이트에 `links` 추가. 말미 빈 줄 40개 제거.
- `.codex/AGENTS.md`, `.codex/README.md`, `.claude/README.md`, `.codex/prompts/{ce-status,ce-research,ce-light,ce-gc}.md`: 진전 원장·닫힘 예산·커밋 형식·`links` 반영.
- `.codex/hooks/paper_links.py` 신설 + `python_harness.py`에 `links` 모드. `repository_harness.py` 필수 경로에 `paper_links.py`, `closure_budget.md` 추가.

### paper
- `paper/진전_원장.md` 신설(§1 목표, §2 단 하나의 표적 + 표적 분열 기록, §3 닫힌 결과 22행, §4 죽은 경로 27행, §5 열린 다리 7행, §6 트랙 6행, §7 세션 로그, §8 갱신 규칙). `paper/README.md` §9.1과 읽기 지도의 0번 입구로 연결.
- `paper/6_최신_연구/용어사전.md` 신설(ce-paper-writer 초안). 읽기 지도 0번에서 연결.
- 양자 보존 원장 E1–E78 앵커(`qnb-eNN-a`) 추가(E4·E5는 라벨 없음). QD-M5-K/L/M1/M2, QD-M5-M3, C4-*, CLG-CONFORMAL-NOGO-04, C4 절 제목 앵커 추가. 10장 앵커 2건 교정.

## 2b. Phase 1b — 목표 추적·workspace·독자 계약 (2026-09-02 추가 지시로 적용)

사용자 추가 지시: "중간에 해결 아이디어가 나오면 매몰된다. 목표를 주고 그 목표를 이루려는 하네스가 필요하다. paper는 한글로 논문 독자 관점에서 써야 한다. 과정(실패 확률 높은 시행착오)은 workspace 폴더로 빼되 절대 비대해지면 안 된다."

- **표적 주입 훅.** `.claude/settings.json`의 UserPromptSubmit이 `goal-reminder.cmd` → `.codex/hooks/goal_reminder.py`를 호출해 매 프롬프트마다 진전 원장 §2(트랙·하위 목표·완료 조건·kill 조건)와 주차장 규율 한 줄을 문맥에 넣는다. 출력 700자 이하.
- **`.codex/harnesses/goal_pursuit.md`.** 아이디어 주차장(실행 금지, 한 줄 기록), 우회 예산(세션당 1회·툴 호출 15개), 정렬 체크포인트(툴 호출 10개마다, `[이탈]` 2연속이면 강제 복귀), 표적 변경은 사용자 `[범위변경]`만, 역할별 적용(math-verifier routes는 주차장 후보), 종료 보고 순서.
- **`_workspace/` 재도입(작게).** `_workspace/README.md`가 상한을 정한다: README 제외 20개, 파일당 48KB, 전체 640KB, `.md`만, 하위 폴더 금지, 무활동 21일. `repository_harness.py`의 `check_workspace_budget`이 초과를 FAIL로 만든다. 노트는 시도 로그·주차장·흡수 판정 세 표만 가지며 흡수 후 삭제한다.
- **논문 독자 계약.** `ce-paper-write`에 "결과 먼저(주장→유도→검증→한계), 과정 표현 금지 목록, 영어 비율 0.25 상한, 앞 장 번호로 시작 금지" 절 추가. `.codex/hooks/paper_lint.py`(`python.cmd lint`)가 과정 표현 횟수와 영어 명사 비율을 파일별로 보고한다.
- CLAUDE.md·AGENTS.md·README·ce-research 스킬·프롬프트에 위 규칙 반영. AGENTS 지침 예산(8192바이트) 안에서 `.codex/AGENTS.md`를 재압축.

## 3. Phase 2 — 구조 변경 (사용자 확인 필요)

메모리 규약상 파일 이동·삭제·디렉터리 변경은 사전 확인 대상이다. 각 항목은 독립이며 하나씩 승인할 수 있다.

| # | 제안 | 이유 | 영향 |
|---|---|---|---|
| 2-1 | **표적 하나 선택.** 05장·DE 원장 §5·36장 중 어느 것을 §2 표적으로 둘지 결정. 나머지 둘은 §6 트랙의 "다음 하나"로 강등 | 닫힘 예산은 표적 하나를 전제 | 진전 원장 §2, 05장·DE 원장·36장의 "다음 표적" 문단 통일 |
| 2-2 | 26–35장을 `6_최신_연구/부록_중력_브리지/`로 이동하고 읽기 지도 본선 목록에서 분리 | 읽기 지도 스스로 본선 이탈로 규정. 본선 독자가 10장을 건너뛰어야 함 | 파일 10개 이동, 링크 갱신(`links`로 검사) |
| 2-3 | 6_최신_연구 장 제목을 한국어 명사구로 개명(영어는 괄호) — 예: `31_정확한_방향_이중벡터와_전체형상_반례(exact_oriented_bivector)` | 한글 설명 부족의 첫 인상이 제목 | 파일명 변경 37건, 링크 갱신 |
| 2-4 | 양자 보존 원장을 E 범위별로 분할: `참조_양자_보존_원장_E01-E40.md`(기록·영수증), `_E41-E53.md`(FLRW mode), `_E54-.md`(QFT-next) + 요약표만 남긴 입구 파일 | 414KB 단일 파일은 편집·검토·앵커 유지 모두 비용 | 원장 분할, 링크 갱신, ce-ledger-write 정본 위치 갱신 |
| 2-5 | 고아 문서 23건 처리: (a) 검증_원장 8건은 README §9.4에서 연결, (b) 4_공학·5_유도·8_리만·참조 15건은 `paper/_보관/`으로 이동하거나 README에 "보관 트랙" 절로 연결 | 링크 없는 정본은 정본이 아님 | 이동 또는 링크 추가 |
| 2-6 | `.claude/skills/clarus-guard-bench/`, `.claude/commands/guard-bench.md` 삭제 | 제거된 제품 참조 | 파일 삭제 |
| 2-7 | `.codex/skills/ce-research/core/`(Rust run 코어)와 `hooks/run.*` 은퇴 처리 | `_workspace/`가 없어 실사용 없음 | 삭제 또는 `_retired/` 이동 |

## 4. Phase 3 — 운영 규율 (확인 후 적용)

| # | 제안 | 기계 강제 |
|---|---|---|
| 3-1 | 커밋 메시지 `<종류>: <항목> — <한 줄>` 강제. 종류 ∉ {닫힘, 기각, 축소, 개시, 정리, 하네스}면 거부 | `check-large-data.cmd --commit`에 메시지 검사 추가 (또는 별도 commit-msg 훅) |
| 3-2 | 원장 파일이 바뀐 커밋은 `paper/진전_원장.md`도 바뀌어야 통과 | 같은 훅에 staged path 검사 |
| 3-3 | `python.cmd links`를 `harness` 검사에 포함(깨진 파일 링크 0 요구) | `repository_harness.py`에서 `paper_links.py` 호출 |
| 3-4 | 한 항목의 하위 게이트 4개째부터는 진전 원장 §2에 "표적 재정의/기각/공리 추가" 기록 링크 필수 | 문서 규율(기계 검사 없음), 세션 종료 메시지에서 사람 검토 |
| 3-5 | 주 1회 `/ce-status`로 §2 갱신일과 마지막 커밋 대조 | 프롬프트 규율 |

## 5. 검증 기록 (2026-09-02)

| 명령 | 결과 |
|---|---|
| `python.cmd links` (적용 전, 첫 실행) | 내부 링크 891, 깨진 파일 링크 0, 깨진 앵커 162(HTML id 미인식 포함), 고아 24 |
| `python.cmd links` (적용 후) | 내부 링크 1074, 깨진 파일 링크 0, 깨진 앵커 0, 고아 22 — `--strict`도 통과 |
| `python.cmd harness` | PASS (AGENTS 지침 8138/8192 바이트. `.codex/AGENTS.md`를 줄여 통과시킴) |
| `python.cmd source .codex/hooks/{paper_links,python_harness,repository_harness}.py` | PASS |
| pytest | 실행하지 않음 (Python 테스트 대상 변경 없음) |

- 고아 22건은 Phase 2-5의 결정 사항이라 남겨 두었다.
- 미커밋 상태로 남긴 사용자 변경: 36장·37장·양자 원장(+1038줄). 이 세션은 stage·commit하지 않았다.
