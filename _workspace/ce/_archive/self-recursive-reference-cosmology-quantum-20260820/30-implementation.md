# 30-implementation — 정본 문서 최소 수정

Status: COMPLETE

Date: 2026-08-22

Implementer: ce-impl-engineer (12-routes §5 문안, 20-audit §4 승인분만 적용)

## 1. 변경 목록 (승인 4건, 무변경 제약 1건)

| # | 승인 항목 (20-audit §4) | 파일 | 적용 내용 | 표지 |
|---|---|---|---|---|
| 1 | §5.1 typed iteration 서두 | `docs/2_경로적분과_응용/14_자기재귀성_대칭.md` | 신설 §0.1 — $X$, $T$, $x_0$, $\operatorname{Fix}(T)$, $\operatorname{Orb}_T(x_0)$, 조건부 readout $\mathfrak R_T(x_0)$ (극한 존재 시에만 정의, 보편 정리 아님 명시) | `[정의]` |
| 2 | §5.2 CPTP 반복 반례 | 같은 파일 | 신설 §0.2 — unitary 2-cycle($\mathcal U_X$, 주변 고유값 $-1$)·완전 dephasing 비유일 고정점 반례, primitive channel은 충분조건임을 명시. 출처: Wolf 2012 강의노트 6장, Watrous 2018 4장 | `[정리]` |
| 3 | §5.3 우주론 §2 축소 | `docs/3_상수/9_우주론_수식_의미와_후보.md` | 절 제목 `자기재귀 고정점` → `Poisson 세대 재귀와 우주론 readout 경계`. [정리]에 selection rule $q_0=0$과 두 고정점 구분($q=1$ 불안정 vs $q_{\rm ext}\in(0,1/D)$ 안정, $D>1$) 명시, `유일한 자기일관해` 무표시 보편 표현 금지 문장 추가 | `[정리]` 내부 |
| 4 | §5.4 SR-6 해소 구현부 | `docs/5_유도/00_선택과_접힘.md` §0.4.1 | 미완성 사슬 6항 → 8항: (신설 5) 실기록 outcome을 갖는 instrument/unravelling 지정 + 비유일성 사유, (신설 7) reproduction count 확률공간과 세대·계통 조건부 독립성. §0.7 총괄표의 "여섯 단계" → "여덟 단계" 정합 수정 | `[미완성]` 목록 |
| 5 | §5.5 무변경 제약 | `docs/3_상수/9_우주론_수식_의미와_후보.md` | legacy `[공리]` 문단(`C-B-LEGACY-01`, $q_{\rm ext}\mapsto\Omega_b$) **byte 무변경** — 정리 승격 없음 | 보존 |

diff 규모(읽기 전용 `git diff --stat HEAD`): 3 files, +47/−8.

## 2. 제약 준수

- **P2-1**: 다형 임계 정리는 이식하지 않았고, 신설 문안 어디에도 offspring 조건 없는 임계 주장 없음. 스칼라 [정리] 문안은 "지정 분지모형($\operatorname{Poisson}(D)$ offspring)" 전제를 유지.
- **P2-2**: 신설 수식 전부 `$$`/`$` 구분자, 백슬래시 원형 유지 — `test_all_docs_use_renderable_math_delimiters_outside_code` 통과로 기계 확인 (§31).
- **코드 무변경**: 프로덕션 코드·artifacts 스크립트 0건 수정 (BR-1 차단·M4-R 동결 유지). run 레인 파일(10/11/12/20) 무수정.
- **기계 판정어·구식 태그 0건**: 신설 문안은 형식 지위 7종 표지만 사용 — policy test에서 수정 3개 파일 위반 0건 (§31의 pre-existing 실패는 범위 밖 파일).
- **CE-5 승격 금지**: $q_{\rm ext}\mapsto\Omega_b$ 정리 승격 경로 차단 유지 — [공리] 문단 무변경으로 이행.

## 3. 불변식 재확인

- physics gates: 실패 후보·수치 은폐 없음(31에 red 결과 원문 기록), 사전등록 기준 변경 없음, 검증 없는 닫힘·제1원리 서술 신설 없음 — 신설 표지는 [정의]·[정리]·[미완성]뿐이고 [정리] 2건은 반례·표준 출처 기반.
- reality_stone/clarus·guard: 해당 코드 무접촉.

## 4. main 인계

- 변경 경로 3건: 위 표의 파일 경로. 신규 산출 1건: `artifacts/verify_brain_recursive_bridge.run-20260822.log`, 본 파일과 31.
- Git 상태 변경 없음 (stage/commit/stash 없음, diff 읽기만).
- 범위 밖 잔존 dirt: `docs/6_뇌/11_리만계량_라우팅_논문.md`(타 레인 수정), `docs/9_등호이전/README.md` "기계 pass는" 어휘, `.claude/.codex` ce-validate SKILL 미러 드리프트 — 본 레인 무접촉, 31 §2 참조.

Status: COMPLETE
