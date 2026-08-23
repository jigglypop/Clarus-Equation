# 20-audit — 형식 지위 감사

Status: COMPLETE

Date: 2026-08-22

Auditor: ce-status-auditor (읽기 전용 레인 — 본 파일은 오케스트레이터 전사. 해시 미계산, 파일 실재·줄 인용 대조만 수행).

## 0. 선행 조건 확인

- 10-sources.md, 11-math.md, 12-routes.md 모두 `Status: COMPLETE` — 감사 개시 조건 충족.
- 00-contract.md `Status: ACTIVE` — run 종료까지 정상.
- artifacts/brain-recursive-bridge-equations.md `Status: DRAFT-FOR-MATH-AUDIT`, artifacts/verify_brain_recursive_bridge.py 실재 확인.
- 20/30/31/40 부재 — 본 문서가 20을 채운다.

## 1. 형식 지위 판정표

### 1.1 SR 주장군 (11-math §8 표기 vs 감사)

| Claim | 표기 지위 | 감사 지위 | 근거 | 판정 |
|---|---|---|---|---|
| SR-1 | PASS-WITH-NARROWING | 일치 — [정의 규약], 보편 정리 아님 | 11-math:23-25 | 정합 |
| SR-2 | PASS-WITH-NARROWING | 일치 — $[0,1]$은 확률 해석 영역 | 11-math:42-43 | 정합 |
| SR-3 | PASS | 일치 — $x_0=1$ 반례와 최소근 selection 명시. 정본 14_자기재귀성_대칭.md:88-93에 이미 올바른 좁은 [정리] 존재 | 11-math:84-92 | 정합 |
| SR-4 | PASS | 일치 — 선형 CPTP vs 비선형 $F_D$ 타입 분리 | 11-math:131-146 | 정합 |
| SR-5 | PASS | 일치 — unitary 2-cycle, dephasing 비유일 반례 | 11-math:149-176 | 정합 |
| SR-6 | REVISE | 일치 — P1, §2 참조 | 11-math:178-191 | REVISE 유지 |
| SR-7 | PASS | 일치 — flow·timebase·제약면 요구, iteration index ≠ 우주시간 | 11-math:195-228 | 정합 |
| SR-8 | PASS | 일치 — $\Omega=cq$ family 반례로 readout 독립성 확인 | 11-math:241-257 | 정합 |

### 1.2 BR/B/E/G 주장군 (artifacts/brain-recursive-bridge-equations.md §18)

| Claim | 표기 지위 | 감사 지위 | 판정 |
|---|---|---|---|
| B1–B9 | 수학 후보, REVISE | [정의] 후보 — BR-1 구현 차단(Dale 부호 방향 모순) 명시, 구현 완료 주장 없음 | 정합 |
| B10 | 충분조건 후보 | [정리 후보: autonomous·zero-noise·delay-consistent 부분계 한정] — 적용범위 자기 제한, Lipschitz 상수 검산 일치 | 정합 |
| E1–E3c | 미완성 | [미완성] — causal receipt·calibration freeze 미구현 명시 | 정합 |
| E4–E8 | 조건부 정리 | [정리: E4 Poisson genealogy 공리 하] — 표준 multitype extinction과 일치 | 정합 |
| H1–H3 | 대안 route | [공리]+[조건부 정리: $\rho(K)<1$ 충분조건], 출처 명시 | 정합 |
| P1–P6 | 구현/model choice | [정의]·[공리] — 독립 target 요구 명시 | 정합 |
| G1–G8 | 정의/model choice | [정의]·[공리: 모델 선택]+[미완성] 명시 | 정합 |
| C1 | typed architecture | [정의] — "자연법칙·AGI 증명 아님" 명시 | 정합 |

### 1.3 기계 판정어 승격 검사

기계 판정어는 run 파일 내부에만 있고 정본 3종(14_자기재귀성_대칭.md, 9_우주론_수식_의미와_후보.md, 00_선택과_접힘.md)에는 형식 지위 7종 표지만 사용 — **승격 위반 0건**.

## 2. SR-6 결함 범위와 해소 경로 (P1)

- 결함: 계약 SR-6 의무 목록(CP reduced dynamics + population 폐쇄 + Markov jump rate + genealogy)만으로는 stochastic transition matrix까지만 얻고 offspring genealogy가 식별되지 않는다. 누락 의무 2건: (i) 실기록 outcome을 갖는 instrument/unravelling 지정 (같은 nonselective channel에 다른 unravelling 대응 — branching matrix 비식별), (ii) reproduction count의 확률공간과 세대 조건부 독립성.
- salvage 좁은 정리 (BLOCKED 불필요): "지정 instrument record + Markov counting + 조건부 독립 계보 + 식별된 $A\ge0$이 주어지면 $F_A$는 기록된 classical genealogy의 확률생성함수이고 E4–E8이 성립한다" — 12-routes §3.1 사슬이 이미 이 형태. bridge 전체는 [미완성] 유지.
- 해소 경로: 계약 판정 수용 기록(오케스트레이터, revise 1/2) + 30단계에서 route §5 항목 4 적용 → REVISE-CLOSED.

## 3. 숨은 공리 검사

주요 브리지·모델 선택 전부 공리로 명시됨 (unravelling 비유일성, E4 Poisson 독립성, G2–G8, C-B-LEGACY-01). 신규 숨은 공리 0건. P2-1: 다형 임계 정리의 offspring 비퇴화 조건이 암묵적 — 정본 이식 시 "Poisson offspring 하" 문구 유지.

## 4. 12-routes §5 정본 수정안 판정

| 항목 | 대상 | 판정 |
|---|---|---|
| §5.1 typed iteration 서두 추가 | 14_자기재귀성_대칭.md | **승인** ([정의] 표지) |
| §5.2 CPTP 반복·unitary/dephasing 반례 추가 | 14_자기재귀성_대칭.md | **승인** ([정리] 표지, 출처 인용) |
| §5.3 자기재귀 고정점 절 축소·$q_0=0$/$q=1$/$q_{\rm ext}<1$ 구분 명시 | 9_우주론_수식_의미와_후보.md §2 | **승인** |
| §5.4 미완성 목록에 instrument/unravelling·genealogy 확률공간 추가 | 00_선택과_접힘.md 0.4.1 | **승인** (SR-6 해소 구현부) |
| §5.5 legacy 공리 표지 보존 | 9_우주론 문서 | **승인 (무변경 제약)** |

전 항목 완전 반례 처리 규약 부합: 보편 부모 주장 축소 + 좁은 정리 보존 + 공리 표지 유지, 식별 공리·수치 상수 무변경.

## 5. 반례별 부모 주장 삭제 범위

| 반례 | 죽는 부모 주장 | 보존되는 좁은 정리 |
|---|---|---|
| CE-1 $x_0=1$ | "모든 초기값에서 최소근 수렴" (무표시 보편 표현) | $x_0\in[0,1)$ basin 정리 (정본 기존재) |
| CE-2 unitary 2-cycle | "CPTP ⇒ 반복 수렴" (정본 내 0건 — 예방적) | 주변 스펙트럼 조건부 수렴 |
| CE-3 dephasing | "CPTP 고정점 유일" | primitive channel 한정 유일성 |
| CE-4 $\dot x=-x$ vs $-x^3$ | "stationary set ⇒ 동역학·작용 선택" | flow+선형화 고유값 판정 |
| CE-5 $\Omega=cq$ family | "$q_{\rm ext}\mapsto\Omega_b$는 고정점 정리의 결론" — **정리 승격 경로 영구 차단** | [공리] C-B-LEGACY-01 조건부 branch 보존 |
| 뇌 CE-1 activation 초과 | 무projection bounded-state 주장 (본 run 코드 범위 밖, BR-1 차단 유지) | B2 projection 하의 bound |

정본 실삭제 0건 (CE-5만 활성 표현이 있었고 이미 [공리]로 강등) — 승격 금지 제약만 유지.

## 6. 30/31 단계 판정

- 30-implementation: **필요** — route §5 항목 1–4의 정본 문서 수정 4건. 코드 변경 없음 (BR-1 차단·M4-R 동결 유지).
- 31-validation: **최소 범위 필요** — artifacts/verify_brain_recursive_bridge.py 단일 실행으로 scalar basin·양자 반례·무차원 검사 수행, 수치 증인($q_{\rm ext}=0.0486467196445741$, multiplier $0.1545875231$) 로그화. 전체 pytest·bench는 SKIPPED (사유: 프로덕션 코드 무변경).

## 7. 결함 목록

- **P1-1** [SR-6] 계약 의무 목록 불완전 — §2 경로로 revise 1회 해소. 담당: 오케스트레이터(계약 수용 기록) + impl-engineer(route §5.4 적용).
- **P2-1** 다형 임계 정리 offspring 비퇴화 암묵 전제 — 정본 이식 시 문구 유지.
- **P2-2** 표기 손상: 11-math 일부 `\qquad` 백슬래시 누락, 12-routes의 `\theta` 탭 손상 — 정본 이식 시 재발 금지 (run 파일은 기록 보존).
- **P2-3** Lalley 자료 강의노트 등급 — 자체 표기 기존재, 유지 확인.

## 8. 집계

검사 명제 68 (SR 8, BR 8, B 10, E 12, H 3, P 6, G 8, C 1, 반례 12). 지위 승격 위반 0. 숨은 공리 0 (P2급 암묵 전제 1). 열린 P0: 0. P1: 1 (SR-6). P2: 3. 삭제·축소 확정 부모 주장 6 (정본 실삭제 0).

## 9. 1차 판정 (이력)

열린 P0 없음. P1 1건(SR-6)은 revise 1회로 해소 가능, 수정 범위 확정. 1차 판정은 REVISE였고 `revise contract` 1/2로 등록되었다. 지목 사항: 오케스트레이터의 계약 판정 수용 기록, impl의 route §5 항목 1–4 최소 변경(항목 5 무변경 제약, P2-1/P2-2 반영), 31의 focused 스크립트 단일 실행.

## 10. 재판정 (2차, 2026-08-22)

Auditor: ce-status-auditor (읽기 전용 재감사; 파일 실재·줄 인용 대조)

### 10.1 (a) SR-6 REVISE-CLOSED — **CLOSED**

| 확인 항목 | 증거 | 판정 |
|---|---|---|
| 계약 판정 수용 기록 | 00-contract.md SR-6 항 — 의무 2건, salvage 문안, bridge [미완성] 유지 명기 | 이행 |
| revise 한도 | revisions log = `contract` 1/2, 한도 내 | 이행 |
| 정본 구현부 | 00_선택과_접힘.md §0.4.1 사슬 6→8항 (instrument/unravelling + 확률공간·조건부 독립성), §0.7 총괄표 정합 | 이행 |

**P1-1 해소.**

### 10.2 (b) 31 witness 커버리지 공백 — **최소 검증 요구 훼손 없음 (P2-4로 강등)**

- 변경분 전부에 기계 증인 존재: 정본 신설 [정리](CPTP 반례 2종·selection rule)는 실행 로그에 `x_unitary_period_two=true`, `dephasing_fixed_states_nonunique=true`, supercritical 최소근 잔차 assert, projection 반례로 기록 — 31 표와 로그 원문 일치.
- 공백 수치($q_{\rm ext}$, multiplier)는 이번 정본 수정 어디에도 신규 전사되지 않음 — 미검증 수치의 정본 승격 0건. 지위 원천은 11-math 유도.
- 정직성 준수: 불일치 분류·미실행 명시·40-final §8 한계 기재 확인. P2-4 (후속: 로그 커버리지 확장)로 강등.

### 10.3 (c) 정본 수정 승인 범위·무변경 제약 — **준수 확인 (실파일 대조)**

§5.1 typed iteration [정의], §5.2 CPTP 반례 [정리] (자족 구성), §5.3 우주론 §2 축소(selection rule·두 고정점 구분), §5.4 미완성 목록, §5.5 legacy [공리] 문단 문자열 동일(무변경), 기계 판정어 승격 0건, P2-1/P2-2 준수, 코드·범위 밖 무접촉. 경미 신규 P2-5: §0.2 인용 출처(Wolf 2012/Watrous 2018)의 10-sources 원장 미등재 — [정리] 지위 영향 없음, 후속 원장 추가 권고.

### 10.4 재집계와 최종 판정

열린 P0: 0. 열린 P1: 0 (P1-1 CLOSED). P2 잔여 4건 목록 유지 (Gate 비차단). 모든 활성 주장의 지위가 근거와 일치 — SR-1~8 종결(SR-6 CLOSED), 정본 신설 표지 정합, bridge [미완성]·legacy [공리] 보존.

Gate: PASS
