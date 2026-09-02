# Q-0006 attempt-01 — 대조표: M1 작용(E54-A) 대 BFR/BFHPR 기준장·pAQFT 조건

기준: `derivations/Q-0006/attempt-01.derivation.md` (S1)–(S5). 문헌 사실은 sourcer 2026-09-02 확인 범위(절·식·정리 번호 미확보). 라벨: 요청의 E58 = 정본 E61-D형 상수-$\bar X$ 평탄 배경, 요청의 E59 = 정본 E62-A 비퇴화 FLRW 국소 patch. E78은 정본에 없음.

## 표 1. BFR 일반 틀 조건 대 M1 $X^A$

| 조건 | 문헌 진술(확인 범위) | M1 $X^A$ (E54-A) | 판정 | 근거 단계 |
|---|---|---|---|---|
| (C1) 공변 스칼라 partial observable 4개 | BFR: 미분동형 공변 스칼라 함수 4개로 단사 사상 $X_\Gamma$ | 기본 동역학 스칼라장 4개, $\delta_{AB}$ 내부 상수 metric | 충족 | (S2.1) |
| (C2) 단사성 $\det(\partial_\mu X^A)\neq0$ — 상수-$\bar X$ 평탄 (E58/E61) | BFR: 배경 근방 단사 | $\partial_\mu\bar X^A=0$, $\det=0$ | 실패 | (S2.2) |
| (C2) 단사성 — 비퇴화 FLRW patch (E59/E62) | 동상 | $\det=\dot T\beta^3=\mu_X^4ua^3b^3\neq0$ iff $u\beta\neq0$; local/noncompact, 유한 과거 singularity bound | 국소 통과 | (S2.3), verify 1 |
| (C3) 섭동적 BV·QME 해결 | BFR·Rejzner: QME = 재규격화 조건 | E70-B…J bounded jet CME witness까지; full M1 CME·QME·loop ST 없음 | 미검증 | (S2.4) |
| (C4) Hadamard 상태 | BFHPR: FLRW 위 Hadamard 다룸 | E63/E65/E67 frozen principal symbol $c^2=1$, $K,G>0$(선언 부문) — 쌍곡성 징후만 | 미검증 | (S2.5) |
| (C5) BFHPR 특수 구성 | $X^0=X^0[\tilde\phi]$, $X^i$ 비국소 조화 좌표; 외부장은 "not useful" | 외부 동역학 물질장 = BFHPR이 배제한 클래스 | 밖 (불가능 아님) | (S2.6)–(S2.7) |

## 표 2. "not useful" 대 "impossible"

| 문헌 문장 | 이유(원문) | 목적 | CE 해석 |
|---|---|---|---|
| "introducing external fields as reference coordinates like in the Brown–Kuchař model is not useful in the context of cosmological perturbation theory" | "these fields would appear in the final gauge-invariant expressions" | 우주론 섭동론: inflaton·metric만의 관측량 | $X^A$는 물질(정본: gauge-only label 아님)이므로 최종식에 남는 것이 의도된 특징. 구성 불가 판정이 아님 |

## 표 3. 4D 미분동형 이상항 부재 출처

| 출처 | 진술 | 범위 | 확인 상태 |
|---|---|---|---|
| Alvarez-Gaumé–Witten 1984 | 중력 이상항은 $d=4k+2$ 카이럴 장에만 | 차원 정리; $4\notin\{4k+2\}$ (verify 2) | 확인(근사) |
| hep-th/0404033 | "there are no field-theoretic diffeomorphism anomalies" (4D) | 4D 장론 일반 | 명시 문장 확인 |
| 보손 물질 전용 문장 (Weinberg §22, Bertlmann) | — | — | 미확인(접근 불가) |
| $H^1(s|d)$ ghost number 1 cohomology 계산 (4D 중력+스칼라) | — | BV 이상항 class | 출처 미확보 |

M1 field content: metric + 실스칼라 5개, 카이럴 장·게이지장 없음. Weyl 이상항은 M1의 게이지 대칭이 아니므로 무관 (S4.3).

## 표 4. kill 판정

| kill | 사전등록 문장 | 판정 | 조건 |
|---|---|---|---|
| 1 | $X^A$가 BFHPR/BFR 클래스 밖이면 문헌 도입 무효 | 부분 | BFR 안(C1 충족, C2 배경 조건부·국소); BFHPR 특수 구성 밖 |
| 2 | 4D 보손 미분동형 BRST 이상항 부재 명시 문장 없으면 M2 조건부도 불가 | 부분 | 4D 명시 문장 있음; 보손 전용·$H^1(s|d)$ 범위 문장 미확인; (36.7)↔QME 동치 미증명 |
| 3 | BFHPR Hadamard 구성이 E59에서 적용 조건 불충족이면 M3 무효 | 부분 | 불충족 증거 없음, 충족 증명 없음 — E62 선형화계 Hadamard 존재가 결정 항목 |

## 표 5. 적응에서 새로 증명할 것 (M3)

| ID | 항목 | 문헌이 주는가 | 정본 현황 |
|---|---|---|---|
| P1 | $X^A$ 요동의 dressed 관측량 대수 기여와 고정 배경 극한의 억제 조건 | 아니오(BFHPR에 없는 항) | 없음 |
| P2 | 강결합 스케일 $\Lambda_{\rm sc}(\mu_X,M_P)$, $f$ 지지 $\ell\gg\Lambda_{\rm sc}^{-1}$ | 아니오 | 36장 "미완성" |
| P3 | E62 patch 전역 쌍곡성 + 게이지 고정 선형화계 normal hyperbolicity + Hadamard 존재 | 자유 KG 정리는 있음(혼합계 확장 출처 미확보) | E63/E65/E67 부분 |
| P4 | 국소 단사 → patch gluing, compact $\Sigma$ 대 noncompact rod | 아니오 | E62-A 정의문이 전역 rod 아님을 명시 |
| P5 | dressing·edge/boundary sector (Donnelly–Giddings) | 아니오 | 없음 |
| P6 | M1 QME 해결 (M2 공유) | 틀만 | E70 계열 유한 witness |

## 검증 상태

symbolic: skipped (sympy 없음). numeric: 두 산술 항등식만 (문헌 판정 아님). 결과는 `hook_result.json`.
