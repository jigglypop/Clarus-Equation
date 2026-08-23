# BA-V3-1 최종 보고 — source lock 실패에 의한 계약 대체와 안정성 정리 salvage

Status: ABANDONED (증거 원천 부재 — 10-sources 재감사에서 게이트 배터리의 출처 잠금 실패로 계약 §5가 채점 도구로서 무효, 20-audit Gate: BLOCKED. 계승: BA-V3-2 신규 계약)

기계 상태: `SOURCE_LOCK_FAIL / CONTRACT_SUPERSEDED / BIO_EVIDENCE_L1_NOT_REACHED / SEEDS_DEV_CONF_UNOPENED` (기계 문자열 — 이론 지위 아님)

Date: 2026-08-23

## 초록

생물 제약 원시연산의 L1 게이트 동시 재현을 시험하려던 계약이 채점 개시 전에 차단되었다. 신규 최우선 하네스(실제 뇌 식 발견 루프)에 따른 출처 재감사가 8개 게이트 중 5개(R1′, R3b′, R4′, R5′, R6′)의 출처 잠금 실패를 확정했고, 감사는 게이트 재설계의 outcome-blind성을 증명할 수 없어 엄격측(새 계약)을 적용했다. 수학 레인은 P0 없이 완료되어 여섯 건의 salvage를 남겼다 — 특히 안정화 기전의 귀속 정정(포화 수면 스케일링 단독으로는 발산하며 항상성 수축이 유일한 안정화 기전)과 일주기 질량 수지 정리, 그리고 L1 비율이 L2 로그정규 폭을 양방향 구속한다는 파생 관계다. development·confirmation seed는 미개봉이다. 부정 결과는 산출이며, 계승 계약 BA-V3-2의 재개 조건을 고정한다.

## 1. 처분과 경위

1. 계약(08-22)은 TS1 처방을 흡수해 동결되었고 math 레인이 게이트 정합성·정상상태 조건·대리 탐색을 완료했다 (P0 없음 — TS1형 게이트 모순 재발 없음).
2. 병행 세션(08-23)이 신규 정본 `real_brain_equation_discovery_loop.md`를 수립하고 10-sources에 출처 재감사를 추가 — "전 게이트 동시 적합은 source lock 불통과, 구현 불가".
3. 감사(20-audit): 재감사 근거는 B-class(출처 문면)이나 대리 탐색(7,167점) 이후 시점이라 outcome-blind 증명 불가 → 하네스 §4·TS1 선례의 엄격측 적용, **run 내 게이트 재정의 기각, 새 계약 처분**. 열린 P0 1건(A-2)으로 Gate: BLOCKED — 30/31은 SKIPPED 유지, 채점 미개시.

## 2. 출처 재감사 판정표 (10-sources 08-23 절 요약)

| Gate | 판정 | 처분 |
|---|---|---|
| R1′ | `UNVERIFIED_AS_GATE` (분자 구성 원문 미확인) | 출처 잠금 후에만 재도입 |
| R2′ | 검증 — metadata(코호트 가중·censoring·interval) 필수 | 보존 (N-계열) |
| R3a′ | 검증 — SBEM 측정모형·hierarchical unit 명기 조건 | 보존 |
| R3b′ | 정의 검증·게이트 미검증 (subgroup CI/SE 부재) | 정밀 대역 $(0,0.05]$ 제거, 불확도와 함께 보고형으로 |
| R4′ | `UNVERIFIED` (longitudinal dataset·estimator 부재) | 제거 |
| R5′ | `UNVERIFIED_AS_RATIO` (1.5 비의 출처 부재) | 제거 |
| R6′ | 검증 — 단 8시간 formation contrast (생존율 비 아님) | 재정의 보존 |

## 3. salvage (20-audit §3 전사)

- **S1 [정리]** $c\equiv1,\ \bar\Delta>\lambda_0\kappa/2$이면 포화 수면 스케일링 단독으로 전 $(\lambda_0,\kappa)$ 발산 — 수면 항은 안정화 기전이 아니다.
- **S2 [정리/미완성]** $\Delta$ 유계 + 항상성 수축 $\bar c<1$ ⇒ 순환 불변측도 존재 (유일성 미완). **기전 귀속 정정: TS1 no-go의 해소자는 $\lambda(w)>0$이 아니라 항상성이다** — 프레임 v3와 계승 계약의 승계 처방을 이 문장으로 정정한다 (P0/P1 이견은 감사가 P1로 조정 — 계약 주장은 "시험"이었으므로).
- **S3 [정리+산출]** 일주기 질량 수지 $(1+\gamma_{\rm top})(1-s_{\rm top})=(1+\gamma)(1-s)$와 정합 창. 파생(목표값 조건부): 각성 순 증가 $\gamma\ge0.2195$, $\lambda_0>0.219$, $\sigma_{\log w}\in(0.108,1.812)$ — **L1 비율이 L2 창발 통계를 양방향 구속**. [예측] 승격은 BA-V3-2 사전 등록 후에만.
- **S4 [산출]** R1′×R2′(N)의 지수 2-혼합 공존 닫힌 조건 — TS1 P0-1형 모순의 해소 구조 (조건부).
- **S5 [산출]** 구 R6′ 정의의 천장 $R6'\le1/R2'_{\rm ad}$ 부분 서로소 — 재정의 필요성의 독립 증거 (무조건 명제 금지: 대리 9.4% 전제 이탈).
- **S6 [산출]** 숨은 파라미터 적발: $\tau_e$(단독으로 R2′-ad 0.403↔0.931), $\beta$(미선언), $w_0$, $K$ 형상 — 계승 계약 자유도 선언의 필수 입력.

## 4. seed 위생

calibration 119001(탐색)·119002–119006(잡음)만 사용. development 119101–116·confirmation 119201–232 **미개봉** — 계승 계약이 승계/신규 중 하나를 명시해 처분. 소각 seed 118101 미사용 확인.

## 5. 동시 집필 사건 (기록)

병행 세션이 11-math를 1회 덮어씀(수치 witness는 독립 재현으로 합치, 등급 이견은 감사 조정) 및 stage 파일 20/30/31/40을 직접 집필 — 단일 작성자 규약 위반으로 판정. 구 판본은 `artifacts/concurrent-writer/`(provenance 헤더)와 `artifacts/math-verify/11-math.concurrent-writer-snapshot.md`에 무손실 보존. 재발 방지 4항: stage 파일 run당 단일 작성자(외부 산출은 artifacts+provenance), 집필 전 status로 소유 확인, `.active-run` 불일치 시 집필 금지, 충돌 시 두 판본 보존+auditor 회부.

## 6. BA-V3-2 재개 조건 (동결)

1. `real_brain_equation_discovery_loop.md` 필수 필드 12종 전부 (`BIO_STARTING_MECHANISM`…`CLAIM_CEILING`).
2. 게이트: R2′(metadata 완비)·R3a′(측정모형 명시)·R6′(formation contrast) 보존; R1′·R3b′는 분자·분모·불확도 출처 잠금 후에만; R4′·R5′는 신규 원자료 없이 제외.
3. **대역 설계 입력은 출처 수치만** — 본 run 대리 탐색 결과(게이트 난이도)의 사용 금지 (target-aware 차단).
4. 실데이터: de Vivo `synapse_data.csv` (MD5 `12e0c2e5ea231619df91a3c8d816d246`) — animal 단위 표본, 식 제안과 독립 confirmation에 동시 사용 금지.
5. P1 이관 7건 반영: λ 귀속 정정(A-1), R2′ 하위 정의 고정(A-4), 항상성 $\beta$·locus 선언(A-5), 숨은 상수 승격/출처 고정(A-8), 목적함수 행별 동결(A-9), 계수 재산정(A-10), 시간·단위 변환표(A-11).

## 7. 잔여 결함·P2

12-routes §8 전사: R3b′ 하한 판별력, R4′ 저판별력, 숨은 상수 4, E2 실 모형 전용성. P2 7건은 20-audit §4 목록 유지.

## 8. 재현성·참조

- run: `_workspace/ce/brainruntime-bio-constrained-l1-20260822`; 기계 결과 `artifacts/math-verify/` (nogo/budget/gate_stats_corrected/identity 등 — 감사 스팟체크 전건 일치).
- 레인: 00(R0 동결), 10(출처 재감사 포함), 11, 12, 20(Gate: BLOCKED 정본) — 2026-08-23 접근. 신규 하네스: `.codex/harnesses/real_brain_equation_discovery_loop.md`.
- 원장: BA-V3-1 행 갱신 (본 보고와 동시).
