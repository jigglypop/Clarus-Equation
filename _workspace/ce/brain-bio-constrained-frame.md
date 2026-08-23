# 뇌 알고리즘 작업 프레임 v3 — 생물 제약 원시연산 + 부분공간 기하 트랙

Status: ACTIVE / [미완성]

Date: 2026-08-22

v2(`brain-two-timescale-frame.md`)를 승계·확장한다. 주축은
`brain_evidence_ladder.md`의 두 질문 — 실제 뇌가 그 연산을 쓰는가, 뇌
데이터로 검증되었는가 — 이며, v2의 2시간척도 골격은 유지하되 원시 연산을
생물 확립 기전으로 교체하고, **부분공간 기하를 별도 트랙으로 보존**한다.
v2는 BA-TS1 계약의 참조 문서로 그대로 남는다.

## 1. 원시 연산 (사다리 하네스 허용 목록 준수)

| # | 원시 연산 | v2 대응물 | 처리 |
|---|---|---|---|
| P1 | 스파이크 이벤트 + 간선별 전도 지연 | 연속 활성 + tick 지연 | 재작성 |
| P2 | STDP 커널 → eligibility trace | 지연 일치 곱 | 교체 (측정 커널) |
| P3 | 3-인자: eligibility × 신경조절 broadcast | "문맥은 공급"(TR22 no-go의 공리) | 채택 — no-go의 생물학적 해답 |
| P4 | 측방 억제 경쟁 (별도 억제 집단, Dale 준수) | 알고리즘적 WTA | 재작성 — WTA는 창발이어야 함 |
| P5 | 항상성 scaling (목표 발화율) | TR9 항 | 생물 형태로 유지 |
| P6 | 2시간척도: 강도 w(분~시간) / 존재 A(일~월) | BA-TS1 골격 | 유지 |
| P7 | 수면 선택적 하향 + replay | TS1 수면 항 | 유지 + replay 후보 |

검증 사다리 목표: 첫 계약은 L1(비율 정합), 이어 L2(창발 통계 —
log-normal 가중치·발화율 분포를 부과하지 않고 얻는가), L3(IBL 등 공개
기록 예측 전이). L4 이전 "뇌가 이렇게 동작한다" 서술 금지.

기전 귀속 정정 (2026-08-23, BA-V3-1 감사 S1/S2): TS1 no-go의 해소자는
수면 항 $\lambda(w)>0$이 아니라 **항상성 수축 $\bar c<1$**이다 — 포화
스케일링 단독으로는 각성 이득이 $\lambda_0\kappa/2$를 넘으면 발산한다
[정리]. 이후 계약의 승계 처방은 "$\lambda>0$ + 항상성 수축($\beta>0$·locus
선언)"으로 쓴다. 상위 정본은 `real_brain_equation_discovery_loop.md`
(2026-08-23) — 실데이터 우선 발견 루프가 본 프레임의 L1 시뮬레이터
경로보다 우선한다.

## 2. 부분공간 기하 트랙 (보존 — 봉인과 생존의 경계)

### 2.1 봉인 유지 (재개 금지)

- 곡률 = 기억 동일성 (TR11 기각 — 순열 gauge).
- TR10 학습족 위 곡률 selector (TR13 폐쇄).
- 한 점 SPD 행렬을 곡률 증거로 읽기 (뇌 논문 §5.4 — 공간 미분 부재).
- "기하가 계산을 몬다"는 인과 방향 일반론.

### 2.2 생존 경로 (전부 부분공간/사영 이후에만 정의됨)

| ID | 객체 | 형식 지위 | 사다리 목표 | 근거 |
|---|---|---|---|---|
| G1 | **신경 다양체(neural manifold)**: 실기록 집단 활동의 저차원 부분공간과 그 위 기하 | 관측 현상 (문헌 확립) — 활동의 기술이지 가중치 기억 아님 | **L3** — 다양체 기하 구조가 held-out 예측을 raw 기술 대비 개선하는가 | 원장 BA-EMP-CLOUD-G `PASS_INPUT_METRIC_ONLY` (기록 내 output-Fisher 기하 시험 허용) |
| G2 | **Fisher pullback 쌍 $(G,R)$**: 조건부 output-Fisher tensor와 held-out routing 점수의 ordered pair | [정의/조건부 정리] — 단일 점수 합산은 교환비 공리 없이는 금지 | L3 | `docs/6_뇌/11_리만계량_라우팅_논문.md` 정리 1–4 |
| G3 | **flow pullback $g_{\rm pass}=J_T^\top G_TJ_T$** | [조건부 정리] (A6-P), 경험 미검 | L1–L2 진단 출력 | 원장 A6-P |
| G4 | **회로 계량장**: $\alpha_{ij}\to S^\Gamma\to(I-\xi^2\Delta_L)k^\Gamma=-2\kappa\xi S^\Gamma\to g^{\rm eff}$ | [공리: 모델 선택] — 우주론 bridge §15와 동일 식 계열 | 모델 선택 route, falsifier 필수 | 뇌 논문 §5.5 = 우주론 run artifacts §15 |
| G5 | **지연 인과 준계량** $d(i\to j)$ = 최소 누적 지연 (방향 비대칭) | [정의] — 시공간 질문의 적정 객체 (리만 아님, Finsler형) | L1 진단 → falsifier 확보 시 승급 | 도착 영수증 인프라 기존 |

### 2.3 생존 조건 (전 경로 공통 falsifier 규율)

1. 기하량은 **원시 간선·지연·활동 서술로 환원되지 않는 예측**을 내야
   생존한다 (TR11–13의 교훈). 환원 가능하면 진단 출력으로만 유지.
2. 계량은 식별된 부분공간/사영 위에서만 정의한다 — 전 공간 계량·시공간
   리만 계량은 불변원리 부재로 자유 convention이라 도입 금지
   (ce-dimensionless: 차원 다른 축 혼합은 기준 스케일 선언 필요).
3. 좌표·순열 gauge 불변 검사를 모든 기하 주장에 의무화한다.

## 3. 우주론 트랙과의 공유 골격 (형식만, 물리 동일시 금지)

탐색(2026-08-22)으로 확인된 실재 공유 구조:

- **typed 무차원 고정점 반복**: 우주론 $F_D(x)=e^{-D(1-x)}$ 최소근
  $q_{\rm ext}$ ↔ 뇌 multitype $q_i=\exp[-\sum_jA_{ij}(1-q_j)]$ — 같은
  반복자 타입.
- **사영 후 계량**: 우주론 scalar 부분공간 축약·projected residual-drive
  ↔ 뇌 probe 평면 pullback·Fisher chart. "부분공간 기하 가능성"의 정확한
  형식적 실체가 이 공유 패턴이다.
- **무차원 비율 게이트**와 7단 지위 태그의 동일 규약.

금지선 (기존 문서가 이미 고정): BR-8 — 우주론 상수($D_{\rm eff}$,
$q_{\rm ext}$, 밀도 분율)를 뇌 target·비율·setpoint로 이식 금지. "양쪽이
무차원이라는 사실은 사상을 유도하지 않는다" (P0 반례 $\Omega_b=cq$,
$c=1,2$ 모두 성립). 공유는 **감사 규약과 수학 골격**이지 물리 브리지가
아니다.

## 4. 우주론 쪽 미결 (탐색에서 확인, 별도 트랙)

- `cosmology-full-closure-unification-20260815`: 레인·게이트 완료
  (`Gate: PASS`)이나 30/31/40 부재. U1–U7 각각에 명시된 미완성 항목
  (migration, species bridge, UV coupling, 절대척도, immutable manifest 등).
- `self-recursive-reference-cosmology-quantum-20260820`: 20/30/31/40 부재,
  SR-6 `REVISE` (instrument/unravelling·genealogy 확률공간 의무), 정본
  최소 수정안 4건 적용 대기 (`12-routes.md` §5).
- 이 둘의 완결은 뇌 트랙과 독립 run으로 처리한다.

## 5. 후보 재정렬 제안 (원장 반영용)

| 순위 | 후보 | 사다리 | 근거 |
|---:|---|---|---|
| 1 | BA-TS1 (평균장 비율 feasibility) | L1 하한 | seed 미개봉, 저비용, 대역 자체 검증 |
| 2 | BA-V3-1: P1–P7 원시연산 재작성 + L1 재도전 | L1→L2 | 본 프레임 첫 계약 |
| 3 | BA-EMP-IBL / BA-EMP-CLOUD-G (G1·G2 실행체) | L3 | 부분공간 기하의 실데이터 시험 — 원장 기존 OPEN |
| 4 | BA-CG1: 지연 인과 준계량 (G5) falsifier 설계 | 진단→L1 | 시공간 질문의 적정형 |
| 5 | 우주론 2 run 완결 | — | 독립 트랙 |

## 6. 금지 재확인

곡률 selector 재개(TR13), simulator 결과의 생물 승격, L4 전 동작 동일성
서술, 우주론↔뇌 상수 이식(BR-8), 검증 없는 "유도됨", SHY/Yang 무근거 채택.
