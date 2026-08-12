# Research contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v14-binding-design-20260812

## 규율

**수학 완전증명 → 구현** 순서를 강제한다 (사용자 지시 2026-08-12). 구현(30)은 11-math의 해당 정리가 닫히기 전에 시작할 수 없다. 수치는 검산용 toy만 허용.

## 질문

뇌형 국소 동역학(V14 route L: 에너지 게이트 + 무손실 latch + HRR binding)과 CE 우주 프레임워크(부트스트랩 3분할 $p^*=(4.87\%,26.2\%,68.9\%)$, 디퓨전 기질)를 하나의 장 — **클라루스장** — 으로 접합하는 형식화가 수학적으로 성립하는가, 그리고 평형 점유율의 $p^*$ 자기수렴 주장은 어떤 조건에서 정리이고 어떤 조건에서 반증 가능한 예측인가.

## 정의역

- 기질: 유한 연결 그래프 $G=(V,E)$, 노드 상태 $s_i \in \mathbb{R}^w$, 스칼라 보조장 $\phi_i \ge 0$.
- 장 방정식(후보): $\partial_\tau \phi = -\Delta_G \phi + s - \lambda\phi$ (디퓨전, 20_DiffusionOrchestration Exact 부분) + 국소 성분 $s_i' = (1-g_i)s_i + g_i\tilde s_i$, $g_i = \sigma(a\|m\odot x_i\|^2 + b)$ (V14 코어). 소스 차원 결손은 11-math 공리 D1로 해소.
- 3상 분할(정의): 활성 $\iff g_i(t)>\theta_g$; 구조 $\iff$ 비활성이고 binding 참여($>\theta_s$); 동결 $\iff$ 그 외(latch 항등). 점유율 $\pi=(\pi_A,\pi_S,\pi_F)$.

## 주장 (수학 레인 완결 전 구현 금지)

- CF-1 (well-posedness): 결합계의 전역 유계 + 동결 항등 보존 충분조건. → 11-math: 조건부 정리(D1·S1–S3).
- CF-2 (게이트-스케줄 안정성, V9 재개 기초): 닫힘 구간 등거리 + 열림 convex 유계 쓰기 ⇒ T-균일 유계·ρ-제어 오차. 균일 수축(NISCC-6A)의 **대체**. → 11-math: 정리.
- CF-3 (점유율 평형 존재): 에르고딕 공리 하 시간 평균 극한. → 11-math: 조건부 정리(A-E1·A-E2; A-E3 명시 권고 P2-a).
- CF-4 ($p^*$ 자기수렴): 유도 경로 부재 시 자유 예측으로 정직 분류 + killing test 명세. → 11-math: 자유 예측. 12-routes: R-A1(항상성 임계 적응, GO-후보)·R-A3(Poisson 소멸확률 재해석, Hypothesis).
- CF-5 (선형 게이트 불가능성의 장 일반화): → 11-math: 정리(정적)+조건부 정리(적응).

## 기호·허용 오차

$q$(수축률), $\rho$(게이트 열림 빈도), $\lambda$(디퓨전 감쇠), $\theta_g,\theta_s$(상 임계 — 사전등록 대상, $\theta_s$는 $\bar r/\lambda$ 단위). 수치 검산 상대오차 $\le 10^{-9}$. 본 채점 규약은 구현 전 별도 사전등록.

## 자유도 원장 (.claude/rules/ce-design-freedom-ledger.md 준수)

- 과제 생성기: 구현 전 사전등록 고정(변경 예산 0). 수학 레인 toy는 검산용 자유.
- 하이퍼파라미터 burned-seed 예산 4 — route 레인 소모 0, 잔여 4.
- 게이트 임계값: killing test 프로토콜에서 실행 전 고정, 사후 조정 금지.
- 아키텍처 변형 상한 3 — R-A1의 공리 AH1(항상성 적응 루프) 채택 시 1/3 소모 (감사 P2-c 반영).
- 1번 게이트는 외부 baseline 절대 기준. 인증-과제 양립 = CF-2가 그 선언.

## 경계

- 뇌·우주 대응은 Bridge 상한. $p^*$ 이식은 CF-4 킬링 테스트 통과 전 Hypothesis.
- V14의 C1/C2/A-SAL은 수학 CONFIRMED로 인용 가능, route L toy 수치는 본 채점처럼 인용 금지(V14 감사 P2-3 상속).
- 동결 파일 무수정. 구현은 감사 승인 순서(P1-1 정본 수리 → 구현 계약 사전등록 → 구현) 엄수.
