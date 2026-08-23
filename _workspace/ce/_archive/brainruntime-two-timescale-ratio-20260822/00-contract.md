# BA-TS1 계약 — 2시간척도 간선 동역학의 관측 비율 동시 재현

Status: COMPLETE

Date: 2026-08-22

## 1. 질문

프레임 v2(`_workspace/ce/brain-two-timescale-frame.md`)의 선언된 간선 동역학
족이, 자유 파라미터 6개의 **단일 전역 적합 1회**로 관측 유래 무차원 비율
7개를 **동시에** 사전 고정 대역 안에서 재현하는가. 재현 실패는 이 족의
STOP이며 산출이다.

우선순위 기록: 원장 순서는 TR31→TR32→TS1이었으나 사용자가 TS1 우선을
지시했다 (2026-08-22). TR31/32는 독립 트랙으로 유지된다.

이 run은 시뮬레이터 실험이다. 통과해도 주장은 "관측 비율과 정합하는 최소
합성 동역학"까지이며 생물학적 기전·수면 기능·기억·AGI 증거가 아니다.

## 2. PREDECESSOR_EVIDENCE

| ID | 상태 | 보존 주장 | 증거 경로 | 재시도 금지 조건 |
|---|---|---|---|---|
| 프레임 v2 | `ACTIVE / 미완성` | F1 기질 동결(공리), F2 2계층, F3 상태 변조(분위수 자유도 격리), F4 기하 파생, F5 잡음 자원 | `_workspace/ce/brain-two-timescale-frame.md` | SHY/Yang 어느 쪽도 공리 승격 금지 |
| BA-TR9/10 | `DEVELOPMENT_GO` | 내생 경쟁·항상성과 국소 확률적 쓰기의 합성 성립 | 원장 행 | gain·decay·seed 재조정 금지 |
| BA-TR14/15 | `STOP / GO` | 지연 일치 eligibility 1회 쓰기, 감쇠 보상 | 원장 행 | 게이트 재조정 금지 |
| BA-TR11–13 | `REJECTED / CLOSED` | 기하량은 파생 진단 — 본 run에서 선택 규칙에 불사용 | 원장 행 | 곡률 selector 재개 금지 |
| 관측 기준선 | sourcer 조사 (2026-08-22) | §5의 비율 7개 (UNVERIFIED 항목은 게이트에서 제외) | `10-sources.md` | 대역의 사후 확장 금지 |

## 3. 동결 모델 (족 선언)

평균장 간선 모집단. 후보 간선 $N_E=10^4$개, 상호작용 없음(최소 apparatus —
BrainRuntime 결합은 이 run의 범위 밖). 시간 단위 = 1일, 각 일은 각성
구간과 수면 구간으로 구성(길이 비 16:8 고정, 설계 상수).

각성 (매일):

$$
w_{ij}\leftarrow w_{ij}+\eta\,\xi_{ij},\qquad
\xi_{ij}\sim\mathrm{Bernoulli}(p_{\rm hit}),\ p_{\rm hit}=0.1\ \text{(설계 상수)}
$$

수면 (매일):

$$
w_{ij}\leftarrow w_{ij}-\lambda_0\,w_{ij}\,\mathbf 1[w_{ij}<q_\theta(\{w\})]
$$

($q_\theta$: 생존 간선 강도 분포의 $\theta$-분위수 — 상위 $1-\theta$ 보존)

존재 (이력 birth–death):

$$
\Pr[A_{ij}:0\to1\ /\text{일}]=\rho(t),\qquad
\Pr[A_{ij}:1\to0]=\mathbf 1[w_{ij}<w_{\min}\ \text{연속}\ \tau_e\ \text{일}]
$$

성숙 스케줄 (형성률 단일 기전):

$$
\rho(t)=\rho_\infty\bigl(1+\kappa\,e^{-t/T_m}\bigr)
$$

신생 간선은 $w=w_0=1.2\,w_{\min}$에서 시작(설계 상수).

**게이지 고정·설계 상수 (적합 금지)**: $w_{\min}\equiv1$ (단위 선택),
$\tau_e\equiv2$일, 각성:수면 = 16:8, $p_{\rm hit}=0.1$, $w_0=1.2$,
$N_E=10^4$, 발달 관찰 구간과 성체 정상상태 판정 구간(§6).

**자유 파라미터 (6개, 1회 전역 적합만)**: $\eta,\ \lambda_0,\ \theta,\
\rho_\infty,\ \kappa,\ T_m$.

## 4. 적합·평가 프로토콜 (동결)

1. calibration seed에서 로그-비율 공간 가중 최소제곱 **1회 전역 적합**으로
   $\theta^\*=(\eta,\lambda_0,\theta,\rho_\infty,\kappa,T_m)$을 얻는다
   (가중치 균등, 알고리즘은 impl이 결정하되 목적함수는 이 문면으로 동결).
2. $\theta^\*$를 동결하고 development seed 16개에서 §5의 비율 7개를
   각 seed마다 평가한다. 전 seed × 전 비율이 대역 안이어야 GO.
3. 비율별 개별 재조정, 대역 확장, 파라미터 추가, seed 선별은 전부 STOP.

## 5. falsifier 비율과 사전 고정 대역 — 형식 지위: [예측]

성체 정상상태(§6의 판정 구간)에서 측정. 근거 관측은 10-sources.md.

| # | 비율 (무차원) | 목표 | 대역 | 근거 |
|---|---|---:|---|---|
| R1 | 월간 존재 turnover (제거+형성)/생존 | 0.04 | [0.02, 0.08] | V1형 저회전 (Grutzendler 2002) |
| R2a | 8일 생존분율, 발달 초기(밀도 피크 시점) | 0.35 | [0.25, 0.45] | Holtmaat 2005 |
| R2b | 8일 생존분율, 성체 — R2a보다 크고 단조 성숙 | 0.73 | [0.60, 0.85] | Holtmaat 2005 |
| R3a | 수면 1회당 총 강도 하강분율 | 0.18 | [0.10, 0.25] | de Vivo 2017 |
| R3b | 수면 중 불변 상위 분위 | ≥0.15 보존 | 상위 15% 이상 무변화 | de Vivo 2017 (상위 ~20%) |
| R4 | 일 주기 순환 안정: 연속 200일 일평균 총 강도의 표류 | 0 | 상대 표류 <5%/100일, 발산 없음 | SHY 정합 조건 |
| R5 | 밀도 과잉: 피크 존재 수/성체 정상 존재 수, 이후 단조 감소 | 1.5 | [1.3, 1.8], 재상승 없음 | Huttenlocher (+50~55%) |
| R6 | 학습 이벤트(η 5배 하루) 후 신생 간선 8일 생존율 / baseline 신생 생존율 | >1 | ≥1.3 | Xu/Yang 2009 (정성) |
| R7 | 고강도($w>q_\theta$) 간선의 월 자발 소실률 | ≈0 | <0.005 | 증거 기반 제거만 |

계수: 자유 파라미터 6 < 게이트 비율 9행(독립 조건 7군: R2a/R2b, R3a/R3b는
각 1군). UNVERIFIED 관측(학습 후 정확 %, Diering 정밀도 등)은 대역 산정에
사용하지 않았다.

## 6. 실행 계획

- 시뮬레이션 길이: 발달 0–300일(피크·성숙 관찰), 성체 판정 구간 500–700일.
  전 구간 하나의 연속 궤적 (구간 경계는 설계 상수).
- seed: calibration `118001`. development `118101..118116`.
  confirmation `118201..118232` — **봉인**.
- calibration 단계 apparatus 결함 수리는 development 개봉 전 revision
  기록下에만. 이후 §4-3 STOP 규율.
- 교정 레지스터: `.codex/harnesses/empirical_calibration_loop.md` 적용.
  식 개정(족 변경)은 §8 프로토콜 — 본 run 안에서 하지 않는다.

## 7. 레인 계획

- physics-sourcer → 10-sources.md: 기왕 수집된 기준선 원자료의 전사 +
  대역 산정 근거. (동일 세션 sourcer 조사 결과 사용, 재검색 불요.)
- math-verifier → 11-math.md: (a) 정상상태 해석식 유도 — R1·R2·R7이
  $(\lambda_0,\theta,\eta,p_{\rm hit},w_{\min},\tau_e)$에 어떻게 의존하는지
  닫힌 근사; (b) 6-파라미터로 7군 동시 만족의 가능성/불능 판정 (모순
  발견 시 P0); (c) 적합 목적함수의 식별성; (d) R4 비발산 조건. routes는
  P0/막힘 시.
- 이후 audit → impl(시뮬레이터+적합+평가) → final.

## 8. 주장 상한

통과: "선언된 6-파라미터 족은 관측 유래 비율 7군과 정합하는 최소 합성
동역학이다" [경험식]. 실패: 이 족의 STOP (산출). 어느 쪽도 생물학적 기전,
수면의 기능, 기억 저장, 곡률, 질병, 물리 에너지, AGI를 확립하지 않는다.
