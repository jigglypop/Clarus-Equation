# BA-TR30 계약 — 독립 후보 스트림 하 다항 차수·잡음 모델 식별

Status: COMPLETE

Date: 2026-08-22

## 1. 질문

사용자 질문 "식이 안 나오고 있음"의 형식화: 지배식(다항 내용 법칙)을 데이터만으로
**식별**할 수 있는가. 구체적으로, 회전 셀 cue/content 과제에서 사전 선언된 차수
집합 $D=\{1,2,3\}$과 잡음 수준 집합 $H=\{0,10^{-3},10^{-2}\}$ 위에서, 동결된
학습 연산자가 (a) 생성기 차수 $d^\*$를 데이터에서 선택하고, (b) held-out query의
내용 벡터를 직접 예측하며, (c) 선언 밖(차수 4) 생성기에는 fail-closed로
기권하는가. 후보 스트림은 held-out 진리와 예측 어느 쪽에서도 유도되지 않아야
한다.

이 run은 시뮬레이터 모델-클래스 식별 실험이다. 실제 뇌·기억·의식·AGI의 증거가
아니며 승격하지 않는다.

## 2. PREDECESSOR_EVIDENCE

| ID | 상태 | 보존 주장 | 증거 경로 | 재시도 금지 조건 |
|---|---|---|---|---|
| BA-TR28 | `CONDITIONAL_LOW_DEGREE_PREDICTION_PASS / RANDOM_PACKET_ENDPOINT_STOP` | 선언된 2차 조건 하 정확 예측(400/400, 최대 오차 `2.74e-15`); 무작위 3-packet endpoint는 STOP; 선언 밖 query-only 완성 $y_q+\delta$ 비식별 | `brainruntime-rotating-low-degree-content-transfer-20260822`, 원장 행 BA-TR28 | threshold·seed·split·control 재조정 금지, STOP 뒤집기 금지 |
| BA-TR29 | `HARD_NEGATIVE_DISCRIMINATION / DEVELOPMENT_GO / CONFIRMATION_SEALED` | 동결 2차 학습자가 nearest 은행과 affine hard-decoy를 모두 구별(400/400); 은행이 환경 중심·decoy가 기준선 유도라는 한계 명시 | `brainruntime-low-degree-hard-negative-transfer-20260822/40-final.md`; 원장 행 BA-TR29; source-freeze `3d6b9a3280e627844e5cd6cc7d16044a6af99b9d49995d45c9e2e498e4b0414e` | 예측·기준선 유도 decoy 재사용 금지; 은행 중심화 금지 |
| BA-TR22 | `THREE_EVENT_RELEVANCE_NO_GO_CONFIRMED` | 국소 규칙에는 외부 지정 관련성 변수가 없음 — 관련성은 공급되어야 함 | 원장 행 BA-TR22 | 과제 재조정·암묵 관련성 proxy 추가 금지 |
| BA-TR20 | `POST_REPAIR_PROVENANCE` | read-before-write 지연 packet 복제 수리; TR21+ 전 run의 runtime 정본 | 동결 runtime SHA-256 `5dc1ba5722ffa430f5c4dd4502defd49130dbe36c7f5522f28a7df36338fed26` | ring alias 재도입 금지 |

closure 공백 기록: 선행 run `brainruntime-low-degree-hard-negative-transfer-20260822`에는
`12-routes.md`가 없고 최종 파일명이 `40-final.md`다. 원장 행 BA-TR29가 정본
지위이며, 본 계약은 원장 행과 40-final의 next-falsifier 문구를 승계 근거로 쓴다.

## 3. 후보 정렬과 기각

원장 "Immediate next priority after BA-TR29" 순서를 따른다.

| 순위 | 후보 | 선택/기각 근거 |
|---:|---|---|
| 1 | **BA-TR30 (본 run)** | 선택. TR28 STOP과 TR29 한계가 남긴 정보(모델-클래스 판별이 살아 있는 질문)를 직접 소비. 독립 falsifier(차수 불일치·잡음·선언 밖 기권) 보유. capability 의존은 동결 runtime뿐 |
| 2 | BA-A8-D-RUST | 기각(연기). `BLOCKED_ABI` — 과학 endpoint 없는 구현 경로 |
| 3 | BA-EMP-IBL | 기각(연기). 외부 데이터 취득 capability 의존, 인과 주장 금지 제약 |
| — | TR28/29 재조정 변형 | 퇴역. endpoint·threshold·seed만 바꾼 재시도는 같은 경로 |

## 4. 동결 설계

### 4.1 apparatus

- runtime: 동결 SHA `5dc1ba5722ffa430f5c4dd4502defd49130dbe36c7f5522f28a7df36338fed26` (BA-TR20 수리판). packet 전달·수령 영수증 규율은 TR28/29와 동일하게 유지한다.
- cue $z\in\mathbb R^2$ (무차원 opaque), content $y\in\mathbb R^6$. cue 표본: $z_i\overset{iid}\sim\mathcal N(0,I_2)$ seeded. admission 영수증: $\operatorname{cond}(\Phi_3)\le10^6$, 초과 시 해당 fold는 `CUE_DEGENERATE`로 fail-closed (R1에서 사전 고정). cond 상한 근거 (R2, P2-c): 일반 위치 $\mathcal N(0,I_2)$ cue의 관측 cond p99는 $N=14$에서도 467이므로 $10^6$은 일반 위치와 구조적 퇴화(원주 배치 rank 붕괴, cond $\sim10^{16}$)를 분리하는 사전 고정 관대 상한이다. TR28의 $10^4$와 다른 값이며 의도적 신규 고정이다.
- seed·fold마다: 생성기 차수 $d^\*\in D$, 잡음 $\eta\in H$. 계수 행렬 $C^\*$ 성분은 $\overset{iid}\sim\mathcal U(-1,1)$ seeded (R2, P1-A — 검산 시뮬레이션과 동일 스케일. $\tau_{\rm class}=8\eta$와 §4.3 상대오차 게이트의 유효성은 이 $O(1)$ 스케일에 조건부이며, 스케일 변경은 계약 개정 사항이다). 훈련 행 $N=24$ (R1): $y_i=\phi_{d^\*}(z_i)^\top C^\*+\eta\,\varepsilon_i$, $\varepsilon_i\overset{iid}\sim\mathcal N(0,I_6)$, query와 독립. held-out query $z_q$는 훈련에서 제외.
- 공리 (R2, P2-b): 학습 연산자는 fold의 참 잡음 수준 $\eta$를 부가 정보로 공급받는다. $\eta$ 추정은 본 run의 주장 범위 밖이다.
- fold 구성: seed당 주 fold 9개($D\times H$ 전조합) + 선언-밖 witness fold 1개(차수 4 생성기, $\eta=10^{-3}$).

### 4.2 학습 연산자 (동결)

$\phi_d(z)$ = 총차수 $\le d$ 단항 특징 (차수 1: 3, 2: 6, 3: 10계수/행).

$$
C_d=\Phi_d^{+}Y,\qquad
s'_d=\frac1N\sum_{i=1}^N\bigl\lVert y_i-\phi_d(z_i)^\top C_d^{(-i)}\bigr\rVert_2\sqrt{1-h_{ii}}
\ \text{(studentized PRESS, R1)},
$$

$$
\hat d=\min\{d\in D:\ s'_d\le(1+\rho)\min_{d'}s'_{d'}+10^{-8}\},\qquad\rho=0.5\ \text{(floor는 R1)},
$$

$$
\hat y_q=\phi_{\hat d}(z_q)^\top C_{\hat d}.
$$

class 기권 게이트: $\min_d s'_d>\tau_{\rm class}=\max(10^{-8},\,8\eta)$이면 예측을
내지 않고 `CLASS_EXTERNAL_ABSTAIN`으로 fail-closed.

### 4.3 endpoint (주 = 직접 벡터 예측) — 형식 지위: [예측] (입력·범위·비교 기준 사전 고정, R2에서 태그 명시)

clean 진리 $y_q^{\rm clean}=\phi_{d^\*}(z_q)^\top C^\*$ 대비 상대오차
$e=\lVert\hat y_q-y_q^{\rm clean}\rVert/\max(\lVert y_q^{\rm clean}\rVert,10^{-12})$.

| 게이트 | $\eta=0$ | $\eta=10^{-3}$ | $\eta=10^{-2}$ |
|---|---:|---:|---:|
| 주 예측 $e\le$ | $10^{-10}$ | $2\times10^{-2}$ | $2\times10^{-1}$ |
| 차수 식별 $\hat d=d^\*$ | 필수 | 필수 | 필수 |

witness fold: `CLASS_EXTERNAL_ABSTAIN` 필수 (예측 발행 시 fold 실패).

### 4.4 독립 후보 스트림 (부 endpoint)

후보 은행 $K=8$: 환경 회전 궤도의 타 셀 clean content들 + seeded 무작위
distractor. **held-out 진리 중심화 금지, 모델 예측·기준선 유도 금지.**

구성 규격 (R2, P1-B):

- 은행 원소 구분 (carve-out 자구 정리): 진리 원소 1개 = 생성기가 query 셀에서
  산출한 $y_q^{\rm clean}$ (모델 비경유). 나머지 $K-1$개 = (seed, 생성기,
  비query 셀)만의 함수 — 타 셀 clean content와 seeded distractor.
- matched-norm 기준: distractor의 norm은 **비query 셀 content들의 평균 norm**에
  맞춘다. 진리 norm을 기준으로 쓰면 §4.4 자기 위반이다.
- 순서 영수증 (필수 산출물): 은행 $\{c_k\}$의 생성·직렬화 SHA-256 해시를
  $\hat y_q$ 계산 **이전에** 기록한다. 영수증에는 은행 해시, 생성 시각 순서
  카운터, $\hat y_q$ 계산 카운터가 포함되어 선행성이 기계 검증 가능해야 한다.
  영수증 부재 또는 순서 위반은 해당 fold `BANK_RECEIPT_FAIL`로 fail-closed.

선택 $=\arg\min_k\lVert\hat y_q-c_k\rVert$, 성공 = 진리 셀 선택.

### 4.5 대조군 (각각 전 fold 필수 결과)

| 대조군 | 필수 결과 |
|---|---|
| cue/content association shuffle | endpoint 전 기각 `0/전 fold` 성공 |
| wrong-cue | 성공 `0` |
| 강제 $d=1$ (affine) on $d^\*\in\{2,3\}$ | 주 예측 게이트 실패 `전 fold` |
| selection ablation ($\rho=0$) | 진단 기록만, 게이트 아님 |

### 4.6 seed 계획

- calibration `117001` — apparatus 결함 수리(R-loop)는 development 개봉 전에만, revision 기록 필수.
- development `117101..117116` (16 seeds × 10 folds).
- confirmation `117201..117232` — **봉인**. 본 run에서 열지 않는다.

### 4.7 STOP 조건 (원장 승계, 동결)

1. 후보 생성이 held-out 진리 또는 어떤 모델 예측을 읽음.
2. decoy가 예측·기준선에서 유도됨.
3. calibration 통과 후 차수·잡음·split·대조군·threshold 변경.
4. endpoint·대조군 약화.
5. development 행 실패 시: 정직하게 STOP 기록 (재조정 재시도 금지).

## 5. 교정 레지스터 (`.codex/harnesses/empirical_calibration_loop.md` 적용)

- 교정 가능 (calibration 단계, revision 기록下): apparatus 구현 결함, §4.3 상수의 계산 착오 정정.
- 교정 불가 (development 개봉 후): §4의 모든 수치·seed·게이트. 불일치 발생 시 D→I→P→C→B→T 분류 루프를 따르고 T는 STOP으로 기록.

## 5.1 Revision 1 (development 개봉 전 설계 정정)

- 사유: 11-math P0-1 — R0의 $\tau_{\rm class}=8\eta$ + mean-L2 LOO 결합이 $N=14,p=10$ 레버리지 꼬리로 $d^\*=3,\eta>0$ fold의 79–80%를 false abstain (반례 `artifacts/math-verify/counterexample.json`). §4.3 "전 fold 필수"와 구조적 모순.
- 채택 수리 (12-routes R-B+R-C+R-D): 기권 통계를 studentized PRESS $s'_d$로 교체, 파시모니에 floor $10^{-8}$ 추가(P1-2 동시 해소), $N\,14\to24$. cue 표본과 condition admission을 명시(P1-1 해소).
- 동결 유지: §4.3 게이트 상수, $\tau_{\rm class}$ 형태 $8\eta$, $\rho=0.5$, seed 계획, 대조군, witness, STOP 조건.
- 확인 근거: R-D 확인 시뮬레이션 (11-math "R-D 확인 시뮬레이션" 절, `artifacts/math-verify/rd_confirm_6000_log.json`) — 전 9셀 실패 0/6000, witness 여유 최소 41.9배, shuffle 기각 3000/3000, 144 fold 전량 통과 보수 하한 0.931. $\eta=10^{-2}$ 게이트 여유 2.24배가 최박임을 기록(P1-3 잔존, 게이트는 완화하지 않음).
- look-elsewhere: 본 수리는 P0 관측 후 구성되었으므로 레인 수치는 설계 근거이지 통과 증거가 아니다. calibration `117001`이 신선한 통과를 제공해야 하며, calibration 통과 후에는 §4.7-3에 따라 어떤 수치도 재변경 불가.
- 채택 시점 (R2, P2-d): R1은 calibration을 포함한 **seed 일체 미개봉** 상태에서 채택되었다. run 내 실행 산출물은 존재하지 않았다.

## 5.2 Revision 2 (감사 Gate: REVISE 대응, seed 일체 미개봉 상태)

- revise 기록: `revise contract` 1/2. 대상: 20-audit P1-A, P1-B + P2 4건 동승.
- P1-A: §4.1에 $C^\*\sim\mathcal U(-1,1)$ seeded와 스케일 조건부성 명시.
- P1-B: §4.4에 은행 구성 규격(carve-out·matched-norm 기준·선행 순서 영수증 `BANK_RECEIPT_FAIL`) 명시. impl은 이 영수증을 의무 산출물로 구현한다.
- P2-a/b/c/d: §4.3 [예측] 태그, η 공급 공리, cond $10^6$ 근거, R1 채택 시점 각 1문.
- 동결 유지: §4.2 연산자(R1 그대로), §4.3 게이트 상수, seed 계획, 대조군, witness, STOP 조건.

## 6. 주장 상한

합격해도 주장은 "선언된 유한 다항 족과 잡음 수준에서, 독립 후보 스트림 하
합성 모델-클래스 식별과 직접 내용 예측"까지다. 자연 후보 생성, 임의 상호작용
발견, 의미 인자 복구, 생물학적 기억·주의·라우팅, curvature-as-memory, 피질
접힘, 질병 기전·치료, 물리 에너지, AGI를 확립하지 않는다. 선언 밖 완성
$y_q+\delta$ 비식별 no-go는 유지된다.
