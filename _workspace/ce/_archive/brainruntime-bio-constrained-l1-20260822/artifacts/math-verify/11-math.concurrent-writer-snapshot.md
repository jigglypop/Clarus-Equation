# 11-math — BA-V3-1 독립 수학 감사

Status: COMPLETE

## 결론

현재 계약은 생물학적 허용 primitive로 만든 합성 동역학이지만, 출처가
고정된 생물 기준식, CE 추가항과 실제 측정모형이 분리되어 있지 않다. 실제
뇌 자료 L1 적합, CE 효과의 식별 또는 wake/sleep 주기 정상상태의 일반적
존재를 아직 주장할 수 없다.

## P0-1 — CE 추가항과 측정모형 미정의

계약에서 weight update로 읽을 수 있는 식은 다음과 같다.

$$
\Delta w_{ij}^{\mathrm{wake}}=\eta g(t)e_{ij}(t),
\qquad
w_{ij}^{\mathrm{sleep}}=w_{ij}^{\mathrm{wake}}
\left(1-\frac{\lambda_0}
{1+(w_{ij}^{\mathrm{wake}}/\kappa)^2}\right).
$$

그러나 별도의 $F_{\mathrm{bio}}$, $\Delta F_{\mathrm{CE}}$, 영점 귀무모형
$\Delta F_{\mathrm{CE}}=0$과 데이터셋별 관측식 $\mathcal H_d$가 없다.
따라서 개선이 나와도 생물 기준식, CE 항, 측정·전처리 중 어디에 기인하는지
식별할 수 없다.

10-sources의 재감사에 따르면 R1′, R3b′, R4′, R5′, R6′의 현재 scoring
정의도 원 관측량과 같지 않다. 분자·분모, time zero, animal/cell/spine
nesting, censoring, acquisition, preprocessing와 uncertainty가 고정되기
전에는 수치 통과·실패 어느 것도 실제 뇌 관측 판정이 아니다.

재개 조건은 1차 자료에서 각 관측 receipt를 잠그고, 분석 전에 데이터셋별
$\mathcal H_d$와 CE delta를 적는 것이다.

## P0-2 — $\lambda(w)>0$는 주기 정상상태 존재조건이 아니다

지속적으로 potentiation되는 한 synapse의 허용된 부분모형을 택한다. 하루
wake increment를 $a>0$라 하면 하루 map은

$$
T(w)=(w+a)\left[
1-\frac{\lambda_0}{1+((w+a)/\kappa)^2}
\right].
$$

큰 $w$에서

$$
T(w)-w
=a-\frac{\lambda_0\kappa^2}{w+a}+O(w^{-3}).
$$

따라서 충분히 큰 $w$에서는 $T(w)-w>a/2>0$이고 $w_n$은 발산할 수 있다.
$\lambda(w)>0$만으로 Lyapunov drift, tightness, invariant distribution 또는
cyclostationary state가 보장되지 않는다.

이 반례를 제거하는 구조적으로 다른 최소 조건은 다음 셋이다.

- high-$w$ wake increment가 $o(1/w)$로 포화한다.
- sleep loss fraction이 large-$w$에서 양의 하한을 갖는다.
- 명시적 전역 homeostatic feedback이 total strength에 음의 drift를 준다.

수치 witness는
`artifacts/math-verify/periodic_drift_counterexample.py`에 둔다.
고정 witness $a=0.1$, $\lambda_0=0.2$, $\kappa=1$, $w_0=100$은 1,000회 뒤
$w=198.6081064$였고 최소 일증분 $0.0980022>a/2=0.05$로 통과했다.

## P0-3 — 시간과 단위 연결 미완성

$w$, $\kappa$, $w_{\min}$, $w_0$는 같은 synaptic-strength 단위이고
$\lambda_0$는 sleep update당 무차원이어야 한다. $\rho_\infty$는
시간$^{-1}$, $T_m$, $\tau_{\mathrm{el}}$, $\tau_\pm$는 시간,
$r^*$는 rate 단위다.

현재 계약은 STDP의 17 ms·34 ms, simulation tick, 16:8 cycle, day/month,
발달·성체 관찰창 사이의 변환을 고정하지 않는다. 따라서 R1′·R3′·R4′·R5′를
하나의 목적함수에서 비교하는 timebase가 닫히지 않았다.

## 질량수지

총 strength를 $W=\sum_iw_i$, night loss를
$L=\sum_i\lambda(w_i)w_i$라 두면

$$
R3a'=\frac{L}{W}.
$$

top set의 strength share를 $s=W_{\mathrm{top}}/W$, top fractional loss를
$\ell_{\mathrm{top}}$이라 하면 bottom loss는

$$
\ell_{\mathrm{bot}}=
\frac{R3a'-s\ell_{\mathrm{top}}}{1-s}.
$$

$R3a'=0.18$과 $\ell_{\mathrm{top}}\le0.05$ 자체는 모순이 아니지만,
top의 strength share와 subgroup uncertainty가 없으면 bottom 부담을 계산할
수 없다. 주기 평균은

$$
\mathbb E[\Delta W_{\mathrm{wake}}+\Delta W_{\mathrm{birth}}
-\Delta W_{\mathrm{death}}-L]=0
$$

을 요구한다. 현재 계약은 이 네 항을 독립 관측량으로 닫지 않는다.
산술 identity witness는 `artifacts/math-verify/mass_balance_witness.py`에 둔다.
top strength share $0.2,0.5,0.8$과 top loss $0,0.025,0.05$의 아홉 조합에서
질량수지 재구성 오차는 $10^{-12}$ 미만이었다. 이는 identity만 검증하며
어느 share가 실제자료에 맞는지는 말하지 않는다.

## R1′과 R2′

R2′를 cohort survival로 읽으면 선행 BA-TS1의 단조 생존 반례가 유지된다.
현재 계약은 이를 전체 spine instance 중 수명 분류인 N-series로 바꾸어 그
직접 모순을 피했다. 2026-08-23 source 재감사는 N-series를 지지하지만 exact
cohort weighting, interval과 censoring metadata가 필요하다. 이것이 고정되기
전에는 R1′과 R2′의 동시 가능성도 수치 채점하지 않는다.

## 자유도와 식별성

`8 < 10`은 식별성 증명이 아니다. 여러 gate가 부등식·질적 조건이고,
$\eta g(t)$는 곱으로만 들어가며, $\lambda_0$과 $\kappa$는 관측 weight range가
좁으면 강하게 상관된다. $\rho_\infty$, $\kappa_m$, $T_m$도 단일 발달창에서
분리되지 않을 수 있다. $w_0$, $\tau_e$, homeostatic gain과 update locus는
gate를 움직이면 이름과 무관하게 유효 freedom이다.

source-locked observable vector $q(\theta)$를 얻은 뒤

$$
J_{ab}=\frac{\partial q_a}{\partial\log\theta_b}
$$

의 singular values, rank, condition number와 profile likelihood를 계산해야
한다. 이 전에는 식별 가능성을 주장하지 않는다.

## P1

- STDP 평균 drift는 pre/post correlation, rate와 trace normalization 없이
  $A_+\tau_+-A_-\tau_-$만으로 정해지지 않는다.
- 항상성 scaling의 gain, update time과 적용 locus가 없어 R4′와 firing-rate
  E2 기여를 분리할 수 없다.
- R1′의 removal-only·all-spine·birth/death 분자 선택은 source lock 전 fit
  target이 아니다.
- 기존 수치탐색은 대리모형 장치 진단이며 실제자료 적합의 근거가 아니다.

## P2

- tick-to-ms/day/month 변환표가 필요하다.
- $w$ gauge와 $\kappa,w_0,w_{\min}$의 동시 scale 변환을 적어야 한다.
- R4′를 net ratio와 log-growth 중 무엇으로 읽는지 convention이 필요하다.

## 판정

실제자료 scoring과 구현은 BLOCKED다. RT-0에서 측정모형을 고정하고 계약을
좁힌 뒤 새로운 수학 감사를 받아야 한다. 이 BLOCKED는 식 family 전체의
불가능성이 아니라 현재 판본의 관측·정상상태 계약 결손이다.

## 6. (e) 설계 상수 민감도 (TS1 P1 승계 검사)

최선점 기준 1-요소 스윕 (`sens.py`/`sens.json`). 대역: R1'[0.02,0.08],
R2'dev[0.25,0.45], R2'ad[0.60,0.85], R3a'[0.10,0.25], R5'[1.3,1.8], R6'≥1.3.

| 상수 | 판정 | 수치 (게이트 이동) |
|---|---|---|
| $w_{\min}\equiv1$ | **진짜 게이지** | 1차 동차, 상대편차 1.7e-15 |
| $w_0=1.2$ | **숨은 적합 파라미터 (P1)** | 1.05→3.0: R2'-ad 0.640→**0.895**, R2'-dev 0.501→0.754, R6' 1.276→1.090, R1' 0.090→0.120 |
| $\tau_e=2$ | **숨은 적합 파라미터 (P1, 최강)** | 1→5: R2'-ad **0.403→0.931** (대역 전폭 초과), R2'-dev 0.311→0.879, R3a' 0.070→0.156, R6' 2.01→1.07. $\tau_e{=}1$에서 통과 게이트 7 (기준 5) |
| $K$ 상수·$\tau_{\rm el}$ (eligibility 형상) | **숨은 적합 파라미터 (P1, 신규)** | 형상 0.5→20: R3a' **0.035→0.316**, R3b' 0.010→0.232, R2'-ad 0.392→0.939, R1' 0.113→0.024. $\eta$는 평균만 흡수하고 분산은 못 흡수 |
| $K$ 상수 부호 | **결정적·적합 불가 (P1)** | $A_+\tau_+=A_-\tau_-$ (17·1 = 34·0.5) ⇒ 비상관 발화의 순 표류 0. 정리 2가 요구하는 $\gamma\ge0.22$는 전부 상관 성분×$g(t)$가 대야 한다 |
| 항상성 이득 $\beta$ (**미선언**) | **숨은 파라미터 (P1)** | 0.2→1.0: R4' 2.62→0.92 %/100일, R1' 0.057→0.088, R2'-dev 0.599→0.503. $\beta{=}0$은 정리 1(iii) 발산 |
| 16:8 | **유효하나 $\eta$와 완전 축퇴 (P2)** | 0.5→1.5배: R3a' 0.012→0.115, R2'-ad 0.160→0.804, R6' 9.19→1.18. TS1의 "무효 상수"(P2-1)는 해소, 대신 축퇴로 이동 |
| $N_E$ | **$S^*$와 축퇴** | 48→90: R1' 0.071→0.125, R3a' 0.083→0.103 (뉴런당 $S^*$ 고정 시 in-degree 변화 → $\bar w$ 변화) |
| 판정 창 (성체) | **비활성 아님 (P2)** | 400–600 vs 500–700: R1' 0.109 vs 0.088 (24%), R5' 1.713 vs 1.813 |
| 판정 창 (발달) | (N-a)에서 무해 | 10–60/20–70/30–80: R2'-dev 0.514/0.503/0.510. **단 (N-b)면 지배적** (§2.2) |

## 7. (f) 목적함수 전행 정의 — 판정: [미완성] (P1)

§4-1은 두 분기(비율 → $\log((x+\epsilon)/(x^*+\epsilon))^2$, 부등식 → hinge)를
선언하지만 **§5의 어느 행이 어느 분기인지 배정하지 않는다**. 결정적 사례
(`objective.py`, 근사해 지점 R1'=0.045, R2'dev=0.33, R2'ad=0.71, R3a'=0.17,
R3b'=0.02, R4'=0.01, R5'=1.45, R6'=1.25):

| 읽기 | 총 목적함수 | R4' 항 | R4' 지분 |
|---|---:|---:|---:|
| R4'·R6'를 부등식 행으로 | 0.0250 | 0.0 | 0% |
| R4'·R6'를 비율 행으로 (목표 0) | **84.90** | **84.83** | **99.91%** |

즉 분기 배정은 표기 문제가 아니라 **적합 결과를 결정한다**
($\log((x+\epsilon)/(0+\epsilon))^2$는 $x=10^{-4}\to0.2$에서 21.3→149.0).

추가 공백:
1. R2'-adult 행의 "R2'-dev보다 크고 **단조 성숙**", R5' 행의 "이후 **단조 감소**" —
   대역·목표·조작정의·목적함수 항이 모두 없다 (TS1 P2-2 재발).
2. R3b'의 강부등식 $>0$은 hinge로 강제되지 않고 §3.2가 구조적으로 보장한다
   (판별력 0). $(0,0.05]$ 내부에서 목적함수는 평탄하다.
3. §5 계수 "유효 조건 10 (L1 8 + L2 2)"은 L2가 §4-1의 적합 대상이 아니므로
   적합 문제의 조건 수와 다르다 (§5 판정).

## 8. (g) L2 게이트의 대리 모형 판정 가능성

| 게이트 | 대리 모형 판정 | 근거 |
|---|---|---|
| **E1** ($|\gamma_1(\log w)|<0.5$ 및 $\gamma_1(w)>1$) | **판정 가능** | LHS(성체 $N>200$) 573점 중 29.0%가 통과. $\gamma_1(\log w)$ 사분위 [−0.657, +0.007], 중앙 −0.242 — 가법 이득의 하단 절단 때문에 좌편향 경향. L1과 교차 구속 $\sigma_{\log w}\in(0.108,1.812)$ (정리 2) |
| **E2** (발화율 왜도 >0.5) | **실 모형 전용으로 분류** | 대리 모형의 발화율 대리지표(뉴런별 입력 질량)는 뉴런별 항상성이 직접 고정한다: CV 중앙값 **0.0069**, p95 0.136. 왜도>0.5이면서 CV>0.05인 점은 2.6%뿐. S3(억제 경쟁·스파이크 부재)로 발화율의 실제 이질성 원천이 없다 |

**주의(P1)**: E2의 성패는 §3.4가 미선언한 항상성의 **적용 locus와 이득**이
결정한다. 뉴런별·완전 보정($\beta{=}1$)이면 발화율 분포가 구조적으로 좁아져
E2는 통과 불가에 가깝고, 전역 또는 $\beta<1$이면 잔여 이질성이 남는다.
이는 "부과하지 않은 창발"이라는 E2의 취지와 직접 충돌한다.

## 9. 숨은 공리·자유도

1. **[공리] 가법·$w$-무관 각성 이득**: §3.1에 가중치 의존이 없다는 사실이
   정리 2의 $\theta_G=0.2$를 만들고 가능 창 (II)를 연다. 실 모형에서 $w$–$\Delta$
   양의 상관이 생기면 $\theta_G$가 커져 창이 좁아지고, $\theta_G\ge0.863$이면
   $f_{\rm top}\le1$과 양립 불가해 **R3a'×R3b'가 서로소가 된다**. 측정 가능한
   반증 조건이다.
2. **[공리] 항상성의 수축성**: 정리 1(iv)–(v). $c<1$이 없으면 $\lambda>0$에도
   발산한다. §3.4에 이득·locus·수축 보장이 선언돼 있지 않다.
3. **[자유도] 설계 상수 4개**: $w_0$, $\tau_e$, eligibility 형상($K,\tau_{\rm el}$),
   $\beta$. §6 표대로 게이트를 대역 전폭 이상 움직인다.
4. **[측정 자유도] R2'(N) 하위 정의**: 같은 동역학에서 0.371 vs 0.730.
5. **[측정 자유도] R1' 분자 구성**: 같은 동역학에서 0.040 / 0.055 / 0.080 / 0.110.
6. **판별력 축소 게이트 2개**: R3b' 하한(구조적 항등), R4'(무작위 상자 90.7% 통과).

## 10. 경계·반례·교차 예측

**반례 1 (정리 1(iii), 모형 무관).** $c\equiv1$, $\bar\Delta>\lambda_0\kappa/2$이면
$\lambda(w)>0$ 전역에서도 발산한다. 수치: $\beta{=}0$, $\bar\Delta=3.77$,
$\lambda_0\kappa/2=2.21$에서 $\bar w=2166.9$, R4'=19.0%/100일, R3a'=4.4e-6.
$\kappa\in\{0.5,4.47,200\}$ 전부 동일. **무너뜨리는 범위**: "§3.2의 $\lambda>0$
강제가 TS1 정리 1을 해소한다"는 승계 처방의 *기전 귀속*만 무너지고, 족의
정상상태 존재 자체는 §3.4로 살아남는다 (정리 1(iv)).

**반례 2 (§2.4, 게이트 집합).** $R2'_{\rm adult}>0.769$이면 $R6'\le1/0.769<1.3$ —
§5의 두 대역은 이 구간에서 서로소다. 반례 값: $R2'_{\rm ad}=0.85$(대역 상한)에서
$R6'\le1.176<1.3$. **무너뜨리는 범위**: "§5 8행이 전 대역에서 동시 만족 가능한
사전 고정 대역이다". 목표 근방 회랑 [1.300,1.370]은 남으므로 전면 반증은 아니다.

**반례 3 (§2.3).** $R1'_A=0.04$(목표)이고 $q_p\le0.85$면
$R1'_{B\pm}=2R1'_A/q_p\ge0.094>0.08$. 4개 읽기 동시 대역화는 $R1'_A\le0.0347$을
강제해 **목표 0.04와 양립하지 않는다**.

**교차 예측 (이 조합이 §5를 만족한다면 반드시 따라오는 것)**

1. 각성 중 총 시냅스 강도의 순 증가가 하루 **≥22%** ($\gamma\ge R3a'/(1-R3a')$),
   그리고 항상성이 매일 $c=1/((1+\gamma)(1-R3a'))<1$로 하향 조정. de Vivo의
   야간 −18%와 짝이 되는 정량 예측이며 각성 측 측정으로 반증 가능.
2. 상위 20%가 받는 각성 이득 분율 $\theta_G$와 질량 분율 $f_{\rm top}$이
   $1.159<f_{\rm top}/\theta_G<4.171$ (동치: 상위 20%의 *분율* 이득이 집단
   평균의 24%–86%). 이 비는 시냅스 크기별 가소성 측정으로 직접 시험된다.
3. 정상 가중치 분포의 폭 $\sigma_{\log w}\in(0.108,1.812)$ — L1이 L2 E1을
   구속한다. 관측 spine ASI의 $\sigma\approx0.7$–1.0은 창 내부.
4. $\lambda_0>0.219$ (관측 무릎 아래 시냅스의 야간 감쇠 상한).
5. (N-a) 해석에서 학습일 신생 spine의 8일 생존율 $\ge0.949$ (R6'≥1.3 회랑).
6. 성체 persistent 풀의 수명 상수 $\tau_p\in[375,1500]$일과
   신생 persistent화 확률의 성숙 증가비 $q_p^{\rm ad}/q_p^{\rm dev}\in[1.33,3.40]$.

## 11. P0 / P1 / P2

**P0: 없음.** §5 게이트 대역 사이에 완전 서로소 쌍이 없고 (LHS 1200점,
쌍별 교집합 최소 3), TS1의 세 P0 중 P0-1은 RT-0의 (N) 재정의로 해석적으로
해소되며(§2.1), P0-2는 §3.4 항상성으로 해소된다(§3(iv)). P0-3(성숙 방향
부호)은 항상성이 $c_{\rm dev}/c_{\rm ad}=(1+\gamma)/(1+1.5\gamma)<1$ (R5'=1.5에서)
로 올바른 부호의 통로를 제공하므로 구조적 부호 반전이 사라졌다.

**P1 (7건, 최소 보조정리 지정)**

1. **§2 기전 귀속 오류** (§3, 반례 1). 최소 보조정리: 승계 처방 문면을
   "$\lambda>0$ + **항상성 수축 $c<1$**"으로 고쳐 §3.4에 $\beta>0$와 수축 보장을
   명시. 미조치 시 impl이 항상성을 약하게 잡으면 TS1 no-go가 그대로 재발한다.
2. **R2'(N) 하위 정의 미고정** (§2.2). 최소 보조정리: (N-a) 출생 코호트 또는
   (N-b) 유병 중 하나를 계약 수준에서 고정하고, (N-b)면 창 길이를 게이트 정의의
   일부로 동결. 이 선택이 R6' 천장(§2.4)의 성패도 결정한다.
3. **R1' 분자 4-읽기 비양립** (§2.3). 최소 보조정리: 게이트 분자를 하나로
   고정하고 병행 보고분에는 대역을 부과하지 않음을 문면에 명시.
4. **R6' × R2'-adult 부분 서로소** (§2.4). 최소 보조정리: R2'-adult 상한을
   R6' 요구와 함께 재검토하거나, R6'의 baseline을 R2'와 다른 정의로 분리 선언.
   본 레인은 대역을 수정하지 않는다 (§5 수정 금지).
5. **숨은 설계 상수 4개** ($w_0,\tau_e$, eligibility 형상, $\beta$; §6).
   최소 보조정리: 자유 파라미터로 승격하거나 관측 유래 값으로 외부 고정하고
   출처를 §3에 기록. 특히 $\tau_e$는 단독으로 R2'-adult를 0.403↔0.931로 옮긴다.
6. **§4-1 목적함수 행→분기 미배정 및 미정의 절** (§7). 최소 보조정리: 행별
   분기 표를 §4에 동결하고, R2'/R5'의 단조 절을 조작정의+항으로 쓰거나 게이트에서
   제거. 미조치 시 R4' 항이 목적함수의 99.9%를 차지할 수 있다.
7. **§5 조건 계수 과대·판별력 축소** (§5, §0 #15–#17). 최소 보조정리: "자유 8 <
   조건 10"을 "등식 5 + 활성 부등식 2 (+비적합 진단 2)"로 정정하고, 해집합이
   3차원 이상임을 §4-2 development 평가 설계에 반영.

추가 P1 (탐색 결과, 조치 불필요·기록용): **대리 모형 7,167 평가에서 8게이트
동시점 미발견** (§4). 반증이 아니며 impl 레인이 실 모형에서 다시 시험해야 한다.
가장 먼 미달은 R2'-dev(로그 0.154)와 R5'(0.076)다.

**P2 (5건, 목록만)**

1. 16:8은 $\eta$와 완전 축퇴 — 설계 상수 선언의 실효는 곱 $\eta\times$각성 틱수
   모듈로만 의미가 있다.
2. §6 성체 판정 창 위치가 R1'를 24% 움직인다.
3. §3.2의 지수 2는 무릎의 예리함을 고정한다 — 본 검산에서 구속력은 확인되지
   않았으나 $s_{\rm bot}/s_{\rm top}$ 요구비가 커지면 형상 자유도 부족이 될 수 있다.
4. R6' 게이트는 전역 스칼라 $g(t)$로 정의되므로 원 관측(Yang 2014의 가지 특이적
   형성)보다 약한 시험이다 — 가법 이득 + 곱셈 정규화는 약한 간선을 일반적으로
   유리하게 만든다.
5. R5'의 "이후 단조 감소"에 정량 임계가 없다 (본 검산은 피크→성체 구간의 증가일
   분율로 조작화, 최선점 0.41).

## 12. 재현

```
sh .claude/hooks/run.sh status _workspace/ce/brainruntime-bio-constrained-l1-20260822
# 이하 repo 루트, A = _workspace\ce\brainruntime-bio-constrained-l1-20260822\artifacts\math-verify
.claude\hooks\python.cmd python -B A\budget.py       # (a)(c) 닫힌 형태 전부 (~5s)
.claude\hooks\python.cmd python -B A\mixwindow.py    # R1'xR2'(N) 2-혼합 창
.claude\hooks\python.cmd python -B A\windowdep.py    # R2'(N-b) 창 길이 의존
.claude\hooks\python.cmd python -B A\smoke.py        # 대리 모형 연기시험 (~0.3s)
.claude\hooks\python.cmd python -B A\search.py 1200  # LHS 1200 (~240s)
.claude\hooks\python.cmd python -B A\refine.py       # 국소정련 840 (~190s)
.claude\hooks\python.cmd python -B A\search2.py 60 50  # 다중시작 ES 3060 (~12min)
.claude\hooks\python.cmd python -B A\search3.py      # 미세탐색 1448 (~6min)
.claude\hooks\python.cmd python -B A\fixr4.py        # R4' 단위 수정 + 게이트 통계 재집계
.claude\hooks\python.cmd python -B A\identity.py     # 정리 2 잔차 검증 + R6' 천장
.claude\hooks\python.cmd python -B A\mkbase.py       # 최선점 고정
.claude\hooks\python.cmd python -B A\sens.py         # (e) 설계 상수 스윕
.claude\hooks\python.cmd python -B A\gauge.py        # w_min 게이지 1차 동차 확인
.claude\hooks\python.cmd python -B A\jac.py          # (d) 야코비·잡음정규화 SVD
.claude\hooks\python.cmd python -B A\nogo.py         # 정리 1(iii) 반례 (beta=0)
.claude\hooks\python.cmd python -B A\het.py          # 폭 가설 시험 (기각)
.claude\hooks\python.cmd python -B A\l2stats.py      # (g) L2 판정 가능성
.claude\hooks\python.cmd python -B A\objective.py    # (f) 목적함수 분기 가격
```

산출: `surrogate.py`(대리 모형), `budget.json`, `mixwindow.json`,
`windowdep.json`, `lhs.csv`/`lhs_summary.json`, `refine.json`, `search2.json`,
`search3.json`, `gate_stats_corrected.json`, `identity.json`, `sens.json`,
`gauge.json`, `jac.json`, `nogo.json`, `het.json`, `l2stats.json`,
`objective.json`, `base_point.json`, 로그 `search2.log`/`search3.log`/`het.log`.
보조: `show.py`/`peek.py`/`peek2.py`/`peek3.py`/`diag.py`, 패치
`patch1..4.py`(재현 시 1회만 적용 — 이미 반영된 소스에 재적용하면 assert로 정지).

**정정 기록**: R4'는 `slope×100/mean`(100일 상대 표류, 분수)이므로 대역은
5%/100일 = **0.05**이며, 초기 스크립트가 5.0으로 두었다. `fixr4.py`가 상수를
고치고 저장된 `lhs.csv`에서 전 통계를 재집계했다 (§4.1은 수정본). 최선점들의
R4' 실측은 0.0047·0.0092로 수정 전후 통과 판정이 바뀌지 않는다.

난수: LHS seed 20260822, 정련 20260823/24/25/26, 시뮬 seed 119001 (calibration).
119002–119006은 잡음 재평가에만 사용. **119101+ (development)와 119201+
(confirmation)은 열지 않았다.**
