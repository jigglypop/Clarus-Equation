# 생물 라인 교정 검정의 수학 설계 — 부속 노트 (run: agi-clarus-field-20260812)

읽기+계산 전용 노트. 대상: docs/7_AGI/25 local temporal memory 게이트
(`reality_stone/python/reality_stone/clarus/local_memory.py`)와 2026-08-12 AML18(GFP) 반례.
검산 스크립트: `_workspace/ce/agi-clarus-field-20260812/artifacts/bio_gate_redesign_check.py`
(재현: `python .../bio_gate_redesign_check.py`, 입력은 `artifacts/agi/local_memory_aml{32,18}_h{1,6}_confirmatory.json`).

---

## 1. 문제의 형식화 — 25번 게이트가 실제로 측정한 것

**[공리 A0 — 관측 모형]** 단위 $i$의 관측열:

$$y_i = P\big[(k * s_i) + \varepsilon_i\big] \ \text{(AML32, GCaMP)}, \qquad y_i = P\big[b_i + \varepsilon_i\big] \ \text{(AML18, GFP)}$$

여기서 $s_i$는 신경 활동, $k$는 indicator kernel(GCaMP6s, 감쇠 $\tau_k \approx 1\text{–}2.5\,$s, 뉴런별 발현량 의존),
$\varepsilon_i$는 광학·운동 잡음, $b_i$는 활동 비의존 형광, $P$는 전처리(Ratio2/ICA 배경보정, I_smooth 저역통과,
광표백 보정) — **양 strain에 동일한 코드 경로** (공리 A1, 코드로 확인됨).

**[보조정리 L1 — 게이트 통계량의 Gaussian 닫힌 형태와 판정 등가]**
평균 0 정상 Gaussian 과정 $y$, 정규화 자기상관 $\rho_\tau$에 대해 선형 예측 부분은

$$R^2_C = \rho_h^2, \qquad R^2_L = c^\top R^{-1} c, \quad c=(\rho_h,\rho_{h+1},\rho_{h+2})^\top,\ R=\mathrm{Toeplitz}(1,\rho_1,\rho_2)$$

$$\Delta = c^\top R^{-1} c - \rho_h^2 \ \ge 0, \qquad \Delta = 0 \iff \rho_{h+j} = \rho_h\,\rho_j\ (j=1,2)\ \text{(AR(1)-Markov 관계)}$$

증명: $R^2_L$은 $(y_t,y_{t-1},y_{t-2})$ 위 직교사영 노름, $R^2_C$는 그 부분사영; 등호 조건은
$\mathrm{Cov}(y_{t+h}, y_{t-j} \mid y_t)=0$과 동치이고 Gaussian에서 위 곱셈 관계와 동치. (표준 사영 논증 — 정리)

**따름 결과:** 25번 게이트의 PASS는 "관측 과정 $y$가 lag-2까지 AR(1)-Markov가 아니다"와 (선형·Gaussian 근사에서) 동치.
$k$, $P$, $\varepsilon$ 어느 것이든 저역통과/스무딩이면 $y$는 비-Markov가 되므로, **$s\equiv \mathrm{const}$여도 PASS한다.**
circular-shift null은 정렬만 파괴하고 이 성질을 검정하지 못한다. 즉 게이트는
$P\circ k$ 합성의 자기상관을 측정했고, $s$의 기억은 식별하지 않았다. AML18 11/11 PASS는 이 분해의 직접 확인.
[판정: 25의 명시적 주장문("aligned past adds held-out prediction")은 관측 과정에 대해 참으로 유지;
"신경 기억" 해석은 근거 상실 — P1, 해석 공백.]

**[보조정리 L2 — 척도 불변과 confound의 통로]** L1의 $\Delta$는 정규화 $\rho$만의 함수이므로 진폭·SNR *척도*에는 불변.
SNR은 오직 합성 자기상관의 **모양** $\rho_y = \frac{\mathrm{SNR}\cdot\rho_{k*s} + \rho_{P\varepsilon}}{1+\mathrm{SNR}}$을 통해서만 들어온다.

**[보조정리 L3 — headroom 항등식]** 정의상 $\Delta = (1-R^2_C)\,\Delta'$, $\Delta' := \frac{R^2_L-R^2_C}{1-R^2_C} \in [0,1]$
(잔차분산 중 lag가 설명하는 비율). 원 $\Delta$의 strain 대조는 $R^2_C$ 차이(headroom)에 오염된다.

**[비식별성 관측]** 단일 파이프라인 내부에서 $y$의 선형 자기상관을 $k*s$ 기원과 $P\varepsilon$ 기원으로 분해하는 것은
비식별 (두 유색 성분의 합은 2차 통계로 분리 불가). 따라서 어떤 within-strain 통계량도 단독으로는 답을 못 준다 —
strain 대조(GFP가 잡음 경로의 $\rho$를 제공)가 유일한 손잡이다.

## 2. 검정 후보의 유도와 식별 가능성

### (a) GFP-matched null (strain 대조)

**매칭 조건 (L1·L2에서 유도):** 유효 대조에 필요한 것은
(M1) 잡음 경로의 정규화 자기상관 모양 $\rho_1,\rho_2,\rho_h,\rho_{h+1},\rho_{h+2}$ (분산·SNR 자체는 불필요 — L2),
(M2) 표본 수·gap 패턴·frame rate (추정량 분산과 ridge 수축), (M3) 주변분포 (비선형 $g$ 항).

**편향 방향 (유도 + 경험적 확인):** L3에 의해 원 $\Delta \le 1-R^2_C$. h=6에서 GCaMP는 kernel 스무딩 탓에
$R^2_C$가 높고(0.17–0.69 vs GFP −0.09–0.36) headroom이 작다 → 원 $\Delta$ 대조는
**"GCaMP < GFP" 방향으로 구조적으로 편향.** 실측(§5)이 이를 확인.
교정: 통계량을 $\Delta'$로. 2차 교정: L1 닫힌 형태에 recording별 $\hat\rho$를 대입한
$\Delta'_{\mathrm{lin}}(\hat\rho)$를 공변량으로 빼서 $\eta_r = \Delta'_r - \Delta'_{\mathrm{lin}}(\hat\rho_r)$ —
스펙트럼 모양 불일치를 선형-Gaussian 차수까지 제거.

**식별 가능성:** 공리 A1(동일 전처리) + A2(잡음 경로 2차 구조가 strain 간 모양까지 동일하거나 $\eta$ 조정으로 흡수됨)
하에서 식별. A2는 $\hat\rho$ 모양 비교로 부분 검증 가능. **판정: 식별 가능(조건부), 구현 최단.**

### (b) 차분/whitening 검정

AML18에서 잡음 정형 필터 $\hat W$ (per-unit AR($p$) 또는 스펙트럼 인수분해)를 추정, 양쪽에 적용 후 $\Delta'$ 재검정.
**over-whitening 경계 (유도):** $\hat W$는 잡음과 모양이 같은 모든 선형 구조를 지운다. 신경 기억이 살아남는 것은
$|K(\omega)|^2 S_s(\omega)$가 $S_{P\varepsilon}(\omega)$와 모양이 **다른** 대역뿐:

$$\text{정보 소실 조건: } |K(\omega)|^2 S_s(\omega) \propto S_{P\varepsilon}(\omega) \text{ 인 대역에서 } s\text{-기억 소거}$$

GCaMP kernel과 I_smooth가 모두 저역통과이고 corner가 비슷하므로(≈6 vol/s 표본화) 겹침이 크다 → 검정력 낮음.
또 $\hat W$ 추정오차 $\delta W$는 H0에서도 잔여 유색성을 만들어 1종 오류를 부풀린다 → null은 반드시
**whitening된 GFP에서 재유도**해야 함. 그러면 (b)는 (a)의 분산축소 변형으로 환원된다.
**판정: 독립 검정 아님; (a)의 확인용 2차.**

### (c) deconvolution 검정

$\hat s = \hat k^{-1} y$ (정칙화 필요). **오차 전파 (유도):** 지수 kernel $K_\tau(\omega)=(1+i\omega\tau)^{-1}$,
$\hat\tau = \tau(1+\delta)$이면 잔여 필터 $K_\tau/K_{\hat\tau}$가 $\omega \sim 1/\tau$에서 $O(\delta)$의 유색성을 남기고,
$\Delta$의 $\rho$-기울기가 일반적으로 0이 아니므로 가짜 $\Delta = O(\delta)$. 문헌상 뉴런별 $\tau$ 산포 $\delta \gtrsim 0.3$
→ 가짜 효과가 h=1 관측 효과(~0.02)와 동급. 게다가 역문제 정칙화(Wiener/Tikhonov)는 검정 대상인 스무딩을 재도입(순환).
**판정: 식별 불가능. 탈락.**

### (d) 스펙트럼 대역 대조

신경 기억이 있으면 조건부 예측 이득의 초과분이 drift 대역 위, $\omega \lesssim 1/\tau_k$ 아래 대역에 집중해야 함.
대역별 (a)와 동일한 대조. **검정력:** $n=7$ vs $11$ rank-sum, $\alpha=0.05$, 80% power에 필요한 이동 ≈ 1.2 SD.
관측 $\Delta'$ SD(h6) ≈ 0.034/0.048 → 전대역에서도 필요한 이동 ≈ 0.05인데 대역 분할은 per-band 분산을 더 키운다.
**판정: (a)의 탐색적 분해로만 유지, 주검정 불가.**

## 3. Killing test (사전등록 판정 기준)

**통계량:** target별 $\delta'_i = \frac{R^2_L - R^2_C}{1-R^2_C}$ (test block; $1-R^2_C < 0.01$인 target 제외 —
h=1 분모 불안정 가드), recording별 $D_r = \mathrm{median}_i\,\delta'_i$.
**대조:** exact one-sided Mann–Whitney $U$, AML32 ($n=7$) vs AML18 ($n=11$), $H_1$: AML32 > AML18.
주검정 $h=6$, 부검정 $h=1$과 $\eta_r$ (L1 공변량 조정판); 총 4검정, 주검정 외 Holm 보정.
(dof: 최소 one-sided p는 $1/\binom{18}{7} \approx 3.1\times10^{-5}$ — 해상도 충분.)

- **KILL ("GCaMP에 GFP 이상의 lag 정보 없음" 확정):** 주검정 $p > 0.05$ **그리고** 중앙값 차 $\le 0.02$
  (검정력 한계 명시: 이 기준으로 놓치는 최대 효과는 잔차분산의 ~5%p).
  죽는 부모 명제의 정확한 범위: docs/7_AGI/25의 **"신경 시간 기억" 해석 전체** (h=1, h=6 모두).
  게이트 자체는 "측정 과정의 비-Markov성" 주장으로 강등되어 존속. CE-AGI의 생물 정합 라인 중
  이 게이트를 인용하는 모든 주장이 함께 강등된다.
- **PASS:** 주검정 $p \le 0.05$ **그리고** $\eta$ 조정 후에도 부호 유지 **그리고** whitened-GFP null (b)에서 재현.
  이 경우에도 결론은 "GFP 잡음 모형 대비 초과 lag 정보" — 메커니즘 식별 아님.

## 4. 권고

**1순위: (a) GFP-matched 대조, 통계량 $\Delta'$ (주) + $\eta$ (L1 조정, 부).**
근거 — 식별 가능성: A1·A2 하 조건부 식별이며 A2가 부분 검증 가능한 유일 후보.
검정력: 전대역 사용으로 (d)보다 우위. 구현: 기존 target-level 산출물 재활용, 새 모형 적합 불필요.
(b)는 확인용 2차, (c) 탈락, (d) 탐색 부속.

사전등록 초안 요지: 위 §3 통계량·대조·판정 그대로; null 절차는 strain 대조 자체가 null
(circular shift는 recording 내부 sanity로만 유지, 증거력 없음); 데이터는 기존 confirmatory 실행의
target-level 재산출(include_targets)로 하고 모형·분할·embargo는 25번과 동일 고정.

## 5. 수치 사전 탐색 (본 채점 아님 — recording 중앙값 기반, ratio-of-medians 주의)

| | $\Delta$ 중앙값 (범위) | $\Delta'$ 중앙값 (범위) |
|---|---|---|
| h=1 AML32 | 0.031 (0.013–0.043) | 0.959 (0.923–0.976) |
| h=1 AML18 | 0.050 (0.024–0.080) | 0.968 (0.956–0.977) |
| h=6 AML32 | 0.214 (0.115–0.256) | 0.337 (0.310–0.402) |
| h=6 AML18 | 0.300 (0.211–0.383) | 0.348 (0.215–0.395) |

exact one-sided MW $p$(AML32>AML18): 원 $\Delta$ — h1: 0.990, h6: 0.998 (역방향 GFP>GCaMP가 $p\approx0.010,\ 0.002$);
정규화 $\Delta'$ — h1: 0.877, h6: **0.430** (완전 겹침).

**[경험적 관측 — 판정]** "GCaMP가 GFP보다 낮다"는 원 $\Delta$의 관찰은 **전적으로 headroom 항등식(L3)으로 설명**된다:
h=6에서 GCaMP의 $R^2_C$가 높아 분모 $(1-R^2_C)$가 작을 뿐, 잔차 중 lag가 설명하는 비율 $\Delta'$은 두 strain이 동일하다.
따라서 "활동으로 인한 희석" 대안 서사는 **불필요**(과잉 설명) — 데이터는 "공통 전처리 유색성 + strain별 합성 스펙트럼
모양 차이"만으로 닫힌다. 또한 h=1에서 $\Delta' \approx 0.96$ (양쪽): 현재값 모형의 잔차 96%가 lag 2개로 선형 설명됨 —
결정론적 스무딩/보간 필터의 전형적 서명으로, 전처리 기원 가설의 독립 정황.
사전 탐색 기준으로 §3의 **KILL 방향 결과가 이미 관측됨**; 확정은 target-level $\delta'$ (median-of-ratios) 재산출 후.

## 숨은 공리·자유도 목록

- A1 전처리 동일 (코드 확인, 강함) / A2 잡음 2차 구조 strain 간 모양 동일 (부분 검증만 가능 — 잔여 자유도)
- L1은 선형·Gaussian 근사: 비선형 $g$ 항과 비-Gaussian 주변분포는 $\eta$가 흡수 못함 (P2 수준 잔여)
- $1-R^2_C$ clip 임계 0.01, 허용 오차 0.02는 이 노트에서 도입한 자유 모수 — 사전등록에서 고정할 것
- recording 중앙값 기반 $\Delta'$은 ratio-of-medians ≠ median-of-ratios (본 채점 전 재산출 필수)

## 본 실행 결과 (2026-08-12, 사전등록 후 확정 채점)

사전등록: `artifacts/agi/local_memory_gfp_matched_preregistration.json` (실행 전 작성·고정).
실행: `_workspace/ce/agi-clarus-field-20260812/artifacts/bio_gate_gfp_matched_run.py`
(동결 `local_memory.py` import만, 무수정 — LF 정규화 sha256이 원 사전등록 해시
`6032a76d…5f86`과 일치 확인). 통계량은 §3 그대로 target-level $\delta'$의
recording별 **median-of-ratios** (§5의 ratio-of-medians와 별개 재산출).
결과: `artifacts/agi/local_memory_gfp_matched_result.json`.

| | AML32 중앙값 (n=7) | AML18 중앙값 (n=11) | 차이 | exact one-sided MW p |
|---|---|---|---|---|
| **h=6 (주검정)** | 0.3564 | 0.3522 | +0.0042 | **0.5351** |
| h=1 (부검정) | 0.9686 | 0.9681 | +0.0004 | 0.3295 |

headroom 가드($1-R^2_C<0.01$ 제외): h=6 제외 0건, h=1 AML32 최대 41/135
(BrainScanner20180709), AML18 최대 4건 — h=1 분모 불안정이 GCaMP 쪽에 집중,
가드 필요성 사후 확인.

**판정: KILL.** 주검정 $p=0.535>0.05$ 그리고 중앙값 차 $0.0042 \le 0.02$ — §3의
KILL 조건 완전 충족. median-of-ratios 재산출이 §5 사전 탐색(ratio-of-medians,
p=0.430)과 같은 방향으로 닫혔고, 두 strain 분포는 사실상 완전히 겹친다
(U=38/77, 기대값 38.5와 구분 불가).

**죽는 것:** docs/7_AGI/25의 "신경 시간 기억" 해석 전체 (h=1, h=6 모두).
GCaMP(활동 의존) 신호가 GFP(활동 비의존) 잡음 대비 잔차분산 정규화 lag 정보를
전혀 초과하지 않으므로, 게이트가 측정한 것은 $P \circ k$ 합성(전처리+indicator)의
비-Markov성이다. **존속하는 것:** 게이트의 명시적 주장문("aligned past adds
held-out prediction")은 관측 과정에 대한 참인 진술로 존속하되 "측정 과정의
비-Markov성"으로 강등. CE-AGI 생물 정합 라인 중 25번 게이트를 신경 기억 근거로
인용하는 모든 주장이 함께 강등된다. 검정력 한계(사전 명시): 이 KILL이 배제하지
못하는 최대 효과는 잔차분산의 ~2%p 미만.

Status: COMPLETE
