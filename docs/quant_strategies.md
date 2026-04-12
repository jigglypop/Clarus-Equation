# CE 우주론 기반 퀀트 전략

> 근거: `docs/경로적분.md`, `docs/3_상수/3_부트스트랩.md`, `docs/3_상수/7_우주론.md`, `docs/7_AGI/11_Quantitative.md`
>
> 코드: `ce_core/src/engine/market.rs`, `clarus/core/src/engine/filter.rs`, `clarus/core/src/engine/manifold.rs`, `clarus/core/src/engine/constants.rs`

---

## 0. 핵심 CE 방정식 요약

### 0.1 부트스트랩 고정점

$$\varepsilon^2 = e^{-(1-\varepsilon^2) D_{\text{eff}}}$$

$D_{\text{eff}} = 3 + \delta$, $\delta = \sin^2\theta_W \cos^2\theta_W = 0.17776$일 때 유일한 물리 해:

$$\varepsilon^2 = -\frac{W_0(-D_{\text{eff}}\, e^{-D_{\text{eff}}})}{D_{\text{eff}}} = 0.04865$$

Lambert $W_0$는 주가지(principal branch). 이 고정점은 바나흐 축소 사상 정리에 의해 존재하고 유일하다.

### 0.2 에너지 분할

$$\Omega_b = \varepsilon^2 = 0.04865, \quad R = \alpha_s D_{\text{eff}}(1+\varepsilon^2\delta) = 0.38063$$

$$\Omega_\Lambda = \frac{1-\varepsilon^2}{1+R} = 0.6891, \quad \Omega_{DM} = \frac{(1-\varepsilon^2)R}{1+R} = 0.2623$$

### 0.3 접힘 생존 함수

가분성(B1) + 연속성(B2) + 정규화(B3)의 유일한 연속해:

$$S(D) = e^{-D}$$

### 0.4 클라루스장 유도 계량 (Conformal Metric)

$$g_{\mu\nu} = e^{-2\alpha\Phi(x)}\,\delta_{\mu\nu}$$

Ricci 스칼라:

$$R(x) = e^{2\alpha\Phi}\left[2(n-1)\alpha\,\nabla^2\Phi - (n-2)(n-1)\alpha^2\,|\nabla\Phi|^2\right]$$

### 0.5 스펙트럼 필터 함수

시간 도메인 펄스열 $y(t) \to$ FFT $\to$ 필터 함수 $|Y(\omega)|^2$.

게인 함수:

$$g(\omega) = \frac{|Y_{\text{CPMG}}(\omega)|^2}{|Y_{\text{CE}}(\omega)|^2}$$

### 0.6 Hopfield 이완 에너지

$$E_{\text{hop}} = -\frac{1}{2}m^\top W m + E_{\text{bias}} + E_{\text{portal}} + E_{\text{bypass}} + E_{\text{codebook}}$$

자연 기울기 업데이트: $\lambda_0, \lambda_\phi, \lambda_{\text{var}}$ + FDT 노이즈.

---

## 1. 곡률-억제 레짐 분류기 (Curvature Regime Classifier)

### 1.1 이론적 기초

CE의 시장 기하학(`market.rs`)에서 시장 상태는 Riemann 곡률 스칼라 $R$로 기술된다. 클라루스장 억제 $\Phi$는 곡률이 높을수록 경로를 접는다:

$$R(t) = \alpha \cdot \frac{\sigma_{\text{Park}}^2(t)}{\ell(t) + \epsilon}$$

여기서:
- $\sigma_{\text{Park}}$: Parkinson 변동성 (High-Low 기반)

$$\sigma_{\text{Park}} = \sqrt{\frac{1}{4\ln 2 \cdot N}\sum_{i=1}^{N}\left(\ln\frac{H_i}{L_i}\right)^2}$$

- $\ell(t) = V_t / \bar{V}$: 상대 유동성 (현재 거래량 / 평균 거래량)
- $\alpha$: 결합 상수 (기본값 5.0)

억제 함수:

$$\Phi(t) = e^{-\beta R(t)}$$

$\beta = 100$ (스케일 인자). $\Phi \to 0$이면 위기, $\Phi \to 1$이면 안정.

### 1.2 레짐 정의

| 레짐 | $R$ 범위 | $\Phi$ 범위 | 시장 상태 | 전략 |
|---|---|---|---|---|
| 저곡률 (평온) | $R < 10^{-4}$ | $\Phi > 0.99$ | 변동성 낮고 유동성 풍부 | 평균회귀 |
| 중곡률 (전이) | $10^{-4} \leq R < 10^{-2}$ | $0.37 < \Phi \leq 0.99$ | 불확실성 증가 | 포지션 축소 |
| 고곡률 (위기) | $R \geq 10^{-2}$ | $\Phi \leq 0.37$ | 급격한 변동, 유동성 고갈 | 역추세 진입 대기 |

임계값 $\Phi = 0.37 = e^{-1}$은 CE 1차원 접힘 인자와 동일하다.

### 1.3 레짐 전환 탐지

부트스트랩 수렴 속도 $\rho = D_{\text{eff}} \cdot \varepsilon^2 = 0.155$를 이용한 지수 이동 평균:

$$R_{\text{smooth}}(t) = \rho \cdot R(t) + (1-\rho) \cdot R_{\text{smooth}}(t-1)$$

$\rho = 0.155$는 부트스트랩 고정점의 최소 축소율이므로, 2-3회 순환(스무딩)으로 잡음을 제거하면서 구조적 전환은 보존한다.

### 1.4 전략 수식

레짐별 목표 포지션:

$$w^*(t) = \begin{cases} w_{\text{MR}} \cdot \Phi(t) & \text{저곡률: 평균회귀} \\ w_{\text{prev}} \cdot \Phi(t) & \text{중곡률: 포지션 축소} \\ w_{\text{CT}} \cdot (1 - \Phi(t)) & \text{고곡률 감쇠 시: 역추세 진입} \end{cases}$$

$\Phi$가 포지션 사이징을 자동 제어한다. 별도 손절 로직 없이 곡률만으로 리스크 관리.

---

## 2. 스펙트럼 필터 신호 정제 (CE Filter Signal Denoising)

### 2.1 이론적 기초

`filter.rs`의 CE 필터는 CPMG(Carr-Purcell-Meiboom-Gill) 대비 특정 주파수 대역에서 노이즈를 더 효과적으로 억제한다. 금융 시계열에 적용하면 가짜 신호(whipsaw)를 줄인다.

### 2.2 펄스 시퀀스 정의

**CPMG** (등간격):

$$t_j^{\text{CPMG}} = \frac{j - 0.5}{N}, \quad j = 1, \ldots, N$$

**UDD** (Uhrig Dynamical Decoupling):

$$t_j^{\text{UDD}} = \sin^2\!\left(\frac{j\pi}{2N+2}\right), \quad j = 1, \ldots, N$$

**CE 펄스**: 부트스트랩 고정점에서 유도된 비등간격 배치. 저주파에서 CPMG보다 강한 억제, 고주파에서 선택적 통과.

### 2.3 금융 시계열 적용

가격 시계열 $p(t)$에 대해:

1. 로그 수익률 $r(t) = \ln p(t) - \ln p(t-1)$ 계산
2. FFT: $\hat{r}(\omega) = \text{FFT}[r(t)]$
3. CE 필터 적용: $\hat{r}_{\text{filtered}}(\omega) = \hat{r}(\omega) \cdot |Y_{\text{CE}}(\omega)|^2$
4. 역 FFT: $r_{\text{clean}}(t) = \text{IFFT}[\hat{r}_{\text{filtered}}(\omega)]$

### 2.4 노이즈 스펙트럼 적분 (시장 잡음 에너지)

$$\mathcal{N} = \int_0^{\omega_{\max}} S(\omega) \cdot |Y(\omega)|^2 \frac{d\omega}{2\pi}$$

$S(\omega) = A/\omega^\gamma$ (핑크 노이즈 모형, 금융 시계열의 전형적 스펙트럼). CE 필터가 CPMG 대비 $\mathcal{N}$을 최소화하는 주파수 대역에서 신호를 추출.

### 2.5 전략 수식

정제된 신호의 방향과 세기:

$$\text{signal}(t) = \text{sign}\!\left(\sum_{k=t-L}^{t} r_{\text{clean}}(k)\right) \cdot \left|\sum_{k=t-L}^{t} r_{\text{clean}}(k)\right|^{1/D_{\text{eff}}}$$

지수 $1/D_{\text{eff}} = 1/3.178 = 0.315$는 비선형 스케일링으로, 큰 신호에 대한 과민 반응을 억제한다. $D_{\text{eff}} > 3$이므로 순수 3차원($1/3 = 0.333$)보다 약간 더 보수적이다.

---

## 3. Riemannian 포트폴리오 최적화

### 3.1 이론적 기초

Markowitz의 평균-분산 최적화는 유클리드 공간을 가정한다. CE의 `manifold.rs`는 Riemannian 다양체 위에서 측지선 경로를 계산한다. 자산 수익률 공간의 곡률이 0이 아니면 최적 경로가 직선이 아니다.

### 3.2 자산 공간의 클라루스장 유도 계량

$n$개 자산의 수익률 벡터 $\mathbf{x} \in \mathbb{R}^n$에 대해, 클라루스장을 변동성 가중 함수로 정의:

$$\Phi(\mathbf{x}) = \sum_{i=1}^{n} \frac{\sigma_i^2(t)}{2} x_i^2 + \sum_{i<j} \rho_{ij}(t) \sigma_i \sigma_j x_i x_j$$

이것은 공분산 행렬 $\Sigma$의 이차형식이므로 $\Phi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \Sigma \mathbf{x}$.

유도 계량:

$$g_{ij}(\mathbf{x}) = e^{-2\alpha\Phi(\mathbf{x})}\,\delta_{ij}$$

### 3.3 리치 스칼라와 리스크

$$R(\mathbf{x}) = e^{2\alpha\Phi}\left[2(n-1)\alpha\,\nabla^2\Phi - (n-2)(n-1)\alpha^2\,|\nabla\Phi|^2\right]$$

$R > 0$: 양의 곡률 = 수렴하는 측지선 = 리스크 집중 영역
$R < 0$: 음의 곡률 = 발산하는 측지선 = 분산 효과 영역
$R \approx 0$: 평탄 = Markowitz와 동일

### 3.4 측지선 리밸런싱

현재 포트폴리오 $\mathbf{w}_{\text{now}}$에서 목표 $\mathbf{w}_{\text{target}}$으로의 최적 경로:

$$\frac{d^2 w^k}{dt^2} + \Gamma^k_{ij} \frac{dw^i}{dt}\frac{dw^j}{dt} = 0$$

크리스토펠 기호 (`manifold.rs`):

$$\Gamma^k_{ij} = \delta^k_i(-\alpha\,\partial_j\Phi) + \delta^k_j(-\alpha\,\partial_i\Phi) - \delta_{ij}(-\alpha\,\partial^k\Phi)$$

Exponential map으로 이산화:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_t \cdot \Delta t + \frac{1}{2}\mathbf{a}_t \cdot \Delta t^2$$

$$a^k = -\sum_{i,j}\Gamma^k_{ij}\, v^i v^j$$

이 경로는 고곡률(고리스크) 영역을 우회하면서 목표에 도달한다. Markowitz의 직선 경로 대비 중간 과정의 리스크가 낮다.

### 3.5 전략 수식

리밸런싱 속도를 곡률로 조절:

$$\Delta t_{\text{rebal}} = \Delta t_0 \cdot e^{-|R(\mathbf{w}_t)|}$$

곡률이 높으면 리밸런싱을 느리게 (급변 시 무리한 거래 방지), 곡률이 낮으면 빠르게 (안정 시 효율적 조정).

---

## 4. Hopfield 에너지 패턴 매칭

### 4.1 이론적 기초

CE 엔진의 Hopfield 이완(`engine.py`)은 hidden state를 에너지 최소화로 안정화한다. 동일 메커니즘으로 과거 가격 패턴을 "기억"하고, 현재 시장 상태를 가장 가까운 기억 패턴으로 이완시켜 예측한다.

### 4.2 에너지 함수

$$E(\mathbf{m}) = -\frac{1}{2}\mathbf{m}^\top W \mathbf{m} + \mathbf{b}^\top \mathbf{m} + E_{\text{portal}} + E_{\text{codebook}}$$

- $\mathbf{m} \in \mathbb{R}^d$: 현재 시장 상태 벡터 (과거 $d$일의 정규화 수익률)
- $W$: 희소 가중치 행렬 (과거 패턴들의 외적 합)
- $\mathbf{b}$: 편향 (시장 드리프트)
- $E_{\text{portal}}$: 클라루스장 포탈 에너지 (레짐 전환 보정)
- $E_{\text{codebook}}$: PQ 코드북 에너지 (양자화된 패턴 유사도)

### 4.3 패턴 저장

$K$개의 과거 패턴 $\{\mathbf{p}_1, \ldots, \mathbf{p}_K\}$ (각각 $d$일 수익률 벡터):

$$W = \frac{1}{K}\sum_{k=1}^{K} \mathbf{p}_k \mathbf{p}_k^\top - \text{diag}$$

### 4.4 이완 (추론)

현재 시장 상태 $\mathbf{m}_0$에서 시작하여 에너지를 최소화:

$$\mathbf{m}_{t+1} = \mathbf{m}_t + \Delta t \left[-\lambda_0 \nabla_m E + \lambda_\phi \cdot \mathbf{f}_{\text{portal}} + \sqrt{\frac{2\tau}{dt}}\,\boldsymbol{\eta}\right]$$

$\tau$: 온도 (탐색-활용 균형)
$\boldsymbol{\eta}$: FDT 노이즈

수렴한 $\mathbf{m}^*$가 가장 가까운 기억 패턴. 그 패턴의 "이후 $h$일 수익률"이 예측.

### 4.5 전략 수식

예측 신뢰도 = 에너지 낙차:

$$\text{confidence} = \frac{E(\mathbf{m}_0) - E(\mathbf{m}^*)}{E(\mathbf{m}_0)}$$

포지션:

$$w(t) = \text{sign}(\hat{r}_{t+h}) \cdot \text{confidence} \cdot \Phi(t)$$

$\hat{r}_{t+h}$: 패턴 기반 예측 수익률, $\Phi(t)$: 곡률 기반 리스크 스케일링.

---

## 5. Sleep 적응형 전략 최적화

### 5.1 이론적 기초

`sleep.py`의 Sleep/Microsleep 메커니즘을 전략 파라미터 자기 최적화에 적용한다. NREM은 안정적 업데이트, REM은 탐색적 업데이트, Guard-set은 성과 저하 시 롤백.

### 5.2 전략 파라미터 벡터

$$\boldsymbol{\theta} = (\alpha, \beta, L, N_{\text{pulse}}, \Delta t_{\text{rebal}}, \tau, \ldots)$$

### 5.3 NREM 단계: Ridge 피팅

최근 $T$ 기간의 데이터로 전략 파라미터를 ridge 회귀:

$$\boldsymbol{\theta}_{\text{NREM}} = \arg\min_{\boldsymbol{\theta}} \left[\sum_{t=1}^{T} \ell(r_t, \hat{r}_t(\boldsymbol{\theta})) + \lambda \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prev}}\|^2\right]$$

정규화 항 $\lambda \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prev}}\|^2$이 과적합을 방지하면서 기존 파라미터에서 크게 벗어나지 않게 한다.

### 5.4 REM 단계: 노이즈 주입 탐색

$$\boldsymbol{\theta}_{\text{REM}} = \boldsymbol{\theta}_{\text{NREM}} + \sigma_{\text{REM}} \cdot \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(0, I)$$

$\sigma_{\text{REM}}$: REM 노이즈 스케일. 새로운 파라미터 조합을 탐색하여 지역 최적에 빠지는 것을 방지.

### 5.5 Guard-set 검증

업데이트 전후로 홀드아웃 데이터에서 성과 지표를 비교:

$$\text{Guard}(\boldsymbol{\theta}) = (\text{Sharpe}, \text{MaxDD}, \text{WinRate})$$

$$\text{if } \text{Guard}(\boldsymbol{\theta}_{\text{new}}) < \text{Guard}(\boldsymbol{\theta}_{\text{old}}) \Rightarrow \text{rollback to } \boldsymbol{\theta}_{\text{old}}$$

이것은 `sleep.py`의 `evaluate_guard_set` + 디코더 스냅샷 롤백과 동일한 메커니즘이다.

### 5.6 수렴 속도 예측

부트스트랩 축소율 $\rho = 0.155$로부터:

| Sleep 순환 $n$ | $\rho^n$ | 잔차 비율 | 의미 |
|---|---|---|---|
| 1 | 0.155 | 15.5% | 초기 적응 |
| 2 | 0.024 | 2.4% | 대부분 수렴 |
| 3 | 0.0037 | 0.37% | 정밀 조정 |

2-3회 Sleep 순환으로 전략이 레짐 변화에 적응 완료.

### 5.7 Microsleep (온라인 적응)

`run_guarded_microsleep_session`과 동일 구조:

1. `PromptReplayBuffer` $\to$ 최근 $B$개 거래 기록 저장
2. 매 `sleep_every` 스텝마다 Sleep 순환 실행
3. Guard 비교 후 accept/reject

---

## 6. Gauge Closure 매크로 헷지

### 6.1 이론적 기초

CE 1층의 게이지 폐합:

$$\alpha_s + \alpha_w + \alpha_{em} = \frac{1}{2\pi}$$

세 결합 상수의 합이 상수. 이것을 세 매크로 지표로 매핑:

| CE 결합 | 비율 | 매크로 대응 | 자산 클래스 |
|---|---|---|---|
| $\alpha_s / \alpha_{\text{total}}$ | 74.1% | 실물 경제 (GDP, 고용) | 주식 |
| $\alpha_w / \alpha_{\text{total}}$ | 21.1% | 금융 정책 (금리, 유동성) | 채권 |
| $\alpha_{em} / \alpha_{\text{total}}$ | 4.8% | 외부 충격 (지정학, 원자재) | 원자재/금 |

### 6.2 폐합 이탈 지표

세 매크로 신호의 정규화 강도 합:

$$C(t) = f_s(t) + f_w(t) + f_{em}(t)$$

CE 폐합: $C = 1/(2\pi) = 0.15916$ (상수). 실제 매크로 지표의 합이 이 범위를 벗어나면 비정상:

$$\Delta C(t) = |C(t) - C_0| / C_0$$

$\Delta C > 0.1$ (10% 이탈): 매크로 헷지 트리거.

### 6.3 에너지 분할 기반 자산 배분

부트스트랩 에너지 분할을 자산 배분에 대응:

$$w_{\text{equity}} = \Omega_b + \Omega_{DM} \cdot f_{\text{risk}}(t) = 0.311 \cdot f_{\text{risk}}(t)$$

$$w_{\text{bond}} = \Omega_\Lambda \cdot (1 - f_{\text{risk}}(t)) = 0.689 \cdot (1 - f_{\text{risk}}(t))$$

$f_{\text{risk}}(t) = \Phi(t)$: 곡률 기반 리스크 선호도.

고곡률(위기): $\Phi \to 0 \Rightarrow$ 주식 비중 $\to 0$, 채권 비중 $\to 69\%$
저곡률(안정): $\Phi \to 1 \Rightarrow$ 주식 비중 $\to 31\%$, 채권 비중 $\to 0$

나머지는 원자재/현금.

---

## 7. 부트스트랩 희소성 기반 팩터 선택

### 7.1 이론적 기초

CE의 부트스트랩 희소율 $\varepsilon^2 = 0.0487$은 "활성 뉴런의 최적 비율"이다. 동일 원리로 팩터 투자에서 "활성 팩터의 최적 비율"을 결정한다.

### 7.2 최적 팩터 수

$N$개 후보 팩터 중 활성 팩터 수:

$$k^* = \lceil \varepsilon^2 \cdot N \rceil$$

| 후보 팩터 수 $N$ | 최적 활성 $k^*$ | 비율 |
|---|---|---|
| 20 | 1 | 5.0% |
| 50 | 3 | 6.0% |
| 100 | 5 | 5.0% |
| 200 | 10 | 5.0% |

CE 예측: 팩터 투자의 효율-정확도 Pareto front의 knee point는 **4-5% 근방**에 나타난다.

### 7.3 팩터 가중치

활성 팩터의 가중치를 CE 에너지 분할로 배분:

$$w_i = \frac{e^{-\Phi_i}}{\sum_{j \in \text{active}} e^{-\Phi_j}}$$

$\Phi_i$: 팩터 $i$의 억압 포텐셜 (변동성, 하락위험 등으로 구성). 억압이 적은 팩터에 더 많은 가중치.

---

## 8. 거시경제 위기 예측 (`MacroEconomy`)

### 8.1 거시 곡률

`market.rs`의 `MacroEconomy` 구조체:

$$R_{\text{macro}} = \alpha_{\text{debt}} \cdot \frac{(\text{Debt/GDP})^2}{\text{Growth Power}} \cdot e^{20 \cdot r_{\text{interest}}} \cdot 10^{-3}$$

성장 동력:

$$\text{Growth Power} = \max\!\left(g_{\text{GDP}} + 0.5 \cdot \beta_{\ell} \cdot \eta \cdot g_{M2},\; 10^{-4}\right)$$

$\eta = 0.1$ if $g_{\text{GDP}} < 0$ (유동성 함정), else $\eta = 1$.

### 8.2 위기 확률

$$P_{\text{crisis}} = 1 - e^{-R_{\text{macro}}}$$

이것은 CE 억제 함수 $\Phi = e^{-R}$의 보수: $P_{\text{crisis}} = 1 - \Phi$.

---

## 9. 통합 전략 아키텍처

```
OHLCV 데이터
    |
    v
[2. CE 필터] --> 정제된 신호
    |
    v
[1. 곡률 레짐 분류] --> R(t), Phi(t), 레짐
    |
    v
+---+---+---+---+
|   |   |   |   |
v   v   v   v   v
[3] [4] [6] [7] [8]
포트 패턴 매크 팩터 거시
폴리 매칭 로헷 선택 위기
오    지    
    |
    v
[5. Sleep 적응] --> 파라미터 자기 최적화
    |
    v
최종 포지션: w*(t) = signal(t) * Phi(t)
```

### 9.1 결합 포지션 산출

$$w^*(t) = \underbrace{\text{signal}_{\text{filter}}(t)}_{\text{방향}} \cdot \underbrace{\text{confidence}_{\text{Hopfield}}(t)}_{\text{확신도}} \cdot \underbrace{\Phi(t)}_{\text{리스크 스케일링}} \cdot \underbrace{w_{\text{asset}}(t)}_{\text{자산 배분}}$$

모든 항이 CE 방정식에서 유도된다.

---

## 10. 구현 우선순위

| 순위 | 전략 | 구현 난이도 | 필요 데이터 | 예상 구현 기간 |
|---|---|---|---|---|
| 1 | 곡률 레짐 + Phi 포지션 사이징 (1, 8) | 낮음 | OHLCV | 1주 |
| 2 | CE 필터 신호 정제 (2) | 중간 | OHLCV | 1-2주 |
| 3 | Sleep 적응형 최적화 (5) | 중간 | 거래 기록 | 2주 |
| 4 | Hopfield 패턴 매칭 (4) | 높음 | 장기 가격 데이터 | 3-4주 |
| 5 | Riemannian 포트폴리오 (3) | 높음 | 다자산 수익률 | 3-4주 |
| 6 | Gauge 매크로 헷지 (6) | 중간 | 매크로 지표 | 2주 |
| 7 | 부트스트랩 팩터 선택 (7) | 낮음 | 팩터 데이터 | 1주 |

### 10.1 최소 실행 가능 전략 (MVP)

1번(곡률 레짐) + 2번(CE 필터)를 결합한 단일 자산 추세추종:

- 입력: 단일 종목/암호화폐 OHLCV
- CE 필터로 신호 정제 $\to$ 추세 방향 결정
- 곡률 $R$로 레짐 분류 $\to$ 포지션 사이징
- $\Phi$로 동적 손절 $\to$ 리스크 관리
- API: Binance/Upbit (무료), VPS 월 5천원

---

## 부록 A. 수치 상수 참조표

| 상수 | 값 | 출처 | 전략에서의 역할 |
|---|---|---|---|
| $\varepsilon^2$ | 0.04865 | 부트스트랩 고정점 | 희소율, 팩터 비율 |
| $D_{\text{eff}}$ | 3.17776 | $3 + \delta$ | 비선형 스케일링 지수 |
| $\rho$ | 0.155 | $D_{\text{eff}} \cdot \varepsilon^2$ | 스무딩 계수, 수렴 속도 |
| $\alpha_s$ | 0.11789 | 게이지 폐합 | 양자 요동 비율 |
| $R = \alpha_s D_{\text{eff}}$ | 0.3747 | DM/DE 비율 | 자산 배분 비율 |
| $e^{-1}$ | 0.3679 | 1차원 접힘 | 위기 임계 |
| $1/(2\pi)$ | 0.15916 | 게이지 총량 | 매크로 폐합 기준 |
| $F = 1 + \alpha_s D_{\text{eff}}$ | 1.375 | QCD 증강 | 시장 증강 인자 |

## 부록 B. 기존 코드 매핑

| 전략 | 기존 코드 | 확장 필요 |
|---|---|---|
| 곡률 레짐 | `ce_core/src/engine/market.rs` `MarketGeometry` | Python 래퍼, API 연결 |
| CE 필터 | `clarus/core/src/engine/filter.rs` `FilterFunction` | 금융 시계열 입력 어댑터 |
| Riemannian | `clarus/core/src/engine/manifold.rs` `SuppressionManifold` | 다자산 공간 확장 |
| Hopfield | `clarus/engine.py` `hopfield_relax` | 가격 패턴 인코더 |
| Sleep | `clarus/sleep.py` `run_sleep_cycle` | 전략 파라미터 디코더 |
| 거시경제 | `ce_core/src/engine/market.rs` `MacroEconomy` | 실시간 데이터 피드 |
