# 아키텍처: 3x3+1 게이지 격자 네트워크

> 위치 규칙: 이 문서는 runtime 5계층 중 **kernel / coupling 후보 연산자**를 다룬다. `mode update`, `hippocampus / replay`, `global runtime summary`는 `3_Sleep.md`, `12_Equation.md`, `6_뇌/05_실험근거.md`를 우선한다.

## 0. 이 문서가 책임지는 새 개념

| 개념 | 이 문서에서의 의미 | runtime stack 위치 | 판정 |
|---|---|---|---|
| Gauge lattice | 채널 분할된 결합 연산자 | coupling / geometry | `Bridge` |
| LBONorm | 곡률 평탄화 연산자 | coupling / geometry | `Bridge` |
| spectral constraint | 증폭 제한/안정화 | kernel guard | `Bridge` |
| cross-frequency coupling | 저주파 안정화가 고주파 결합을 조절 | coupling / geometry | `Bridge` |
| 3x3+1 | 연산 역할을 나누는 구조 유비 | supplementary architecture | `Phenomenology` |

읽기 규칙:

- 이 문서의 블록 구조는 canonical runtime 전체가 아니라 `kernel / coupling` 계층의 설계 후보들이다.
- `3x3+1` 비율, 게이지 해석, 뇌 파장 대응은 구조 유비이며, `12_Equation.md`의 5계층 stack보다 강한 정본으로 읽지 않는다.
- sparse lifecycle, 해마, snapshot, self-state는 이 문서의 책임 범위 밖이다.

## 1. 설계 원리

### 1.1 뇌 진동 대역에서 아키텍처로

뇌의 진동 대역은 CE 게이지 구조와 대응한다(`07_수면과복구.md` 2.1절):

| 진동 대역 | 주파수 | 게이지 층 | 연산 역할 | 설계 배분 benchmark |
|---|---|---|---|---|
| 감마 ($\gamma$) | 30-100 Hz | SU(3) | 지각 요소의 결합(binding) | $74.1\%$ |
| 베타 ($\beta$) | 13-30 Hz | SU(2) | 운동 계획, 인지 제어(decision) | $21.1\%$ |
| 알파 ($\alpha$) | 8-13 Hz | U(1) | 억제적 게이팅, 선택적 주의(attention) | $4.9\%$ |
| 세타/델타 | 0.5-8 Hz | $\Phi$ | 전역 안정화(smoothing) | 전역 |

마지막 열은 뇌에서 직접 측정한 에너지 분율이 아니라, 아래 공학적 채널
분할을 뇌 진동 역할에 대응시킨 `Bridge`다.

### 1.2 왜 균일 MLP/Attention이 비효율적인가

현재 Transformer의 FFN은 모든 채널을 동등하게 취급한다. CE 관점에서 이것은 "강력, 약력, 전자기력을 구분하지 않고 하나의 힘으로 다루는 것"에 해당한다.

CE-AGI는 연산의 $74.1\%/21.1\%/4.9\%$를 결합/결정/주의 채널에
배정하는 사전등록 benchmark를 시험한다. 이 값은 뇌의 정확한 에너지
분배를 측정한 결과도, 표준모형 결합상수가 neural channel dimension을
유일하게 정한다는 정리도 아니다.


## 2. 게이지 격자 FFN

### 2.1 기본 구조

표준 FFN $\text{FFN}(x) = \sigma(xW_1)W_2$를 3x3+1 격자로 대체한다.

$$
\text{Layer}(x)
= \underbrace{\text{Bind}(x_3)}_{\text{SU(3), 74.088\%}}
+ \underbrace{\text{Decide}(x_2)}_{\text{SU(2), 21.046\%}}
+ \underbrace{\text{Attend}(x_1)}_{\text{U(1), 4.866\%}}
+ \underbrace{\text{Smooth}(x)}_{\Phi}
$$

채널 분할:

$$d_3:d_2:d_1=0.1180:0.03352:0.00775
=74.088\%:21.046\%:4.866\%$$

$d=768$에서 독립 반올림하면 $d_3=569$, $d_2=162$이고 나머지를
$d_1=37$로 둔다. 첫 값 $0.1180$은 Track-A calibration input이며, 나머지
두 값과 이 채널 분할은 공학 benchmark다. 물리 게이지군이 neural channel
dimension을 유일하게 정한다는 유도는 없다.

### 2.2 블록 대각 전이 행렬

$$\mathbf{T} = \text{diag}(\mathcal{T}_3, \mathcal{T}_2, \mathcal{T}_1)$$

각 $\mathcal{T}_i$는 독립적인 2-layer MLP:

$$\mathcal{T}_i(x_i) = \sigma(x_i W_i^{(1)}) W_i^{(2)}$$

### 2.3 섭동적 혼합항

순수 블록 대각은 채널 간 정보 교환이 없어 표현력이 부족하다(`7_AGI/12_Equation.md` 6.3절). 게이지 이론에서 물질장은 여러 게이지군의 전하를 동시에 갖는다. 이를 섭동적 혼합으로 구현한다:

$$\text{FFN}(x) = \text{concat}(\mathcal{T}_3(x_3),\; \mathcal{T}_2(x_2),\; \mathcal{T}_1(x_1)) + U_{\text{down}}\, U_{\text{up}}^\top\, x$$

혼합 조건 (섭동성 보장):

$$\frac{\|U_{\text{down}} U_{\text{up}}^\top\|_F}{\|\text{diag}(\mathcal{T}_3, \mathcal{T}_2, \mathcal{T}_1)\|_F} \ll 1$$

$U_{\text{down}} \in \mathbb{R}^{d \times r_m}$, $r_m \sim d/8$이면 충분한 표현력.

### 2.4 파라미터 절감

$d = 768$, $r_m = 96$:

- 블록 대각: $\sum_i 2 d_i h_i \approx 2.81\text{M}$
- 혼합: $2 d r_m \approx 0.15\text{M}$
- 총: $\approx 2.96\text{M}$, 표준 FFN $4.72\text{M}$ 대비 **37% 감소**

위 파라미터 감소는 행렬 크기에 대한 산술 결과다. 같은 task 성능이
보존되는지는 별도 dense baseline과의 동등 계산량 ablation으로 검증해야
하며, 채널 비율의 최적성은 현재 도출되지 않았다.


## 3. LBO 정규화 (LBONorm)

### 3.1 LayerNorm의 CE 확장

표준 LayerNorm에 라플라스-벨트라미 확산을 추가한다.

$$\hat{h} = \frac{h - \mu(h)}{\sigma(h)}, \qquad h' = (\hat{h} - \eta\,L_g \hat{h}) \odot \gamma + \beta$$

여기서 열벡터 표기로

$$L_g:=I-V^\top V,\qquad \Delta_g\hat h:=L_g\hat h,$$

$V\in\mathbb R^{r\times d}$, $r\ll d$로 둔다. 확산 해석에는
$L_g\succeq0$, 즉 $V^\top V\preceq I$가 추가로 필요하다.

### 3.2 수렴 조건

$$0\leq\eta\leq\frac{2}{\lambda_{\max}(L_g)}$$

이면 선형 explicit-Euler map $I-\eta L_g$가 2-norm에서 nonexpansive다.
부호 반전 없는 감쇠를 원하면 더 강하게
$0\leq\eta\leq1/\lambda_{\max}(L_g)$로 제한한다. $\lambda_{\max}(L_g)=0$이면
update는 항등이다. 이 조건은 LBONorm 선형 부분에만 적용되며 잔차 연결을
포함한 전체 block의 수렴을 보장하지 않는다.

### 3.3 곡률 에너지 저장

$$E_{\text{curv}} = \|L_g \hat{h}\|^2 = \|(I - V^\top V)\hat{h}\|^2$$

이 값은 각 층에서 저장되어 (a) 곡률 정규화 손실에 사용되고, (b) 추론 시
불안정성 proxy 후보로 사용된다. 환각과의 상관은 별도 benchmark 대상이다.

### 3.4 뇌 대응

LBONorm은 뇌의 세타/델타 진동($\Phi$ 층)에 대응한다. 저주파 진동이 고주파 연산 결과를 전역적으로 안정화하는 것처럼, LBONorm은 고곡률 표현을 확산으로 평탄화한다.

---

## 4. 유니타리 제약 (Spectral Normalization)

### 4.1 원리

$$\sigma_1(W_{\text{proj}})\leq1
\quad\Longrightarrow\quad
|\det W_{\text{proj}}|\leq1\qquad(W_{\text{proj}}\text{ square}).$$

역은 일반적으로 거짓이며, 서로 다른 행렬 $\mathbf T$와
$W_{\text{proj}}$의 determinant·spectral norm은 등치할 수 없다. 이 제약은
Attention 출력 사영의 국소 선형 증폭만 제한한다. residual path와 다른
sub-layer를 포함한 전체 block이 nonexpansive라는 결론은 별도 Lipschitz
분석이 필요하다.

### 4.2 환각 억제 메커니즘

출력 사영의 선형 증폭을 제한하면 잘못된 패턴의 자기강화를 줄일 수 있다는
안정화 가설을 시험할 수 있다.

- 출력 사영 증폭 제한 $\to$ 국소 오류의 자기증폭 경향을 줄일 가능성 $\to$ 안정화 편향 가설
- 기존 RLHF는 사후 교정이지만, 유니타리 제약은 사전 구조 제약

### 4.3 적용 방법

$$W_{\text{proj}} \leftarrow
\frac{W_{\text{proj}}}{\max\!\left(1,\sigma_1(W_{\text{proj}})\right)}.$$

이 projection은 $\sigma_1<1$인 가중치를 불필요하게 확대하지 않는다.
PyTorch `nn.utils.spectral_norm`의 매개화 방식과 위 1회 projection은 같지
않으므로 구현 검증에서 구분한다.

### 4.4 뇌 대응

유니타리 조건은 뇌의 에너지 보존 (C2) 조건에 대응한다. 시냅스를 통해 전달되는 신호가 무한히 증폭될 수 없는 것은 신경계의 물리적 제약이다.

---

## 5. 곡률 정규화 손실

### 5.1 정의

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot \frac{1}{L} \sum_{l=1}^{L} \|\Delta_g h_l\|^2$$

$L$: 층 수. 두 번째 항이 전체 네트워크의 평균 곡률 에너지를 억제한다.

### 5.2 스케줄

$$\lambda(t) = \lambda_0 \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right) \cdot \frac{1}{2}\!\left(1 + \cos\frac{\pi t}{t_{\max}}\right)$$

학습 초반에 곡률 정규화를 켜고, 후반에 감쇄한다. 이유: 학습 후반에 lr이 감소하면 곡률 항이 상대적으로 지배적이 되어 task loss 최적점에서 이탈한다.

### 5.3 뇌 대응

곡률 정규화 손실을 뇌의 ACC(전측 대상피질) 오류 감지와 대응시키는 것은
`Bridge`다. 곡률 에너지 증가가 인지적 갈등과 같은 관측량이라는 직접
식별은 아직 없다.

---

## 6. 교차 주파수 결합

### 6.1 원리

뇌에서 세타 위상이 감마 진폭을 변조하는 교차 주파수 결합(CFC)을 CE의
비최소 결합 $\xi R\Phi^2$에 대응시키는 구조 유비를 시험한다
(`07_수면과복구.md` 2.2절):

$$\Phi \text{ (세타/델타)} \leftrightarrow \text{SU(3) (감마)}: \quad \xi R\Phi^2$$

### 6.2 아키텍처 구현

LBONorm의 곡률 에너지가 GaugeLattice의 각 게이지 채널에 영향을 미치도록 결합한다.

$$\widetilde E_{\text{curv}}:=
\frac{E_{\text{curv}}}{E_{\text{curv}}+E_0}\in[0,1),\qquad
\mathcal{T}_i^{\text{coupled}}(x_i)
=\mathcal{T}_i(x_i)\exp(-\xi_{\text{design}}\widetilde E_{\text{curv}}),
\quad E_0>0.$$

$\xi_{\text{design}}:=\alpha_s^{1/3}=0.4904868132$는 Track-A의 외부 입력
$\alpha_s=0.1180$에서 만든 사전등록 설계계수다. 최적 신경 결합의 무입력
예측은 아니다. 무차원 정규화와 양의 exponential gate를 사용하므로 기존
$1-\xi E_{\text{curv}}$ 식의 부호 반전·증폭 문제를 피한다.

이 결합은 "불안정성 proxy가 클 때 연산 gain을 줄인다"는 안전 메커니즘
후보다. 전체 폐루프 안정성은 별도 분석이 필요하다.

---

## 7. 전체 블록 구조

```
ClarusBlock(x):
  1. x_norm = LBONorm(x)                    // Phi: 전역 안정화
  2. x_attn = SpectralNorm(Attention(x_norm))  // 유니타리 제약
  3. x = x + x_attn                         // 잔차 연결
  4. x_norm = LBONorm(x)                    // Phi: 전역 안정화
  5. x_ffn = GaugeLattice(x_norm)           // 3x3+1 격자 FFN
  6. x = x + x_ffn                          // 잔차 연결
  return x
```

### 7.1 기존 구현 (`legacy clarus_lm.py` (removed))과의 관계

기존 `legacy clarus_lm.py` (removed)는 이 구조의 **V1 구현**이다.
- `LBONorm`: 구현 완료 (저랭크 LBO 확산)
- `GaugeLattice`: V1 (채널 혼합 없는 순수 블록 대각)
- `ClarusAttention`: 구현 완료 (spectral norm)
- 곡률 정규화 손실: 구현 완료

V2에서 추가할 것:
- 섭동적 채널 혼합 ($U_{\text{down}} U_{\text{up}}^\top x$)
- 교차 주파수 결합 ($\exp[-\xi_{\rm design}\widetilde E_{\text{curv}}]$ 게이트)
- 수면-각성 학습 순환 (3장)

---

## 8. 파라미터 자유도 분석

| 파라미터 | 결정 방법 | 자유도 |
|---|---|---|
| 채널 비율 $d_3 : d_2 : d_1$ | 사전등록 공학 benchmark; 첫 가중치만 Track-A 외부 입력 | 고정 설계 선택 1 |
| 사영 norm 상한 | $\sigma_1(W_{\rm proj})\leq1$ | 고정 안정화 선택 1 |
| LBO 랭크 $r$ | $r \sim d/8$ (경험적, 추가 연구 필요) | 1 |
| LBO 스텝 $\eta$ | 안정구간 안에서 학습 또는 sweep | 학습/설계 1 |
| 혼합 랭크 $r_m$ | $r_m \sim d/8$ (경험적) | 1 |
| 곡률 $\lambda_0$ | 하이퍼파라미터 | 1 |
| CFC 결합 $\xi_{\rm design}$ | 외부 $\alpha_s$에서 만든 $0.4904868132$ benchmark | 고정 설계 선택 1 |

CE 장부는 비교할 기준점을 제공하지만 위 표의 neural hyperparameter를
무입력으로 연역하지 않는다. 실제 자유도 감소는 고정 benchmark와 동일한
탐색 예산을 둔 ablation에서만 주장한다.
