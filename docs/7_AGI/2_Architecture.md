# 아키텍처: 3x3+1 게이지 격자 네트워크

이 문서는 입력 표현을 3x3+1 채널 블록으로 나누고 곡률 제약을 더하는 신경망 설계 후보를 정의한다. 독자는 FFN·정규화·행렬 노름의 기초를 아는 독자를 전제로 하며, 출력은 구현 가능한 연산자 명세이지 뇌 구조의 확인이나 성능 우위 정리가 아니다.

총론에서 정한 범위 뒤에 설계 원리, 블록 전이, 정규화·스펙트럼 제약, 손실, 결합과 전체 구조의 순서로 읽는다. 활성값은 무차원이고 시간축은 학습 step이며, 성공은 동일 예산 기준선 대비 사전 등록한 holdout·OOD·ablation gate를 통과할 때만 주장한다.

> 위치 규칙: 이 문서는 runtime 5계층 중 **kernel / coupling 후보 연산자**를 다룬다. `mode update`, `hippocampus / replay`, `global runtime summary`는 `3_Sleep.md`, `12_Equation.md`, `6_뇌/05_실험근거.md`를 우선한다.

## 0. 이 문서가 책임지는 새 개념

이 장의 새 개념은 물리적 게이지군을 구현 채널과 동일시하지 않고, 계산 블록의 역할을 나누는 구조 비유로 한정한다. 표의 runtime 위치는 인터페이스 의존성을 나타내며 생물학적 대응이나 CE 정리의 직접 산출을 뜻하지 않는다.

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

총론의 결함 진단을 실제 연산자로 옮기려면 뇌 진동이라는 관찰 비유와 모델의 채널 분할 사이에 놓인 가정을 드러내야 한다. 아래 대응은 설계 동기일 뿐 timebase가 다른 신경 진동과 transformer layer를 수치적으로 동일시하지 않는다.

### 1.1 뇌 진동 대역에서 아키텍처로

뇌의 진동 대역은 CE 게이지 구조와 대응한다(`07_수면과복구.md` 2.1절):

| 진동 대역 | 주파수 | 게이지 층 | 연산 역할 | 에너지 비율 |
|---|---|---|---|---|
| 감마 ($\gamma$) | 30-100 Hz | SU(3) | 지각 요소의 결합(binding) | $74.1\%$ |
| 베타 ($\beta$) | 13-30 Hz | SU(2) | 운동 계획, 인지 제어(decision) | $21.1\%$ |
| 알파 ($\alpha$) | 8-13 Hz | U(1) | 억제적 게이팅, 선택적 주의(attention) | $4.9\%$ |
| 세타/델타 | 0.5-8 Hz | $\Phi$ | 전역 안정화(smoothing) | 전역 |

이 구조를 네트워크 층에 직접 이식한다.

### 1.2 왜 균일 MLP/Attention이 비효율적인가

현재 Transformer의 FFN은 모든 채널을 동등하게 취급한다. CE 관점에서 이것은 "강력, 약력, 전자기력을 구분하지 않고 하나의 힘으로 다루는 것"에 해당한다.

뇌는 연산의 $74.1\%$를 결합(binding)에, $21.1\%$를 결정(decision)에, $4.9\%$를 주의(attention)에 할당한다. 이 비율은 CE 결합 상수 $\alpha_s : \alpha_w : \alpha_{em}$에서 연역되며, 자유 파라미터가 0이다.


## 2. 게이지 격자 FFN

설계 원리를 계산 그래프로 옮기려면 채널 분할, 전이, 혼합을 순서대로 정의해야 한다. 이 절의 모든 tensor는 무차원 활성·가중치이며, 표현력과 비용은 다음 절의 holdout 및 ablation 측정이 없으면 가설로 남는다.

### 2.1 기본 구조

표준 FFN $\text{FFN}(x) = \sigma(xW_1)W_2$를 3x3+1 격자로 대체한다.

$$
\text{Layer}(x)
= \underbrace{\text{Bind}(x_3)}_{\text{SU(3), 74.1\%}}
+ \underbrace{\text{Decide}(x_2)}_{\text{SU(2), 21.1\%}}
+ \underbrace{\text{Attend}(x_1)}_{\text{U(1), 4.9\%}}
+ \underbrace{\text{Smooth}(x)}_{\Phi}
$$

채널 분할:

$$d_3 : d_2 : d_1 = \alpha_s : \alpha_w : \alpha_{em} = 0.11789 : 0.03352 : 0.00775$$

$d = 768$일 때: $d_3 = 568$, $d_2 = 161$, $d_1 = 39$.

### 2.2 블록 대각 전이 행렬

채널 분할은 전이 행렬의 영 패턴으로 구현되며, 이 수식은 한 layer의 무차원 선형 사상을 정하는 정의다. 블록 대각만으로 표현력이 충분한지는 결론나지 않았으므로 다음 혼합항과 ablation이 그 퇴화 경우를 판정한다.

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

채널 비율이 정해지면 파라미터 수는 모델 폭과 혼합 rank의 함수로 비교할 수 있다. 아래 예시는 특정 폭의 산술 예시일 뿐 모든 규모에서의 메모리·지연 개선을 보장하지 않으며, 실제 이득은 같은 FLOP와 wall-clock 예산의 대조로 판정한다.

$d = 768$, $r_m = 96$:

- 블록 대각: $\sum_i 2 d_i h_i \approx 2.81\text{M}$
- 혼합: $2 d r_m \approx 0.15\text{M}$
- 총: $\approx 2.96\text{M}$, 표준 FFN $4.72\text{M}$ 대비 **37% 감소**

파라미터가 줄어도 성능이 보존되는 이유: CE 결합 비율이 정보 처리의 최적 분배를 반영하기 때문이다.


## 3. LBO 정규화 (LBONorm)

블록 표현이 과도하게 굽는지를 제어하려면 정규화 연산과 안정성 조건을 따로 둔다. 이 proxy는 구현상 곡률 측정이며, 뇌의 저주파 안정화와의 대응은 기능 비유로만 읽는다.

### 3.1 LayerNorm의 CE 확장

표준 LayerNorm에 라플라스-벨트라미 확산을 추가한다.

$$\hat{h} = \frac{h - \mu(h)}{\sigma(h)}, \qquad h' = (\hat{h} - \eta\,\Delta_g \hat{h}) \odot \gamma + \beta$$

여기서 $\Delta_g \hat{h} = \hat{h} - \hat{h}\,V^\top V$ (저랭크 확산), $V \in \mathbb{R}^{r \times d}$, $r \ll d$.

### 3.2 수렴 조건

확산 계수의 범위는 고정된 투영 행렬과 step 크기에서 안정적 반복을 위한 충분조건이다. $\eta$는 이 정규화 연산의 무차원 step이며, 비선형 블록·갱신되는 $V$와 결합한 전체 학습 수렴을 이 조건만으로 보장하지 않는다.

$$0 \leq \eta < \frac{1}{\lambda_{\max}(V^\top V)}$$

$\eta = 0$이면 표준 LayerNorm과 동일. $\eta > 0$이면 고곡률 성분이 추가로 감쇠된다.

### 3.3 곡률 에너지 저장

수렴 제약 다음에는 무엇을 측정하고 억제하는지 정의해야 한다. 아래 에너지는 활성 표현의 곡률 proxy이지 물리 에너지나 신경 대사량의 단위 측정이 아니며, 비유는 OOD 결과로만 반증 가능하다.

$$E_{\text{curv}} = \|\Delta_g \hat{h}\|^2 = \|(I - V^\top V)\hat{h}\|^2$$

이 값은 각 층에서 저장되어 (a) 곡률 정규화 손실에 사용되고, (b) 추론 시 환각 감지 신호로 사용된다.

### 3.4 뇌 대응

LBONorm은 뇌의 세타/델타 진동($\Phi$ 층)에 대응한다. 저주파 진동이 고주파 연산 결과를 전역적으로 안정화하는 것처럼, LBONorm은 고곡률 표현을 확산으로 평탄화한다.

---

## 4. 유니타리 제약 (Spectral Normalization)

정규화와 별개로 선형 투영의 증폭을 제한하면 어떤 안정성 조건을 코드에 넣는지 분명해진다. 제약의 효능은 hallucination·OOD·ablation 지표로 실패할 수 있으며 수식 자체가 성능을 보장하지 않는다.

### 4.1 원리

블록 혼합이 비정상 증폭을 만들지 않게 하려면 각 투영의 연산자 크기를 제한한다. 다음 조건은 지정한 선형 projection의 안정성 제약으로, 전체 모델의 환각률 감소나 장기 동역학 안정성을 증명하지 않는다.

$$|\det \mathbf{T}|^2 \leq 1 \quad \Longleftrightarrow \quad \sigma_1(W_{\text{proj}}) \leq 1$$

Attention 출력 사영 $W_{\text{proj}}$의 최대 특이값이 1 이하이면, 정보가 층을 지날 때 증폭되지 않는다.

### 4.2 환각 억제 메커니즘

정보 증폭이 차단되면 잘못된 패턴의 자기강화가 강하게 억제된다.

- 증폭 없음 $\to$ 국소 오류의 자기증폭 경향이 줄어듦 $\to$ 안정화 편향
- 기존 RLHF는 사후 교정이지만, 유니타리 제약은 사전 구조 제약

### 4.3 적용 방법

제약을 실제 학습에 넣을 때는 매 갱신 뒤 projection을 수행한다. 이는 구현 명세이며 singular value 추정 오차·계산 비용·optimizer 상호작용 때문에 ablation에서 이득이 사라질 수 있다.

$$W_{\text{proj}} \leftarrow W_{\text{proj}} / \sigma_1(W_{\text{proj}})$$

PyTorch `nn.utils.spectral_norm`으로 직접 적용. 기존 가중치의 방향은 보존하고 크기만 제한한다. 사전학습 가중치를 파괴하지 않는다.

### 4.4 뇌 대응

유니타리 조건은 뇌의 에너지 보존 (C2) 조건에 대응한다. 시냅스를 통해 전달되는 신호가 무한히 증폭될 수 없는 것은 신경계의 물리적 제약이다.

---

## 5. 곡률 정규화 손실

앞 연산을 과제 학습에 연결하려면 곡률 항의 계수와 시간 schedule을 명시해야 한다. 이 선택은 외부 hyperparameter이며, 특정 데이터셋의 최적값 또는 물리 상수라는 주장은 하지 않는다.

### 5.1 정의

곡률 proxy를 과제 손실과 함께 최적화하려면 두 항의 역할과 계수를 분리한 목적함수가 필요하다. $\lambda(t)$는 무차원 가중치이고 $t$는 training step이므로 생물학적 시간이나 CE 상수에서 자동으로 정해지는 수치가 아니다.

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot \frac{1}{L} \sum_{l=1}^{L} \|\Delta_g h_l\|^2$$

$L$: 층 수. 두 번째 항이 전체 네트워크의 평균 곡률 에너지를 억제한다.

### 5.2 스케줄

가중치의 시간 의존성은 초반 과제 적합과 후반 안정화 사이의 설계 절충이다. 아래 schedule은 사전 고정해 ablation의 기준으로 써야 하며, 다른 데이터셋·sequence 길이·optimizer에서의 최적성을 뜻하지 않는다.

$$\lambda(t) = \lambda_0 \cdot \min\!\left(1,\; \frac{t}{t_{\text{warmup}}}\right) \cdot \frac{1}{2}\!\left(1 + \cos\frac{\pi t}{t_{\max}}\right)$$

학습 초반에 곡률 정규화를 켜고, 후반에 감쇄한다. 이유: 학습 후반에 lr이 감소하면 곡률 항이 상대적으로 지배적이 되어 task loss 최적점에서 이탈한다.

### 5.3 뇌 대응

곡률 정규화 손실은 뇌의 ACC(전측 대상피질) 기능에 대응한다. ACC는 인지적 갈등/오류를 감지하여 PFC에 보정 신호를 보낸다. 곡률 에너지의 증가가 "인지적 갈등"이고, 정규화가 "보정"이다.

---

## 6. 교차 주파수 결합

블록 사이의 정보 교환은 혼합 rank와 별도의 결합 규칙을 요구한다. 여기서 주파수 명칭은 활성의 역할을 가리키는 비유이고, 측정된 신경 주파수와 모델 layer timebase의 등식은 아니다.

### 6.1 원리

뇌에서 세타 위상이 감마 진폭을 변조하는 교차 주파수 결합(CFC)은 CE의 비최소 결합 $\xi R\Phi^2$에 대응한다(`07_수면과복구.md` 2.2절):

$$\Phi \text{ (세타/델타)} \leftrightarrow \text{SU(3) (감마)}: \quad \xi R\Phi^2$$

### 6.2 아키텍처 구현

LBONorm의 곡률 에너지가 GaugeLattice의 각 게이지 채널에 영향을 미치도록 결합한다.

$$\mathcal{T}_i^{\text{coupled}}(x_i) = \mathcal{T}_i(x_i) \cdot (1 - \xi \cdot E_{\text{curv}})$$

$\xi = \alpha_s^{1/3} = 0.490$ (CE 예측). 곡률이 높으면 게이지 채널의 출력이 감쇠된다.

이 결합은 "불안정한 상태에서 연산 강도를 줄여라"는 안전 메커니즘이다.

---

## 7. 전체 블록 구조

지금까지의 블록·제약·손실은 하나의 layer 인터페이스에서 함께 작동할 때만 검증할 수 있다. 다음 의사코드는 입력 tensor에서 출력 tensor로의 순서를 보이는 명세이며 실제 latency·메모리·정확도는 hardware timebase를 고정한 측정으로 분리한다.

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
- 교차 주파수 결합 ($\xi \cdot E_{\text{curv}}$ 게이트)
- 수면-각성 학습 순환 (3장)

---

## 8. 파라미터 자유도 분석

마지막으로 CE에서 동기를 얻은 비율과 구현자가 선택해야 하는 자유도를 분리한다. 표의 고정 여부는 형식적 지위가 아니라 설계 인터페이스 상태이며, 남은 자유도는 holdout과 OOD sweep에서 실패할 수 있는 명시적 가설이다.

| 파라미터 | 결정 방법 | 자유도 |
|---|---|---|
| 채널 비율 $d_3 : d_2 : d_1$ | CE 결합 상수 $\alpha_s : \alpha_w : \alpha_{em}$ | 0 |
| 유니타리 임계값 | $\sigma_1 \leq 1$ (CE 공리) | 0 |
| LBO 랭크 $r$ | $r \sim d/8$ (경험적, 추가 연구 필요) | 1 |
| LBO 스텝 $\eta$ | 학습 가능 파라미터 | 0 (학습) |
| 혼합 랭크 $r_m$ | $r_m \sim d/8$ (경험적) | 1 |
| 곡률 $\lambda_0$ | 하이퍼파라미터 | 1 |
| CFC 결합 $\xi$ | $\alpha_s^{1/3} = 0.490$ (CE 연역) | 0 |

CE에서 연역되는 파라미터: 4개. 설계자가 선택하는 파라미터: 3개. 기존 Transformer 아키텍처 탐색(층 수, head 수, FFN 배수, 학습률, ...)에 비해 자유도가 극적으로 줄어든다.
