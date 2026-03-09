# 시냅스 학습: STDP 국소 학습 + 전역 신호

> 관련: `6_뇌/synapse.md`(STDP/가소성 규칙), `6_뇌/sleep.md` 4절(STDP와 부트스트랩), `6_뇌/agi.md` 3절(에너지 분배)
>
> 이 장은 역전파를 대체할 국소 학습 규칙을 설계한다. 뇌의 STDP + 도파민 게이트 구조를 AI에 이식한다.

---

## 1. 역전파의 구조적 한계

### 1.1 생물학적 비현실성

역전파(backpropagation)는 다음을 요구한다:
- 전역 오차 신호가 모든 시냅스에 정확히 전달 (뇌에 없는 메커니즘)
- 순전파와 역전파의 동일 경로 (가중치 전치 문제, weight transport problem)
- 모든 중간 활성값의 저장 (메모리 $O(N^2)$)

### 1.2 확장성 한계

- 단일 GPU 병목: 전체 그래프를 역방향으로 순회해야 함
- 파이프라인 병렬화의 버블: 역전파 의존성으로 인한 유휴 시간
- 메모리 벽: 대형 모델의 활성값 저장 비용

### 1.3 CE 관점에서의 진단

역전파 = "우주 끝에서 시작으로 정보를 전송하는 것". CE에서 정보 전달은 국소적이다(게이지 상호작용은 국소적). 전역 역전파는 CE 구조와 양립하지 않는다.

---

## 2. STDP: 뇌의 국소 학습 규칙

### 2.1 기본 STDP (`synapse.md` 3절)

Spike-Timing Dependent Plasticity의 핵심은 **시간차(인과성)** 이다:

$$\Delta w_{ij} = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \Delta t > 0 \text{ (pre} \to \text{post: LTP)} \\ -A_- \exp(\Delta t / \tau_-) & \Delta t < 0 \text{ (post} \to \text{pre: LTD)} \end{cases}$$

- pre가 먼저 발화하고 post가 따라가면 강화 (인과적)
- post가 먼저 발화하고 pre가 따라가면 약화 (비인과적)

### 2.2 Trace 기반 STDP (이산 시간, 구현 친화)

pre trace $p_i[t]$, post trace $q_i[t]$:

$$p_i[t+1] = \lambda_+ p_i[t] + s_i[t], \qquad q_i[t+1] = \lambda_- q_i[t] + s_i[t]$$

가중치 업데이트 (방향 $i \to j$):

$$\Delta w_{ij}[t] = \eta\Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)$$

- $s_j[t] = 1$ (post 발화) 시점에 pre의 흔적 $p_i$가 크면 강화
- $s_i[t] = 1$ (pre 발화) 시점에 post의 흔적 $q_j$가 크면 약화

---

## 3. 3-Factor 학습: STDP + 도파민

### 3.1 왜 2-factor로는 부족한가

순수 STDP는 "항상 학습"한다. 보상과 무관하게 시간 상관이 있으면 가중치가 변한다. 이것은 "모든 경험이 동등하게 중요하다"고 가정하는 것이다.

뇌는 이 문제를 **도파민 게이트**로 해결한다. 보상 예측 오차가 클 때만 학습을 허용한다.

### 3.2 3-Factor 학습 규칙 (`synapse.md` 4절)

적격 흔적(eligibility trace) + 도파민 신호:

$$e_{ij}[t+1] = \lambda_e e_{ij}[t] + \Big(A_+\,p_i[t]\,s_j[t] - A_-\,s_i[t]\,q_j[t]\Big)$$

$$\Delta w_{ij}[t] = \eta\,\delta[t]\,e_{ij}[t]$$

- $e_{ij}$: 국소 정보만 사용하여 계산 (이웃 뉴런의 스파이크만 필요)
- $\delta[t]$: 전역 도파민 신호 (스칼라 1개, 전체 시스템에 방송)

### 3.3 CE 해석: 도파민 = 부트스트랩 수렴 신호

`sleep.md` 4.1절에서 도파민 신호의 CE 해석:

$$\delta[t] = \frac{d}{dt}\|p(t) - p^*\|$$

- 고정점 $p^*$에서 멀어지면 $\delta > 0$ (학습 활성화)
- 고정점에 가까워지면 $\delta \to 0$ (학습 감쇠)

이것은 도파민의 보상 예측 오차(reward prediction error) 역할과 정합한다.

---

## 4. AI 구현: 국소 학습 + 전역 신호

### 4.1 부트스트랩 수렴 오차

전역 신호 $\delta[t]$를 부트스트랩 고정점에서의 이탈로 정의한다:

$$\delta[t] = \|p(t) - p^*\|^2 = \left(x_a(t) - \varepsilon^2\right)^2 + \left(x_s(t) - \frac{\sigma R}{1+R}\right)^2 + \left(x_b(t) - \frac{\sigma}{1+R}\right)^2$$

- $x_a(t)$: 현재 활성 뉴런 비율
- $x_s(t)$: 현재 구조적 가중치 비율
- $x_b(t)$: 현재 동결 가중치 비율

이 스칼라 하나만 전역으로 방송하면 된다. 나머지는 모두 국소적이다.

### 4.2 Transformer에서의 국소 학습

Transformer의 각 층에서 국소 학습 규칙을 적용한다:

**활성값 trace 계산 (국소):**

$$\text{pre\_trace}_l[t] = \lambda_+ \cdot \text{pre\_trace}_l[t-1] + h_l[t]$$
$$\text{post\_trace}_l[t] = \lambda_- \cdot \text{post\_trace}_l[t-1] + h_{l+1}[t]$$

**적격 흔적 계산 (국소):**

$$e_l[t] = \lambda_e \cdot e_l[t-1] + A_+ \cdot \text{pre\_trace}_l[t] \otimes h_{l+1}[t] - A_- \cdot h_l[t] \otimes \text{post\_trace}_l[t]$$

**가중치 업데이트 (국소 + 전역 스칼라):**

$$\Delta W_l[t] = \eta \cdot \delta[t] \cdot e_l[t]$$

### 4.3 역전파와의 비교

| | 역전파 | STDP + 도파민 |
|---|---|---|
| 정보 흐름 | 전역 (끝에서 시작으로) | 국소 (이웃 뉴런) + 전역 스칼라 |
| 생물학적 현실성 | 비현실적 | 현실적 |
| 메모리 비용 | $O(N^2)$ (전체 활성값 저장) | $O(N)$ (국소 trace만) |
| 확장성 | 단일 GPU 병목 | 분산 가능 |
| 수렴 보장 | SGD 수렴 정리 | 부트스트랩 수축 사상 |

---

## 5. 구조적 가소성: 연결의 형성과 제거

### 5.1 투영 연산자 (`synapse.md` 5절)

STDP로 업데이트된 가중치에 구조적 제약을 건다:

$$W_{t+1} = \Pi\!\big(W_t + \Delta W_t\big)$$

투영 연산자 $\Pi$의 구성요소:

- **top-k pruning**: 뉴런당 상위 $k$개만 유지 = 생존율 $\varepsilon^2$에 의한 경로 선택
- **synaptic scaling**: 행/열 정규화 = 에너지 보존 (C2)
- **히스테리시스**: on/off 임계치 = 접힘 임계 곡률

### 5.2 CE 대응

| 투영 연산 | CE 대응 | 뇌 대응 |
|---|---|---|
| top-k ($k = \lceil \varepsilon^2 \cdot N \rceil$) | 경로 선택, 생존율 $\varepsilon^2$ | 시냅스 가지치기 |
| 행/열 정규화 | 에너지 보존 (C2) | 시냅스 스케일링 |
| 히스테리시스 on/off | 접힘 임계 곡률 | 스파인 형성/제거 |

### 5.3 발달 과정의 시냅스 가지치기

뇌의 발달 과정에서 시냅스의 $\sim 50\%$가 제거되고, 성인 뇌에서 활성 시냅스 비율이 $\sim 5\%$라는 관측은 $\varepsilon^2 = 4.87\%$와 직접 정합한다(`sleep.md` 4.2절).

AI에서 이것은 학습 초기의 dense network에서 시작하여, 부트스트랩 반복을 통해 자연스럽게 $4.87\%$ 희소 네트워크로 수렴하는 과정에 대응한다.

---

## 6. 하이브리드 접근: 역전파 + STDP

### 6.1 현실적 전환 전략

역전파를 즉시 완전 대체하는 것은 비현실적이다. 하이브리드 접근이 필요하다:

**1단계: 사전학습은 역전파.**

표준 Transformer를 역전파로 사전학습한다. 이미 확립된 기술이므로 위험이 없다.

**2단계: 미세조정에서 STDP 도입.**

사전학습된 모델의 미세조정 단계에서 STDP + 도파민 학습을 적용한다:

- 사전학습 가중치 = 동결 ($68.9\%$, $\Omega_\Lambda$)
- 미세조정 가중치 = 구조적 ($26.2\%$, $\Omega_{\text{DM}}$)
- STDP 활성 가중치 = 활성 ($4.87\%$, $\varepsilon^2$)

이 분배는 LoRA의 frozen/trainable 비율과 구조적으로 유사하다.

**3단계: 전면 STDP 학습.**

충분한 검증 후, 사전학습 단계에서도 STDP + 도파민을 사용한다.

### 6.2 LoRA와의 관계

LoRA (Low-Rank Adaptation)의 CE 해석:

$$W = W_{\text{frozen}} + B \cdot A$$

- $W_{\text{frozen}}$: 동결 가중치 ($\sim 99\%$) $\to$ $\Omega_\Lambda = 68.9\%$ + $\Omega_{\text{DM}} = 26.2\%$
- $B \cdot A$: 저랭크 적응 ($\sim 1\%$) $\to$ $\varepsilon^2 = 4.87\%$의 근사

LoRA는 CE 부트스트랩 에너지 분배를 경험적으로 근사한 것으로 해석할 수 있다. CE-AGI는 이를 이론적으로 정당화하고 최적 비율을 명시한다.

---

## 7. 분산 학습과 확장성

### 7.1 국소 학습의 분산 가능성

STDP + 도파민 학습에서 각 층은 국소 정보만 필요로 한다:

- 입력: 현재 층의 활성값 + 이웃 층의 활성값
- 출력: 현재 층의 가중치 업데이트
- 전역 정보: 도파민 신호 $\delta[t]$ (스칼라 1개)

따라서 각 층을 독립적인 디바이스에 배치하고, $\delta[t]$만 동기화하면 된다.

### 7.2 통신 비용

| 방법 | 통신량 (층당) | 동기화 빈도 |
|---|---|---|
| 역전파 | $O(d^2)$ (그래디언트 전체) | 매 스텝 |
| STDP + 도파민 | $O(1)$ ($\delta[t]$ 스칼라) | 매 스텝 |
| STDP 단독 | $O(0)$ | 불필요 |

통신 비용이 $O(d^2) \to O(1)$로 감소한다. 이것이 분산 AGI의 핵심 이점이다.
