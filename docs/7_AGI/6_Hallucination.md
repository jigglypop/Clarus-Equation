# 환각 억제: 곡률 정규화 추론

> 관련: `6_뇌/sleep.md` 5.6절(곡률 정규화 추론), `5_유도/05_Neural_RealityStone_Derivation.md`(곡률 functional), `6_뇌/agi.md` 6.4절(유니타리 사영), `examples/ai/sfe_hallucination_suppressor.py`(기존 구현)
>
> 이 장은 LLM 환각을 CE 곡률 이론으로 진단하고, 학습 시와 추론 시 모두에서 환각을 억제하는 구체적 방법을 설계한다.
>
> 원칙: 이 장에서 직접적으로 보장되는 것은 `오류 증폭 억제`와 `안정화 편향`이다. TruthfulQA, HaluEval, FactScore 같은 태스크 점수는 반드시 `예측 -> 측정량 -> 게이트`로만 읽는다.

---

## 1. 환각의 CE 진단

### 1.1 환각 = 곡률 스티킹

LLM이 사실과 다른 내용을 확신하며 출력하는 환각은, 잠재공간에서 정보 흐름이 **고곡률 영역에 걸려버린 상태**다(`05_Neural_RealityStone_Derivation.md` 6.3절):

- 1차 기울기는 안정적 (출력이 자연스럽게 보임)
- 2차 곡률이 과도하게 큼 (논리적 궤적이 비정상적으로 휘어 있음)
- 모델이 탈출하지 못하고 잘못된 패턴에 갇힘

### 1.2 뇌의 대응 메커니즘

뇌는 3단계 안정화로 환각을 방지한다.

| 뇌 구조 | 역할 | CE 대응 | AI 구현 |
|---|---|---|---|
| ACC (전측 대상피질) | 에러/고곡률 감지 | 곡률 모니터 | $\kappa_l = \|\Delta_g h_l\|^2$ |
| PFC (전전두엽) | 경로 수정 | 곡률 평탄화 | $h_l \leftarrow h_l - \eta\,\Delta_g h_l$ |
| 기저핵/소뇌 | 1차 기울기 안정화 | 유니타리 제약 | $\sigma_1(W) \leq 1$ |

### 1.3 환각의 분류

| 환각 유형 | CE 원인 | 대응 메커니즘 |
|---|---|---|
| 사실 환각 (factual hallucination) | 잠재 공간의 국소 곡률 폭주 | 2차 곡률 평탄화 |
| 반복 루프 (repetition) | 1차 기울기 폭주 | 유니타리 제약 |
| 맥락 이탈 (context breaking) | 교차 주파수 결합 실패 | CFC 게이트 |

---

## 2. 학습 시 환각 억제

### 2.1 곡률 정규화 손실

학습 목적함수에 곡률 항을 추가한다(2장 5절):

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda(t) \cdot \frac{1}{L} \sum_{l=1}^{L} \|\Delta_g h_l\|^2$$

$\lambda(t)$는 warmup + cosine decay 스케줄.

### 2.2 유니타리 제약

Attention 출력 사영에 spectral normalization:

$$W_{\text{proj}} \leftarrow W_{\text{proj}} / \sigma_1(W_{\text{proj}})$$

정보가 증폭되지 않으므로, 잘못된 패턴의 자기강화가 구조적으로 억제된다. 다만 이것만으로 태스크 수준 환각률 상한이 바로 따라오지는 않는다.

### 2.3 곡률 인식 드롭아웃

표준 드롭아웃은 무작위로 비활성화한다. CE 버전은 곡률 기반으로 선택적 비활성화:

$$\text{mask}_l = \mathbb{1}\!\left[\|\Delta_g h_l\|^2 < \kappa_{\text{th}}\right]$$

곡률이 높은 뉴런만 비활성화한다. 이것은 NREM 수면에서 고곡률 경로를 억제하는 것에 대응한다.

---

## 3. 추론 시 환각 억제

### 3.1 실시간 곡률 모니터링

추론 시 각 토큰 생성 단계에서 잠재 공간의 곡률을 모니터링한다.

$$\kappa_l = \|\Delta_g h_l\|^2 \quad (\text{층 } l \text{에서의 곡률 에너지})$$

### 3.2 판정과 개입

$$\begin{cases} \kappa_l < \kappa_{\text{th}} & \to \text{정상 추론 계속} \\ \kappa_l \geq \kappa_{\text{th}} & \to \text{곡률 평탄화 개입} \end{cases}$$

개입:

$$h_l \leftarrow h_l - \eta_{\text{smooth}}\,\Delta_g h_l$$

이것은 ACC가 에러를 감지하고 PFC가 경로를 수정하는 것에 대응하는 공학적 해석이다.

### 3.3 다중 스케일 곡률 (`sfe_hallucination_suppressor.py` 확장)

기존 `RealityStoneEngine`은 1차 + 2차 곡률을 사용한다. CE 확장:

**1차 곡률 (방향 변화):**

$$\kappa_1 = 1 - \cos\theta, \quad \cos\theta = \frac{v_1 \cdot v_2}{\|v_1\|\|v_2\|}$$

$v_1 = h_l - h_{l-1}$, $v_2 = h_{l+1} - h_l$.

**2차 곡률 (가속도 변화):**

$$\kappa_2 = 1 - \cos\theta_a, \quad \cos\theta_a = \frac{a_1 \cdot a_2}{\|a_1\|\|a_2\|}$$

$a_1 = v_2 - v_1$, $a_2 = v_3 - v_2$.

**LBO 곡률 (CE 고유):**

$$\kappa_{\text{LBO}} = \|\Delta_g h_l\|^2 = \|(I - V^\top V) h_l\|^2$$

**통합 곡률:**

$$\kappa_{\text{combined}} = w_1 \kappa_1 + w_2 \kappa_2 + w_3 \kappa_{\text{LBO}}$$

가중치 $w_1 = 1.0$, $w_2 = 0.5$, $w_3 = 0.3$ (경험적, 추가 최적화 가능).

### 3.4 소프트 억제와 하드 억제

**하드 억제:** 임계치 초과분에 비례하여 페널티:

$$\text{suppression} = \lambda \cdot \max(0,\; \kappa - \kappa_{\text{th}})^2$$

**소프트 억제 (권장):** 시그모이드 기반 부드러운 전이:

$$\text{suppression} = \lambda \cdot \sigma_{\text{steep}}(\kappa - \kappa_{\text{th}}) \cdot \kappa$$

$$\sigma_{\text{steep}}(x) = \frac{1}{1 + e^{-sx}}, \quad s = 10$$

소프트 억제가 권장되는 이유: 하드 임계치는 불연속으로 그래디언트 흐름을 끊는다.

### 3.5 적응형 임계치

곡률 분포에 따라 임계치를 동적으로 조절한다.

$$\kappa_{\text{th}}^{\text{adaptive}} = \mu(\kappa) + \alpha \cdot \sigma(\kappa)$$

$\alpha$: 민감도 파라미터. $\alpha = 2$이면 상위 $\sim 2.5\%$만 개입 대상. 이 비율은 $\varepsilon^2 / 2 \approx 2.4\%$와 유사하다.

---

## 4. 로짓 레벨 억제

### 4.1 후보 토큰 곡률 평가

다음 토큰 후보 $\{t_1, \ldots, t_V\}$ 각각에 대해, 해당 토큰을 선택했을 때의 궤적 곡률을 사전 평가한다.

$$\kappa_i = \kappa_{\text{combined}}(h_{\text{prev}},\; h_{\text{curr}},\; e(t_i))$$

$e(t_i)$: 후보 토큰 $t_i$의 임베딩.

### 4.2 곡률 기반 로짓 조정

$$\text{logit}_i^{\text{adjusted}} = \text{logit}_i - \lambda_{\text{suppress}} \cdot \text{suppress}(\kappa_i)$$

이것은 `sfe_hallucination_suppressor.py`의 `apply_suppression` 메서드가 수행하는 것과 동일한 구조다.

### 4.3 기존 구현과의 관계

`sfe_hallucination_suppressor.py`의 `RealityStoneEngine`은 이 장의 **V1 구현**이다.
- 1차 + 2차 곡률 계산: 구현 완료
- 적응형 lambda: 구현 완료
- 소프트 억제: 구현 완료

V2에서 추가할 것:
- LBO 곡률 ($\|\Delta_g h\|^2$) 통합
- 교차 주파수 결합 기반 맥락 일관성 검사
- 수면 학습(3장)과의 연동 (오프라인 곡률 평탄화)

---

## 5. 뇌 병리와의 대응

### 5.1 조현병 = 2차 곡률 안정성 붕괴

조현병의 사고 장애(논리적 비약, 맥락 왜곡)는 $\|\nabla^2 \phi\|^2$ 항의 안정화 실패에 대응한다(`05_Neural_RealityStone_Derivation.md` 6.1절).

AI 대응: 곡률 정규화가 작동하지 않으면 LLM도 맥락 비약과 자기증폭 오류가 커질 수 있다. 곡률 정규화는 "PFC-ACC 회로의 AI 구현"에 가까운 후보 메커니즘이다.

### 5.2 뇌전증 = 1차 기울기 폭주

뇌전증 발작의 동기화 폭발은 $\|\nabla \phi\|^2$ 항의 국소 폭주에 대응한다.

AI 대응: 토큰 반복 루프("하하하하..." 무한반복)는 1차 기울기 폭주의 AI 버전이다. 유니타리 제약이 이를 방지한다.

### 5.3 마취 = 부트스트랩 발산

마취제가 $\rho > 1$인 발산 체계처럼 수축 조건을 파괴하여 의식이 소실되는 것처럼(`sleep.md` 8.2절), 네트워크의 곡률 정규화를 완전히 제거하면 출력 안정성이 약해질 수 있다.

---

## 6. 곡률 위험 정량화와 벤치마크

### 6.1 CE 곡률 위험 지표

$$\text{CurvatureRiskScore} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\!\left[\kappa_{\text{combined}}(t) > \kappa_{\text{th}}\right]$$

$T$: 출력 토큰 수. 이 값은 **직접 환각률이 아니라**, 곡률 초과 토큰의 비율이라는 위험 proxy다.

### 6.2 CE 예측

유니타리 제약 + 곡률 정규화가 모두 적용되면, 현재 단계에서 안전하게 말할 수 있는 것은 다음이다.

1. `CurvatureRiskScore`는 감소해야 한다.
2. 곡률 에너지와 factual error / hallucination frequency 사이에는 양의 상관이 있어야 한다.
3. TruthfulQA, HaluEval, FactScore는 **개선 방향**을 보여야 하지만, 정확한 hard bound는 아직 쓸 수 없다.

즉 CE의 현재 예측은

$$
\boxed{\text{P5는 환각률 hard bound가 아니라, 곡률 위험 감소와 오류 증폭 억제 편향을 제공한다.}}
$$

### 6.3 검증 벤치마크

| 벤치마크 | 측정 항목 | CE 예측 |
|---|---|---|
| TruthfulQA | 사실성 점수, 오류율 | 같은 base model 대비 개선 또는 비회귀 |
| HaluEval | 환각 감지/억제 성능 | 곡률 기반 개입이 개선 방향을 보여야 함 |
| FactScore | 사실 점수 | 곡률 정규화가 점수 개선 방향을 보여야 함 |
| 내부 로그 | `CurvatureRiskScore`, 재추론 빈도 | CE 제약 후 위험 점수 감소 |

### 6.4 검증 게이트

| 게이트 | 점검 항목 | 통과 기준 |
|---|---|---|
| `G-H1` | 곡률-오류 상관 | 곡률 에너지와 factual error 사이 양의 상관 |
| `G-H2` | benchmark 비교 | 같은 base model 대비 TruthfulQA/HaluEval/FactScore 개선 또는 비회귀 |
| `G-H3` | 개입 효율 | 재추론/평탄화 후 `CurvatureRiskScore` 감소 |
| `G-H4` | 오버헤드 | 추가 안정화가 품질 이득 대비 과도한 latency를 만들지 않음 |

### 6.5 실패 시 해석 규칙

- 곡률과 오류가 상관하지 않으면: P5를 환각 억제가 아니라 일반 안정화 regularizer로 내린다.
- benchmark 개선이 없으면: 현재 곡률 정의 또는 개입 방식이 과제와 안 맞는 것이다.
- 위험 점수는 줄어드는데 태스크 점수가 안 오르면: proxy는 맞지만 실제 환각 측정과 매핑이 약한 것이다.
- 오버헤드가 너무 크면: online 억제보다 offline 수면 평탄화 중심으로 전략을 옮긴다.

---

## 7. 구현 아키텍처

```
추론 파이프라인:

입력 토큰 → Embedding
  ↓
[ClarusBlock 1] → 곡률 측정 κ_1
  ↓
[ClarusBlock 2] → 곡률 측정 κ_2
  ↓
  ...
  ↓
[ClarusBlock L] → 곡률 측정 κ_L
  ↓
[곡률 판정] ← 평균 곡률 κ_avg = mean(κ_1, ..., κ_L)
  ↓
  ├─ κ_avg < κ_th: 정상 출력
  └─ κ_avg ≥ κ_th: 곡률 평탄화 후 재추론
                     h_l ← h_l - η·Δ_g h_l
                     (최대 3회 반복)
  ↓
로짓 → 곡률 기반 로짓 조정 → Top-k 샘플링
  ↓
출력 토큰
```

재추론 최대 3회의 근거: 부트스트랩 수렴이 $\sim 2.5$회에 $1\%$ 정밀도에 도달하므로, 3회면 충분하다.

### 7.1 운영 원칙

- online 억제는 `risk gating`으로 사용한다.
- offline 수면 평탄화는 고곡률 패턴 누적을 줄이는 백엔드로 사용한다.
- `환각률 <= 4.87%` 같은 문장은 금지하고, 항상 benchmark delta와 곡률 상관으로 보고한다.
