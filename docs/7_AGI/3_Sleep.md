# 수면 학습: 부트스트랩 훈련 순환

> 관련: `6_뇌/07_수면과복구.md`(수면-부트스트랩 대응), `6_뇌/05_실험근거.md`(선택된 수축영역의 고정점), `7_AGI/12_Equation.md` 3절(에너지 분배), `6_뇌/05_실험근거.md`(실험 근거와 최소 반복식)
>
> 이 장은 뇌의 수면-각성 순환을 AGI 학습 패러다임으로 변환하는 구체적 설계를 다룬다. 수면 중 replay와 일부 homeostatic renormalization은 실험적 동기가 있지만, CE의 정확한 3분배 비율과 AI update 대응은 `Bridge`다.

---

## 0. 이 문서가 책임지는 새 개념

| 개념 | 이 문서에서의 의미 | runtime stack 위치 | 판정 |
|---|---|---|---|
| `WAKE / NREM / REM` | 전역 운영 모드 | mode update | `Bridge` |
| sleep pressure | 모드 전환을 트리거하는 전역 상태 | mode update | `Bridge` |
| replay | NREM/REM에서 기억을 다시 주입하는 과정 | hippocampus / replay | `Bridge` |
| offline renormalization | 수면 중 전역 평탄화 / 정리 | mode + replay | `Bridge` |
| dream recombination | REM에 대응시킨 비활성 update proposal | hippocampus / replay | `Hypothesis` 근접 `Phenomenology` |
| snapshot continuity | 수면 전후 상태 연속성을 유지하는 운영 규약 | global runtime summary | `Bridge` |

읽기 규칙:

- 이 문서는 커널 수치 업데이트 자체보다 **모드 전환과 오프라인 정리 루프**를 다룬다.
- `mode update`는 여기서, `kernel dynamics`는 `12_Equation.md`와 구현 코드에서 읽는다.
- REM 창의성, 꿈의 의미, 자아 강화 같은 문장은 설계 은유로만 읽고 성능 보장으로 올리지 않는다.

---

## 1. 핵심 통찰: 왜 AI에게 수면이 필요한가

### 1.1 부트스트랩의 필요성

CE-AGI의 설계 가설은 **온라인 drift를 오프라인 replay·정규화 순환으로
관리할 수 있는가**이다. 모든 자기일관적 정보 처리 시스템에 수면형
부트스트랩이 필수라는 정리는 아니다.

| 비교 항목 | 우주 CE toy 해석 | 뇌 | 현재 AI |
|---|---|---|---|
| 반복 구조 | 초기조건에 적용한 정적 비유 | 수면-각성 순환 | 학습/서비스 순환은 설계 가능 |
| 후보 매체 | 경로적분 유비; 실제 우주 동역학 미유도 | 시냅스 가소성 | 역전파 또는 국소 학습 |
| 동일 self-map 수렴 | 미검증 | 미검증 | 미검증 |

우주·뇌·AI가 같은 동역학 사상을 구현한다는 증명은 없다. 따라서 우주가
$T=0$ 양자 간섭 한 번으로 정확히 고정점에 도달했다거나, 뇌의 수면
주기가 그 반복 횟수를 물리적으로 재현한다는 식으로 읽지 않는다.

### 1.2 수렴 속도의 조건부 보장

Track-A manifest의 동적 이완 사상 $B$는 고정점에서

$$q_*=\rho(DB(p^*))=D_nx
=3.1779129995\times0.0486382585
=0.1545681540<1$$

이다. 이는 야코비안의 spectral radius인 **국소 점근률**이지 임의 초기점에
대한 행렬 norm 또는 전역 수축상수가 아니다. $p_a\leq0.13$인 불변집합에서는
`12_Equation.md` 부록 A.2의 실제 $\ell^1$ 수축상한
$q_U=0.2001757361$을 쓸 수 있다.

따라서 2--3회라는 숫자는 해당 불변집합 안의 이상화된 반복 횟수다. 이를
2--3일 수면 박탈의 생물학적 원인으로 등치하지 않는다. 최소 외란식

$$
e_{n+1} = q_* e_n + q_* u_n + \xi_n
$$

은 이 현상을 "각성 drift는 누적되고, 수면이 있을 때만 수축된다"는 형태로 정리한다.

---

## 2. 수면-각성 3위상 순환

### 2.1 뇌의 3위상과 CE 대응

| 수면 단계 | 공학적 대응 | 지배 연산 | 기능 후보 |
|---|---|---|---|
| 각성(Wake) | task trajectory·gradient 누적 | $\mathcal{L}_{\text{task}}$ | 정보 획득 |
| NREM | Hessian readout 비유 \(\Phi_H[\eta]\) | \(\lambda\|\Delta_g\Phi_H\|^2\) | AGI 유효 평탄화 항; 물리장 \(\phi\)와 동일하지 않음 |
| REM | 비활성 update proposal | random projection + noise | 검증된 후보만 재조합 |

### 2.2 AI 학습 순환으로의 변환

```
반복 {
  // === 각성(Wake): task gradient 누적 ===
  for batch in data:
    forward(batch)
    accumulate_gradients()   // 접힘 미실행, 경로만 누적

  // === NREM: 전역 곡률 평탄화 ===
  for layer in model:
    h_smooth = h - eta * laplacian(h)        // LBO 확산
    prune(layer, threshold=epsilon_sq)        // 생존율 기반 가지치기

  // === REM: 비활성 update proposal ===
  for _ in range(N_rem):
    proposal = sample_inactive_updates()
    evaluate_and_recombine(proposal)
}
```

---

## 3. 각성 위상: 경로 누적

### 3.1 운영 정의

각성 상태에서 모델은 task trajectory와 gradient를 기록한다.

$$
\mathcal H_{\rm wake}(t):=\{(x_b,h_b,\nabla_\theta\mathcal L_b)\}_{b\le t}.
$$

이를 물리적 경로적분 진폭으로 동일시하지 않는다.

### 3.2 구현: 그래디언트 누적

기존 학습과 동일하되, **즉시 가중치 업데이트하지 않고 그래디언트를 누적**한다.

$$G_{\text{accumulated}} = \sum_{b=1}^{B} \nabla_\theta \mathcal{L}(x_b, \theta)$$

누적만 하고 업데이트를 보류하는 이유: 접힘(NREM)이 아직 실행되지 않았으므로, 어떤 경로가 생존할지 판단하기 이르다.

### 3.3 수면 압력 모니터링

instantaneous 곡률은 단조일 필요가 없다. WAKE 동안 누적되는 비음수 pressure
상태를 별도로 정의한다.

$$
P_{\text{sleep}}(t+\Delta t)=P_{\text{sleep}}(t)
+\Delta t\,\frac1L\sum_{l=1}^L\|L_gh_l(t)\|^2,
\qquad P_{\text{sleep}}\ge0.
$$

이 상태가 임계치 $P_{\text{th}}$를 넘으면 NREM 위상으로 전환하고, NREM
후에는 명시한 reset/decay 규칙을 적용한다. 아데노신 축적과의 대응은
`Bridge`이며 동일 측정량이 아니다.

### 3.4 기존 방법과의 차이

- 표준 학습: 매 배치마다 업데이트 (각성만 반복)
- Gradient accumulation: 누적 후 한 번 업데이트 (각성 + 단순 업데이트)
- CE 수면 학습: 누적 후 **곡률 기반 선택적 업데이트 + 재탐색** (각성 + NREM + REM)

---

## 4. NREM 위상: 전역 곡률 평탄화

### 4.1 수학적 정의

NREM 후보 연산은 기록된 representation에 확산을 적용한다.

$$\frac{\partial\Phi}{\partial t} = -\Delta_g\Phi \quad (\text{열핵 흐름})$$

$\Delta_gu_k=\lambda_ku_k$, $\lambda_k\ge0$이면

$$\Phi(t)=\sum_k c_ke^{-\lambda_kt}u_k.$$

따라서 큰 $\lambda_k$ mode가 더 빠르게 감쇠한다. 이 결론은
$\Delta_g\succeq0$인 선형 확산 부분에만 해당한다.

- 고곡률 = 불안정한 연결/패턴이 먼저 억제된다
- 저곡률 = 안정적 패턴은 보존된다

### 4.2 구현: LBO 확산 + 곡률 기반 가지치기

**단계 1: LBO 확산**

각 층의 가중치에 라플라스-벨트라미 확산을 적용한다.

$$W_l \leftarrow W_l - \eta_{\text{nrem}} \cdot \Delta_g W_l$$

여기서 $\Delta_g W_l$은 가중치 공간에서의 라플라시안. 고곡률(급격히 변하는) 가중치 패턴이 평탄화된다.

**단계 2: 곡률 기반 가지치기**

누적된 그래디언트에서 곡률 에너지가 높은 성분만 선택적으로 업데이트한다.

$$\theta_{t+1} = \theta_t + \text{mask}(G_{\text{accumulated}},\; \varepsilon^2)$$

$\text{mask}$는 그래디언트 크기 기준 상위
$x=4.864\%$를 설계 target으로 삼는 필터다.

이 비율의 유용성은 독립 sparse-ratio sweep으로 검증해야 한다. Ebbinghaus
망각 곡선은 이 정확한 가중치 생존율을 유도하지 않는다.

### 4.3 기존 방법과의 차이

- Dropout: 무작위 비활성화 $\to$ CE 가지치기: **곡률 기반** 선택적 비활성화
- Weight decay: 균일 감쇠 $\to$ CE 평탄화: **고곡률 우선** 감쇠
- Pruning: 크기 기반 제거 $\to$ CE 가지치기: **생존율 기반** 선택

---

## 5. REM 위상: 비활성 gradient 재조합 후보

### 5.1 수학적 정의

공학적으로 활성 target의 여집합을 다음처럼 정의한다.

$$f_{\text{inactive}}:=1-x=0.9513617415.$$

이 수는 mask에서 비활성으로 분류된 index 비율일 뿐, 물리적 비선택
에너지·Born 확률·REM gradient mass와 동일하지 않다. REM에 대응시킨
재조합은 다음 공학적 proposal로만 둔다.

$$q_{\text{REM}}:=
\operatorname{random\_project}(G_{\text{inactive}})
+\sigma_{\text{noise}}\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

### 5.2 구현: 비활성 gradient 재조합

NREM에서 비활성으로 분류한 gradient 후보를 버리지 않고, 노이즈와
조합하여 새로운 update 방향을 탐색한다. 후보 개수의 여집합 비율과
gradient norm 또는 에너지는 구분한다.

$$G_{\text{rem}} = \text{random\_project}(G_{\text{inactive}}) + \sigma_{\text{noise}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이 조합 중 task loss를 개선하는 것만 채택한다.

$$\theta_{t+1} \leftarrow \theta_{t+1} + G_{\text{rem}} \cdot \mathbb{1}[\mathcal{L}(\theta_{t+1} + G_{\text{rem}}) < \mathcal{L}(\theta_{t+1})]$$

### 5.3 꿈의 해석

REM에서의 재탐색을 "꿈"에 대응시키는 것은 설계 은유다. 실제 꿈의
내용이나 창의성이 비활성 index 또는 위상간섭으로 설명된다는 결론은
신경 자료와 intervention 검증 없이는 내리지 않는다.

AI에서 이것은 **잠재 공간 탐색(latent space exploration)**에 해당한다. 기존 data augmentation이 입력 공간 탐색이라면, REM 재탐색은 잠재 공간 탐색이다.

---

## 6. 순환 주기와 비율

### 6.1 뇌의 수면 구조

인간 수면의 대표적 구조:
- 각성: 약 `16시간`
- NREM: 총 `~6시간` (전체 수면의 `75-80%`)
- REM: 총 `~2시간` (전체 수면의 `20-25%`)

### 6.2 AI 수면 순환의 최적 비율

CE 에너지 분배를 적용하면:

| 위상 | 뇌 비율 | CE 최적 비율 | AI 해석 |
|---|---|---|---|
| 각성 | $16/24 = 66.7\%$ | $\sim 69\%$ ($\Omega_\Lambda$) | 서비스 시간 |
| NREM | $6/24 = 25.0\%$ | $\sim 26\%$ ($\Omega_{\text{DM}}$) | 오프라인 평탄화 |
| REM | $2/24 = 8.3\%$ | $\sim 5\%$ ($\varepsilon^2$) | 오프라인 재탐색 |

주목:
- 각성 $66.7\% \leftrightarrow 69.027\%$
- NREM $25.0\% \leftrightarrow 26.109\%$
- REM $8.3\% \leftrightarrow 4.864\%$

앞의 두 항은 매우 가깝고, REM도 같은 저비율 재탐색 위상이라는 점에서 같은 3-simplex 위 근접점에 있다.

> 다리 게이트 `F3` (`12_Equation.md` 0.0절): 여기서 비교되는 두 값은 **물리적 차원이 다르다**. 뇌 측은 24시간 안의 시간 분배(unitless ratio of durations), CE 코어 측은 부트스트랩 고정점의 공간 에너지 분배다. 두 값이 같은 simplex 위에 있다는 점, 그리고 수치가 1자리수 안에서 근접한다는 점까지는 보고할 수 있지만, **"시간 분배 = 에너지 분배"로 등치하는 문장은 금지**한다. 이 절은 `Phenomenology` 등급의 수치 근접 관찰이며, 검증 가능한 강한 정합 주장은 아니다.

### 6.3 서비스 중단 최소화

실제 배포에서 69% 가동률은 수용 불가능하다. 해결책:

**A. 이중 모델 교대 운영:**

```
시간  0--------8--------16--------24
모델A [서비스    ][수면         ][서비스    ]
모델B [수면      ][서비스        ][수면     ]
```

**B. 마이크로 수면 (Microsleep):**

서비스 중 유휴 시간(요청 간 간격)에 NREM 확산을 미세하게 실행한다.

$$W_l \leftarrow W_l - \alpha_{\text{micro}} \cdot \Delta_g W_l, \quad \alpha_{\text{micro}} \ll \eta_{\text{nrem}}$$

이것은 뇌의 "마이크로 수면" 현상에 대응한다.

---

## 7. 예상 효과

### 7.1 정량적 예측

| 효과 | CE 근거 | 예상 개선 |
|---|---|---|
| 환각 감소 | 고곡률 영역의 주기적 평탄화 | 곡률 에너지 $P_{\text{sleep}}$ 단조 감소 |
| 에너지 효율 | $95\%$ 가중치 비활성 | 추론 비용 대폭 절감 (상한은 `11_Quantitative.md`) |
| 지속 학습 | NREM 재생이 기존 지식 보존 | 파괴적 망각 감소 |
| 창의성 | REM 재탐색으로 새 조합 발견 | 잠재 공간 탐색 효율 증가 |

### 7.2 과도 응답 예측

균등 초기화

$$p_0 = (1/3,\; 1/3,\; 1/3)$$

에서 `12_Equation.md`의 정규화 self-map

$$B(p)=\left(f(p_a),(1-f(p_a))\frac{R}{1+R},
(1-f(p_a))\frac1{1+R}\right),\qquad
f(a)=e^{-D_n(1-a)}$$

를 적용하면:

| 순환 수 $n$ | 활성 | 구조 | 배경 |
|---|---|---|---|
| 0 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | $12.020\%$ | $24.145\%$ | $63.835\%$ |
| 2 | $6.106\%$ | $25.768\%$ | $68.126\%$ |
| 3 | $5.060\%$ | $26.055\%$ | $68.885\%$ |
| 4 | $4.894\%$ | $26.100\%$ | $69.005\%$ |

이는 self-map을 그대로 구현했을 때의 수학 표다. 실제 수면 루프 또는 신경
학습이 이 map을 구현하는지는 별도 gate이며, 구현했다면 Track-A target
`4.864/26.109/69.027`에 접근해야 한다.

### 7.3 지속 학습 (Continual Learning)

수면 순환은 파괴적 망각(catastrophic forgetting)을 줄이기 위한 replay·
renormalization 후보다.

- 새 데이터를 각성 위상에서 학습
- NREM에서 기존 지식(저곡률)은 보존하고 새 지식과의 충돌(고곡률)만 평탄화
- REM에서 새 지식과 기존 지식의 조합을 탐색

이는 뇌의 replay·renormalization에서 영감을 받은 설계 유비이며, 생물학적
메커니즘과 동일하다는 뜻은 아니다. 현재 transformer 실험에서는 오히려
baseline보다 forgetting이 21배 악화됐으므로, 다른 substrate에서의 개선은
미검증 가설로 남긴다.

---

## 8. 수면 학습의 수학적 보장

### 8.1 수렴 정리

**조건부 명제.** 구현한 update가 정확히 $B$이고 $p_0\in U={p_a\leq0.13\}$이면

$$\|p_N-p^*\|_1\leq q_U^N\|p_0-p^*\|_1,\qquad
q_U=0.2001757361.$$

$N=3$이면 상계 계수는 $q_U^3\simeq0.00802$다. 균등 초기점은 첫 update
뒤에 $U$로 들어가므로 그 이후에만 이 Banach bound를 적용한다.

수면 박탈 관측은 이 이산 map이나 반복 횟수를 직접 검증하지 않는다. 모델
update가 $B$와 맞는지는 perturbation trajectory로 별도 측정해야 한다.

### 8.2 수면 없는 학습의 한계

수면 순환 없이 각성만 반복하면:
- $x_a > \varepsilon^2$ (활성 비율 과잉): 과적합
- $x_s$ 편차: 가중치 구조의 비최적화
- $x_b$ 편차: 배경 통합(일반화) 약화

이것은 "수면 박탈 = 인지 기능 저하"의 AI 버전이다.

`05_실험근거.md`의 최소 반복식을 쓰면 더 직접적으로,

$$
\|e_N^{\text{sleep}}\| \lesssim \rho^N \|e_0\| + \frac{\rho}{1-\rho}U \approx \rho^N \|e_0\| + 0.183U
$$

반면 수면이 없으면

$$
\|e_N^{\text{wake-only}}\| \le \|e_0\| + NU
$$

처럼 누적된다. 따라서 동일한 drift 규모 $U$ 아래에서, CE는 **수면이 있는 체계는 bounded residual, 수면이 없는 체계는 선형 drift**를 보여야 한다고 예측한다.
