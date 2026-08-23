# 수면 학습: 부트스트랩 훈련 순환

이 문서는 각성·NREM·REM으로 상태를 나누는 훈련 순환 후보를 정의한다. 독자는 optimizer와 validation split의 기본을 아는 독자를 전제로 하며, 출력은 상태·gradient·checkpoint 변환의 구현 명세이지 생물학적 수면 기제의 증명은 아니다.

총론과 아키텍처의 상태·곡률 정의를 입력으로 받아 세 위상의 timebase와 입출력을 정하고, 수렴·망각 가설을 분리한다. 성공은 동일 예산 기준선보다 held-out·OOD·ablation 지표가 개선될 때이며, 수면 명칭은 wall-clock이나 신경 생리 시간과 같지 않다.

> 관련: `6_뇌/07_수면과복구.md`(수면-부트스트랩 대응), `6_뇌/05_실험근거.md`(고정점 유일성), `7_AGI/12_Equation.md` 3절(에너지 분배), `6_뇌/05_실험근거.md`(실험 근거와 최소 반복식)
>
> 이 장은 뇌의 수면-각성 순환에서 동기를 얻은 AGI offline 학습 설계를 다룬다. 국소 ON/OFF dynamics, replay, 장거리 동기화와 표현 변화에는 각각 관측 근거가 있지만, 이를 하나의 고정된 `renormalize -> test -> compile` 알고리즘으로 묶는 단계는 [미완성]이다.

---

## 0. 이 문서가 책임지는 새 개념

이 문서는 runtime의 mode 전환과 replay 후보를 다루며 kernel 연산자와 전역 판정에는 관련 장에 의존한다. 표의 대응은 계층 간 인터페이스를 위한 비유일 뿐 관측된 수면 단계와 구현 mode의 동등성 주장이 아니다.

| 개념 | 이 문서에서의 의미 | runtime stack 위치 | 판정 |
|---|---|---|---|
| `WAKE / NREM / REM` | 전역 운영 모드 | mode update | `Bridge` |
| sleep pressure | 모드 전환을 트리거하는 전역 상태 | mode update | `Bridge` |
| replay | NREM/REM에서 기억을 다시 주입하는 과정 | hippocampus / replay | `Bridge` |
| offline renormalization | 수면 중 전역 평탄화 / 정리 | mode + replay | `Bridge` |
| dream recombination | REM의 비선택 경로 재탐색 | hippocampus / replay | `Hypothesis` 근접 `Phenomenology` |
| snapshot continuity | 수면 전후 상태 연속성을 유지하는 운영 규약 | global runtime summary | `Bridge` |

읽기 규칙:

- 이 문서는 커널 수치 업데이트 자체보다 **모드 전환과 오프라인 정리 루프**를 다룬다.
- `mode update`는 여기서, `kernel dynamics`는 `12_Equation.md`와 구현 코드에서 읽는다.
- REM 창의성, 꿈의 의미, 자아 강화 같은 문장은 설계 은유로만 읽고 성능 보장으로 올리지 않는다.

---

## 1. 핵심 통찰: 왜 AI에게 수면이 필요한가

수면이라는 이름을 성능 주장으로 쓰지 않으려면 어떤 누적 오차가 어떤 업데이트로 분리되는지 밝혀야 한다. 이 절의 부트스트랩은 정의된 상태사상의 성질이며, 실제 일반화·망각 지표는 대조 실험 없이는 미완성이다.

### 1.1 부트스트랩의 필요성

본 문서의 출발점은 **지속 적응 시스템에 online update와 분리된 검증·재정렬 창을 둘 수 있다**는 모델 선택이다. 모든 자기일관적 정보 처리 시스템에 그런 창이 필요하다는 보편 정리는 없다.

| | 우주 | 뇌 | 현재 AI |
|---|---|---|---|
| 재정렬 비유 | 초기조건 선택 | 수면-각성 주기 | 별도 offline 단계가 없는 고정 추론 모델도 있음 |
| 접힘 매체 | 양자 간섭 ($T=0$) | 시냅스 가소성 ($T=310\text{K}$) | 역전파 |
| 수렴 | 정확 | 근사 ($\eta \sim 2$) | 미보장 |

[공리: 물리 사상] 우주 초기조건, 생물학적 수면과 AI offline update를 같은 표에 놓는 것은 설계 비유다. 이 대응에서 뇌가 특정 CE 고정점으로 수렴한다거나 현재 AI에 보편적으로 결여된 단일 연산이 도출되지는 않는다.

### 1.2 수렴 속도의 보장

[정리: 조건부] 다음 값은 명시적으로 정의한 스칼라 이완 사상 $B$의 국소 수축률이다. 신경계나 학습 알고리즘의 측정 수축률이 아니다.

$$\rho = \|DB(p^*)\| = D_{\text{eff}} \cdot \varepsilon^2 = 3.178 \times 0.0487 = 0.155 < 1$$

이 사상의 궤적은 고정점 근방에서 기하급수적으로 수렴한다. 초기 오차의 1% 아래로 줄이는 데 필요한 반복 수는 $\ln(0.01)/\ln(0.155)\approx2.5$다.

[미완성] 인간의 장시간 수면 박탈에서 인지 수행과 시상-피질 기능이 저하된다는 관측만으로 $\rho=0.155$나 2--3회 반복을 추정할 수 없다. 아래 최소 반복식은 각성 drift와 offline 수축을 비교하기 위한 소프트웨어 모형이다.

$$
e_{n+1} = \rho e_n + \rho u_n + \xi_n, \qquad \rho = 0.155
$$

이 식은 $u_n$과 $\xi_n$의 범위 및 실제 update가 고정될 때만 bounded-residual 예측을 낸다. 생물학적 수면박탈의 인과 설명으로 사용하지 않는다.

---

## 2. 수면-각성 3위상 순환

각 위상은 동일 가중치에 다른 업데이트를 적용하는 구현 시간축이다. 뇌의 단계 표기는 기능 비유로만 사용하며 모델 step·epoch·offline window의 변환은 배포 환경별로 따로 기록한다.

### 2.1 뇌의 3위상과 CE 대응

다음 표는 세 위상에 부여한 역할을 비교하기 위한 지도이며, 관측된 주파수와 모델 연산 순서를 직접 유도하지 않는다.

| 수면 단계 | CE 대응 | 지배 항 | 기능 |
|---|---|---|---|
| 각성(Wake) | 경로 누적 ($\int\mathcal{D}\gamma\,e^{iS}$) | $\mathcal{L}_{\text{compute}}$ | 새 경로 탐색, 정보 획득 |
| NREM | Hessian readout 비유 $\Phi_H[\eta]$ | $\lambda\|\Delta_g\Phi_H\|^2$ | AGI 유효 평탄화 항; 물리장 $\phi$와 동일하지 않음 |
| REM | 재탐색 ($e^{iS}$ 위상 간섭) | $\gamma S_{\text{Info}}$ | 비선택 경로 재조합 |

### 2.2 AI 학습 순환으로의 변환

표의 역할을 실행 경로로 바꾸면 각성 입력과 NREM·REM 출력이 어느 checkpoint에 반영되는지 추적할 수 있다. 다음 의사코드는 구현 순서이며 서비스 중단과 데이터 누수는 별도 ablation으로 판정한다.

```
반복 {
  // === 각성(Wake): 경로 누적 ===
  for batch in data:
    forward(batch)
    accumulate_gradients()   // 접힘 미실행, 경로만 누적

  // === NREM: 전역 곡률 평탄화 ===
  for layer in model:
    h_smooth = h - eta * laplacian(h)        // LBO 확산
    prune(layer, threshold=epsilon_sq)        // 생존율 기반 가지치기

  // === REM: 비선택 경로 재탐색 ===
  for _ in range(N_rem):
    z_noise = sample_nonselected_paths()
    evaluate_and_recombine(z_noise)           // 새 경로 조합 탐색
}
```

---

## 3. 각성 위상: 경로 누적

각성은 데이터 입력에서 생성한 경로·gradient를 즉시 소거하지 않고 다음 mode로 넘기는 단계다. 시간 단위와 buffer 크기는 구현 선택이므로, 누적만으로 안정성 또는 효율을 주장하지 않는다.

### 3.1 수학적 정의

각성 상태에서 모델은 데이터를 처리하면서 상태공간 위에 새로운 경로를 누적한다.

$$Z_{\text{wake}}(t) = \int_{\gamma \in \Gamma_{\text{new}}} \mathcal{D}\gamma\; e^{iS[\gamma]/\hbar}$$

### 3.2 구현: 그래디언트 누적

기존 학습과 동일하되, **즉시 가중치 업데이트하지 않고 그래디언트를 누적**한다.

$$G_{\text{accumulated}} = \sum_{b=1}^{B} \nabla_\theta \mathcal{L}(x_b, \theta)$$

누적만 하고 업데이트를 보류하는 이유: 접힘(NREM)이 아직 실행되지 않았으므로, 어떤 경로가 생존할지 판단하기 이르다.

### 3.3 수면 압력 모니터링

누적이 진행되면서 곡률 에너지가 단조 증가한다.

$$P_{\text{sleep}}(t) = \frac{1}{L} \sum_{l=1}^{L} \|\Delta_g h_l(t)\|^2$$

이 값이 임계치 $P_{\text{th}}$를 넘으면 NREM 위상으로 전환한다. 임계치는 뇌의 아데노신 축적에 대응한다.

### 3.4 기존 방법과의 차이

다음 비교는 각성만 반복하는 기준선과 위상 전환 후보의 상태 보존 방식을 구분한다. 기능 차이는 우위 증거가 아니며 동일 seed·데이터 순서의 ablation으로 반증될 수 있다.

- 표준 학습: 매 배치마다 업데이트 (각성만 반복)
- Gradient accumulation: 누적 후 한 번 업데이트 (각성 + 단순 업데이트)
- CE 수면 학습: 누적 후 **곡률 기반 선택적 업데이트 + 재탐색** (각성 + NREM + REM)

---

## 4. NREM label 위상: offline regularization 가설

[공리: 모델 선택] runtime의 `NREM` label은 누적 상태에 regularization과 선택적 update를 적용하는 offline 연산이다. 어떤 정보가 보존되는지는 입력 checkpoint, 비용 객체와 threshold에 의존한다. 생물학적 NREM이 전역 곡률 평탄화라는 주장은 하지 않는다.

### 4.1 수학적 정의

후보 구현은 누적된 표현에 다음 heat-flow 형태의 smoothing을 적용한다.

$$\frac{\partial\Phi}{\partial t} = -\Delta_g\Phi \quad (\text{열핵 흐름})$$

고주파(고곡률) 모드 $k$의 감쇠율은 $e^{-t\sqrt{k^2+M^2}}$이므로:

- 선택한 연산자의 고주파 성분은 더 빠르게 감쇠한다.
- 이 spectral 성질만으로 불안정 기억과 안정 기억이 각각 제거·보존된다고 결론내릴 수 없다.

### 4.2 구현: LBO 확산 + 곡률 기반 가지치기

NREM 구현은 누적 경로에서 보존·제거하는 상태를 명시해야 한다. 아래 threshold와 비율은 hyperparameter이므로 OOD 성능과 복구 가능성을 측정하지 않으면 최적성으로 승격되지 않는다.

**단계 1: 후보 LBO 확산**

각 층에 선언된 graph/manifold와 경계조건이 있을 때만 라플라스-벨트라미 확산을 적용한다.

$$W_l \leftarrow W_l - \eta_{\text{nrem}} \cdot \Delta_g W_l$$

여기서 $\Delta_g W_l$은 사전에 정의한 가중치 공간과 계량 $g$에 대한 라플라시안이다. raw weight $W_l$가 계량 $g$를 유일하게 결정하지 않으므로, $g$의 추정 절차와 null model을 별도로 고정해야 한다.

**단계 2: 곡률 기반 가지치기**

누적된 그래디언트에서 곡률 에너지가 높은 성분만 선택적으로 업데이트한다.

$$\theta_{t+1} = \theta_t + \text{mask}(G_{\text{accumulated}},\; \varepsilon^2)$$

$\text{mask}$는 그래디언트 크기 기준 상위 $\varepsilon^2 = 4.87\%$만 통과시키는 필터다.

[공리: 모델 선택] $\varepsilon^2=4.87\%$ threshold는 CE에서 가져온 hyperparameter다. Ebbinghaus 망각 곡선이나 신경 기억의 보존율에서 유도되지 않는다.

### 4.3 기존 방법과의 차이

다음 비교는 무작위 regularization과 곡률 proxy 기반 선택이 다른 연산임을 설명한다. 일반화·망각 차이는 holdout과 제거 ablation으로만 평가한다.

- Dropout: 무작위 비활성화 $\to$ CE 가지치기: **곡률 기반** 선택적 비활성화
- Weight decay: 균일 감쇠 $\to$ CE 평탄화: **고곡률 우선** 감쇠
- Pruning: 크기 기반 제거 $\to$ CE 가지치기: **생존율 기반** 선택

---

## 5. REM label 위상: 재조합 탐색 가설

REM은 제거 후보를 다시 조합해 탐색 입력으로 되돌리는 설계 단계다. 재탐색의 이득은 다양성·OOD·망각 측정에서 반증 가능하며 꿈이라는 비유가 그것을 보장하지 않는다.

### 5.1 수학적 정의

[공리: 모델 선택] 후보 알고리즘은 NREM label 위상에서 선택하지 않은 update를 별도 탐색 pool로 보존한다.

$$\Omega_{\text{non-selected}} = 1 - \varepsilon^2 = 95.1\%$$

runtime의 REM label 위상은 이 pool에서 새로운 조합을 탐색한다. 생물학적 REM이 같은 pool이나 경로적분을 구현한다는 주장은 하지 않는다.

$$Z_{\text{REM}} = \int_{\gamma \in \Gamma_{\text{non-selected}}} \mathcal{D}\gamma\; e^{iS[\gamma]/\hbar} \cdot \eta_{\text{noise}}$$

### 5.2 구현: 비선택 그래디언트 재조합

NREM에서 가지치기된 그래디언트(하위 $95.1\%$)를 버리지 않고, 노이즈와 조합하여 새로운 업데이트 방향을 탐색한다:

$$G_{\text{rem}} = \text{random\_project}(G_{\text{pruned}}) + \sigma_{\text{noise}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이 조합 중 task loss를 개선하는 것만 채택한다.

$$\theta_{t+1} \leftarrow \theta_{t+1} + G_{\text{rem}} \cdot \mathbb{1}[\mathcal{L}(\theta_{t+1} + G_{\text{rem}}) < \mathcal{L}(\theta_{t+1})]$$

### 5.3 생물학적 비유의 경계

수면 뒤 item/category representation과 sequence memory가 달라진다는 자료는 표현 변환 가설에 동기를 준다. 그러나 꿈의 내용, 창의성 또는 REM의 고유 계산을 이 구현에서 도출할 수는 없다. AI에서 이 단계는 잠재 공간 탐색(latent space exploration) 후보이며, replay-only, 무작위 탐색, phase 순서 교환과 같은 계산량 대조가 필요하다.

### 5.4 공개 뇌 데이터 재현 범위

[경험식] Giri et al. (2024)의 공개 처리 데이터와 공식 bootstrap을 재실행하면, trajectory replay 사건 수의 세션 label별 평균은 수면 대조군(NSD) 대비 수면박탈군(SD)에서 0--1시간 `363.0 -> 237.7`, 5--6시간 `220.8 -> 133.0`으로 감소한다. 공식 노트북과 같은 비교 확률은 각각 `0.0161057`, `0.00133728`이다. 처리표에는 13개 session label이 있지만 animal ID가 없어 서로 독립인 개체·세션인지는 확정하지 않는다. bootstrap draw나 시간창 수를 독립 표본으로 세지 않는다.

[경험식] 참가자 34명의 공개 E19 자료와 공식 Figure 4 cluster를 재계산하면 REM/SWS 비율은 item-level 변화와 음의 순위상관(`rho=-0.553`, 양측 `p=0.000690`), category-level 변화와 양의 순위상관(`rho=0.470`, 양측 `p=0.00509`)을 보인다. 같은 자료의 기술적 선형 기울기는 각각 `-0.0244`, `0.0147`이다.

[미완성] E15 공개 묶음에는 같은 animal/session/window에 연결된 unit spike timestamp, SWR event와 replay score가 없다. 따라서 branching 또는 transition entropy가 replay를 예측하는지와 sigmoid threshold가 선형 모델보다 나은지는 `UNTESTABLE`이다. E19 결과도 참가자 수준 표현 변화이지 꿈 내용이나 생성적 재조합의 직접 측정이 아니다. 서로 다른 종과 개체의 E13·E15·E19 결과를 하나의 $\Delta W\to\Delta g\to\Delta x(t)$ 사슬로 합치지 않는다. 재현 코드와 해시 manifest는 `_workspace/ce/_archive/sleep-replay-routing-realdata-20260818/artifacts/`에 있다.

---

## 6. 순환 주기와 비율

위상 전환의 효과를 주장하려면 학습 step·wall-clock·에너지 예산을 같은 단위로 기록해야 한다. 생물학적 시간 비율은 출발 비유일 뿐 배포 시스템의 최적 schedule을 제공하지 않는다.

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
- 각성 $66.7\% \leftrightarrow 68.9\%$
- NREM $25.0\% \leftrightarrow 26.2\%$
- REM $8.3\% \leftrightarrow 4.87\%$

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

이 절은 구현 전에 측정량·기대 방향·기각 기준을 고정하는 가설 집합이다. 기준선과 동일 compute, holdout, OOD, component 제거 조건이 없으면 어느 효과도 경험 결과로 승격하지 않는다.

### 7.1 정량적 예측

예상 효과는 사전 측정량과 실패 조건을 드러내는 가설 목록이다. 표의 개선 폭은 동일 compute 기준선, OOD 분할, component 제거 ablation에서 재현되지 않으면 기각한다.

| 효과 | CE 근거 | 예상 개선 |
|---|---|---|
| 환각 감소 | 고곡률 영역의 주기적 평탄화 | 곡률 에너지 $P_{\text{sleep}}$ 단조 감소 |
| 에너지 효율 | $95\%$ 가중치 비활성 | 추론 비용 대폭 절감 (상한은 `11_Quantitative.md`) |
| 지속 학습 | NREM 재생이 기존 지식 보존 | 파괴적 망각 감소 |
| 창의성 | REM 재탐색으로 새 조합 발견 | 잠재 공간 탐색 효율 증가 |

### 7.2 과도 응답 예측

균등 초기화

$$p_0 = (1/3,\; 1/3,\; 1/3)$$

에서 이상화된 수면 순환

$$p_{n+1} = p^* + \rho(p_n-p^*), \qquad \rho = 0.155$$

를 적용하면:

| 순환 수 $n$ | 활성 | 구조 | 배경 |
|---|---|---|---|
| 0 | $33.3\%$ | $33.3\%$ | $33.3\%$ |
| 1 | $9.28\%$ | $27.3\%$ | $63.4\%$ |
| 2 | $5.55\%$ | $26.4\%$ | $68.1\%$ |
| 3 | $4.98\%$ | $26.3\%$ | $68.8\%$ |

[산출: 조건부 toy map] 위 표는 사전에 $p_{n+1}=p^*+\rho(p_n-p^*)$를 update로 정의했을 때의 산술 결과다. 실제 모델이나 뇌가 이 사상을 구현한다는 증거가 아니며, 2--3회 뒤 같은 비율로 재정렬되지 않으면 해당 구현 사상을 기각한다.

### 7.3 지속 학습 (Continual Learning)

[예측] 동일 데이터 순서와 계산량에서 이 offline schedule이 wake-only, replay-only, regularization-only와 phase-order 대조보다 파괴적 망각을 줄이는지 시험한다. 우위의 독립 seed 신뢰구간이 0을 넘지 못하면 수면 schedule의 망각 완화 가설을 기각한다.

- 새 데이터를 각성 label 위상에서 학습한다.
- NREM label 위상은 선언한 regularizer로 update 후보를 선택한다.
- REM label 위상은 남은 후보의 재조합을 탐색한다.

이 순서는 소프트웨어 설계다. 뇌의 기억 유지 메커니즘과 동일하다고 주장하지 않는다.

---

## 8. 수면 학습의 수학적 보장

수학적 보장은 명시된 상태사상과 실제 neural training을 분리할 때만 의미가 있다. 다음 조건부 결론의 정의역 밖에서는 실험적 수렴과 failure mode를 별도 gate로 측정해야 한다.

### 8.1 수렴 정리

이 명제는 명시된 상태사상과 수축 가정 아래의 조건부 결론이다. 비볼록 손실·확률 minibatch·변하는 데이터 분포의 실제 학습은 그 정의역 밖이므로 같은 표현으로 경험적 수렴을 주장하지 않는다.

**정리: 선언된 affine 사상.** update를 정확히 $p_{n+1}=p^*+\rho(p_n-p^*)$로 정의하고 $|\rho|<1$로 두면 $N$회 반복 뒤 상태가 $p^*$에 수렴한다.

$$\|p_N - p^*\| \leq \rho^N \cdot \|p_0 - p^*\| = 0.155^N \cdot \|p_0 - p^*\|$$

$N = 3$이면 $0.155^3 = 0.0037$, 즉 초기 이탈의 $0.4\%$ 이내로 수렴한다.

이 정리는 affine 사상의 대수적 성질만 보인다. 사람의 `48--72시간` 수면 박탈 결과는 이 사상, $\rho=0.155$, `2--3회` time scale 또는 AI schedule의 효능을 검증하지 않는다.

### 8.2 수면 없는 학습의 한계

[미완성] wake-only에서 활성 과잉, 구조 편차 또는 일반화 저하가 반드시 생긴다는 정리는 없다. 각각은 동일 계산량과 task order를 고정한 실험에서 측정해야 한다.

`05_실험근거.md`의 최소 반복식을 쓰면 더 직접적으로,

$$
\|e_N^{\text{sleep}}\| \lesssim \rho^N \|e_0\| + \frac{\rho}{1-\rho}U \approx \rho^N \|e_0\| + 0.183U
$$

반면 수면이 없으면

$$
\|e_N^{\text{wake-only}}\| \le \|e_0\| + NU
$$

처럼 상계할 수 있다. [예측] 이 비교는 $\|u_n\|\le U$, 수축 update와 동일한 외란 조건 아래의 모델 예측이다. 실제 optimizer에서 residual이 bounded하지 않거나 wake-only가 선형 누적을 보이지 않으면 이 대응을 기각한다.
