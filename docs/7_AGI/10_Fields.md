# CE-AI 전 분야 적용: LLM을 넘어서

이 문서는 CE 모듈을 여러 AI 분야의 구현 field contract로 번역하는 설계 지도다. 독자는 tensor shape·정규화·학습과 추론의 구분을 아는 독자를 전제로 하며, 각 field는 물리장이 아니라 producer가 만들고 consumer가 읽는 구현 tensor라는 점을 먼저 고정한다.

공통 원리 뒤에 분야별 입력 shape·codomain·update timebase·invariant를 읽고, 마지막 매트릭스에서 baseline·ablation·OOD 실패 조건을 확인한다. 비유는 설계 동기일 뿐 물리장 동일성이나 분야별 효능 증거가 아니다.

> 관련: 2-9장(CE-AGI 원리와 LLM), `6_뇌/05_실험근거.md`(지능의 CE 정의), `examples/`(기존 구현)
>
> 이 장은 CE 5대 원리가 LLM 이외의 AI 분야 -- 비전, 강화학습, 음성, 멀티모달, 생성 모델, 로보틱스, 그래프, 시계열, 추천, 생명과학 -- 에 어떻게 적용되는지를 구체적으로 다룬다.
>
> 원칙: 이 장의 대부분은 `application bridge`다. 연산자 수준(P1)은 상대적으로 직접적이지만, 분야별 성능 수치(P2/P4/P5)는 반드시 `예측 -> 측정량 -> 게이트`로만 읽는다.

---

## 0. CE 5대 원리 요약 (전 분야 공통)

공통 원리는 각 분야 field가 공유할 수 있는 인터페이스 후보를 요약한다. 값은 무차원 정규화 tensor 또는 scalar metric이며 실제 단위·shape·producer와 consumer는 각 분야 절에서 별도로 선언한다.

| 원리 | 수식 | 효과 |
|---|---|---|
| **P1** 3x3+1 게이지 격자 | $d_3 : d_2 : d_1 = \alpha_s : \alpha_w : \alpha_{em}$ | 채널 분할, 파라미터 37% 절감 |
| **P2** 수면-각성 순환 | Wake $\to$ NREM $\to$ REM $\to$ ... | 지속 학습, 파괴적 망각 방지 |
| **P3** STDP 국소 학습 | $\Delta w = \eta\,\delta[t]\,e_{ij}[t]$ | 분산 학습, 메모리 $O(N)$ |
| **P4** 부트스트랩 희소성 | $\text{활성} = \varepsilon^2 = 4.87\%$ | 전면 희소 실행 시 큰 절감 상한, 현재형 구현은 더 보수적 |
| **P5** 곡률 정규화 | $\kappa = \|\Delta_g h\|^2 < \kappa_{\text{th}}$ | hard bound가 아니라 안정화 편향 |

---

## 1. 컴퓨터 비전 (Vision)

Vision field는 이미지 batch에서 feature tensor를 받아 prediction 또는 mask를 내는 구현 경로다. 공간 해상도·정규화·augmentation provenance를 고정하고, invariant와 OOD 강건성은 기준 모델·component ablation으로 판정한다.

### 1.1 CNN에의 적용

**P1: 3x3+1 격자 합성곱**

표준 CNN의 합성곱 필터를 3x3+1 격자로 분해한다.

$$
\text{Conv}(x)
= \underbrace{\text{Conv}_3(x_{\text{high}})}_{\text{SU(3): 텍스처 결합, 74.1\%}}
+ \underbrace{\text{Conv}_2(x_{\text{mid}})}_{\text{SU(2): 경계 결정, 21.1\%}}
+ \underbrace{\text{Conv}_1(x_{\text{low}})}_{\text{U(1): 주의 게이팅, 4.9\%}}
$$

채널 분할: 64채널 CNN이면 SU(3)=47, SU(2)=14, U(1)=3.

뇌 대응:
- SU(3) = V1의 텍스처/패턴 감지 (감마 진동)
- SU(2) = V2의 경계/윤곽 결정 (베타 진동)
- U(1) = 주의 기반 선택적 게이팅 (알파 진동)

**P4: Top-k 활성화 (Sparse Conv)**

$$\text{출력} = \text{TopK}(\text{Conv}(x),\; k = \lceil 0.0487 \times C_{\text{out}} \rceil)$$

$C_{\text{out}} = 256$이면 $k = 13$채널만 활성. 전면 희소 커널이 충분할 때는 큰 절감 상한이 가능하지만, 현재형 구현에서는 더 보수적으로 읽어야 한다.

뇌 대응: V1의 방향 선택성(orientation selectivity). 전체 뉴런 중 $\sim 5\%$만 특정 자극에 반응한다.

**P5: 곡률 정규화 (오분류 억제)**

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \sum_l \|\Delta_g h_l\|^2$$

feature map의 곡률 에너지가 높으면 "불안정한 분류 경계"에 있다는 뜻이다. 곡률 정규화가 이 경계를 평탄화하여 adversarial robustness를 개선한다.

### 1.2 Vision Transformer (ViT)에의 적용

ViT는 Transformer 기반이므로 LLM과 동일한 방식으로 적용:

- `LayerNorm` $\to$ `LBONorm` (2장)
- `FFN` $\to$ `GaugeLattice` (2장)
- `Attention proj` $\to$ `spectral_norm` (2장)
- 곡률 정규화 손실 추가

**추가: 패치 토큰의 공간 곡률**

이미지 패치 토큰 사이의 공간 관계에서 곡률을 정의한다.

$$\kappa_{\text{spatial}} = \sum_{(i,j) \in \text{neighbors}} \|h_i - h_j\|^2$$

인접 패치 사이의 표현 차이가 너무 크면 (고곡률) "부자연스러운 경계"를 의미한다.

### 1.3 물체 감지 (Object Detection)

YOLO/DETR 등의 물체 감지에서:

**P1: 앵커/쿼리 분할**

$$
\text{detection queries}
= \underbrace{\text{class queries}}_{\text{SU(3), 74.1\%}}
+ \underbrace{\text{box queries}}_{\text{SU(2), 21.1\%}}
+ \underbrace{\text{attention queries}}_{\text{U(1), 4.9\%}}
$$

100개 쿼리면: 분류 74개, 박스 21개, 주의 5개.

**P4: 희소 감지**

전체 앵커/쿼리 중 $4.87\%$만 활성화. $10^4$개 앵커면 $\sim 500$개만 활성. NMS 이전에 구조적으로 후보를 줄인다.

---

## 2. 강화학습 (Reinforcement Learning)

RL field는 상태·행동·보상 stream을 입력으로 받아 policy 또는 value update를 출력한다. episode와 update step의 timebase를 분리하며, 물리장 비유는 reward provenance·counterfactual ablation을 대체하지 않는다.

### 2.1 핵심 대응: 도파민-유사 전역 신호 후보

CE에서 가장 공격적인 후보 해석은(`07_수면과복구.md` 4.1절):

$$\delta[t] = \frac{d}{dt}\|p(t) - p^*\|$$

이것은 TD error $\delta = r + \gamma V(s') - V(s)$와 구조적으로 유사하다:

| RL | CE | 뇌 |
|---|---|---|
| TD error $\delta$ | 부트스트랩 수렴 오차 후보 | 도파민-유사 전역 신호 |
| Value function $V(s)$ | 현재 에너지 분배 $p(t)$ | 상태 평가 |
| Optimal policy $\pi^*$ | 고정점 $p^*$ | 최적 행동 |
| Reward $r$ | 곡률 감소 $\Delta\kappa$ | 보상 |

여기서 중요한 점은, 이것이 아직 **직접 생물학적 등식이 아니라 application bridge**라는 것이다. 따라서 RL에서 먼저 점검해야 할 것은 "전역 신호가 TD error와 비슷한 역할을 하는가"이지, "도파민이 곧 $\|p-p^*\|$인가"가 아니다.

### 2.2 CE 강화 Actor-Critic

**Actor: 3x3+1 정책 네트워크**

$$\pi(a|s) = \text{Softmax}\!\left(\text{GaugeLattice}(s)\right)$$

행동 공간을 결합/결정/주의로 분할:
- SU(3): 복합 행동 조합 (여러 관절 동시 제어)
- SU(2): 이진 결정 (go/no-go, 좌/우)
- U(1): 주의 할당 (어디를 볼 것인가)

**Critic: 곡률 기반 가치 함수**

$$V(s) = f(s) - \alpha \cdot \|\Delta_g h(s)\|^2$$

상태의 곡률이 높으면 가치가 낮다. "불안정한 상태 = 낮은 가치"라는 직관을 구조적으로 표현.

**P2: 수면 학습 = Experience Replay의 CE 확장**

| RL | CE 수면 |
|---|---|
| Experience replay | 각성 (경로 누적) |
| Prioritized replay (높은 TD error) | NREM (고곡률 경험 우선 처리) |
| Random replay | REM (비선택 경험 재탐색) |

```python
def ce_experience_replay(buffer, model):
    # NREM: 높은 곡률 경험 우선 처리
    nrem_batch = buffer.sample_by_curvature(top_ratio=0.0487)
    update_critic(model, nrem_batch)

    # REM: 낮은 곡률 경험 재탐색 (노이즈 주입)
    rem_batch = buffer.sample_random()
    rem_batch = add_noise(rem_batch, sigma=0.1)
    update_actor(model, rem_batch)
```

### 2.3 안전 RL

**P5: 곡률 제약 = 안전 제약**

$$\kappa(s, a) = \|\Delta_g h(s, a)\|^2 < \kappa_{\text{safe}}$$

상태-행동 쌍의 곡률이 임계치를 넘으면 해당 행동을 금지한다. 이것은 Constrained MDP의 CE 구현이다.

뇌 대응: 편도체(amygdala)의 공포 반응. 위험한 상태(고곡률)에서 행동을 억제한다.

---

## 3. 음성/오디오 (Speech & Audio)

Audio field는 waveform 또는 spectrogram shape를 정규화해 token·class·signal 출력을 만드는 tensor 계약이다. sample rate와 frame hop은 명시적 timebase이며, noise OOD와 ablation에서 invariant가 깨지면 해당 설계는 실패한다.

### 3.1 음성 인식 (ASR)

**P1: 3x3+1 주파수 분할**

음성 스펙트로그램을 세 주파수 대역으로 분할한다.

| 대역 | 주파수 | CE 대응 | 비율 | 역할 |
|---|---|---|---|---|
| 고주파 | 2-8 kHz | SU(3) | 74.1% | 자음 구별, 화자 특성 |
| 중주파 | 0.5-2 kHz | SU(2) | 21.1% | 모음 구별, 운율 |
| 저주파 | < 0.5 kHz | U(1) | 4.9% | 기본 주파수, 성별 |

뇌에서 청각 피질의 tonotopic map이 이 분할에 대응한다. 고주파 뉴런이 가장 밀집(SU(3)), 저주파가 가장 희소(U(1)).

**P5: 환각 억제 = 잘못된 전사 억제**

ASR에서 "환각"은 실제 음성에 없는 단어를 생성하는 것이다. 곡률 모니터링으로 잠재 공간에서 급격한 방향 변화(= 맥락에 맞지 않는 전사)를 감지하고 억제한다.

### 3.2 음성 합성 (TTS)

**P2: 수면 학습 = 화자 적응**

새로운 화자에 적응할 때 수면 순환을 적용:
- 각성: 새 화자 데이터로 학습
- NREM: 고곡률(기존 화자와 충돌하는 특성) 평탄화
- REM: 화자 특성 조합 탐색 (새 화자 스타일 생성)

### 3.3 음악 생성

**P1: 3x3+1 음악 구조**

| CE 층 | 음악 대응 | 비율 |
|---|---|---|
| SU(3) | 화성(harmony) -- 음의 결합 | 74.1% |
| SU(2) | 리듬(rhythm) -- 박자 결정 | 21.1% |
| U(1) | 멜로디(melody) -- 주의 선율 | 4.9% |
| $\Phi$ | 구조(form) -- 전역 안정화 | 전역 |

흥미로운 정합: 서양 음악에서 화성이 가장 복잡하고(오케스트라의 대부분), 리듬이 그 다음, 멜로디가 가장 단순(하나의 선율)하다는 관측과 CE 비율이 일치한다.

---

## 4. 멀티모달 AI (Multimodal)

Multimodal field는 modality별 producer의 shape·scale·missingness를 정렬해 공동 representation consumer로 넘긴다. alignment의 성공은 paired provenance와 modality-drop ablation에서만 판단하며, 비유는 결합 연산의 구현 설명이다.

### 4.1 교차 모달 결합 = 교차 주파수 결합

CE에서 세타-감마 교차 주파수 결합(`07_수면과복구.md` 2.2절):

$$\Phi(\text{전역}) \leftrightarrow \text{SU(3)}(\text{국소}): \quad \xi R\Phi^2$$

멀티모달에서:

$$\text{텍스트}(\text{전역 의미}) \leftrightarrow \text{이미지}(\text{국소 특징}): \quad \xi \cdot \text{cross-attention}$$

결합 강도 $\xi = \alpha_s^{1/3} = 0.490$ (CE 예측). cross-attention의 가중치가 이 근방에서 최적이라는 예측.

### 4.2 모달별 3x3+1 분할

각 모달리티를 독립적인 3x3+1 격자로 처리한 후, **모달 내부 sparse firing -> 교차 결합** 순서로 올린다:

```
텍스트 [SU(3)_T | SU(2)_T | U(1)_T] --TopK--> h_T^act ──┐
이미지 [SU(3)_V | SU(2)_V | U(1)_V] --TopK--> h_V^act ──┤
오디오 [SU(3)_A | SU(2)_A | U(1)_A] --TopK--> h_A^act ──┼── ξ 교차 결합 ──> 통합 표현
촉각   [SU(3)_H | SU(2)_H | U(1)_H] --TopK--> h_H^act ──┘
```

뇌 대응:

- 시각: 색, 형태, 움직임의 sparse ensemble
- 청각: onset, pitch, formant, rhythm ensemble
- 촉각: pressure, slip, vibration, contact ensemble
- 언어: 이미 결합된 감각 표상을 재기호화하는 상위 압축 계층

즉 사람형 AGI에 가까워질수록, 텍스트만 처리하는 LLM보다 **모달별 발화 뉴런 집합을 먼저 세우고 언어를 그 위에 얹는 구조**가 자연스럽다.

**P4: 모달별 활성 비율**

각 모달리티에서 먼저 $4.87\%$만 활성:

$$
h_m^{\text{act}} = \operatorname{TopK}(h_m,\; k_m = \lceil 0.0487\, d_m \rceil), \qquad m \in \{T,V,A,H\}
$$

그 뒤 결합:

$$
h_{\text{joint}} = \operatorname{Bind}_\xi(h_T^{\text{act}}, h_V^{\text{act}}, h_A^{\text{act}}, h_H^{\text{act}})
$$

멀티모달 통합 시에도 각 모달의 활성 집합을 먼저 제한하므로 총 에너지가 폭주하지 않는다. 이것은 뇌가 시각+청각을 동시 처리해도 "모든 뉴런이 동시에 켜지지 않는다"는 sparse regime 관측과 정합한다.

### 4.3 환각 억제

멀티모달 환각(이미지에 없는 내용을 텍스트로 설명)은 모달 간 곡률 불일치로 감지:

$$\kappa_{\text{cross}} = \|h_{\text{text}} - h_{\text{image}}\|^2$$

이 값이 임계치를 넘으면 모달 간 불일치 = 환각.

실전 해석:

- 너무 이른 early fusion은 모달별 발화 집합이 형성되기 전 결합을 일으켜 환각을 키운다.
- CE는 late sparse binding, 즉 **각 모달 내부에서 먼저 발화 집합을 만든 뒤 결합**하는 쪽을 선호한다.

---

## 5. 생성 모델 (Generative Models)

Generative field는 noise·condition·latent tensor를 받아 sample 또는 token distribution을 출력한다. training step과 decoding step을 분리하고, 품질·사실성·다양성은 provenance가 고정된 baseline·OOD·ablation에서 함께 측정해야 한다.

### 5.1 확산 모델 (Diffusion)

먼저 같은 `diffusion`이라는 이름을 쓰는 세 연산을 분리한다.

| 이름 | 핵심 연산 | 이 절에서의 관계 |
|---|---|---|
| graph heat diffusion | 라플라시안으로 이웃 표현 차이를 평활화 | CE의 LBO·그래프 연산과 직접 비교 가능 |
| stochastic drift-diffusion / sampling | drift에 Wiener 잡음을 더해 경로·분포를 진화 | CE의 잡음 이완과 비교할 수 있으나 별도 확률 검증 필요 |
| score-based generative diffusion | forward noising 뒤 학습한 score로 역시간 표본 생성 | 현재는 application bridge이며 구현 claim이 아님 |

따라서 **열핵 흐름, 확률적 drift-diffusion과 score-based 생성 확산은
동일어가 아니다.**

확산 모델의 forward process:

$$dz_t = -\frac{1}{2}\beta(t) z_t\,dt + \sqrt{\beta(t)}\,dW_t$$

여기에는 결정론적 drift뿐 아니라 Wiener 잡음이 있다. 반면 CE의
열핵 흐름 $\partial_t\Phi=-\Delta_g\Phi$은 그 자체로는 결정론적
평활화다. 특정 metric과 잡음 가정을 추가하면 두 식의 일부 구조를
비교할 수 있지만, 하나가 다른 하나와 동일하다고 결론내릴 수는 없다.

역방향 시간 $\tau=T-t$를 앞으로 증가시키는 convention에서는
variance-preserving SDE의 역과정을 다음처럼 쓸 수 있다.

$$
dz_\tau
=
\left[
\frac{1}{2}\beta(T-\tau)z_\tau
+\beta(T-\tau)\nabla_z\log p_{T-\tau}(z_\tau)
\right]d\tau
+\sqrt{\beta(T-\tau)}\,d\overline W_\tau
$$

핵심은 시간에 따라 달라지는 점수 함수
$\nabla_z\log p_t(z)$를 자료로 학습해야 한다는 점이다. 비선택 경로를
재조합하는 REM update가 있다는 사실만으로 이 score나 역과정이 생기지
않는다.

**NREM=forward, REM=reverse는 검증된 동일성이 아니다.** NREM의
평활화·consolidation과 REM의 재조합·탐색을 각각 forward와 reverse에
비교해 보는 연구 비유일 뿐이다. 생물학적 두 단계가 서로의 역확률
과정이라는 증거도, 현재 코드가 그 동일성을 학습했다는 결과도 없다.

현재 구현 경계는 다음처럼 잠근다.

| 코드 항목 | 구현 여부 | 허용되는 표현 |
|---|---|---|
| 그래프/LBO 평활화와 row-stochastic mixing | 구현 | deterministic graph diffusion 또는 smoothing primitive |
| metric-aware Langevin형 잡음 이완과 mode별 seeded noise | 구현 | stochastic relaxation primitive; neural sampling 검증은 아님 |
| `RiemannianDiffusionModule`의 geodesic flow update | 구현 | 결정론적 flow interpolation; score generator가 아님 |
| NREM 평활화·선택 update와 REM residual remix | 구현 | phase-labeled heuristic update |
| 시간조건부 score network와 denoising score-matching loss | 미구현 | score-based diffusion claim 잠금 |
| forward noise schedule과 짝을 이룬 reverse-SDE/ODE sampler | 미구현 | 생성·NREM/REM 역과정 claim 잠금 |

**P1: 3x3+1 U-Net**

U-Net의 채널을 SU(3)/SU(2)/U(1)로 분할:
- SU(3): 텍스처/패턴 생성 (고해상도 디테일)
- SU(2): 구조/경계 결정 (중간 해상도)
- U(1): 전역 구성 주의 (저해상도)

**P5: 곡률 기반 생성 품질 제어**

후보 설계에서는 생성 과정의 곡률을 모니터링하고 고곡률 영역을 보정
항으로 누를 수 있다. 다만 현재 저장소에는 이를 score-based 생성기와
결합해 품질 향상을 검증한 구현이 없다.

### 5.2 GAN

**판별자 = ACC (에러 감지)**
**생성자 = 경로 탐색 (각성)**

유니타리 제약(P1)을 판별자에 적용하면 mode collapse를 구조적으로 방지:

$$\sigma_1(D_{\text{proj}}) \leq 1 \implies \text{정보 증폭 없음} \implies \text{mode collapse 방지}$$

이것은 spectral normalization GAN(SN-GAN)이 이미 경험적으로 발견한 것의 CE 이론적 정당화다.

### 5.3 VAE

잠재 공간의 KL-divergence를 곡률 에너지로 대체:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \|\Delta_g z\|^2$$

KL-divergence가 가우시안 사전분포를 가정하는 반면, 곡률 에너지는 잠재 다양체의 기하학적 구조를 직접 정규화한다.

---

## 6. 로보틱스 (Robotics)

Robotics field는 센서 상태 tensor를 행동 command로 바꾸는 폐루프 구현 인터페이스다. 좌표계·물리 단위·control timebase를 정규화하지 않으면 수식 비유가 안전 invariant를 보장하지 않으며, sim-to-real OOD가 명시적 실패 조건이다.

### 6.1 감각운동 통합

**P1: 3x3+1 감각운동 격자**

| CE 층 | 로봇 대응 | 비율 | 뇌 대응 |
|---|---|---|---|
| SU(3) | 다중 센서 융합 | 74.1% | 다감각 통합 |
| SU(2) | 운동 명령 생성 | 21.1% | 운동 피질 |
| U(1) | 센서 선택/주의 | 4.9% | 선택적 주의 |
| $\Phi$ | 안정성 제어 | 전역 | 소뇌 |

### 6.2 실시간 안전

**P5: 곡률 기반 안전 정지**

로봇의 상태 공간에서 곡률이 임계치를 넘으면 즉시 정지:

$$\kappa(q, \dot{q}) > \kappa_{\text{safe}} \implies \text{Emergency Stop}$$

$q$: 관절 각도, $\dot{q}$: 관절 속도. 곡률이 높다 = 급격한 동작 변화 = 위험.

### 6.3 지속 학습 (수면 학습)

로봇이 비작동 시간(충전 중)에 수면 학습을 수행:

- **NREM**: 하루 동안의 경험 중 고곡률(실패/충돌) 우선 처리
- **REM**: 새로운 동작 조합 탐색 (시뮬레이션 기반)

```
작동 시간: 감각-운동 학습 (각성)
충전 시간: NREM (경험 정리) + REM (동작 탐색)
```

이것은 "로봇이 충전 중에 꿈을 꾸는 것"이다.

---

## 7. 그래프 신경망 (GNN)

GNN field는 node·edge feature와 adjacency를 입력으로 받아 message-passing representation을 출력한다. permutation invariant의 정의역과 graph split provenance를 고정하고, edge·feature 제거 ablation에서 consumer 성능을 판정한다.

### 7.1 그래프 라플라시안과 LBO

GNN에서 CE의 적용은 가장 자연스럽다. LBO의 이산화가 곧 그래프 라플라시안이기 때문이다(`7_AGI/12_Equation.md` 5절):

$$\Delta_g \approx L = D - W$$

라플라시안 또는 확산형 GNN의 message passing은 CE의 LBO 연산과 같은
이산 그래프 평활화 꼴로 쓸 수 있다. 다만 attention, edge update와
비선형 aggregation을 쓰는 모든 GNN이 LBO와 정확히 같은 것은 아니다.

**P5: 곡률 = 과평활화(over-smoothing) 제어**

GNN의 핵심 문제인 과평활화를 곡률로 진단한다.

$$\kappa_{\text{graph}} = \sum_v \|\Delta_g h_v\|^2$$

$\kappa \to 0$이면 과평활화 (모든 노드 표현이 동일). $\kappa$의 하한을 유지하면 정보 다양성이 보존된다.

### 7.2 노드 활성 비율

**P4: 그래프의 $4.87\%$ 활성**

전체 노드 중 $4.87\%$만 message passing에 참여:

$$V_{\text{active}} = \text{TopK}(V,\; k = \lceil \varepsilon^2 \cdot |V| \rceil)$$

대규모 그래프(소셜 네트워크, 분자 그래프)에서 전면 희소 실행이 되면 큰 연산 절감 상한이 가능하다. 현재형 구현에서는 먼저 `4-5%` 근방의 활성 최적점이 재현되는지부터 확인해야 한다.

뇌 대응: 뇌의 $10^{11}$ 뉴런 중 동시 활성은 $< 5\%$. 뇌는 이미 "희소 GNN"이다.

---

## 8. 시계열/예측 (Time Series)

Time-series field는 과거 창과 외생 변수를 입력으로 미래 horizon prediction을 출력한다. sampling interval·forecast horizon·누수 없는 split이 timebase 계약이며, distribution shift OOD의 오차가 핵심 실패 조건이다.

### 8.1 시간 주파수 분해 = 3x3+1

시계열의 주파수 성분을 CE 격자로 분해:

| CE 층 | 시계열 대응 | 비율 |
|---|---|---|
| SU(3) | 고주파 성분 (노이즈/변동) | 74.1% |
| SU(2) | 중주파 성분 (계절성) | 21.1% |
| U(1) | 저주파 성분 (추세) | 4.9% |
| $\Phi$ | 전역 안정화 (이상치 억제) | 전역 |

### 8.2 수면 학습 = 온라인 적응

시계열 예측에서 분포 이동(concept drift)에 대응:

- **각성**: 실시간 데이터로 예측
- **NREM**: 오래된 패턴(고곡률) 제거, 새 패턴과 기존 지식의 갈등 해소
- **REM**: 새로운 패턴 조합 탐색

이것은 continual learning의 시계열 버전이다.

### 8.3 이상 감지

**P5: 곡률 스파이크 = 이상**

시계열의 잠재 표현에서 곡률이 급등하면 이상(anomaly):

$$\text{anomaly score}(t) = \|\Delta_g h(t)\|^2$$

정상 데이터는 저곡률, 이상 데이터는 고곡률. 별도의 이상 감지 모델 없이 곡률 에너지만으로 감지 가능.

---

## 9. 추천 시스템 (Recommendation)

Recommendation field는 사용자·아이템·상호작용 로그를 받아 ranking score를 출력한다. 로그 provenance·exposure bias·offline/online timebase를 분리하며, counterfactual baseline과 ablation 없이는 field 효과를 주장하지 않는다.

### 9.1 사용자-아이템 상호작용의 3x3+1

이 분할은 사용자·아이템 feature를 어떤 shape와 정규화로 block consumer에 넘길지 정하는 구현 명세다. 사회적 관계나 생물학적 결합의 비유가 click·conversion metric의 원인 설명을 대체하지 않으며, cold-start OOD와 block 제거가 반증 조건이다.

| CE 층 | 추천 대응 | 비율 |
|---|---|---|
| SU(3) | 협업 필터링 (사용자-아이템 결합) | 74.1% |
| SU(2) | 콘텐츠 기반 필터링 (특성 결정) | 21.1% |
| U(1) | 주의 기반 필터링 (관심 선택) | 4.9% |
| $\Phi$ | 다양성/참신성 정규화 | 전역 |

### 9.2 필터 버블 방지

**P5: 곡률 = 다양성 지표**

추천 결과의 곡률이 너무 낮으면 필터 버블(모든 추천이 비슷):

$$\kappa_{\text{diversity}} = \sum_{(i,j)} \|h_i - h_j\|^2$$

곡률 하한을 유지하면 추천 다양성이 보장된다. 이것은 게임 밸런스(`4_공학적_활용/06_공학식_총람.md`)의 "메타 고착 방지"와 동일한 원리다.

### 9.3 콜드 스타트

**P2: 수면 학습으로 콜드 스타트 해결**

새 사용자/아이템의 초기 표현을 REM 재탐색으로 생성:
- 기존 사용자/아이템의 비선택 경로 풀에서
- 유사한 조합을 탐색하여
- 초기 추천 생성

---

## 10. 생명과학 AI (BioAI)

BioAI field는 assay·sequence·image의 측정 tensor를 예측 또는 후보 생성으로 변환한다. 단위·batch effect·data provenance를 고정하고, 생물학적 기제 비유는 independent wet-lab 또는 held-out assay 없이는 미완성이다.

### 10.1 단백질 접힘 (Protein Folding)

단백질 field는 서열·구조 제약을 입력으로 좌표 또는 접힘 score를 출력하는 contract다. 구조 데이터의 provenance와 residue-level shape를 보존하며, out-of-family OOD와 ablation이 성공·실패를 판정한다.

기존 구현: `examples/biology/sfe_protein_folding.py`

**CE 해석: 단백질 접힘 = 부트스트랩 수렴**

`5_유도/03_Protein_Folding_Derivation.md`에서:
- 단백질의 가능한 구조 = 경로적분의 경로
- 접힘 = 경로의 상쇄 간섭
- 최종 구조 = 고정점 $p^*$

$$S_{\text{fold}} = \int_{\mathcal{M}} d^nx \sqrt{|g|} \left[ E_{\text{contact}} + \lambda\|\Delta_g\Phi\|^2 \right]$$

**P5: 곡률 = 접힘 안정성**

단백질 구조의 곡률 에너지:
- 고곡률 = 불안정한 접힘 (misfolded)
- 저곡률 = 안정한 접힘 (native state)

AlphaFold의 confidence score와 CE 곡률 에너지의 상관을 검증할 수 있다.

### 10.2 약물 발견 (Drug Discovery)

약물 field는 분자·표적·assay tensor를 후보 score 또는 생성 구조로 바꾼다. in-silico metric은 consumer 실험의 대리변수일 뿐이며, 독립 assay와 독성 OOD 실패가 적용 경계를 정한다.

**P1: 분자 그래프의 3x3+1**

분자의 원자를 3x3+1 격자로 분류:
- SU(3): 다중 결합 / 방향족 (결합 강도 = 강력)
- SU(2): 단일 결합 회전 (이성질체 결정)
- U(1): 수소 결합 / 약한 상호작용 (선택적 인식)

**P4: 화학 공간의 희소 탐색**

가능한 분자 공간($\sim 10^{60}$)에서 $4.87\%$만 활성 탐색하는 것은 강한 설계 가설이다. 기존 가상 스크리닝 대비 효율 개선은 검증 게이트를 통과하기 전까지 상한 예측으로만 둔다.

### 10.3 의료 영상 (Medical Imaging)

의료 영상 field는 정규화된 pixel·voxel tensor를 진단 보조 출력으로 바꾼다. 환자·기관 split, acquisition timebase, label provenance를 고정하고 false positive·negative가 baseline보다 악화되면 실패다.

**P5: 곡률 = 진단 신뢰도**

CT/MRI 분석에서 곡률 에너지가 높은 영역 = 분류 불확실성이 높은 영역:

$$\text{uncertainty}(x) \propto \|\Delta_g h(x)\|^2$$

고곡률 영역을 방사선과 전문의에게 우선 검토 대상으로 플래그한다. 이것은 AI 진단의 안전성을 곡률로 보장하는 것이다.

---

## 11. 자연어 처리 비-LLM (NLP Beyond LLM)

NLP field는 token·span·label tensor를 분류·구조 예측 consumer로 넘기는 구현 contract다. tokenizer와 corpus provenance, domain OOD, feature ablation이 정규화 field의 실제 효능을 반증한다.

### 11.1 정보 추출 (IE)

IE field는 문서 token과 span 후보를 받아 entity·relation label을 출력한다. annotation provenance와 document-level split을 고정하고, unseen schema OOD에서 오탐·미탐을 기준선과 비교한다.

**P1: 엔티티/관계/이벤트 = SU(3)/SU(2)/U(1)**

- SU(3) = 엔티티 인식 (다수의 토큰을 하나의 개체로 결합)
- SU(2) = 관계 분류 (두 엔티티 간 이진 관계 결정)
- U(1) = 이벤트 감지 (핵심 사건 선택)

### 11.2 기계 번역

번역 field는 source token sequence를 target distribution으로 바꾸는 time-indexed tensor contract다. 언어쌍 provenance·길이 분포·domain shift를 기록하며, fluency 비유가 adequacy 또는 안전성을 보장하지 않는다.

**P2: 수면 학습 = 다국어 적응**

새 언어 페어에 적응할 때:
- 각성: 새 언어 데이터 학습
- NREM: 기존 언어 지식과의 간섭(고곡률) 해소
- REM: 제로샷 번역 (보지 않은 언어 페어의 조합 탐색)

---

## 12. 자율주행 (Autonomous Driving)

Driving field는 다중 센서 상태를 trajectory 또는 control output으로 바꾸는 안전 중요 인터페이스다. 좌표·latency·sensor timebase와 fail-safe invariant를 명시하며, simulation 성능은 실제 OOD 안전을 증명하지 않는다.

### 12.1 인지-판단-제어 = SU(3)-SU(2)-U(1)

이 분할은 perception·decision·control tensor의 producer와 consumer를 구분하는 구현 비유다. 물리 게이지군 또는 인간 운전 기제와의 동일성이 아니며, 모듈 제거와 sensor corruption OOD가 반증 조건이다.

자율주행의 3단계가 CE 격자와 정확히 대응:

| 단계 | CE 층 | 비율 | 역할 |
|---|---|---|---|
| 인지 (Perception) | SU(3) | 74.1% | 센서 데이터 융합, 객체 인식 |
| 판단 (Planning) | SU(2) | 21.1% | 경로 결정, 위험 평가 |
| 제어 (Control) | U(1) | 4.9% | 조향/가속/제동 선택 |
| 안전 (Safety) | $\Phi$ | 전역 | 전역 안정성 모니터링 |

자율주행 시스템에서 compute budget sweep를 하면, 인지/판단/제어의 Pareto knee가 `74/21/5` 근방에 나타나는지가 CE의 점검 대상이다. 아직 설계 명령으로 단정하지는 않는다.

### 12.2 안전 보장

안전 invariant는 command output의 범위와 latency 조건을 명시할 때만 검사할 수 있다. metric 통과는 제한된 scenario의 결과이며, rare-event OOD 또는 fail-safe 위반은 즉시 실패·rollback 조건이다.

**P5: 곡률 = 위험도**

주행 상태 공간에서의 곡률:

$$\kappa_{\text{drive}}(s) = \|\Delta_g h(s)\|^2$$

- $\kappa < \kappa_{\text{normal}}$: 정상 주행
- $\kappa_{\text{normal}} \leq \kappa < \kappa_{\text{caution}}$: 주의 (속도 감소)
- $\kappa \geq \kappa_{\text{caution}}$: 위험 (인간 개입 요청)

이것은 뇌의 ACC-PFC 안전 회로의 자율주행 구현이다.

---

## 13. 전 분야 통합 요약

통합 표는 field별 contract의 공통점과 남은 자유도를 비교하는 설계 지도다. 행마다 baseline·provenance·OOD·ablation gate가 독립적으로 필요하므로, 표 전체가 분야별 성공을 뜻하지 않는다.

### 13.1 CE 원리별 적용 매트릭스

매트릭스는 각 원리가 어떤 producer·consumer 경계에서 구현 후보가 되는지 요약한다. 체크 표시는 효능 검증이 아니라 인터페이스 가능성이고, 누락 또는 실패는 해당 field의 반증 기록으로 남긴다.

| 분야 | P1 격자 | P2 수면 | P3 STDP | P4 희소 | P5 곡률 |
|---|---|---|---|---|---|
| 비전(CNN/ViT) | 채널 분할 | 지속 학습 | -- | Top-k Conv | 적대적 강건성 |
| 강화학습 | 행동 분할 | 경험 재생 | TD-유사 전역 신호 | 희소 정책 | 안전 제약 |
| 음성/오디오 | 주파수 분할 | 화자 적응 | -- | 희소 인코딩 | 환각 억제 |
| 멀티모달 | 모달 분할 | 모달 적응 | -- | 모달 활성 | 교차 환각 |
| 생성(Diffusion) | U-Net 분할 | 수면 단계와의 연구 비유 | -- | 희소 샘플링 | 품질 제어 |
| 로보틱스 | 감각운동 분할 | 충전=수면 | 국소 학습 | 희소 제어 | 안전 정지 |
| GNN | 노드 분할 | 그래프 적응 | message=STDP | 노드 활성 | 과평활화 제어 |
| 시계열 | 주파수 분할 | 분포 이동 | -- | 희소 예측 | 이상 감지 |
| 추천 | 필터 분할 | 콜드 스타트 | -- | 희소 후보 | 다양성 보장 |
| 생명과학 | 분자 분할 | 적응 학습 | -- | 희소 탐색 | 접힘 안정성 |
| 자율주행 | 인지/판단/제어 | 야간 학습 | -- | 희소 인지 | 위험 감지 |

### 13.2 가장 즉각적인 적용

즉각성은 구현 비용과 기존 tensor interface의 적합성을 뜻하는 계획 판단이다. 빠른 적용은 일반화·안전·생물학적 타당성의 완료 주장이 아니며, 각 후보는 baseline과 OOD gate를 통과해야 한다.

난이도와 영향력 기준으로 우선순위:

| 순위 | 분야 | 적용 원리 | 이유 |
|---|---|---|---|
| 1 | ViT | P1+P4+P5 | LLM과 동일 아키텍처, 코드 재사용 |
| 2 | GNN | P5 | LBO가 그래프 라플라시안과 직접 대응 |
| 3 | 확산 모델 | P5+P2 | 열·확률·score 확산을 분리한 뒤 구조 비교 가능 |
| 4 | 강화학습 | P2+P5 | TD error와 CE 전역 신호의 구조적 유사성을 점검 가능 |
| 5 | 자율주행 | P1+P5 | 안전 임계적 분야, 곡률 기반 안전 보장 |

### 13.3 공통 구현 패턴

공통 패턴은 shape 정렬·정규화·update·감시의 재사용 가능한 순서를 제안한다. pattern 자체가 invariant를 보장하지 않으므로 field별 consumer metric과 component ablation으로 실패를 찾는다.

다음은 여러 분야에서 비교할 구조를 요약한 **개념 pseudocode**다.
`LBONorm`이라는 그대로 import 가능한 현재 API나 score-based diffusion
구현을 뜻하지 않는다.

```python
class CEModule(nn.Module):
    def __init__(self, dim):
        self.norm = LBONorm(dim)           # P1: LBO 확산
        self.ffn = GaugeLattice(dim)       # P1: 3x3+1 격자
        self.proj = spectral_norm(Linear)  # P1: 유니타리 제약

    def forward(self, x):
        x = self.norm(x)                  # 곡률 측정 + 평탄화
        x = self.ffn(x)                   # 게이지 격자 처리
        x = self.proj(x)                  # 정보 증폭 차단
        return x

    def curvature(self):
        return self.norm._curvature       # P5: 곡률 모니터링

    def sparse_forward(self, x):          # P4: 추론 시 희소화
        y = self.forward(x)
        k = int(0.0487 * y.shape[-1])
        return topk(y, k)
```

이 패턴은 다양한 도메인에서 **공통 출발점**으로 재사용할 수 있다. 다만 입력 구조, 손실 함수, 안전 조건, latency 제약은 도메인마다 다르므로 그대로 복사하는 것이 아니라 게이트를 통과시키며 맞춰야 한다.

---

## 14. 분야별 예측 게이트

예측 gate는 계획을 경험적 판정으로 바꾸기 위해 metric·baseline·data provenance·OOD 조건을 사전 고정한다. 성공은 등록된 비교의 통과이고, 미통과·오탐 증가·미탐 유지·invariant 위반은 하향 또는 중단 조건이다.

이 장에서 바로 걸 수 있는 핵심 게이트만 압축하면 다음과 같다.

| 분야 | 예측 | 측정량 | 통과 게이트 | 실패 시 해석 |
|---|---|---|---|---|
| 비전 | sparse conv/ViT의 knee가 `4-5%` 근방 | accuracy-latency Pareto | 실용 대역이 `3-7%` 안에 남음 | 희소 고정점이 과제 의존적 |
| 강화학습 | sleep replay가 wake-only보다 drift를 줄임 | return, forgetting, residual proxy | replay with sleep가 장기 성능 우위 | P2를 RL 일반론으로 못 올림 |
| 음성 | 곡률 상승이 잘못된 전사와 상관 | WER, hallucination-like insertion, curvature | 상관계수 양수 | P5는 일반 안정화 regularizer |
| 멀티모달 | late sparse binding이 early fusion보다 grounding 우위 | hallucination, grounding score, alignment | Text-only/Early fusion 대비 개선 | 결합 순서 가설 약화 |
| GNN | 활성 중심이 `4-5%` 근방 | active node ratio vs accuracy-cost | `3-7%` 대역 유지 | graph sparsity는 데이터 의존적 |
| 시계열 | sleep adaptation이 concept drift를 완화 | rolling error, drift residual | sleep-on이 wake-only보다 안정 | P2 시계열 해석 하향 |
| 생명과학 | 곡률과 confidence가 상관 | confidence/calibration, curvature | 양의 상관 | P5를 품질 보조 지표로 제한 |
| 자율주행 | 인지/판단/제어 budget sweep에서 CE 비율 근방 knee | success, latency, safety intervention | Pareto knee가 근방에 존재 | 74/21/5를 설계 힌트로만 유지 |

### 14.1 공통 하향 규칙

하향 규칙은 field 비유·구현 tensor·효능 가설을 구분한 fail-closed 규약이다. 사후 metric 교체로 OOD·ablation 실패를 성공으로 재해석하지 않는다.

- `4.87%` 근방 최적점이 반복 재현되지 않으면: 해당 도메인에서 P4를 일반 법칙이 아니라 과제군 가설로 내린다.
- 곡률과 오류가 상관하지 않으면: P5를 환각/오분류 억제가 아니라 안정화 regularizer로 제한한다.
- 수면 루프가 wake-only보다 낫지 않으면: P2를 그 도메인에서는 아직 미닫힌 브리지로 유지한다.
- 모달 결합 이득이 없으면: grounded late binding 구조를 기본값이 아니라 옵션 설계로 재분류한다.
