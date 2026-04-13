# 뇌 검증 매트릭스

> 역할: `anatomy.md`, `measure.md`, `control.md`, `graph.md`, `evidence.md`에 흩어진 주장을 `무엇이 증명 완료인지` 기준으로 묶는다.
>
> 목적: 각 부위별, 중추별, 알고리즘 모듈별로 `수학적 일관성 / 통계적 식별 / 인과 개입 / 예측력`을 모두 통과해야만 "맞다"로 올리는 기준을 고정한다.

---

## 1. 무엇을 증명해야 하는가

이 프로젝트에서 "증명"은 한 층이 아니라 네 층을 모두 통과해야 한다.

$$
\Pi_r
=
G_{\text{formal},r}
\cdot
G_{\text{obs},r}
\cdot
G_{\text{causal},r}
\cdot
G_{\text{pred},r}
$$

여기서 `r`는 특정 부위, 중추, 또는 제어축이다.

| 게이트 | 의미 | 통과 조건 |
|---|---|---|
| `G_formal` | 식 자체가 내부적으로 말이 되는가 | 상태공간 보존, 부호 조건, 안정성 조건, 식별 가능한 연산자 정의 |
| `G_obs` | 실제 데이터에서 그 변수가 분리되는가 | 재현 가능한 관측 proxy, 홀드아웃에서 방향 일치, bootstrap 부호 안정성 |
| `G_causal` | 그 부위를 흔들면 모델이 예측한 방향으로 변하는가 | 자극/병변/약물/수면박탈/자율신경 조작 등에서 부호 일치 |
| `G_pred` | 넣었을 때 실제 예측력이 올라가는가 | baseline 대비 out-of-sample 성능 증가, CI가 0 초과 |

따라서 `\Pi_r = 1`일 때만 그 부위/중추/제어축에 대해 "증명 완료"라고 부른다.

### 1.1 현재 상태 표기

아래 표의 `pass / partial / fail`은 **현재 진행도**를 뜻한다.  
최종 의미는 다음처럼 고정한다.

| 상태 | 의미 |
|---|---|
| `pass` | `Formal / Observable / Causal / Predictive` 네 게이트가 모두 통과 |
| `partial` | `Formal`은 닫혔고, 나머지 게이트 중 적어도 하나는 부분적으로 살아 있음 |
| `fail` | 사실상 `Formal`만 남아 있거나, 관측/인과/예측이 모두 비어 있어 현재 채택 불가 |

즉 이 문서에서 `partial`은 "가능성 있음"이지 "증명 완료"가 아니다.

---

## 2. 전역 알고리즘 게이트

부위별 검증 전에 전체 알고리즘이 먼저 통과해야 하는 전역 게이트를 둔다.

$$
\Pi_{\text{alg}}
=
G_{\Delta^2}
\cdot
G_{p}
\cdot
G_{q}
\cdot
G_{G}
\cdot
G_{s}
\cdot
G_{w}
$$

| 모듈 | 핵심 주장 | 수학적으로 증명해야 할 것 | 실험적으로 확인해야 할 것 | 즉시 반증 조건 | 현재 상태 |
|---|---|---|---|---|---|
| `p_r` simplex | `p_r=(x_a,x_s,x_b)`가 항상 유효한 상태다 | `x_a+x_s+x_b=1`, 비음수 보존 | multimodal normalization이 안정적으로 동작 | 성분 하나가 음수/합이 1에서 반복 이탈 | `bridge` |
| `x_a` proxy | task recruitment와 국소 cortical overdrive를 `x_a`로 읽는다 | `x_a` 증가가 `s_r`를 올리는 부호 일관성 | EEG/fMRI/PET와, GBM이면 PBZ hyperexcitability proxy가 같은 방향을 가리킴 | 과제 burden과 분리되지 않거나 PBZ signal과도 무관 | `supported/bridge` |
| `x_s` decomposition | maintenance/plasticity/replay/support를 합쳐 `x_s`로 읽는다 | 세분화해도 정규화가 닫힘 | 분해한 항이 지역별로 구분 가능 | 모든 항이 한 값으로만 뭉쳐 식별 불가 | `bridge` |
| `q_n` control vector | sleep/autonomic/endocrine/immune/metabolic burden을 하나의 느린 제어벡터로 묶는다 | `\ell_r = d_r^\top(q_n-q^*)_+`가 well-defined | 각 축이 독립적이지 않더라도 추정 가능 | 축들이 완전히 구분되지 않거나 방향이 뒤집힘 | `bridge` |
| `\Delta_G` graph coupling | 인접 회로가 spread/buffering을 만든다 | graph-Laplacian 항으로 안정성 조건 유도 | connectivity-informed model이 flat index보다 예측력 우위 | graph 항을 넣어도 성능 이득 0 | `bridge` |
| `s_r` vulnerability | `s_r`가 과활성 + 구조저하 + 배경저하 + body-loop burden 합이다 | 각 항의 부호가 취약도 증가 방향과 일치 | `s_r`가 실제 stress/질병 burden과 함께 증가 | 한 항의 부호가 반복적으로 반대 | `bridge` |
| `w_r` coupling | `s_r` 증가, 특히 GBM의 `V_ctx/PBZ` 과흥분이 `w_r` 또는 `M_eff` 증가로 연결된다 | 결합계 안정성, `K_r` 조건 | 암 데이터에서 취약도 및 PBZ direct-drive와 mismatch 방향 일치 | 취약도나 PBZ 과흥분이 높아도 mismatch와 무관 | `bridge/hypothesis` |

### 2.1 전역 알고리즘 현재 진행도

| 모듈 | Formal | Obs | Causal | Pred | 전체판정 | 현재 이유 |
|---|---|---|---|---|---|---|
| `p_r` simplex | `pass` | `partial` | `partial` | `fail` | `partial` | stage-1 `\lambda_r` 정리 닫힘. simplex invariance 충분조건 (정리 9.1), 지수 수렴 (정리 10.1), noise bound (따름정리 10.2) 닫힘. ds000201 v3 방향 유지. 예측 미완 |
| `x_a` proxy | `pass` | `pass` | `partial` | `fail` | `partial` | sparse/task evidence 강함. ds000201 v3에서 sleepiness task 방향 유지 (`delta=+0.006, pos=67%, p=0.39`). fMRI proxy noise가 큼 |
| `x_s` decomposition | `pass` | `partial` | `fail` | `fail` | `partial` | 분해식은 닫혔지만 항별 식별이 아직 약함 |
| `q_n` control vector | `pass` | `partial` | `partial` | `fail` | `partial` | 축별 근거는 있으나 하나의 벡터로서 예측 성능은 미검증 |
| `\Delta_G` graph coupling | `pass` | `partial` | `fail` | `fail` | `partial` | 해부학 연결은 있으나 graph 항의 실측 이득은 아직 없음 |
| `s_r` vulnerability | `pass` | `partial` | `fail` | `fail` | `partial` | 합성식은 닫히나 composite 자체의 실측 추정이 부족 |
| `w_r` coupling | `pass` | `fail` | `fail` | `fail` | `fail` | tumor mismatch는 있으나 brain-side `s_r -> w_r` 및 `V_ctx/PBZ -> w_r` direct path 결합은 직접 검증되지 않음 |

### 2.2 `p_r`의 stage-1 joint theorem

`measure.md`에서는 `x_a/x_b`의 첫 실측을

$$
u_{a,r,n} := \hat x_{a,r}^{(1)}(n),
\qquad
u_{b,r,n} := \hat x_{b,r}^{(1)}(n)
$$

로 두고, pair share를

$$
\lambda_{r,n}
:=
\frac{u_{a,r,n}}{u_{a,r,n}+u_{b,r,n}}
$$

로 정의한다.

이때 `x_s = x_s^*`를 고정한 stage-1 simplex

$$
\hat p_{r,n}^{\text{stage-1}}
=
\Big(
(1-x_s^*)\lambda_{r,n},
\;
x_s^*,
\;
(1-x_s^*)(1-\lambda_{r,n})
\Big)
$$

는 자동으로 `\Delta^2` 안에 있고,

$$
\big\|\hat p_{r,n}^{\text{stage-1}} - p^*\big\|_2
=
\sqrt{2}\,(1-x_s^*)\,|\lambda_{r,n}-\lambda^*|
$$

를 만족한다. 여기서

$$
\lambda^*
=
\frac{x_a^*}{1-x_s^*}
\approx 0.0660
$$

이다.

따라서 `p_r`의 최소 형식 게이트는 이미 stage-1에서 닫혀 있다.
- nonnegative `u_a,u_b`는 stage-1 simplex의 비음수와 합=1을 바로 준다
- `\lambda_r` 하나만 추정하면 `x_a/x_b` pair를 `p_r` 좌표계로 옮길 수 있다
- 현재 `MSC` pilot의 `\lambda=0.0701`은 `\lambda^*=0.0660`과 가깝고, 실제로 `\|\hat p_{\text{stage-1}}-p^*\|_2=0.0042`가 나온다

즉 `p_r`는 현재 문서 체계에서 "아직 전부 관측된 변수"는 아니지만, 적어도 `x_a/x_b`와 `x_s^*`를 묶는 최소 기하학은 이미 정리로 닫혀 있다.

---

## 3. 부위별 증명 매트릭스

### 3.1 피질 `V_ctx`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 피질은 task recruitment와 intrinsic background의 주 관측창이며, GBM에서는 PBZ hyperexcitability를 통해 종양측 직접 입력이 될 수 있다 |
| 핵심 변수 | `x_a`, `x_b`, `\Delta_G`, 일부 `x_s` |
| `G_formal` | `x_a` 증가가 `s_r` 증가로, `x_b` 감소가 `s_r` 증가로 가는 부호가 닫혀야 함 |
| `G_obs` | task-positive EEG/fMRI/PET와 resting/DMN proxy가 분리되어야 하며, GBM에서는 PBZ hyperexcitability / seizure burden / neuron-glioma synaptic marker와도 정렬 가능해야 함 |
| `G_causal` | 감각 자극, 과제 부하, TMS, 수면박탈, anti-epileptic/AMPAR modulation에서 `x_a`/`x_b` 또는 tumor-adjacent cortical drive 방향 변화가 예측과 일치해야 함 |
| `G_pred` | 피질 변수 추가 시 cognitive burden 또는 disease burden, 특히 GBM `w_r` / recurrence burden 예측력이 baseline보다 좋아야 함 |
| 즉시 반증 조건 | 과제 반응과 배경 활동이 일관되게 분리되지 않거나 PBZ hyperexcitability가 tumor burden과 무관 |
| 현재 판정 | `supported/bridge` |

### 3.2 시상 `V_thal`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 시상은 relay/gating과 bandwidth 재배분의 중심이다 |
| 핵심 변수 | `x_a`, `x_b`, `\Delta_G` |
| `G_formal` | relay node가 graph에서 gating hub로 작동할 때 coupled relaxation이 잘 정의되어야 함 |
| `G_obs` | thalamo-cortical proxy가 cortical-only model보다 분리된 정보를 줘야 함 |
| `G_causal` | arousal modulation, sensory gating 과제, thalamic lesion/stimulation에서 예측 부호가 맞아야 함 |
| `G_pred` | 시상 항 추가 시 vigilance, relay failure, state transition 예측이 개선되어야 함 |
| 즉시 반증 조건 | 시상 정보를 넣어도 cortical-only baseline과 차이가 없음 |
| 현재 판정 | `bridge` |

### 3.3 해마 `V_hip`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 해마는 replay, episodic indexing, memory binding의 중심이다 |
| 핵심 변수 | `x_s`, `x_a`, `R_r`, `P_r` |
| `G_formal` | replay burden과 plasticity burden이 `x_s` 안에 안정적으로 묶여야 함 |
| `G_obs` | replay proxy와 memory/plasticity proxy가 지역적으로 구분 가능해야 함 |
| `G_causal` | 수면 교란, stress, memory task intervention에서 `x_s`, `s_r` 변화 방향이 맞아야 함 |
| `G_pred` | replay 항을 넣으면 memory-related burden 또는 downstream vulnerability 예측이 향상되어야 함 |
| 즉시 반증 조건 | replay를 넣어도 `x_s` 설명력이 늘지 않음 |
| 현재 판정 | `bridge` |

### 3.4 살리언스 허브 `V_sal`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | salience hub는 switching, gain control, value/risk tagging을 담당한다 |
| 핵심 변수 | `x_a`, `x_s`, `q_arousal`, `q_endo` |
| `G_formal` | gain shift가 `x_a` 또는 `x_s` 방향으로 일관되게 투사되어야 함 |
| `G_obs` | salience-related task에서 별도 proxy가 안정적으로 분리되어야 함 |
| `G_causal` | threat/reward manipulation, dopaminergic modulation에서 부호가 맞아야 함 |
| `G_pred` | salience 항 추가 시 switching failure 또는 stress reactivity 예측력이 올라가야 함 |
| 즉시 반증 조건 | salience 지표가 피질 activation과 완전히 중복 |
| 현재 판정 | `bridge` |

### 3.5 시상하부 `V_hyp`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 시상하부는 circadian, endocrine, metabolic set-point의 핵심이다 |
| 핵심 변수 | `q_sleep`, `q_endo`, `q_met`, `x_b` |
| `G_formal` | 느린 제어축이 `\ell_r`를 통해 지역 취약도로 투사되는 식이 well-defined여야 함 |
| `G_obs` | sleep/endocrine/metabolic marker가 독립 축으로 추정 가능해야 함 |
| `G_causal` | 수면 제한, 식이/호르몬 조작, circadian shift에서 예측 부호가 맞아야 함 |
| `G_pred` | hypothalamic control 항을 넣으면 stress-homeostasis 예측이 개선되어야 함 |
| 즉시 반증 조건 | 제어축 변화가 cortical vulnerability와 무관 |
| 현재 판정 | `bridge` |

### 3.6 뇌간 `V_stem`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 뇌간은 arousal tone과 autonomic correction의 중심이다 |
| 핵심 변수 | `q_arousal`, `q_aut`, 일부 `x_b` |
| `G_formal` | arousal/autonomic forcing이 지역 상태와 결합될 때 부호가 일관되어야 함 |
| `G_obs` | pupil, HRV, vigilance EEG가 별도 축으로 작동해야 함 |
| `G_causal` | vagal stimulation, sleep deprivation, neuromodulatory challenge에서 부호 일치 |
| `G_pred` | brainstem control 축 추가 시 vigilance/autonomic burden 예측이 향상 |
| 즉시 반증 조건 | autonomic marker 변화가 뇌 상태 변화와 분리되지 않음 |
| 현재 판정 | `bridge` |

### 3.7 자율신경 출력 `V_aut`

| 항목 | 내용 |
|---|---|
| 모델상 주장 | 자율신경 출력은 body-coupling의 직접 관문이다 |
| 핵심 변수 | `q_aut`, `\ell_r`, `s_r \to w_r` |
| `G_formal` | 자율신경 burden이 지역 취약도로 투사되는 map이 well-defined여야 함 |
| `G_obs` | HRV, blood pressure variability, respiration coupling이 안정적으로 측정되어야 함 |
| `G_causal` | vagal/sympathetic intervention에서 예측 부호가 맞아야 함 |
| `G_pred` | autonomic 항이 disease progression 또는 vulnerability 예측을 개선해야 함 |
| 즉시 반증 조건 | autonomic 축을 넣어도 예측 이득이 전혀 없음 |
| 현재 판정 | `bridge/hypothesis` |

### 3.8 부위별 현재 진행도

| 부위/중추 | Formal | Obs | Causal | Pred | 전체판정 | 현재 이유 |
|---|---|---|---|---|---|---|
| `V_ctx` | `pass` | `pass` | `partial` | `fail` | `partial` | cortical sparse activity와 DMN 근거는 강하지만 PBZ hyperexcitability를 포함한 tumor-side 예측 검증은 아직 없음 |
| `V_thal` | `pass` | `partial` | `fail` | `fail` | `partial` | relay/gating 해석은 자연스럽지만 직접 분리 근거가 아직 약함 |
| `V_hip` | `pass` | `partial` | `partial` | `fail` | `partial` | replay/engram/plasticity 방향은 있으나 정량 closure는 없음 |
| `V_sal` | `pass` | `fail` | `fail` | `fail` | `fail` | salience hub를 별도 노드군으로 세운 것은 아직 구조 가정 비중이 큼 |
| `V_hyp` | `pass` | `partial` | `partial` | `fail` | `partial` | circadian/endocrine/metabolic 중심축이라는 방향은 강하지만 모델 예측은 미검증 |
| `V_stem` | `pass` | `partial` | `partial` | `fail` | `partial` | arousal/autonomic 중심축으로 읽는 건 자연스럽지만 정량식은 약함 |
| `V_aut` | `pass` | `partial` | `partial` | `fail` | `partial` | HRV 등 관측은 강하지만 brain node family로의 통합은 아직 bridge |

---

## 4. 제어축별 증명 매트릭스

| 축 | 모델상 주장 | `G_obs` 핵심 | `G_causal` 핵심 | `G_pred` 핵심 | 즉시 반증 조건 | 현재 판정 |
|---|---|---|---|---|---|---|
| `q_sleep` | 수면 부채와 circadian misalignment가 `x_b`, `s_r`, 일부 perivascular clearance를 흔든다 | slow-wave/activity, actigraphy, latency, melatonin, resting EEG posterior alpha reactivity, glymphatic proxy 재현성 | 수면 제한/회복, circadian shift에서 부호 일치 | sleep 축 추가 시 burden 또는 GBM immune/mech mismatch 예측 향상 | sleep/glymphatic 지표가 취약도와 무관 | `supported/bridge` |
| `q_arousal` | hyperarousal이 `x_a`, `s_r`를 민다 | pupil, vigilance EEG, arousal index 분리 | arousal challenge, wake extension | arousal 축 추가 시 상태전환 예측 개선 | arousal marker가 과제부하와 구분되지 않음 | `bridge` |
| `q_aut` | sympathetic-vagal imbalance가 취약도를 높인다 | HRV, BPV, respiration coupling | vagal/sympathetic 조작 | autonomic 축 추가 시 예측 개선 | HRV 축이 방향성을 못 가짐 | `supported` |
| `q_endo` | endocrine stress가 `s_r`를 올린다 | cortisol profile 안정성 | HPA modulation, stress paradigm | endocrine 축 추가 시 stress-burden 예측 향상 | cortisol 축이 방향 불일치 | `supported` |
| `q_immune` | inflammatory burden이 `s_r \to w_r`를 민다 | cytokine/CRP signal 안정성 | inflammatory perturbation | immune 축 추가 시 mismatch 예측 개선 | immune marker가 mismatch와 무관 | `supported` |
| `q_met` | metabolic reserve 부족이 취약도와 mismatch를 민다 | glucose/perfusion/temperature 분리 | metabolic perturbation | metabolic 축 추가 시 예측 개선 | metabolic proxy가 불안정 | `bridge` |

### 4.1 제어축 현재 진행도

| 제어축 | Formal | Obs | Causal | Pred | 전체판정 | 현재 이유 |
|---|---|---|---|---|---|---|
| `q_sleep` | `pass` | `pass` | `pass` | `fail` | `partial` | `ds004902` EEG: alpha reactivity `p=0.040, n=19`. `ds000201` fMRI v3 (`n=18`): KSS `p=0.0003`, `x_a` 방향 유지 (`+0.006, pos=67%`), `x_b` segregation 방향 유지 (`-0.005, pos=67%`). 개별 proxy는 유의하지 않으나 3개 proxy 방향이 모두 일치. holdout 예측 미완 |
| `q_arousal` | `pass` | `partial` | `partial` | `fail` | `partial` | arousal marker는 있으나 과제부하와 분리된 예측은 미약 |
| `q_aut` | `pass` | `pass` | `partial` | `fail` | `partial` | HRV 축은 강하지만 본 모델 안에서의 홀드아웃 이득은 없음 |
| `q_endo` | `pass` | `pass` | `partial` | `fail` | `partial` | endocrine stress 축은 강하나 region-level prediction은 비어 있음 |
| `q_immune` | `pass` | `pass` | `partial` | `fail` | `partial` | inflammatory burden 자체는 강하지만 brain-side causal chain은 약함 |
| `q_met` | `pass` | `partial` | `fail` | `fail` | `partial` | metabolic reserve 방향은 타당하지만 측정과 개입이 아직 약함 |

### 4.2 `q_sleep`의 현재 실증 근거

`measure.md`의 `ds004902` pilot은 현재 레포 안에서 재현 가능한 `q_sleep`의 첫 observable/causal closure다.

운영상 핵심 관측량은

$$
r_\alpha
:=
\frac{\alpha_{\text{closed}}}{\alpha_{\text{open}}}
$$

인 posterior alpha reactivity ratio다.

현재 세션에서 eyes-open/eyes-closed complete pair `19`명 기준으로

$$
\text{median NS}=2.1224,
\qquad
\text{median SD}=1.2490,
\qquad
\Delta_{\text{mean}}=-1.1582,
\qquad
p=0.0401
$$

이 나왔고, same-dataset 행동축도

$$
\text{SSS}: 2.0 \to 5.0,\qquad p=2.75\times 10^{-4}
$$

$$
\text{PVT RT}: 320 \to 359,\qquad p=1.45\times 10^{-5}
$$

로 함께 악화됐다.

따라서 현재 판정은 다음처럼 읽는다.
- `G_obs`: deprivation label이 행동 지표와 EEG reactivity 양쪽에서 분리된다
- `G_causal`: sleep deprivation 자체가 개입이며, 그 개입에서 `r_\alpha \downarrow`와 vigilance burden 증가가 함께 나온다
- `G_pred`: 아직 `q_sleep`를 넣었을 때 홀드아웃 예측이나 GBM immune/mech mismatch 예측이 실제로 얼마나 좋아지는지는 검증되지 않았다

즉 `q_sleep`는 현재 문서 체계에서 이미 `관측 가능`하고 `개입 부호`도 확인된 축이지만, 예측 게이트가 비어 있으므로 전체 등급은 여전히 `partial`이다.

### 4.2.1 `q_sleep`의 ds000201 fMRI 실증

`ds004902` EEG에 이어, `ds000201` fMRI cohort에서 `x_a/x_b` joint proxy까지 확인했다.

**v2 (n=9) pilot**: KSS `p=0.008`, `x_a` `+0.020` (`p=0.43`), `x_b` segregation `-0.024` (`p=0.074`), active share `+0.039` (`p=0.16`)

**v3 (n=18) 확장 cohort**: KSS 역방향 1명 제외, label 추론 로직 추가 (한쪽만 label이 있으면 반대쪽 추론, 양쪽 다 없으면 KSS 차이로 추론)

$$
\text{KSS}: \Delta_{\text{mean}} = +1.71,
\qquad
p = 2.9\times 10^{-4}
$$

$$
\text{active responsive fraction}: \Delta_{\text{mean}} = +0.006,
\qquad
\text{positive frac} = 0.67,
\qquad
p = 0.39
$$

$$
\text{network segregation}: \Delta_{\text{mean}} = -0.005,
\qquad
\text{positive frac} = 0.33,
\qquad
p = 0.32
$$

$$
\text{active share}: \Delta_{\text{mean}} = +0.009,
\qquad
\text{positive frac} = 0.61,
\qquad
p = 0.37
$$

해석:
- 수면 압력 조작은 표본이 커지면서 훨씬 강해졌다 (`p = 0.008` -> `p = 0.0003`)
- `x_a`, `x_b`, active share의 방향은 모두 CE 이론과 일치한다 (3개 proxy 방향 동시 일치)
- 다만 개별 fMRI proxy의 effect size는 n=9 pilot보다 줄었고, 유의하지 않다
- n=9에서 보였던 강한 effect는 표본 선택 편향일 가능성이 있다
- fMRI voxel-level proxy의 개인 간 variability가 크다
- EEG (`ds004902`) + fMRI (`ds000201`) 두 modality에서 같은 방향이 나온다는 점은 여전히 유효하다
- proxy 정밀도 향상을 위해 atlas 기반 parcellation 또는 regional proxy 전환이 필요하다

### 4.3 `q_sleep`의 형식 정리

`control.md`와 `graph.md`에서는 `q_sleep`를

$$
z_n := (q_{\text{sleep},n}-q_{\text{sleep}}^*)_+
$$

로 낮추고, sleep forcing를

$$
h_r^{\text{sleep}} = (\alpha_r,\; 0,\; -\alpha_r),
\qquad
\alpha_r \ge 0
$$

로 두었다.

이 선택의 의미는 분명하다.

$$
q_{\text{sleep}} \uparrow
\Longrightarrow
x_a \uparrow,
\qquad
x_b \downarrow,
\qquad
x_s \text{ unchanged}
$$

그리고 `graph.md`의 명제 8.1과 따름정리 8.2에 의해, threshold kink를 제외한 점에서

$$
\frac{\partial x_{b,r,n+1}}{\partial z_n} = -\alpha_r
$$

$$
\frac{\partial s_{r,n+1}}{\partial z_n}
=
\eta_a \alpha_r \mathbf 1_{\{x_{a,r,n+1}>x_a^*\}}
+
\eta_b \alpha_r \mathbf 1_{\{x_{b,r,n+1}<x_b^*\}}
+
\eta_q d_{r,\text{sleep}}
\ge
\eta_q d_{r,\text{sleep}}
\ge 0
$$

가 성립한다.

따라서 `q_sleep`에 대해 필요한 최소 형식 부호

$$
q_{\text{sleep}} \uparrow
\Longrightarrow
x_b \downarrow
\Longrightarrow
s_r \uparrow
$$

는 현재 문서 체계 안에서 이미 닫혀 있다.

이 절이 닫아 주는 것은 `Formal` 게이트뿐이다. 실제로 그 부호가 데이터와 개입에서 맞는지는 `4.2`의 `ds004902` 결과가 받쳐 주고, 예측 게이트는 아직 남아 있다.

---

## 5. 결합 알고리즘 증명 매트릭스

| 모듈 | 알고리즘 주장 | 수학적 게이트 | 실험/데이터 게이트 | 인과 게이트 | 즉시 반증 조건 | 현재 상태 |
|---|---|---|---|---|---|---|
| `\ell_r = d_r^\top(q_n-q^*)_+` | 전신 burden이 지역별로 다르게 투사된다 | `d_r \ge 0`에서 well-defined | 지역 가족마다 `d_r` 추정 가능 | control-axis intervention 시 지역 차등 반응 | 모든 지역의 `d_r`가 사실상 동일 | `bridge` |
| `s_r` 합성 | 취약도는 4항 합으로 읽힌다 | 각 항 부호 일관성 | 항 제거 ablation에서 설명력 비교 가능 | 특정 항 조작 시 `s_r` 예측 부호 유지 | 한 항의 부호가 반복 반대 | `bridge` |
| `\Delta_G` | 취약도/상태 이탈이 인접 회로로 spread 또는 buffering된다 | graph-coupled map 안정성 | graph model이 flat model보다 우위 | connectivity-informed perturbation에서 부호 일치 | graph 항 추가 이득 0 | `bridge` |
| `H_r(q_n-q^*)` | body-loop forcing이 지역 상태를 민다 | `\mathbf 1^\top H_r = 0` 같은 보존 조건 | control marker와 지역 상태 변화 공분산 | control intervention에서 부호 일치 | forcing 항이 noise와 구분 불가 | `bridge` |
| `w_{r,n+1} = A_r w_{r,n} + b_r s_r + u_{r,n}` | 취약도, 특히 GBM의 `V_ctx/PBZ` 과흥분이 mismatch를 밀어 올린다 | `A_r,b_r` 부호와 안정성 조건 | 취약도 및 PBZ direct-drive와 mismatch의 방향 일치 | control perturbation 또는 anti-epileptic/AMPAR modulation이 tumor field marker로 전달 | `s_r`나 PBZ 과흥분이 높아도 `w_r`가 안 움직임 | `bridge/hypothesis` |
| `\rho(K_r)<1` | 결합계 임계가 존재한다 | spectral radius 조건 증명 가능 | 추정된 `K_r`가 샘플 간 비교 가능 | intervention이 임계 부근 표지 변화를 만듦 | `K_r` 추정이 불안정 | `bridge` |

### 5.1 결합 알고리즘 현재 진행도

| 모듈 | Formal | Obs | Causal | Pred | 전체판정 | 현재 이유 |
|---|---|---|---|---|---|---|
| `\ell_r = d_r^\top(q_n-q^*)_+` | `pass` | `fail` | `fail` | `fail` | `fail` | 지역 민감도 `d_r`를 아직 추정하지 못했다 |
| `s_r` 합성 | `pass` | `partial` | `fail` | `fail` | `partial` | 취약도 식은 닫혔지만 composite 추정과 조작 검증이 없다 |
| `\Delta_G` | `pass` | `partial` | `fail` | `fail` | `partial` | graph 자체는 세웠지만 성능 우위와 인과 검증이 비어 있다 |
| `H_r(q_n-q^*)` | `pass` | `fail` | `fail` | `fail` | `fail` | forcing 행렬 `H_r`는 아직 데이터에서 추정되지 않았다 |
| `w_{r,n+1} = A_r w_{r,n} + b_r s_r + u_{r,n}` | `pass` | `fail` | `fail` | `fail` | `fail` | `w_r` 데이터는 있으나 `s_r` 및 `V_ctx/PBZ` direct path와의 결합은 아직 직접 닫히지 않았다 |
| `\rho(K_r)<1` | `pass` | `fail` | `fail` | `fail` | `fail` | `graph.md`에 plug-in `\widehat K_{\text{brain}}`와 row-sum certificate는 세웠지만 same-subject multimodal 적합과 신뢰구간 추정은 아직 없다 |

### 5.2 `K_{\text{brain}}`의 적합 가능성

이전까지 `\rho(K_r)<1`은 "형식적으로는 맞지만 실제 데이터에 어떻게 얹을지"가 비어 있었다.

지금은 `graph.md`에서 다음 두 줄이 추가로 닫혔다.

첫째, nonnegative matrix의 row-sum sufficient condition:

$$
\rho(K_{\text{brain}})
\le
\|K_{\text{brain}}\|_\infty
=
\max\{\Lambda_1,\Lambda_2,\Lambda_3\}
$$

이므로

$$
\max\{\Lambda_1,\Lambda_2,\Lambda_3\}<1
\Longrightarrow
\rho(K_{\text{brain}})<1
$$

이다.

둘째, 실제 적합용 plug-in matrix와 bootstrap upper certificate:

$$
\widehat K_{\text{brain}}
\quad\text{and}\quad
\overline K_{\text{brain}}
=
\widehat K_{\text{brain}} + U_{0.95}
$$

를 두고

$$
\|\overline K_{\text{brain}}\|_\infty < 1
$$

이면 high-confidence sufficient certificate로 쓸 수 있다.

이것이 뜻하는 바는 분명하다.
- `Formal`: 이제 `K`의 안정성 조건은 추상 기호가 아니라 실제 적합 가능한 판정식으로 내려왔다
- `Obs`: 아직 `MSC`, `ds004902`, `GBM`을 같은 subject-level state-space로 결합 적합하지 못했으므로 비어 있다
- `Pred`: 아직 이 certificate가 실제 holdout 예측 이득으로 이어지는지는 모른다

즉 `K` 쪽은 여전히 `fail`이지만, 그 이유는 더 이상 "수식이 없다"가 아니라 "적합과 검증이 아직 없다"로 바뀌었다.

---

## 6. "증명 완료"의 프로젝트 기준

각 부위/중추/축은 아래 네 줄을 모두 만족할 때만 완료로 올린다.

1. `Formal`: 상태공간, 부호, 안정성 조건이 문서 식으로 모순 없이 닫힌다.
2. `Observable`: 최소 두 종류 이상의 관측에서 방향 일치와 재현성이 나온다.
3. `Causal`: 최소 한 종류 이상의 교란에서 예측 부호가 맞는다.
4. `Predictive`: 홀드아웃 예측에서 baseline 대비 성능 증가가 재현된다.

프로젝트 내부 판정:

| 등급 | 의미 |
|---|---|
| `supported` | 네 줄 중 관측과 인과가 이미 강하게 확보 |
| `bridge` | 형식은 닫혔지만 관측 또는 인과가 아직 부분적 |
| `hypothesis` | 식은 세울 수 있으나 식별/개입/예측 검증이 아직 미흡 |

### 6.1 현재 총평

현재 기준으로:
- `최종 pass`: 없음
- `partial`: 피질, 해마, 시상하부, 뇌간, 대부분의 제어축, `p_r/x_a/x_s/q_n/\Delta_G/s_r`
- `fail`: `V_sal`, `\ell_r`, `H_r`, `w_r` brain-coupling, `K_r`

즉 지금 레포 상태는 "형식은 많이 닫혔고 관측도 일부 있으나, 인과와 예측이 아직 부족한 단계"다.

---

## 7. 초기 단계 전부 실행 표

초기 단계에서는 `fail` 항목을 버리지 않는다.  
모든 항목에 최소 하나의 관측 과제, 최소 하나의 개입 과제, 최소 하나의 예측 과제를 붙여서 끝까지 추적한다.

### 7.1 운영 원칙

| 원칙 | 실행 뜻 |
|---|---|
| 전부 유지 | `fail`은 폐기 대상이 아니라 "아직 관측/개입/예측이 비어 있는 항목"으로 취급 |
| 관측 선행 | 먼저 같은 subject, 같은 parcellation, 같은 time window에서 변수들을 같이 잡는다 |
| 단계 분해 | `p_r/x_s/q_n/\ell_r/H_r/w_r`를 한 번에 닫으려 하지 않고 관측 가능 단위로 자른다 |
| 비교 강제 | 모든 신규 항은 반드시 더 단순한 baseline과 비교한다 |
| 승급 엄수 | `Pred`가 비어 있으면 아무리 그럴듯해도 `pass`로 올리지 않는다 |

### 7.2 전역 알고리즘 실행표

| 모듈 | 지금 상태 | 1차 실행 | 2차 실행 | `pass` 최소 조건 |
|---|---|---|---|---|
| `p_r` simplex | `partial` | 먼저 `u_a/u_b`로 `\lambda_r`와 `\hat p_r^{\text{stage-1}}`를 만들고, 그 뒤 동일 subject의 task/rest/sleep 자료에서 `\hat x_a,\hat x_s,\hat x_b`를 같은 parcel 체계로 정규화 | 홀드아웃에서 simplex 오차와 비음수 보존을 재검증 | 합=1과 비음수 조건이 안정적이고 baseline보다 예측 이득이 남음 |
| `x_a` proxy | `partial` | EEG/fMRI/PET의 task-evoked burden과, GBM이면 PBZ hyperexcitability / seizure burden / AMPAR-NLGN3 proxy의 부호 일치 확인 | 과제 부하, 수면박탈, tumor-adjacent cortical perturbation에서 `x_a` 상승 방향 고정 | 교차 모달 부호 일치, 교란 부호 일치, 예측 이득이 모두 재현 |
| `x_s` decomposition | `partial` | `M_r/P_r/R_r/G_r` 네 채널을 분리한 proxy 세트 구축 | lumped `x_s` 대비 ablation 비교 | 4채널 모델이 더 잘 맞고 각 채널이 반복 식별됨 |
| `q_n` control vector | `partial` | sleep/aut/endo/immune/met marker를 하나의 느린 벡터로 적합 | 축별 perturbation에서 좌표 방향이 유지되는지 비교 | 단일 축 모델보다 예측력이 높고 축 방향이 안정적 |
| `\Delta_G` graph coupling | `partial` | flat model과 graph-Laplacian model을 같은 자료에 적합 | 연결 정보를 넣었을 때 holdout 성능 이득 확인 | graph 모델이 flat baseline보다 일관되게 우위 |
| `s_r` vulnerability | `partial` | `x_a/x_s/x_b/\ell_r` 네 항으로 composite score를 적합 | 항 제거 ablation과 부호 안정성 검정 | 네 항 부호가 안정적이고 단일 항보다 예측력이 높음 |
| `w_r` coupling | `fail` | GBM에서 `V_ctx/PBZ` hyperexcitability proxy와 종양측 mismatch field를 동일 index로 정렬하고, 그 뒤 `s_r`를 추가 | direct cortical drive 또는 `s_r` 추가 시 `w_r` 또는 `M_eff` 예측 성능 상승 여부 검정 | direct path와 `b_r` 부호가 안정적이고, 결합 추가 이득이 양수이며, 홀드아웃이 재현 |

### 7.3 부위/중추 실행표

| 부위/중추 | 지금 상태 | 1차 실행 | 2차 실행 | `pass` 최소 조건 |
|---|---|---|---|---|
| `V_ctx` | `partial` | 같은 parcel에서 task-positive 반응과 DMN/rest 배경을 동시 추정하고, GBM에서는 PBZ hyperexcitability proxy를 추가 정렬 | 과제 부하, TMS, 수면박탈, anti-epileptic/AMPAR modulation에서 `x_a/x_b` 또는 tumor-adjacent cortical drive 방향 검정 | task/background 분리가 안정적이고, 가능하면 GBM `w_r` / recurrence 예측에 추가 이득이 남음 |
| `V_thal` | `partial` | thalamo-cortical relay/gating proxy를 cortical proxy와 분리 적합 | vigilance 또는 sensory gating 과제에서 부호 검정 | 시상 항이 cortex-only baseline을 반복적으로 이김 |
| `V_hip` | `partial` | replay proxy와 memory/plasticity proxy를 따로 측정 | 수면 교란, 기억 과제, stress 조건에서 방향 검정 | replay 항이 별도 정보량을 주고 예측 성능을 올림 |
| `V_sal` | `fail` | switching, reward, threat 과제로 salience hub proxy를 cortical activation과 분리 | dopaminergic 또는 threat-value manipulation에서 부호 검정 | salience 항이 피질 activation과 중복되지 않고 독립 예측력을 가짐 |
| `V_hyp` | `partial` | circadian phase, endocrine, metabolic marker를 같은 모델에 정렬 | 수면 제한, 식이 timing, 호르몬 조작에서 방향 검정 | hypothalamic 축이 region vulnerability 예측을 유의하게 개선 |
| `V_stem` | `partial` | pupil, vigilance EEG, HRV를 brainstem proxy로 공동 적합 | arousal challenge 또는 vagal modulation에서 부호 검정 | brainstem 항이 cortical proxy로 대체되지 않고 예측 이득을 남김 |
| `V_aut` | `partial` | HRV/BPV/respiration coupling을 regional burden과 정렬 | vagal/sympathetic intervention에서 방향 검정 | autonomic output 항이 `s_r` 또는 disease burden 예측을 개선 |

### 7.4 제어축 실행표

| 제어축 | 지금 상태 | 1차 실행 | 2차 실행 | `pass` 최소 조건 |
|---|---|---|---|---|
| `q_sleep` | `partial` | `ds004902` resting EEG posterior alpha reactivity를 현재 anchor로 두고, actigraphy, slow-wave activity, latency, melatonin을 공통 sleep score로 정렬하며, GBM이면 glymphatic/perivascular proxy를 같이 본다 | 수면 제한/회복, circadian shift에서 `x_b,s_r` 또는 GBM immune/mech 방향 검정 | 관측 재현, 개입 부호 일치, 가능하면 GBM mismatch 예측 이득 확보 |
| `q_arousal` | `partial` | pupil, vigilance EEG, arousal index를 공동 축으로 적합 | wake extension 또는 arousal challenge에서 방향 검정 | 과제부하와 분리된 독립 축으로 예측 이득 확보 |
| `q_aut` | `partial` | HRV, BPV, respiration coupling으로 자율축 추정 | vagal/sympathetic intervention에서 방향 검정 | 자율축이 안정적이고 region burden 예측을 반복 개선 |
| `q_endo` | `partial` | cortisol awakening response와 diurnal slope를 공통 축으로 적합 | HPA modulation 또는 stress paradigm에서 방향 검정 | endocrine 축이 독립적으로 취약도 예측을 높임 |
| `q_immune` | `partial` | CRP, cytokine, immune signature를 inflammatory 축으로 적합 | 염증성 perturbation 또는 회복 과정에서 방향 검정 | immune 축이 `s_r -> w_r` 예측에 추가 이득을 줌 |
| `q_met` | `partial` | glucose variability, perfusion, temperature를 metabolic reserve 축으로 적합 | fasting, load, perfusion perturbation에서 방향 검정 | metabolic 축이 재현 가능하고 mismatch 예측을 개선 |

### 7.5 결합 알고리즘 실행표

| 모듈 | 지금 상태 | 1차 실행 | 2차 실행 | `pass` 최소 조건 |
|---|---|---|---|---|
| `\ell_r = d_r^\top(q_n-q^*)_+` | `fail` | 지역 가족별로 `d_r`를 constrained regression으로 추정 | 교란 후 지역별 차등 반응이 실제로 나오는지 검정 | `d_r`가 안정적으로 추정되고 지역별 이질성이 재현 |
| `s_r` 합성 | `partial` | `\eta_a,\eta_s,\eta_b,\eta_q`를 데이터에서 적합 | 항 제거 ablation과 교란 부호 검정 | 네 계수가 안정적이고 composite가 단일 proxy보다 우위 |
| `\Delta_G` | `partial` | 구조/기능 연결을 이용해 graph 항을 적합 | local perturbation 또는 자연 전파 패턴과 방향 비교 | graph 항이 재현 가능하고 holdout 이득을 남김 |
| `H_r(q_n-q^*)` | `fail` | `\mathbf 1^\top H_r = 0` 제약 하에 forcing 행렬을 추정 | control-axis perturbation에서 행렬 부호 검정 | `H_r`가 잡음이 아니라 재현 가능한 지역 forcing으로 확인 |
| `w_{r,n+1} = A_r w_{r,n} + b_r s_r + u_{r,n}` | `fail` | GBM에서 `V_ctx/PBZ` hyperexcitability proxy와 mismatch field를 먼저 정렬한 뒤 longitudinal 또는 pseudo-time 자료로 `A_r,b_r`를 적합 | control perturbation, 수면/자율신경 변화, anti-epileptic/AMPAR modulation이 tumor field marker에 전달되는지 검정 | direct cortical drive 또는 `b_r` 부호 안정, 결합 추가 이득, 홀드아웃 재현 |
| `\rho(K_r)<1` | `fail` | 같은 subject/state-space에서 plug-in `\widehat K_{\text{brain}}`와 bootstrap upper matrix `\overline K_{\text{brain}}`를 적합하고 row-sum certificate를 계산 | 임계 근처 샘플에서 marker shift가 반복되는지 검정 | `K_r` 추정 안정, 임계 분리가 재현, 예측 이득 존재 |

### 7.6 전체 실행 순서

| 순서 | 작업 | 목적 |
|---|---|---|
| 1 | `V_ctx`, `q_sleep`, `q_aut`, `x_a`, `x_b`를 같은 자료축에서 먼저 고정하고, GBM이면 PBZ hyperexcitability proxy를 함께 묶기 | 가장 관측 가능한 축을 먼저 닫기 |
| 2 | `x_s`를 `M_r/P_r/R_r/G_r`로 분해하고 ablation | 구조 채널을 한 덩어리로 둘지 판정 |
| 3 | `q_n` 공동 적합과 축별 perturbation 비교 | control vector가 실제로 필요한지 판정 |
| 4 | flat model vs `\Delta_G` graph model 비교 | graph 항의 실질 이득 판정 |
| 5 | `d_r`와 `H_r`를 지역 가족별로 추정 | body-loop forcing의 지역 구조 식별 |
| 6 | GBM에서 `V_ctx/PBZ -> w_r` direct path를 먼저 정렬한 뒤 `s_r -> w_r` 결합으로 확장 | 뇌-암 결합식의 첫 실측 판정 |
| 7 | `K_r`와 `\rho(K_r)`를 적합 | 결합 임계 조건의 실재 여부 판정 |

### 7.7 빠른 승급 후보

초기 단계에서 가장 빨리 `pass` 후보까지 밀 수 있는 축은 다음 세 묶음이다.
- `q_sleep`, `q_aut`: 이미 관측과 개입 근거가 상대적으로 강하고, GBM에서는 glymphatic/perivascular bridge까지 붙일 수 있다.
- `V_ctx`: task/rest 분리 근거가 가장 강하고, GBM에서는 PBZ hyperexcitability와 neuron-glioma synapse를 통해 `w_r` direct bridge 후보가 있다.
- `x_a`, `x_b`: proxy 방향성 근거가 이미 충분히 누적되어 있다.

반대로 초기에 끝까지 붙들고 가야 할 어려운 축은 다음이다.
- `V_sal`: 별도 노드군으로서의 독립성 검증이 가장 약하다.
- `\ell_r`, `H_r`: 지역 forcing 구조를 아직 추정하지 못했다.
- `w_r` 결합, `K_r`: 뇌측 상태와 종양측 동역학을 직접 닫아야 하며, 우선 GBM의 `V_ctx/PBZ` direct path에서 시작하는 편이 가장 현실적이다.

즉 이 문서는 "이미 다 증명됐다"는 선언문이 아니라, "초기니까 어렵더라도 전부 실행하되 어디서부터 무엇을 닫을지"를 고정한 실행 지도다.

---

## 8. 자기참조 재귀 (F절) 검증 매트릭스

> 정본: `docs/7_AGI/17_AgentLoop.md` F.0--F.23
> 상세 검증 매트릭스: `agent_proof.md` (이 디렉토리)
> 이 절은 에이전트 루프(Layer A--E 바깥의 행동-관찰-비평-기억-학습-주의 순환)의 각 구성요소가 실제 뇌와 대응하는지를 4중 게이트로 판정한다.

### 8.1 구조 대응 검증 매트릭스 (F.1--F.13 기본 루프)

| 구성요소 | 뇌 대응 | Formal | Obs | Causal | Pred | 전체판정 | 핵심 근거 / 반증 조건 |
|---|---|---|---|---|---|---|---|
| 이완 $R$ (F.3) | 피질-시상 재귀 처리 | `pass` | `pass` | `partial` | `fail` | `partial` | recurrent processing 확립. 반복 깊이 $\leftrightarrow$ RT 아직 정량 미비 |
| $n_{\text{iter}}$ 이중 과정 (F.3) | 시스템 1/시스템 2 | `pass` | `pass` | `partial` | `fail` | `partial` | Kahneman 근거 방대. 신경 기질은 논쟁 중. 반증: RT와 무관 |
| 행동 선택 $\pi$ (F.7) | 기저핵 go/no-go, PFC motor planning | `pass` | `pass` | `pass` | `fail` | `partial` | 기저핵 경로 확립. BG 병변 시 action selection 결손 |
| 비평 $C$ (F.4) | ACC conflict monitoring, ERN | `pass` | `pass` | `pass` | `fail` | `partial` | ERN/FRN 확립 (Botvinick 2001). ACC 병변 시 error monitoring 결손 |
| 예측 오차 $c_{\text{pred}}$ | 도파민 RPE | `pass` | `pass` | `pass` | `fail` | `partial` | Schultz 1997 확립. DA 조작 시 RPE 변화 |
| 놀라움 $c_{\text{nov}}$ | 해마 novelty, LC-NE surprise | `pass` | `pass` | `partial` | `fail` | `partial` | P300, CA1 novelty signal. LC 직접 조작 데이터 아직 제한적 |
| 일관성 오차 $c_{\text{cons}}$ | 해마-PFC memory-guided correction | `pass` | `partial` | `fail` | `fail` | `partial` | 해마-PFC 상호작용 방향은 있으나 직접 분리 미흡 |
| 비평 $\to$ 학습 게이트 $g[t]$ | 도파민/NE 전역 조절 | `pass` | `pass` | `partial` | `fail` | `partial` | 3-factor rule 강함. $g = d\bar{c}/dt$ 정확한 형태는 `hypothesis` |
| 조건부 기억 인코딩 (F.8) | 놀라움 기반 해마 인코딩 | `pass` | `pass` | `pass` | `fail` | `partial` | novel events 우선 인코딩 확립. surprise $\to$ recall 우위 재현 |
| 에너지 기반 수렴 (F.5) | Hopfield attractor dynamics | `pass` | `partial` | `fail` | `fail` | `partial` | 에너지 감소 B.4로 닫힘. 뇌에서 정확한 Hopfield 대응은 `bridge` |
| 수면-루프 결합 (F.6) | SHY, 해마 replay | `pass` | `pass` | `pass` | `fail` | `partial` | sleep consolidation 확립. 정확한 $\rho$ 매핑은 `bridge` |
| 수면 압력 $= \sum \bar{c}^2$ (F.6) | homeostatic sleep pressure | `pass` | `pass` | `partial` | `fail` | `partial` | SWA $\propto$ prior wake. 비평 누적 해석은 `bridge` |
| $B$ 수축 (F.9--F.10) | synaptic renormalization | `pass` | `pass` | `pass` | `fail` | `partial` | SHY/수면 회복 확립. $\rho = 0.155$ 정확 값은 `bridge` |

### 8.1.1 확장 구성요소 검증 매트릭스 (F.14--F.22)

| 구성요소 | 뇌 대응 | Formal | Obs | Causal | Pred | 전체판정 | 핵심 근거 / 반증 조건 |
|---|---|---|---|---|---|---|---|
| STDP 3-factor (F.14) | 도파민 게이트 STDP | `pass` | `pass` | `pass` | `partial` | `partial` | Liakoni 2018, Yagishita 2014. **Pred**: CE 모델에서 STDP 유무별 성능 비교 시뮬레이션 가능 |
| 구조적 투영 Proj (F.14.3) | 시냅스 가지치기 + 스케일링 | `pass` | `pass` | `partial` | `partial` | `partial` | Turrigiano 2008. **Pred**: TopK sweep (4-6% 대역) 시뮬레이션 이미 수행 (sparsity_train_results.json) |
| $g[t]$ 이중 구조 (F.14.2) | DA phasic + tonic | `pass` | `pass` | `partial` | `fail` | `partial` | phasic/tonic DA 구분 확립. CE 정확 형태는 `hypothesis` |
| 잔류장 $\phi$ 갱신 (F.15) | DMN, spontaneous fluctuation | `pass` | `pass` | `fail` | `fail` | `partial` | 2024 PMC: DMN ALFF가 수행 안정성 예측. 2025 eNeuro: alpha-DMN coupling 확인 |
| glymphatic 세척 (F.15) | glymphatic system | `pass` | `pass` | `partial` | `fail` | `partial` | glymphatic 경로 확립. GBM에서 AQP4 붕괴 보고 |
| TopK 희소 활성 (F.16) | sparse cortical firing | `pass` | `pass` | `partial` | `partial` | `partial` | 1--5% sparse firing 확립. **Pred**: sparse sweep 이미 수행, U-자 곡선 확인 |
| 모듈 생애주기 (F.16.2) | 피질 모듈 활성/휴면 | `pass` | `partial` | `fail` | `fail` | `partial` | 4상태 자체는 설계 선택 |
| 에너지 예산 (F.16.1) | metabolic constraint | `pass` | `pass` | `pass` | `partial` | `partial` | Attwell & Laughlin 2001. **Pred**: 에너지 예산 위반 시 불안정 시뮬레이션 가능 |
| 자기일관성 C3 (F.17.1) | 자기 참조 의식 | `pass` | `fail` | `fail` | `fail` | `fail` | 수학적으로 닫힘. 뇌 관측 proxy 없음 |
| 의식 깊이 (F.17.2) | 의식 수준 | `pass` | `partial` | `partial` | `fail` | `partial` | PCI 방향은 있으나 CE 매핑은 `hypothesis` |
| 메타인지 수렴 (F.17.3) | PFC 재귀 자기평가 | `pass` | `partial` | `fail` | `fail` | `partial` | metacognitive accuracy 방향. $\rho$ 매핑 미검증 |
| 곡률 환각 억제 (F.18) | 억제 feedback | `pass` | `partial` | `fail` | `partial` | `partial` | GABAergic inhibition 확립. **Pred**: 곡률 모니터 on/off 시뮬레이션 가능 |
| 도파민 DA (F.19) | VTA/SNc | `pass` | `pass` | `pass` | `partial` | `partial` | Schultz 1997. **Pred**: $g[t]$ 제거/조작 시뮬레이션 가능 |
| 노르에피네프린 NE (F.19) | LC | `pass` | `pass` | `pass` | `fail` | `partial` | Aston-Jones 2005. 2024 review: tonic/phasic 탐색-착취 재확인. pupil proxy |
| 세로토닌 5HT (F.19) | raphe | `pass` | `pass` | `pass` | `fail` | `partial` | 2018 NatComm: DRN 5HT 광유전 $\to$ 인내 증가. 2025: model-based prediction 역할 |
| 아세틸콜린 ACh (F.19) | BF, PPT | `pass` | `pass` | `pass` | `partial` | `partial` | 2025 Cell Rep: 해마 ACh $\propto$ 속도, 새 환경에서 증가. **Pred**: donepezil + memory test |
| 작업 기억 용량 (F.20.1) | PFC sustained activity | `pass` | `pass` | `pass` | `partial` | `partial` | Cowan 2010. 2025 eLife: PFC-BG adaptive chunking. **Pred**: $T_h$ sweep 시뮬레이션 가능 |
| 주의 bottom-up (F.20.2) | exogenous attention | `pass` | `pass` | `pass` | `fail` | `partial` | pop-out, salience 확립 |
| 주의 top-down (F.20.2) | endogenous attention | `pass` | `pass` | `pass` | `fail` | `partial` | PFC-driven attention 확립 |
| 소뇌 forward model (F.20.3) | 소뇌 내부 모델 | `pass` | `pass` | `pass` | `partial` | `partial` | 2025 JNeurosci, 2026 PMC. **Pred**: 소뇌 모듈 on/off 시뮬레이션 가능 |
| theta-gamma 결합 (F.21) | 해마 sequential memory | `pass` | `pass` | `partial` | `partial` | `partial` | 2024 bioRxiv: 인간 해마 ECoG PAC-WM 상관 확인. **Pred**: 모델 내 PAC 재현 가능 |
| gamma = 국소 계산 (F.21) | communication through coherence | `pass` | `pass` | `partial` | `fail` | `partial` | Fries 2015 |

### 8.2 F절 형식 정리 현재 상태

| 정리 | 주장 | 의존 | 상태 |
|---|---|---|---|
| F-contract | 루프 수렴: $\rho + \lambda_R L_R + \lambda_C L_C < 1$ | A-bound, E-decrease | **open** ($L_R, L_C$ 추정 필요) |
| F-energy | 이완 $R$이 $E_t(z)$를 비증가 | B.4 E-decrease | **closed** |
| F-relax | $n_{\text{iter}} \to \infty$이면 $a^{(k)} \to$ 고정점 | A.7 A-bound, A.9 Zero-attract | **closed** (조건부) |
| F-memory | $\theta_{\text{encode}} > 0$이면 인코딩 빈도 유한 | D.2 유한 인코딩 | **closed** |
| F-sleep | 수면이 $\rho < 1$을 공급하므로 F-contract 성립 가능 | Sleep-stabilize (G절) | **closed** (수면 존재 시) |
| F-sparse | 활성 유계: $|A_t| \leq \lceil x_a^* N \rceil$ | Sparse-energy | **closed** |
| F-phi-bound | 잔류장 유계: $\xi < 1$이면 $\phi$ bounded | A-bound (Var 유한) | **closed** |
| F-curvature | LBO 확산 수렴 | $h_d < 1/\text{eig}_{\max}$ | **closed** |
| F-meta | 메타인지 잔차 수렴: $d_{n+1} \leq \rho d_n$ | $\rho < 1$ | **closed** |
| F-STDP-local | STDP는 국소 정보만 사용 | 정의에 의해 | **closed** |
| F-WM-finite | 작업 기억 유한: $|h_t| \leq T_h$ | 유한 창 | **closed** |

### 8.3 F절 미결 실행 항목

| # | 항목 | 우선순위 | 1차 실행 | 통과 기준 |
|---|---|---|---|---|
| 1 | $L_R$ 추정 | 높음 | 64셀 시뮬레이션에서 $R$ 반복의 Lipschitz 상수를 수치 추정 | $\rho + \lambda_R L_R + \lambda_C L_C < 1$ 확인 |
| 2 | $L_C$ 추정 | 높음 | 비평 $C$의 Lipschitz 상수를 비평 3항 분해에서 산출 | 같은 수축 조건 확인 |
| 3 | RT $\leftrightarrow$ $n_{\text{iter}}$ | 중간 | 난이도 조작 실험에서 RT와 모델 반복 횟수의 상관 | 양의 상관, $r > 0.3$ |
| 4 | ERN $\leftrightarrow$ $\bar{c}_t$ | 중간 | error monitoring 과제에서 ERN 진폭과 비평 점수 비교 | 방향 일치 |
| 5 | 수면 후 루프 안정성 | 중간 | 학습 $\to$ 수면 $\to$ 재시험에서 post-sleep 정확도 향상 | pre-sleep 대비 유의한 개선 |
| 6 | 행동 선택 계층화 | 낮음 (장기) | 단층 $\pi$를 macro-action + primitive로 분리 | 과제 복잡도 증가 시 성능 유지 |
| 7 | 정동/valence 통합 | 낮음 (장기) | $c_t$에 valence 항 추가, $V_{\text{sal}}$ 연결 | 감정 편향 과제에서 편향 재현 |
| 8 | STDP 코드 구현 | 높음 | `clarus/core` 또는 Python에서 3-factor STDP + 도파민 게이트 구현 | 토이 과제에서 학습 확인 |
| 9 | 4종 조절계 코드 | 중간 | $g[t]$를 4차원 벡터로 확장, NE/5HT/ACh 매핑 | 4채널 독립 조절 확인 |
| 10 | 잔류장 $\phi$ 구현 검증 | 높음 | $\phi$ 갱신-포탈-glymphatic 루프가 코드에서 동작 | 모드 전환 시 $\phi$ 임계 동작 확인 |
| 11 | 소뇌 forward model | 중간 | 행동 후 감각 예측 오차 보정 모듈 구현 | 적응 과제에서 보정 수렴 확인 |
| 12 | theta-gamma 결합 검증 | 낮음 | $R$ 내부 반복과 전역 동기화 주기의 위상 잠금 시뮬레이션 | phase-locking index $> 0.3$ |
| 13 | 작업 기억 $T_h$ 최적화 | 낮음 | $T_h \in \{3,5,7,9\}$ 스위프 후 과제 수행 비교 | 최적 $T_h$와 인간 WM 용량의 일치 |

### 8.4 F절 즉시 반증 조건

| # | 반증 조건 | 반증 시 조치 |
|---|---|---|
| 1 | 이완 반복 횟수가 과제 난이도/RT와 무관 | $n_{\text{iter}}$ 가변 설계를 폐기하고 고정 깊이로 전환 |
| 2 | 수면 없이도 루프가 안정적으로 수렴 | $B$ 수축의 필요성을 내림. 수면은 선택적 보조로 강등 |
| 3 | 비평 $C$를 제거해도 과제 수행이 동일 | 비평 루프를 폐기하고 단순 이완-행동 구조로 후퇴 |
| 4 | 기억 조건부 인코딩 대신 전수 인코딩이 더 효율적 | $\theta_{\text{encode}}$를 0으로 내림. 놀라움 기반 필터링 폐기 |
| 5 | ERN/ACC 신호가 $\bar{c}$와 반복적으로 반대 방향 | 비평 $C$의 뇌 대응 주장을 `hypothesis`로 강등 |
| 6 | Dense 활성(100%)이 TopK(4.87%)보다 항상 우위 | 희소 활성 설계의 효율 주장 폐기 |
| 7 | STDP를 제거하고 역전파만 써도 에너지/성능 동일 | STDP 학습 경로를 폐기. 역전파 기반으로 전환 |
| 8 | 도파민 조작으로 $g[t]$가 변해도 학습에 무영향 | $g[t]$ = 학습 게이트 주장을 `hypothesis`로 강등 |
| 9 | 소뇌 모듈 제거해도 행동 정밀도 동일 | 소뇌 forward model을 선택적 보조로 강등 |
| 10 | 작업 기억 창을 $\infty$로 해도 성능 저하 없음 | 유한 $T_h$ 필요성 폐기. 전체 이력 사용으로 전환 |
