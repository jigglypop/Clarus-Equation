# 뇌 관측 정의

> 역할: `evidence.md`에서 쓰는 `p_r`, `x_a`, `x_s`, `x_b`, `s_r`의 operational definition을 전용으로 정리한다.
>
> 검증 기준: `proof.md`

---

## 1. 지역 상태 `p_r`

메인 상태는

$$
p_r(t) = \big(x_{a,r}(t), x_{s,r}(t), x_{b,r}(t)\big) \in \Delta^2
$$

로 둔다.

직접 한 번에 재는 값은 아직 없으므로, 현재는 multimodal score를 정규화해 읽는 것이 가장 안전하다.

$$
\hat x_{a,r}(t),\;\hat x_{s,r}(t),\;\hat x_{b,r}(t)\ge 0,
\qquad
p_r(t)
=
\frac{1}{\hat x_{a,r}+\hat x_{s,r}+\hat x_{b,r}}
\big(\hat x_{a,r},\hat x_{s,r},\hat x_{b,r}\big)
$$

기준점은

$$
p^* = (x_a^*, x_s^*, x_b^*) = (0.0487,\; 0.2623,\; 0.6891)
$$

로 둔다.

---

## 2. 세 성분의 최소 proxy

| 성분 | 지금 바로 쓸 수 있는 proxy | 주 관측 도구 | 현재 판정 |
|---|---|---|---|
| `x_a` | event-locked firing fraction, task-evoked BOLD/PET increment, task-positive gamma/beta burden | EEG/MEG, fMRI, FDG-PET | 방향성은 `supported`, 지역 비율화는 `bridge` |
| `x_s` | maintenance/plasticity burden, replay/renormalization burden, repair support | sleep EEG, PET, molecular assay | 전역 해석은 `supported`, 지역 정량화는 `bridge` |
| `x_b` | resting metabolism, DMN/intrinsic activity, tonic spontaneous background | rs-fMRI, PET, resting EEG | 방향성은 `supported`, node-wise simplex 변수화는 `bridge` |

### 2.1 첫 번째 실행: `x_a`

초기 단계에서 가장 먼저 닫을 변수는 `x_a`다.
이유는 task-positive burden, sparse firing, sleep deprivation 쪽 근거가 가장 직접적이기 때문이다.

ds000201 v2 pilot에서는 sleepiness task responsive fraction이 hands task보다 SD/NS 구분력이 높았으므로, 같은 세션에 sleepiness task가 있으면 이를 1순위 `x_a` 센서로 쓴다.

단, 이 단계에서는 `p_r` 전체를 닫지 않고 `\hat x_{a,r}`만 먼저 고정한다.

같은 parcel `r`에 대해 modality `m`의 task burden surrogate를

$$
u_{a,r}^{(m)}(t)
=
\big(z_{m,r}^{\text{task}}(t)-z_{m,r}^{\text{rest}}-\tau_m\big)_+,
\qquad
m\in\{\text{EEG/MEG},\text{fMRI},\text{PET}\}
$$

로 둔다.

초기 결합 추정량은

$$
\hat x_{a,r}^{(1)}(t)
=
\sum_{m\in \mathcal M_t}\omega_m u_{a,r}^{(m)}(t),
\qquad
\omega_m\ge 0,
\qquad
\sum_{m\in \mathcal M_t}\omega_m = 1
$$

로 둔다.

- modality가 하나뿐이면 그 값을 그대로 쓴다.
- 이 값은 stage-1 active burden이며, 아직 최종 simplex component `x_{a,r}`는 아니다.
- `\hat x_{s,r}`, `\hat x_{b,r}`가 같이 준비된 뒤에만 `p_r` 정규화에 넣는다.

초기 단계에서는 `V_ctx`를 우선한다.
피질에서는 task-positive signal과 rest/DMN background 분리가 가장 잘 보이기 때문이다.

| 게이트 | 지금 바로 확인할 것 | 통과 기준 |
|---|---|---|
| `Formal` | `u_{a,r}^{(m)} \ge 0`, `\hat x_{a,r}^{(1)} \ge 0` | 부호 위반 없음 |
| `Obs` | task > rest, modality 간 부호 일치 | 최소 두 modality에서 방향 일치 |
| `Causal` | 과제 부하 증가 또는 수면박탈 | global active burden 증가 |
| `Pred` | vigilance lapse 또는 task error 예측 | rest-only baseline보다 성능 우위 |

이 네 줄이 다 통과되면 `x_a`는 이 프로젝트에서 가장 먼저 `pass` 후보로 승급된다.

### 2.2 두 번째 실행: `x_b`

이번 단계의 `x_b`는 **논문 근거가 이미 있는 resting / intrinsic background proxy만** 사용한다.

허용 근거는 `evidence.md`에 이미 정리된 공개 문헌으로 제한한다.
- `The restless brain`
- `The Brain's Default Mode Network`
- `48 and 72 h of sleep deprivation on waking human regional brain activity`
- `Sleep deprivation, vigilant attention, and brain function: a review`

즉 이 단계에서는 임의로 만든 reserve 점수나 설명 불가능한 내부 점수는 쓰지 않는다.

같은 parcel `r`에 대해 modality `m`의 resting background surrogate를

$$
u_{b,r}^{(m)}(t)
=
\big(z_{m,r}^{\text{rest}}(t)-\tau_m\big)_+,
\qquad
m\in\{\text{rs-fMRI},\text{PET},\text{resting EEG}\}
$$

로 둔다.

여기서
- `rs-fMRI`: resting network segregation (coarse parcel connectivity의 within/between module 차이) 또는 DMN amplitude
- `PET`: resting glucose metabolism
- `resting EEG`: tonic spontaneous background power 또는 resting stability

현재 v2 pilot에서 가장 잘 동작한 fMRI 기반 `u_b` proxy는 **rest network segregation**이다. 이것은 coarse spatial parcel로 voxel을 묶고, parcel 간 상관행렬의 spectral bipartition으로 within-module과 between-module 평균 상관의 차이를 구한 것이다. 수면박탈 시 network segregation이 감소하며 (`p = 0.074`, `n = 9`), 이전에 시도한 split-half RMS 안정성이나 global connectivity map 안정성보다 방향이 정확하다.

초기 결합 추정량은

$$
\hat x_{b,r}^{(1)}(t)
=
\sum_{m\in \mathcal M_t}\omega_m u_{b,r}^{(m)}(t),
\qquad
\omega_m\ge 0,
\qquad
\sum_{m\in \mathcal M_t}\omega_m = 1
$$

로 둔다.

- modality가 하나뿐이면 그 값을 그대로 쓴다.
- 이 값은 stage-1 background burden이며, 아직 최종 simplex component `x_{b,r}`는 아니다.
- `\hat x_{a,r}`, `\hat x_{s,r}`가 같이 준비된 뒤에만 `p_r` 정규화에 넣는다.
- 수면박탈은 `x_b`를 직접 정의하는 값이 아니라, `x_b`의 교란 검증용 개입으로만 쓴다.

초기 단계에서는 `V_ctx`와 DMN-associated parcel을 우선한다.
여기가 resting / intrinsic background 문헌 근거가 가장 강한 영역이기 때문이다.

| 게이트 | 지금 바로 확인할 것 | 통과 기준 |
|---|---|---|
| `Formal` | `u_{b,r}^{(m)} \ge 0`, `\hat x_{b,r}^{(1)} \ge 0` | 부호 위반 없음 |
| `Obs` | rest 반복 측정 안정성, modality 간 방향 일치 | 최소 두 modality 또는 반복 run에서 방향 일치 |
| `Causal` | 수면박탈 또는 prolonged wake | background proxy 감소 또는 resting organization 악화 |
| `Pred` | vigilance stability 또는 recovery slope 예측 | task-only baseline보다 성능 우위 |

이 네 줄이 다 통과되면 `x_b`는 `x_a` 다음의 두 번째 `pass` 후보가 된다.

### 2.3 `x_a/x_b`의 stage-1 joint closure

초기 실데이터에서는 `x_s`를 바로 적합하지 못하는 경우가 많으므로, 먼저

$$
u_{a,r,n} := \hat x_{a,r}^{(1)}(n),
\qquad
u_{b,r,n} := \hat x_{b,r}^{(1)}(n)
$$

를 묶는 stage-1 simplex를 둔다.

pair share를

$$
\lambda_{r,n}
:=
\frac{u_{a,r,n}}{u_{a,r,n}+u_{b,r,n}},
\qquad
u_{a,r,n}+u_{b,r,n}>0
$$

로 두면, `x_s = x_s^*`를 고정한 최소 상태는

$$
\hat p_{r,n}^{\text{stage-1}}
:=
\Big(
(1-x_s^*)\lambda_{r,n},
\;
x_s^*,
\;
(1-x_s^*)(1-\lambda_{r,n})
\Big)
$$

로 쓴다.

이 식의 장점은 세 가지다.
- `u_a,u_b`가 비음수이면 자동으로 `\hat p_{r,n}^{\text{stage-1}} \in \Delta^2`가 된다
- stage-1 geometry가 `\lambda_{r,n}` 하나로 압축된다
- `x_a/x_b` 쌍의 실측을 `p_r` 좌표계로 즉시 옮길 수 있다

기준 share를

$$
\lambda^*
:=
\frac{x_a^*}{x_a^*+x_b^*}
=
\frac{x_a^*}{1-x_s^*}
\approx 0.0660
$$

로 두면,

$$
p^*
=
\Big(
(1-x_s^*)\lambda^*,
\;
x_s^*,
\;
(1-x_s^*)(1-\lambda^*)
\Big)
$$

이고,

$$
\big\|\hat p_{r,n}^{\text{stage-1}} - p^*\big\|_2
=
\sqrt{2}\,(1-x_s^*)\,|\lambda_{r,n}-\lambda^*|
$$

를 얻는다.

따라서 stage-1 오차는 결국 `pair share` 오차 하나로 측정된다.

또한 interior region에서

$$
\frac{\partial \hat x_{a,r,n}^{\text{stage-1}}}{\partial \lambda_{r,n}}
=
1-x_s^*
>
0,
\qquad
\frac{\partial \hat x_{b,r,n}^{\text{stage-1}}}{\partial \lambda_{r,n}}
=
-(1-x_s^*)
<
0
$$

이므로, `\lambda`가 커질수록 stage-1 state는 `x_a` 쪽으로 움직인다.

`control.md`의 sleep forcing를 background 쪽에만 반영해

$$
u_{b,r,n}^{\text{eff}}
:=
u_{b,r,n} - \alpha_r z_n
$$

라고 두고 `u_{b,r,n}^{\text{eff}}>0`인 구간을 보면,

$$
\lambda_{r,n}^{\text{eff}}
:=
\frac{u_{a,r,n}}{u_{a,r,n}+u_{b,r,n}^{\text{eff}}}
$$

에 대해

$$
\frac{\partial \lambda_{r,n}^{\text{eff}}}{\partial z_n}
=
\frac{\alpha_r u_{a,r,n}}
{\big(u_{a,r,n}+u_{b,r,n}^{\text{eff}}\big)^2}
\ge 0
$$

이다.

즉 sleep debt가 background reserve를 깎으면, stage-1 pair share는 active 쪽으로 기울고, 그 기울어짐이 곧 `p_r`의 stage-1 이동으로 읽힌다.

---

## 3. 구조 채널 `x_s` 세분화

`x_s`는 한 덩어리 값이 아니라, 적어도 다음 네 부담의 합으로 읽는다.

$$
\hat x_{s,r}
=
\alpha_{\text{maint}} M_r
+
\alpha_{\text{plast}} P_r
+
\alpha_{\text{replay}} R_r
+
\alpha_{\text{support}} G_r,
\qquad
\alpha_i \ge 0
$$

| 항 | 의미 | 대표 회로 | 지금 가능한 proxy | 판정 |
|---|---|---|---|---|
| `M_r` | 막 유지, 펌프, 단백질 turnover 같은 maintenance cost | cortex-thalamus backbone | housekeeping metabolism, baseline synaptic cost | `supported/bridge` |
| `P_r` | STDP, eligibility, LTP/LTD 같은 plasticity burden | cortex, hippocampus, basal ganglia | plasticity marker, LTP/LTD assay | `bridge` |
| `R_r` | 수면 replay와 renormalization burden | hippocampo-cortical loop | sleep EEG + replay proxy | `bridge` |
| `G_r` | astrocyte/microglia/myelin/repair support | glia-vascular support system | glial marker, repair assay | `bridge/hypothesis` |

정규화는

$$
x_{s,r}
=
\frac{\hat x_{s,r}}{\hat x_{a,r}+\hat x_{s,r}+\hat x_{b,r}}
$$

처럼 읽는다.

---

## 4. 취약도 `s_r`

`s_r`는 단순 과활성 지표가 아니라, 과활성 + 구조 저하 + 배경 항상성 저하 + body-loop burden을 묶은 값이다.

$$
s_r(n)
=
\eta_a\big(x_{a,r}(n)-x_a^*\big)_+
\;+\;
\eta_s\big(x_s^*-x_{s,r}(n)\big)_+
\;+\;
\eta_b\big(x_b^*-x_{b,r}(n)\big)_+
\;+\;
\eta_q \ell_r(n)
$$

여기서 `\ell_r(n)`의 정의는 `control.md`를 따른다.

항별 해석:
- `x_a > x_a^*`: 병적 과흥분 또는 응급성 대사 부담
- `x_s < x_s^*`: 구조 유지/복원 여력 감소
- `x_b < x_b^*`: 배경 항상성/완충 여력 감소
- `\ell_r > 0`: 수면, 자율신경, endocrine, immune, metabolic burden이 해당 회로에 투사됨

---

## 5. 보조 관측량

현재 `\|\Delta_g \Phi\|^2`의 안전한 proxy 후보는 다음 네 가지다.
- slow-wave activity
- population synchrony / desynchrony
- functional connectivity roughness
- replay burden

가장 안전한 global 조절 오차 후보는

$$
\delta[t] = a \cdot \text{RPE}(t) + b \cdot \text{surprise}(t) + c \cdot \text{novelty}(t)
$$

이다. 다만 이것만으로 `q_n`을 대신하지는 않는다. dopamine/novelty는 빠른 학습 신호이고, `q_n`은 느린 제어 배경이다.

---

## 6. 실데이터 검증 세트

실제 검증은 "아이디어에 맞는 데이터"가 아니라, 아래 공개 데이터셋들에 같은 계산을 반복 적용하는 방식으로 한다.

| 검증 대상 | 공개 데이터셋 | 바로 읽을 수 있는 관측 | 용도 |
|---|---|---|---|
| `x_a`, `x_b` | Midnight Scan Club `ds000224` | repeated rest/task fMRI | 같은 subject 안에서 task-positive vs resting background 분리 |
| `x_a`, `x_b` | HCP Young Adult | rest fMRI + task fMRI | 대규모 cohort에서 task/rest 분해 재현 |
| sparse activity scale | Allen Brain Observatory Visual Coding / Neuropixels | single-neuron spiking, responsive fraction | `4.87%`가 low single-digit sparse regime 안에 있는지 직접 체크 |
| `q_sleep` | Sleepy Brain Project I `ds000201` | rest/task fMRI, polysomnography, sleep restriction | 수면 부족이 `x_b`, vigilance burden을 흔드는지 확인 |
| `q_sleep` | OpenNeuro `ds004902` | resting EEG, normal sleep vs deprivation | 수면 박탈에서 resting organization과 active burden 변화를 확인 |
| `w_r` GBM mismatch | 10x Genomics Visium CytAssist human glioblastoma FFPE dataset | raw feature matrix + spatial coordinates | 실제 spot-level `M_eff`, shell peak, `\hat A_{\text{tumor}}`, `\hat\rho(A_{\text{tumor}})` 계산 |

즉 초기 단계의 최소 검증 조합은 다음처럼 둔다.
- `MSC` 또는 `HCP`로 `x_a/x_b`
- `Sleepy Brain` 또는 `ds004902`로 `q_sleep`
- `10x GBM`으로 `w_r`

### 6.1 현재 레포에서 바로 재현 가능한 GBM 실행 예

현재 레포의 `examples/biology/cancer_mismatch.py`는 raw 10x output을 바로 받을 수 있다.

실행 예:

```bash
./.venv/Scripts/python.exe examples/biology/cancer_mismatch.py \
  --real-gbm-h5 data/gbm/CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_raw_feature_bc_matrix.tar.gz \
  --real-gbm-spatial data/gbm/CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_spatial.tar.gz
```

현재 세션에서 실제로 얻은 pilot 결과:

$$
\hat\rho(A_{\text{tumor}}) = 1.184,
\qquad
h^\dagger = 27
$$

$$
M_{\text{eff}}(h^\dagger) - M_{\text{edge}} = 1.061,
\qquad
M_{\text{eff}}(h^\dagger) - M_{\text{core}} = 0.382
$$

즉 이 공개 GBM 샘플 하나에서는:
- inner shell peak가 실제로 edge/core보다 높게 나왔다
- dominant region은 `stromal`, dominant axis는 `mech`였다
- `\hat\rho(A_{\text{tumor}}) > 1`도 같이 나왔다

해석:
- 이것은 `n=1` pilot이므로 일반 결론은 아니다
- 다만 "실데이터를 현재 코드로 바로 돌릴 수 있는가"와 "최소 GBM shell/mismatch 판정이 공개 샘플에서 동작하는가"는 이미 통과한 셈이다

### 6.2 현재 레포에서 바로 재현 가능한 MSC 실행 예

`x_a`, `x_b` 쪽의 첫 pilot은 `examples/physics/brain_cosmos.py`로 바로 재현할 수 있다.

실행 예:

```bash
./.venv/Scripts/python.exe examples/physics/brain_cosmos.py
```

현재 세션에서 `MSC01`, `ses-func01`로 얻은 pilot 결과는 다음과 같다.

$$
\text{motor contrast responsive fraction}
=
0.0494
$$

$$
u_a/(u_a+u_b) = 0.0701,
\qquad
u_b/(u_a+u_b) = 0.9299
$$

구조 채널을 아직 직접 적합하지 않았으므로, 이 단계에서는 `x_s = x_s^*`를 고정한 stage-1 simplex만 본다:

$$
\hat p_{\text{stage-1}}
=
(0.0517,\; 0.2623,\; 0.6860),
\qquad
\|\hat p_{\text{stage-1}} - p^*\|_2 = 0.0042
$$

해석:
- `Allcondition_avg` motor contrast의 responsive fraction `0.0494`는 CE의 `x_a^* = 0.0487`와 거의 같은 스케일이다
- raw rest/task BOLD로 만든 stage-1 pair에서도 active share는 low single-digit to low-ten-percent regime에 머물고, background share가 압도적으로 크다
- `x_s`를 CE prior로 고정하면 전체 상태는 이미 `p^*` 근방에 놓인다
- 아직 이것만으로 `x_b` causal gate가 닫힌 것은 아니며, 그 부분은 `Sleepy Brain` 또는 `ds004902` 같은 수면 교란 데이터로 이어서 검증해야 한다

즉 지금 레포는 종양 쪽 `GBM mismatch`뿐 아니라, 뇌 쪽 `x_a/x_b`도 공개 데이터 한 샘플에서 바로 재현 가능한 상태까지는 도달했다.

### 6.3 현재 레포에서 바로 재현 가능한 `ds004902` 실행 예

`q_sleep` 쪽의 첫 resting EEG pilot도 같은 스크립트로 바로 재현할 수 있다.

실행 예:

```bash
./.venv/Scripts/python.exe examples/physics/brain_cosmos.py ds004902
```

현재 세션에서 `participants.tsv` 전체와, eyes-open/eyes-closed complete pair를 가진 `19`명 EEG subset으로 얻은 결과는 다음과 같다.

행동 축에서는 수면박탈 조건 자체가 강하게 드러난다:

$$
\text{SSS median: } 2.0 \to 5.0,
\qquad
\Delta_{\text{mean}} = +1.657,
\qquad
p = 2.75\times 10^{-4}
$$

$$
\text{PVT RT median: } 320 \to 359,
\qquad
\Delta_{\text{mean}} = +41.217,
\qquad
p = 1.45\times 10^{-5}
$$

$$
\text{PVT lapse median: } 57.965 \to 83.795,
\qquad
\Delta_{\text{mean}} = +22.458,
\qquad
p = 1.53\times 10^{-4}
$$

resting EEG에서는 open-only fraction보다 `eyes-closed / eyes-open` posterior alpha reactivity가 더 안정적인 `q_sleep` proxy로 나왔다:

$$
\text{posterior alpha reactivity ratio}
:=
\frac{\alpha_{\text{closed}}}{\alpha_{\text{open}}}
$$

$$
\text{median NS} = 2.1224,
\qquad
\text{median SD} = 1.2490,
\qquad
\Delta_{\text{mean}} = -1.1582,
\qquad
p = 0.0401
$$

보조로 본 difference metric

$$
(\alpha_{\text{closed}}-\alpha_{\text{open}})
$$

도

$$
\text{median NS} = 0.2650,
\qquad
\text{median SD} = 0.0412
$$

까지 내려가지만, 현재 pilot에서는 ratio 쪽이 더 잘 닫혔다.

해석:
- `participants.tsv` 전체에서 `SSS`, `PVT`, `PANAS_N`이 함께 악화되므로 `ds004902`의 deprivation label은 실제 조작으로 보아도 무방하다
- EEG subset에서는 posterior alpha reactivity가 `NS`보다 `SD`에서 유의하게 낮아져, resting organization이 수면박탈에서 약해진다는 방향이 나온다
- 현재 pilot에서는 open-only resting fraction보다 `closed/open reactivity`가 더 강한 `q_sleep` proxy였다
- 따라서 `x_b`의 causal gate는 `ds004902`에서 바로 이어서 닫을 수 있고, `q_sleep`은 행동 악화와 함께 움직이는 실제 perturbation axis로 취급할 수 있다

### 6.4 현재 레포에서 바로 재현 가능한 `ds000201` v2 실행 예

`x_a`, `x_b`, `q_sleep`을 동시에 잡는 joint pilot은 같은 스크립트로 바로 재현할 수 있다.

실행 예:

```bash
./.venv/Scripts/python.exe examples/physics/brain_cosmos.py sleepybrain
```

현재 세션에서 `sub-9001` 외 9개 subject pair (NS/SD 각 1세션)으로 얻은 v2 결과는 다음과 같다.

**수면 압력 조작 (KSS)**

$$
\text{KSS median: } 4.33 \to 6.17,
\qquad
\Delta_{\text{mean}} = +1.98,
\qquad
p = 0.0078
$$

**stage-1 active sensor (`x_a`)**

sleepiness task responsive fraction을 1순위 센서로, hands task는 보조로 쓴다. 이유: hands 단독은 SD/NS 구분이 거의 없었으나 (`mean delta = -0.006`, `p = 0.57`), sleepiness task는 기대 방향으로 움직인다 (`mean delta = +0.024`, `positive frac = 0.63`).

$$
\text{active responsive fraction median: } 0.4619 \to 0.4761,
\qquad
\Delta_{\text{mean}} = +0.020,
\qquad
p = 0.43
$$

**stage-1 background sensor (`x_b`)**

coarse rest network segregation을 background proxy로 쓴다. parcel bin size `6`, 최소 `24` voxel 이상 parcel만 사용하고, spectral bipartition으로 within/between module 평균 상관 차이를 구한다.

$$
\text{network segregation median: } 0.1135 \to 0.0911,
\qquad
\Delta_{\text{mean}} = -0.024,
\qquad
p = 0.074
$$

9쌍 중 7쌍에서 SD에서 segregation이 감소했다.

이전 시도에서 쓴 split-half RMS 안정성 (`p = 0.65`)과 global connectivity map 안정성 (`p = 0.65`)은 모두 방향이 안 나왔다.

**stage-1 simplex 상태**

$$
\text{active share median: } 0.7908 \to 0.8174,
\qquad
\Delta_{\text{mean}} = +0.039,
\qquad
p = 0.16
$$

해석:
- `q_sleep` 조작은 강하게 잡힌다 (`p = 0.008`)
- `x_a` 센서는 sleepiness 쪽이 hands보다 낫다. 단독으로는 아직 유의하지 않지만 방향은 맞다
- `x_b`는 network segregation 정의에서 처음으로 기대 방향 (SD에서 감소)이 나왔고, `p = 0.074`까지 왔다
- stage-1 simplex에서 active share가 SD에서 올라가므로, 전체적으로 `x_a↑, x_b↓` 방향이 보인다
- 이것은 `n = 9` pilot이므로 일반 결론은 아니다

**확장 cohort (v3, n=18)**

9명에서 18명으로 확장 후 재분석한 결과:

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
- 수면 압력 조작은 표본이 커지면서 훨씬 강해졌다 (p = 0.008 -> p = 0.0003)
- `x_a`, `x_b`, active share의 방향은 모두 CE 이론과 일치한다
- 다만 개별 fMRI proxy의 effect size는 n=9 pilot보다 줄었고, 유의하지 않다
- 이는 fMRI의 coarse voxel-level proxy가 개인 간 variability에 민감하다는 뜻이다
- proxy 정밀도를 올리려면 (a) parcel 정의를 atlas 기반으로 교체하거나 (b) regional proxy로 전환하거나 (c) modality를 바꾸는 것이 필요하다

### 6.5 변수별 식별 원칙

초기 실데이터 식별은 다음 순서를 따른다.

1. `x_a`: task-rest 차이 또는 responsive fraction 같은 **증가량**부터 고정
2. `x_b`: resting / intrinsic background의 **기저량**을 별도 추정
3. `q_sleep`: 수면 제한/회복, circadian shift 같은 **개입축**으로 검증
4. `w_r`: spatial omics에서 shell profile과 region mismatch를 먼저 계산
5. 마지막에만 `s_r -> w_r`와 `K_r`를 적합

즉 `p_r`, `q_n`, `w_r`를 한 번에 닫으려 하지 않고,

$$
x_a/x_b
\;\to\;
q_{\text{sleep}}
\;\to\;
w_r
\;\to\;
s_r \to w_r
\;\to\;
K_r
$$

순으로 올린다.

---

## 7. 사용 규칙

| 문장 | 판정 |
|---|---|
| 뇌에서 `x_a`, `x_b` 방향성을 현재 proxy로 읽을 수 있다 | `supported` |
| `x_s`를 maintenance/plasticity/replay/support 합으로 읽는다 | `bridge` |
| 동일 위치에서 multimodal data로 `p_r`와 `s_r`를 완전히 닫는다 | `hypothesis` |
