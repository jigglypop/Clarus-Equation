# 항상성 제어축

> 역할: `evidence.md`의 `q_n`과 body-loop forcing을 전용으로 정리한다.
>
> 검증 기준: `proof.md`

---

## 1. 느린 제어 상태 `q_n`

full-stack 최소 상태는 다음처럼 둔다.

$$
q_n
=
\big(
q_{\text{sleep},n},
q_{\text{arousal},n},
q_{\text{aut},n},
q_{\text{endo},n},
q_{\text{immune},n},
q_{\text{met},n}
\big)
$$

건강한 기준은 `q^*`로 둔다.

---

## 2. 제어축별 실제 부품

| 축 | 핵심 구조 | 대표 관측 | CE에서 직접 밀어 올리는 항 | 현재 판정 |
|---|---|---|---|---|
| `q_sleep` | SCN, VLPO, orexin/LH | actigraphy, sleep latency, slow-wave activity, melatonin phase, resting EEG posterior alpha reactivity, glymphatic/perivascular proxy | `x_b`, `s_r` | `supported/bridge` |
| `q_arousal` | LC, raphe, basal forebrain | pupil, vigilance EEG, arousal index | `x_a`, `s_r` | `bridge` |
| `q_aut` | NTS, DMV, vagal/sympathetic loop | HRV, blood pressure variability, respiration coupling | `x_b`, `s_r` | `supported` |
| `q_endo` | hypothalamus PVN, HPA axis | cortisol awakening response, diurnal cortisol slope | `s_r` | `supported` |
| `q_immune` | neuroimmune / inflammatory loop | CRP, cytokine panel, immune signature | `s_r \to w_r` | `supported` |
| `q_met` | hypothalamus-metabolic-vascular reserve | glucose variability, perfusion, temperature | `s_r \to w_r` | `bridge` |

### 2.1 `q_sleep`의 부호와 현재 실측 anchor

이 문서에서 `q_sleep`는 "잘 잤다"의 양이 아니라, burden 방향 좌표다.

$$
q_{\text{sleep},n}\uparrow
\quad\Longleftrightarrow\quad
\text{sleep debt 또는 circadian misalignment}\uparrow
$$

현재 레포에서 바로 재현되는 첫 관측 anchor는 `OpenNeuro ds004902`의 resting EEG posterior alpha reactivity다.

$$
r_\alpha
:=
\frac{\alpha_{\text{closed}}}{\alpha_{\text{open}}}
$$

eyes-open/eyes-closed complete pair를 가진 `19`명 기준으로

$$
\text{median NS}=2.1224,
\qquad
\text{median SD}=1.2490,
\qquad
\Delta_{\text{mean}}=-1.1582,
\qquad
p=0.0401
$$

이 나왔다.

같은 데이터셋의 행동축도

$$
\text{SSS}: 2.0 \to 5.0,\qquad p=2.75\times 10^{-4}
$$

$$
\text{PVT RT}: 320 \to 359,\qquad p=1.45\times 10^{-5}
$$

로 함께 악화되므로, 현재 단계에서는

$$
q_{\text{sleep},n}\uparrow
\Longrightarrow
r_\alpha \downarrow,
\qquad
\text{vigilance burden}\uparrow
$$

를 `q_sleep`의 가장 직접적인 실측 부호로 둔다.

### 2.2 `q_sleep`의 최소 좌표화

실제 계산에서는 sleep burden의 최소 좌표를

$$
z_n
:=
\big(q_{\text{sleep},n} - q_{\text{sleep}}^*\big)_+
\ge 0
$$

로 둔다.

그리고 sleep-sensitive node `r`에 대한 최소 forcing는

$$
H_r \delta q_n
=
z_n\, h_r^{\text{sleep}}
+
\widetilde H_r\, \delta \widetilde q_n,
\qquad
h_r^{\text{sleep}}
:=
(\alpha_r,\; 0,\; -\alpha_r),
\qquad
\alpha_r \ge 0
$$

로 둔다. 여기서 `\delta \widetilde q_n`은 sleep 좌표를 제외한 나머지 control-axis 이탈이다.

이 최소식의 의미는 단순하다:
- `z_n`이 커질수록 local mass가 `x_b`에서 빠져 `x_a`로 간다
- `x_s`는 이 최소 모델에서는 직접 건드리지 않는다
- `\mathbf 1^\top h_r^{\text{sleep}} = 0`이므로 simplex 총량은 보존된다
- `\alpha_r`는 해당 node의 sleep sensitivity다

지역 burden 투사도 같은 방식으로 분리한다.

$$
\ell_r(n)
=
d_{r,\text{sleep}}\, z_n
+
\widetilde \ell_r(n),
\qquad
d_{r,\text{sleep}} \ge 0
$$

즉 `q_sleep`는 하나의 추상 라벨이 아니라, local background reserve를 깎고 body-loop burden을 동시에 올리는 단일 좌표 `z_n`으로 읽힌다.

관측식도 최소형으로 다음처럼 둔다.

$$
r_{\alpha,n}
=
r_\alpha^*
-
c_\alpha z_n
+
\varepsilon_{\alpha,n},
\qquad
c_\alpha > 0
$$

여기서

$$
r_{\alpha,n}
:=
\frac{\alpha_{\text{closed}}}{\alpha_{\text{open}}}
$$

이다.

`ds004902`에서 수면박탈 후 `r_\alpha`가 실제로 감소했으므로, 현재 단계에서는 `c_\alpha > 0`를 첫 실측 부호로 둔다.

즉 `q_n`은 피질 뒤 배경이 아니라, 실제로 body-wide set-point를 다시 맞추는 제어층이다.

---

## 3. 지역 투사 `\ell_r`

전신 burden이 각 회로에 걸리는 정도를

$$
\ell_r(n) = d_r^\top (q_n-q^*)_+,
\qquad
d_r \ge 0
$$

로 둔다.

여기서 `d_r`는 지역별 민감도다.

| 지역 가족 | 크게 받는 축 | 이유 |
|---|---|---|
| cortex / thalamus | `q_sleep`, `q_arousal` | wake pressure, vigilance overload, relay fatigue |
| hippocampus | `q_sleep`, `q_endo` | replay failure, stress-sensitive memory loop |
| salience hub | `q_arousal`, `q_endo` | gain shift, threat/value bias |
| hypothalamus / brainstem / autonomic output | `q_aut`, `q_endo`, `q_met` | 제어 중심축 자체 |

따라서 `q_n`은 전신 공통 burden이고, `\ell_r`는 그것이 어느 회로에 얼마나 세게 실리는지의 지역 가중치다.

---

## 4. 암 취약도로 넘어가는 최소 경로

가장 짧은 causal chain은 다음처럼 쓴다.

$$
\text{sleep debt}
\to
\text{LC/orexin overdrive}
\to
\text{sympathetic + HPA shift}
\to
\text{immune/metabolic drift}
\to
\ell_r \uparrow
\to
s_r \uparrow
\to
w_r \uparrow
$$

바로 안전하게 말할 수 있는 부분:
- 수면 부족이 각성계, 자율신경, endocrine stress를 흔든다.
- 그 결과가 inflammatory/metabolic burden과 연결된다.
- GBM에서는 이 수면축이 `AQP4` / perivascular clearance / antigen drainage를 통해 `w_{\text{immune}}`, `w_{\text{mech}}`로 이어지는 브리지가 될 수 있다.
- `ds004902`에서는 수면박탈 개입 자체가 `posterior alpha reactivity ratio` 저하와 `SSS/PVT` 악화를 함께 만들어, 적어도 `q_sleep`의 첫 causal gate는 현재 레포에서 바로 재현된다.

아직 `bridge`인 부분:
- 특정 암종에서 어떤 `d_r`가 가장 큰가
- 그 투사가 `stromal / hypoxic / mech` mismatch를 얼마나 밀어 올리는가

---

## 5. 읽기 규칙

| 문장 | 판정 |
|---|---|
| sleep, autonomic, endocrine, immune, metabolic 축이 뇌 취약도에 영향을 준다 | `supported/bridge` |
| `q_n`을 하나의 control vector로 묶는다 | `bridge` |
| `q_n`만으로 암 예후를 단독 예측한다 | `hypothesis` |
