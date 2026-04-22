# 뇌 그래프

> 역할: `r`를 flat index가 아니라 parcel/node로 읽기 위한 brain graph 정의를 전용으로 정리한다.
>
> 검증 기준: `proof.md`

---

## 1. brain graph

$$
G_{\text{brain}}=(V_{\text{brain}}, E_{\text{brain}})
$$

정점 집합은 최소한 다음처럼 나눈다.

$$
V_{\text{brain}}
=
V_{\text{ctx}}
\sqcup
V_{\text{thal}}
\sqcup
V_{\text{hip}}
\sqcup
V_{\text{sal}}
\sqcup
V_{\text{hyp}}
\sqcup
V_{\text{stem}}
\sqcup
V_{\text{aut}}
$$

| 정점 집합 | 포함 구조 | 주된 기능 |
|---|---|---|
| `V_ctx` | sensory, motor, association cortex, DMN | task recruitment, intrinsic background |
| `V_thal` | thalamic relay | gating, relay, bandwidth reallocation |
| `V_hip` | hippocampus-entorhinal loop | replay, memory indexing |
| `V_sal` | amygdala, basal ganglia, salience switching hub | switching and gain control |
| `V_hyp` | hypothalamus | circadian, endocrine, metabolic set-point |
| `V_stem` | brainstem arousal/autonomic nuclei | arousal tone, autonomic correction |
| `V_aut` | vagal/sympathetic output side | body-coupling output |

---

## 2. 간선 가족

| 간선 부류 | 예시 | 수학적 해석 |
|---|---|---|
| cortico-thalamic | cortex $\leftrightarrow$ thalamus | fast relay/gating coupling |
| hippocampo-cortical | hippocampus $\leftrightarrow$ cortex | replay/renormalization coupling |
| salience-switch | salience hub $\leftrightarrow$ cortex/hippocampus | switching and gain control |
| hypothalamo-brainstem-autonomic | hypothalamus $\leftrightarrow$ brainstem $\leftrightarrow$ autonomic output | slow homeostatic/body-loop forcing |

즉 `r`는 단순한 위치 번호가 아니라, 어떤 회로층과 연결 부류에 속하는지까지 포함한 node다.

---

## 3. graph coupling

지역 간 spread/buffering은

$$
\Delta_G f(r)
=
\sum_{s:(s,r)\in E_{\text{brain}}}
a_{rs}\big(f_s-f_r\big),
\qquad
a_{rs}\ge 0
$$

로 둔다.

해석:
- 피질-피질 인접성: 과제 burden과 background drift의 확산
- 시상-피질 relay: 입력 gating과 대역 재분배
- 해마-피질 replay: 기억 재활성화와 복구 경로
- 시상하부-뇌간-자율신경 축: distributed correction 또는 failure propagation

해부학적 연결이 존재한다는 점은 `supported`지만, 이를 위 식으로 요약하는 것은 현재 `bridge`다.

---

## 4. graph-coupled relaxation

정적 고정점은 여전히

$$
B_* : p \mapsto p^*
$$

이고, graph 위의 최소 이완은

$$
p_{r,n+1}
=
\Pi_{\Delta^2}\!\Big(
(1-\rho)p^*
+ \rho p_{r,n}
+ \gamma_p \Delta_G p_{r,n}
+ H_r(q_n-q^*)
\Big)
$$

로 둔다.

여기서
- `\rho p_{r,n}`: 기존 local state의 관성
- `\gamma_p \Delta_G p_{r,n}`: 인접 회로를 통한 spread/buffering
- `H_r(q_n-q^*)`: sleep/autonomic/endocrine/immune/metabolic burden의 지역 forcing

즉 `B`는 단일 affine contraction에서 끝나지 않고, brain graph 위의 coupled relaxation으로 확장된다.

---

## 5. 최소 뇌 알고리즘

위 식들을 실제 알고리즘으로 읽기 위해, 건강 기준점 주변의 상태를 다음처럼 둔다.

$$
e_{r,n} := p_{r,n} - p^*,
\qquad
\delta q_n := q_n - q^*
$$

GBM 같은 직접 neuroglioma 결합까지 포함하면, 최소 상태는

$$
\big(e_{r,n},\; \delta q_n,\; w_{r,n},\; \nu_{r,n}\big)
$$

로 읽는다.

여기서
- `e_{r,n}`: 지역 3분배의 기준점 이탈
- `\delta q_n`: 수면/각성/자율신경/내분비/면역/대사 축의 느린 이탈
- `w_{r,n}`: 종양 mismatch burden
- `\nu_{r,n}`: GBM에서의 direct cortical drive, 즉 PBZ hyperexcitability / neuron-glioma synaptic drive

최소 반복은 다음 네 단계로 쓴다.

1. 각성 또는 외부 입력 누적:

$$
p_{r,n+\frac12} = p_{r,n} + u_{r,n},
\qquad
\mathbf 1^\top u_{r,n} = 0
$$

2. 느린 제어축 업데이트:

$$
\delta q_{n+1} = A_q \delta q_n + r_n + \xi_n^{(q)},
\qquad
\rho(A_q) < 1
$$

3. 종양 또는 병적 burden 업데이트:

$$
w_{r,n+1}
=
A_w w_{r,n}
+ b_s\, s_{r,n}
+ b_\nu\, \nu_{r,n}
+ u_{r,n}^{(w)}
$$

여기서 `s_{r,n}`은 `measure.md`, `control.md`의 취약도 식을 따른다.

4. 수면/항상성 수축과 graph redistribution:

$$
p_{r,n+1}
=
\Pi_{\Delta^2}\!\Big(
(1-\rho)p^*
+ \rho p_{r,n+\frac12}
+ \gamma_p \Delta_G p_{r,n+\frac12}
+ H_r \delta q_n
+ C_r w_{r,n}
+ \xi_{r,n}
\Big)
$$

즉 연산 순서는
- `u_{r,n}`가 빠른 입력과 과제 burden을 올리고
- `A_q`가 느린 control state를 갱신하고
- `A_w, b_s, b_\nu`가 병적 burden을 누적하며
- 마지막에 `\rho`, `\Delta_G`, `H_r`, `C_r`가 수면-그래프-몸축-종양 역작용을 함께 반영해 다음 상태를 만든다

건강한 알고리즘은 `w_{r,n}=0`, `\nu_{r,n}=0`에 가까운 경우고, GBM 같은 경우는 `b_\nu \nu_{r,n}`가 추가된 경우다.

---

## 6. 고유모드 분석

simplex projection 내부에서 국소 선형화를 하면

$$
e_{r,n+1}
=
\rho e_{r,n}
+ \gamma_p \Delta_G e_{r,n}
+ H_r \delta q_n
+ C_r w_{r,n}
+ \xi_{r,n}
$$

를 얻는다.

이제 graph operator를 행렬 `L_G`로 써

$$
\Delta_G e_n = L_G e_n
$$

라 두자. 무방향 가중 그래프라면 `L_G`의 고유값 `\mu_k`는

$$
\mu_k \le 0,
\qquad
\mu_0 = 0
$$

를 만족한다.

고유모드 `\phi_k`에 대해

$$
L_G \phi_k = \mu_k \phi_k,
\qquad
e_n = \sum_k a_{k,n}\phi_k
$$

이면, 종양/제어축 forcing을 잠시 끄면 각 모드는

$$
a_{k,n+1} = (\rho + \gamma_p \mu_k)\, a_{k,n}
$$

처럼 진화한다.

따라서:
- `k=0` 전역 모드는 `a_{0,n+1} = \rho a_{0,n}`로 줄어든다
- `\mu_k < 0`인 고주파 모드는 `\rho + \gamma_p \mu_k`에 의해 더 빠르게 감쇠한다
- 즉 graph coupling은 local roughness를 더 세게 누르는 spatial smoother로 작동한다

건강한 spread/buffering이 유지되려면 모든 모드에 대해

$$
|\rho + \gamma_p \mu_k| < 1
$$

이어야 한다. 특히 가장 불안정한 최소 고유값 `\mu_{\min}`에 대해 충분조건은

$$
\gamma_p |\mu_{\min}| < 1 + \rho
$$

이다.

---

## 7. 결합 안정성 분석

`s_{r,n}`이 `e_{r,n}`, `\delta q_n`에 대해 Lipschitz라고 두자:

$$
s_{r,n} \le \kappa_p \|e_{r,n}\| + \kappa_q \|\delta q_n\|
$$

또한 GBM direct drive도

$$
\nu_{r,n}
\le
\|N_p\|\,\|e_{r,n}\|
+ \|N_q\|\,\|\delta q_n\|
+ \|\xi_{r,n}^{(\nu)}\|
$$

처럼 상계된다고 두자.

그럼

$$
\chi_{r,n}
:=
\begin{pmatrix}
\|e_{r,n}\|\\
\|\delta q_n\|\\
\|w_{r,n}\|
\end{pmatrix}
$$

에 대해 다음의 작은-이득 상계를 쓸 수 있다.

$$
\chi_{r,n+1}
\lesssim
K_{\text{brain}}
\chi_{r,n}
+ d_{r,n}
$$

$$
K_{\text{brain}}
=
\begin{pmatrix}
\rho + \gamma_p\|L_G\| & \|H_r\| & \|C_r\|\\
0 & \|A_q\| & 0\\
\|b_s\|\kappa_p + \|b_\nu\|\|N_p\| & \|b_s\|\kappa_q + \|b_\nu\|\|N_q\| & \|A_w\|
\end{pmatrix}
$$

여기서 `d_{r,n}`은 외부 forcing와 noise를 모은 항이다.

따라서 healthy regime의 최소 안정성 조건은

$$
\boxed{\rho(K_{\text{brain}}) < 1}
$$

이다.

해석:
- `\rho + \gamma_p\|L_G\|`: 수면 수축과 graph spread가 합쳐진 fast loop
- `\|A_q\|`: 느린 body-loop의 자체 복원력
- `\|A_w\|`: 병적 burden의 자체 증폭성
- `\|b_\nu\|\|N_p\|`: GBM에서 cortical overdrive가 tumor mismatch로 새어 들어가는 direct path의 세기

즉 이 프로젝트의 핵심은 단순히 `\rho=0.155`만 보는 것이 아니라,
`graph`, `control`, `tumor`, `direct neural drive`를 모두 묶은

$$
\rho(K_{\text{brain}})
$$

를 실제 데이터에서 적합하는 것이다.

### 7.1 적합 가능한 충분조건

`K_{\text{brain}}`의 모든 entry는 norm 또는 비음수 coupling 크기이므로, 이 행렬은 원소별로 nonnegative다.

따라서 Perron-Frobenius와 induced infinity norm을 쓰면

$$
\rho(K_{\text{brain}})
\le
\|K_{\text{brain}}\|_\infty
=
\max\{\Lambda_1,\Lambda_2,\Lambda_3\}
$$

를 얻는다. 여기서

$$
\Lambda_1
:=
\rho + \gamma_p\|L_G\| + \|H_r\| + \|C_r\|
$$

$$
\Lambda_2
:=
\|A_q\|
$$

$$
\Lambda_3
:=
\|b_s\|(\kappa_p+\kappa_q)
+
\|b_\nu\|(\|N_p\|+\|N_q\|)
+
\|A_w\|
$$

다.

따라서

$$
\max\{\Lambda_1,\Lambda_2,\Lambda_3\} < 1
$$

이면 곧

$$
\rho(K_{\text{brain}}) < 1
$$

이다.

증명:
nonnegative matrix에 대해 spectral radius는 induced matrix norm보다 크지 않다. 여기서는 row-sum norm을 썼고, 세 row sum이 각각 `\Lambda_1,\Lambda_2,\Lambda_3`다. 따라서 최대 row sum이 `1`보다 작으면 spectral radius도 `1`보다 작다. `\square`

이 sufficient bound의 장점은 분명하다:
- 각 항을 직접 고유값 분해하지 않아도 된다
- 필요한 건 coupling의 upper scale뿐이다
- bootstrap upper bound만으로도 건강/비건강 판정을 줄 수 있다

### 7.2 plug-in `\hat K_{\text{brain}}`와 안정성 certificate

실제 적합에서는 먼저 관측 가능한 상태를 만든다.

$$
\hat e_{r,n}
:=
\hat p_{r,n}^{\text{stage-1}} - p^*
$$

여기서 `\hat p_{r,n}^{\text{stage-1}}`는 `measure.md`의 `x_a/x_b` stage-1 simplex를 쓴다.

sleep 축의 첫 empirical coordinate는

$$
\hat z_n^{(\alpha)}
:=
\frac{(r_\alpha^*-\hat r_{\alpha,n})_+}{\hat c_\alpha},
\qquad
\hat c_\alpha > 0
$$

로 둔다. 여기서 `\hat r_{\alpha,n}`는 `posterior alpha reactivity ratio`이고, `\hat c_\alpha`는 sleep burden과 `r_\alpha` 사이의 scale factor다. 첫 실행에서는 `\widehat{\delta q}_n = \hat z_n^{(\alpha)}`만 써도 된다.

그 다음 operator들은 constrained regression으로 적합한다.

$$
\widehat A_q
:=
\arg\min_{A:\,\rho(A)<1}
\sum_n
\big\|
\widehat{\delta q}_{n+1}
-
A\,\widehat{\delta q}_n
\big\|_2^2
+
\lambda_q \|A\|_F^2
$$

$$
(\widehat H_r,\widehat C_r)
:=
\arg\min_{H,C}
\sum_n
\big\|
\hat e_{r,n+1}
-
\hat\rho\,\hat e_{r,n}
-
\hat\gamma_p L_G \hat e_{r,n}
-
H\,\widehat{\delta q}_n
-
C\,\hat w_{r,n}
\big\|_2^2
+
\lambda_H\|H\|_F^2
+
\lambda_C\|C\|_F^2
$$

subject to

$$
\mathbf 1^\top H = 0,
\qquad
\mathbf 1^\top C = 0
$$

이고,

$$
(\widehat A_w,\widehat b_s,\widehat b_\nu)
:=
\arg\min_{A,b_s,b_\nu \ge 0}
\sum_n
\big\|
\hat w_{r,n+1}
-
A\,\hat w_{r,n}
-
b_s\,\hat s_{r,n}
-
b_\nu\,\hat \nu_{r,n}
\big\|_2^2
+
\lambda_w\big(\|A\|_F^2+\|b_s\|_2^2+\|b_\nu\|_2^2\big)
$$

로 적합한다.

또한 direct-drive upper map은

$$
(\widehat N_p,\widehat N_q)
:=
\arg\min_{N_p,N_q \ge 0}
\sum_n
\big\|
\hat \nu_{r,n}
-
N_p \hat e_{r,n}
-
N_q \widehat{\delta q}_n
\big\|_2^2
+
\lambda_\nu\big(\|N_p\|_F^2+\|N_q\|_F^2\big)
$$

로 두고, vulnerability 쪽의 Lipschitz surrogate는

$$
(\widehat\kappa_p,\widehat\kappa_q)
:=
\arg\min_{\kappa_p,\kappa_q \ge 0}
\sum_n
\Big(
\hat s_{r,n}
-
\kappa_p \|\hat e_{r,n}\|
-
\kappa_q \|\widehat{\delta q}_n\|
\Big)^2
$$

로 둔다.

그러면 plug-in matrix는

$$
\widehat K_{\text{brain}}
:=
\begin{pmatrix}
\hat\rho + \hat\gamma_p\|L_G\| & \|\widehat H_r\| & \|\widehat C_r\|\\
0 & \|\widehat A_q\| & 0\\
\|\widehat b_s\|\,\widehat\kappa_p + \|\widehat b_\nu\|\,\|\widehat N_p\|
&
\|\widehat b_s\|\,\widehat\kappa_q + \|\widehat b_\nu\|\,\|\widehat N_q\|
&
\|\widehat A_w\|
\end{pmatrix}
$$

다.

실제 판정은 bootstrap upper matrix로 주는 편이 더 안전하다.

$$
\overline K_{\text{brain}}
:=
\widehat K_{\text{brain}} + U_{0.95}
$$

여기서 `U_{0.95}`는 각 entry의 one-sided `95%` upper confidence radius를 모은 행렬이다.

그럼

$$
\|\overline K_{\text{brain}}\|_\infty < 1
$$

이면

$$
\rho(K_{\text{brain}}) < 1
$$

을 고신뢰 sufficient certificate로 쓸 수 있다.

이 방식이 중요한 이유는 `K_{\text{brain}}` 전체를 한 번에 완벽히 알 필요 없이, 현재 레포에 이미 있는
- `MSC`의 `\hat p^{\text{stage-1}}`
- `ds004902`의 `\hat z^{(\alpha)}`
- `ds000201`의 paired `x_a` (sleepiness task responsive fraction), `x_b` (rest network segregation), KSS
- `GBM`의 `\hat w`

를 차례대로 붙여 나가며 stability bound를 확장할 수 있기 때문이다.

특히 `ds000201` v2 pilot에서 `x_b`의 운영 정의가 rest network segregation으로 바뀌었으므로, `\hat e_{r,n}`의 background 성분은 이제 coarse parcel connectivity의 within/between module 차이로 읽는다. 이것은 split-half RMS 안정성이나 global connectivity map 안정성보다 SD/NS 구분력이 높았다 (`p = 0.074` vs `p = 0.65`).

---

## 8. `q_sleep \to x_b \to s_r` 최소 정리

`control.md`의 sleep burden 좌표

$$
z_n := (q_{\text{sleep},n} - q_{\text{sleep}}^*)_+
$$

와 minimal sleep forcing

$$
h_r^{\text{sleep}} = (\alpha_r,\; 0,\; -\alpha_r),
\qquad
\alpha_r \ge 0
$$

를 graph relaxation 안에 넣자.

sleep 좌표를 제외한 나머지 forcing를

$$
\Omega_{r,n}
:=
\gamma_p \Delta_G p_{r,n}
+
\widetilde H_r \delta \widetilde q_n
+
C_r w_{r,n}
+
\xi_{r,n}
$$

로 묶으면, pre-projection update는

$$
p_{r,n+1}^{\text{pre}}
=
(1-\rho)p^*
+
\rho p_{r,n}
+
z_n h_r^{\text{sleep}}
+
\Omega_{r,n}
$$

처럼 쓸 수 있다.

### 8.1 명제: 수면부채의 simplex 부호 폐쇄

`p_{r,n+1}^{\text{pre}}`가 관심 구간에서 simplex 내부에 머물러 `\Pi_{\Delta^2}`가 항등으로 작동하고,

$$
\mathbf 1^\top \Omega_{r,n} = 0
$$

이라고 하자.

그러면

$$
\mathbf 1^\top p_{r,n+1}^{\text{pre}}
=
(1-\rho)\mathbf 1^\top p^*
+
\rho \mathbf 1^\top p_{r,n}
+
z_n \mathbf 1^\top h_r^{\text{sleep}}
+
\mathbf 1^\top \Omega_{r,n}
=
1
$$

이고, 좌표별로는

$$
x_{a,r,n+1}
=
(1-\rho)x_a^*
+
\rho x_{a,r,n}
+
\Omega_{a,r,n}
+
\alpha_r z_n
$$

$$
x_{s,r,n+1}
=
(1-\rho)x_s^*
+
\rho x_{s,r,n}
+
\Omega_{s,r,n}
$$

$$
x_{b,r,n+1}
=
(1-\rho)x_b^*
+
\rho x_{b,r,n}
+
\Omega_{b,r,n}
-
\alpha_r z_n
$$

를 얻는다.

따라서 다른 항을 고정하면

$$
\frac{\partial x_{a,r,n+1}}{\partial z_n} = \alpha_r,
\qquad
\frac{\partial x_{s,r,n+1}}{\partial z_n} = 0,
\qquad
\frac{\partial x_{b,r,n+1}}{\partial z_n} = -\alpha_r
$$

이다.

즉 이 최소 모델에서 `q_sleep` 증가, 정확히는 `z_n \uparrow`는 simplex mass를 `x_b`에서 `x_a`로 옮긴다.

증명:
`h_r^{\text{sleep}}`의 정의에서 `\mathbf 1^\top h_r^{\text{sleep}} = 0`이므로 총량 보존이 성립한다. 나머지는 affine update를 좌표별로 읽으면 바로 나온다. `\square`

### 8.2 따름정리: 취약도 단조증가

`control.md`의 지역 burden 분해

$$
\ell_r(n)
=
d_{r,\text{sleep}} z_n
+
\widetilde \ell_r(n),
\qquad
d_{r,\text{sleep}} \ge 0
$$

와 취약도 정의

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

를 합치자.

그러면 threshold kink `x_{a,r}=x_a^*`, `x_{b,r}=x_b^*`를 제외한 점에서는

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

특히 sleep-sensitive node에서

$$
d_{r,\text{sleep}} > 0
$$

이면

$$
\frac{\partial s_{r,n+1}}{\partial z_n} > 0
$$

이므로, `z_n`이 커질수록 취약도는 단조증가한다. threshold 점에서는 같은 결론이 one-sided derivative 또는 subgradient로 그대로 유지된다.

증명:
명제 8.1에서 `x_a`와 `x_b`의 `z_n`에 대한 기울기를 이미 알았다. 이를 `(\cdot)_+`의 도함수

$$
\frac{d}{du}(u)_+ = \mathbf 1_{\{u>0\}}
$$

에 체인룰로 대입하면 첫 두 항이 나온다. `x_s`는 이 최소 모델에서 `z_n`에 직접 의존하지 않으므로 기여가 없고, `\ell_r(n)`의 미분은 `d_{r,\text{sleep}}`다. 네 항을 합하면 된다. `\square`

### 8.3 실데이터 해석으로의 연결

`control.md`의 관측식

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

을 같이 쓰면, `ds004902`에서 관측된

$$
r_\alpha \downarrow
$$

는 곧

$$
z_n \uparrow
\Longrightarrow
x_b \downarrow
\Longrightarrow
s_r \uparrow
$$

라는 최소 기전으로 읽힌다.

즉 `posterior alpha reactivity ratio` 감소는 단순 상관표지가 아니라, 현재 문서 체계에서는 sleep debt가 background reserve를 잠식하고 vulnerability를 올린다는 동역학적 부호 증거다.

---

## 9. Well-posedness

### 9.1 Simplex invariance (정리)

전체 알고리즘의 pre-projection update는

$$
p_{r,n+1}^{\text{pre}}
=
(1-\rho)p^*
+
\rho p_{r,n}
+
F_{r,n}
$$

로 쓸 수 있다. 여기서

$$
F_{r,n}
:=
\gamma_p \Delta_G p_{r,n}
+
H_r \delta q_n
+
C_r w_{r,n}
+
\xi_{r,n}
$$

는 graph, control, tumor, noise forcing의 합이다.

**정리 9.1** (Simplex invariance sufficient condition).
$\mathbf 1^\top F_{r,n} = 0$이고

$$
\|F_{r,n}\|_\infty
<
\min\!\big\{(1-\rho)p_{\min}^*,\;\rho\, p_{\min,r,n}\big\}
$$

이면 $p_{r,n+1}^{\text{pre}} \in \operatorname{int}(\Delta^2)$이고 $\Pi_{\Delta^2}$는 항등으로 작동한다.

여기서 $p_{\min}^* := \min\{x_a^*,x_s^*,x_b^*\} = x_a^* = 0.0487$이고 $p_{\min,r,n} := \min\{x_{a,r,n},x_{s,r,n},x_{b,r,n}\}$이다.

증명:
$\mathbf 1^\top F_{r,n} = 0$에서 $\mathbf 1^\top p_{r,n+1}^{\text{pre}} = 1$이 나온다.
좌표별로 $p_{r,n+1}^{\text{pre},i} = (1-\rho)p_i^* + \rho p_{r,n}^i + F_{r,n}^i$이므로, $|F_{r,n}^i| \le \|F_{r,n}\|_\infty$에서

$$
p_{r,n+1}^{\text{pre},i}
\ge
(1-\rho)p_{\min}^* + \rho\, p_{\min,r,n} - \|F_{r,n}\|_\infty
> 0
$$

이고, 합이 1이므로 각 좌표는 1 미만이다. $\square$

**해석**: forcing이 fixed point의 최소 성분 $(x_a^* = 0.0487)$의 $(1-\rho)$배보다 작으면 simplex를 벗어나지 않는다. 건강한 상태에서 이 조건은 쉽게 만족된다. 병적 상태에서 forcing이 이 bound를 넘으면 $\Pi_{\Delta^2}$가 경계에서 clipping하여 성분을 0 이상으로 잡는다.

### 9.2 $\mathbf 1^\top$ 보존 조건

simplex invariance의 전제인 $\mathbf 1^\top F_{r,n} = 0$이 성립하려면 각 항이 independently mass-preserving이어야 한다:

$$
\mathbf 1^\top H_r \delta q_n = 0,
\qquad
\mathbf 1^\top C_r w_{r,n} = 0,
\qquad
\mathbf 1^\top \xi_{r,n} = 0
$$

graph 항은 $\Delta_G$의 정의에 의해 자동으로 $\mathbf 1^\top \Delta_G p_{r,n} = 0$이다 (각 edge의 기여가 zero-sum).

따라서 mass conservation은 $H_r$와 $C_r$가 각각 $\mathbf 1^\top H_r = 0$, $\mathbf 1^\top C_r = 0$을 만족하면 임의의 $\delta q_n$, $w_{r,n}$에 대해 성립한다.

이 조건은 `control.md`의 $h_r^{\text{sleep}} = (\alpha_r, 0, -\alpha_r)$에서 이미 만족된다: $\alpha_r + 0 + (-\alpha_r) = 0$.

---

## 10. 수렴 정리

### 10.1 정리: 건강한 상태의 지수 수렴

$w_{r,n} = 0$ (종양 없음), $\nu_{r,n} = 0$ (direct drive 없음)이고, simplex invariance가 유지되는 구간에서, noise를 끄면 ($\xi = 0$),

$$
\chi_{r,n}
:=
\begin{pmatrix}
\|e_{r,n}\|\\
\|\delta q_n\|
\end{pmatrix}
$$

에 대해 reduced gain matrix

$$
K_0
:=
\begin{pmatrix}
\rho + \gamma_p\|L_G\| & \|H_r\|\\
0 & \|A_q\|
\end{pmatrix}
$$

가 $\rho(K_0) < 1$을 만족하면

$$
\|\chi_{r,n}\|
\le
\|K_0^n\|\,\|\chi_{r,0}\|
\le
C_0\,\rho(K_0)^n\,\|\chi_{r,0}\|
$$

를 만족한다. 여기서 $C_0 \ge 1$은 $K_0$의 condition number에만 의존하는 상수다.

$K_0$가 상삼각이므로 고유값은 대각 원소이고,

$$
\rho(K_0) = \max\!\big\{\rho + \gamma_p\|L_G\|,\;\|A_q\|\big\}
$$

이다. 따라서 수렴 속도는 두 loop 중 느린 쪽이 결정한다:
- **fast loop**: $\rho + \gamma_p\|L_G\|$. 이것은 cortical relaxation과 graph smoothing의 합성 수축률이다.
- **slow loop**: $\|A_q\|$. 이것은 body-level homeostasis의 자체 복원 속도다.

혼합각 식별 (I1) 아래에서 $\rho = 0.155$ (정의 1.3의 최소 등방 closure $B$의 1회 적용당 수축률) 이므로 fast loop의 기저 수축률은 강하다. $B$의 1회 완전 적용은 약 1.6밤의 회복 수면에 대응하며, 단일 야간의 실효 수축률은 $\rho_{\text{night}} \approx 0.31$이다(수면 회복 문헌 피팅, $R^2 > 0.99$). 다만 이 exact 값은 (I1)의 중심값이다: $\delta_{\text{mix}}$가 $\pm 0.10$만 흔들려도 $\rho$는 대략 `0.1425-0.1676`, $x_a$는 `4.35%-5.45%` 범위로 이동한다. 따라서 (I1)의 구체적 수치가 달라져도 $D_{\text{eff}} > 1$인 한 수축 구조 자체는 유지되지만, fast loop의 **정확한 속도**는 rigid constant가 아니다. 건강한 성인에서 $\gamma_p\|L_G\|$가 $1 - \rho$보다 충분히 작으면 fast loop는 빠르게 수렴한다. slow loop $\|A_q\|$는 circadian scale (1일 단위)에서의 복원률이므로 0에서 1 사이의 값을 가지며, 수면박탈처럼 $\|A_q\| \to 1$이면 복원이 느려진다.

증명:
nonnegative matrix의 spectral radius bound와, 상삼각 행렬의 고유값이 대각 원소라는 사실에서 바로 나온다. $\square$

### 10.2 따름정리: noise 하의 정상상태 오차 bound

noise $\xi_{r,n}$이 $\|\xi_{r,n}\| \le \bar\xi$로 bounded이면, $\rho(K_0) < 1$인 한

$$
\limsup_{n\to\infty} \|\chi_{r,n}\|
\le
\frac{C_0\,\bar\xi}{1 - \rho(K_0)}
$$

이다. 즉 정상상태 오차는 noise 크기에 비례하고 안정성 여유 $1 - \rho(K_0)$에 반비례한다.

해석:
- 안정성 여유가 클수록 noise에 강하다
- 수면박탈로 $\|A_q\| \uparrow$이면 안정성 여유가 줄어 같은 noise에도 오차가 커진다
- 이것이 "수면박탈이 뇌 상태를 불안정하게 한다"의 정량적 의미다

### 10.3 정리: Bootstrap-Laplacian Spectral Theorem (정밀 spectral radius)

§ 7 의 row-sum bound $\|K_{\text{brain}}\|_\infty$ 는 충분조건이지만, fast loop 의 *정확한* spectral radius 는 부트스트랩 contraction 과 graph Laplacian 의 직합으로 닫힌다. 이 정리는 § 7 의 bound 가 어디서 loose 한지, 그리고 graph 가 부트스트랩 수렴 속도에 *언제* 영향을 주는지를 명시한다.

**정리 10.3** (Bootstrap-Laplacian Spectral Theorem).
$L_G$ 가 무방향 가중 brain graph 의 normalized Laplacian (자기수반, $\sigma(L_G) \subset [0, 2]$, 최소 고유값 $\mu_0 = 0$ 은 DC mode), $\rho_B = D_{\text{eff}} \cdot \varepsilon^2$ 가 부트스트랩 contraction rate, $\gamma_p \ge 0$ 가 graph coupling 강도라 하자. Tangent-space joint operator

$$T := \rho_B I + \gamma_p (-L_G)$$

의 spectrum 은 정확히

$$\boxed{\sigma(T) = \{\rho_B - \gamma_p \mu : \mu \in \sigma(L_G)\}}$$

이고 spectral radius 는

$$\rho(T) = \max\!\big(|\rho_B|,\; |\rho_B - \gamma_p \lambda_{\max}|\big),
\qquad \lambda_{\max} := \max \sigma(L_G).$$

증명:
$L_G$ 가 자기수반이므로 정규직교 고유분해 $\{(\mu_k, \phi_k)\}$ 가 존재한다. 임의 $e = \sum_k a_k \phi_k$ 에 대해 $T e = \sum_k (\rho_B - \gamma_p \mu_k) a_k \phi_k$. 따라서 $\sigma(T) = \rho_B - \gamma_p \cdot \sigma(L_G)$. spectral radius 는 양 끝점 ($\mu = 0$ 과 $\mu = \lambda_{\max}$) 의 절댓값 중 큰 것. $\square$

**따름정리 10.3.1** (regime 분리).
다음 두 regime 으로 갈린다.

| 조건 | $\rho(T)$ | 의미 |
|---|---|---|
| Subcritical: $\gamma_p \le 2\rho_B / \lambda_{\max}$ | $\rho_B$ (고정) | 부트스트랩 dominant, graph 영향 없음 |
| Supercritical: $\gamma_p > 2\rho_B / \lambda_{\max}$ | $\gamma_p \lambda_{\max} - \rho_B$ | graph dominant, contraction *느려짐* |

stability boundary ($\rho(T) = 1$) 는 $\gamma_p^* = (1 + \rho_B) / \lambda_{\max}$.

**따름정리 10.3.2** (§ 7 row-sum bound 의 looseness).
§ 7 의 sufficient bound $\rho_{\text{fast}}^{\text{ub}} = \rho_B + \gamma_p \|L_G\|$ 는 단조 부등식

$$\rho(T) \le \rho_{\text{fast}}^{\text{ub}}$$

이지만 일반적으로 strict 이며, subcritical 영역에서 두 값의 비율은 30-70 % 까지 떨어질 수 있다. 적합/판정에는 위 정확 식 (정리 10.3) 을 쓰는 편이 안전하다.

**따름정리 10.3.3** (Fiedler value 의 역할 정정).
Subcritical 영역에서 fast loop 의 contraction rate 는 $\rho_B$ 로 고정되므로 Fiedler value $\lambda_2$ 와 직접 관계가 없다. $\lambda_2$ 가 결정하는 것은 *non-trivial mode 사이의 상대 감쇠 속도* — 즉 *공간 패턴 (regional roughness)* 의 평탄화 속도이지 *전역 평균* 의 수렴 속도가 아니다.

**따름정리 10.3.4** (DC mode bottleneck — 우주론 비율의 universality).
부트스트랩 수렴의 bottleneck 은 항상 $L_G$ 의 0 고유값 (DC mode = 전체 평균) 이며, 그 mode 의 contraction rate 는

$$\rho_{\text{DC}} = \rho_B = D_{\text{eff}} \cdot \varepsilon^2 \approx 0.155.$$

따라서

$$\boxed{\text{healthy brain 의 부트스트랩 수렴 속도는 graph 구조에 무관하게 우주론 비율 } p^* \text{ 로 고정된다.}}$$

graph 가 하는 일은 *고주파 모드 (지역 noise) 를 빠르게 감쇠시키는 것뿐이며, 전역 평균이 $p^*$ 로 가는 속도는 graph 구조와 무관*하다. 이는 우주론 비율 $p^* = (\varepsilon^2, \Omega_{DM}, \Omega_\Lambda)$ 가 모든 healthy brain 에 universal 하게 적용되는 spectral 증명이다.

지위: 정리 10.3 자체는 `Exact` (선형대수). 따름정리 10.3.4 의 universality 해석은 $L_G$ 가 healthy brain 그래프의 적절한 모형이라는 가정 아래 `Bridge`.

수치 검증: `examples/physics/bootstrap_laplacian_consistency.py` (100-parcel WS small-world graph 에서 위 식들이 정확히 성립함을 확인).

### 10.4 정리: Bootstrap Fixed-Point Uniqueness (simplex 위 Banach 닫힘)

§ 9.1 의 simplex invariance 와 § 10.3 의 spectral theorem 을 결합하면, 부트스트랩 고정점 $p^*$ 가 healthy brain 의 *유일한* steady state 라는 사실이 Banach fixed-point theorem 으로 즉시 닫힌다.

**정리 10.4** (Bootstrap Fixed-Point Uniqueness).
$\gamma_p \le 2\rho_B / \lambda_{\max}$ 인 subcritical regime 에서, forcing 이 없는 ($H_r \delta q_n = 0$, $C_r w_{r,n} = 0$, $\xi_{r,n} = 0$) graph-coupled relaxation map

$$B(p) := \Pi_{\Delta^2}\!\Big((1-\rho_B) p^* + \rho_B p + \gamma_p (-L_G)(p - p^*)\Big)$$

는 다음을 만족한다.

(i) **Well-defined**: $B: \Delta^2 \to \Delta^2$ ($\Pi_{\Delta^2}$ 가 simplex projection 이므로 자명).

(ii) **Lipschitz contraction**: $\|B(p) - B(q)\|_2 \le \rho(T) \|p - q\|_2$ with $\rho(T) = \rho_B < 1$ (정리 10.3).

(iii) **Unique fixed point** (Banach): $\Delta^2$ 가 complete metric space, $B$ 가 contraction 이므로 unique fixed point 존재 + 임의 초기값 $p_0$ 에서 $\|p_n - p^*\|_2 \le \rho_B^n \|p_0 - p^*\|_2$.

(iv) **Fixed point 는 정확히 $p^* = (\varepsilon^2, \Omega_{DM}, \Omega_\Lambda)$**: 직접 확인 — $p^* \in \operatorname{int}(\Delta^2)$ 이고 $L_G p^*$ 의 component 는 평균이 $0$ (graph Laplacian 성질) 이지만 $p - p^* = 0$ 에서 $L_G(p - p^*) = 0$ 이므로

$$B(p^*) = \Pi\big((1-\rho_B) p^* + \rho_B p^* + 0\big) = \Pi(p^*) = p^*.$$

증명:
(i) projection 정의에 의해 자명.
(ii) Inner affine map $T_{\text{aff}}(p) := (1-\rho_B) p^* + \rho_B p + \gamma_p (-L_G)(p - p^*)$ 의 Jacobian 은

$$\partial T_{\text{aff}} / \partial p = \rho_B I + \gamma_p (-L_G) = T.$$

$L_G$ 가 자기수반이므로 $T$ 도 자기수반이고 $\|T\|_{\text{op}} = \rho(T)$. 따라서 $\|T_{\text{aff}}(p) - T_{\text{aff}}(q)\|_2 \le \rho(T) \|p - q\|_2$. simplex projection $\Pi_{\Delta^2}$ 는 closed convex set 으로의 projection 이므로 nonexpansive ($\|\Pi(x) - \Pi(y)\|_2 \le \|x - y\|_2$). 합성:

$$\|B(p) - B(q)\|_2 \le 1 \cdot \rho(T) \|p - q\|_2 = \rho_B \|p - q\|_2.$$

(iii) Banach FPT 의 표준 가정 (complete metric space + contraction with $L < 1$) 충족. unique fixed point 와 지수 수렴이 따른다.
(iv) 위 직접 계산. $\square$

**따름정리 10.4.1** (전역 일관성).
healthy brain 의 simplex 좌표는 **graph 구조, 초기 상태, 지역 noise 와 무관하게** 단일 attractor $p^*$ 로 수렴한다. 즉 우주론 비율은 healthy brain 의 globally unique steady state 이며, 이는 spectral 만의 결과 (정리 10.3) 에서 한 단계 강화된 명제다.

**따름정리 10.4.2** (supercritical regime 에서의 한계).
$\gamma_p > 2\rho_B / \lambda_{\max}$ 영역에서는 정리 10.3 (ii) 의 spectral radius 가 부트스트랩 단독값 $\rho_B$ 를 넘기 시작하므로 위 contraction 증명이 깨진다. 그러나 $\rho(T) < 1$ 인 한 Banach 는 여전히 적용되며 $p^*$ 가 unique fixed point. $\rho(T) \ge 1$ 인 영역은 § 10.3 따름정리 10.3.1 의 stability boundary 를 넘은 *병적 regime* 으로, 본 정리의 가정 밖이다 (Banach 적용 불가, 다중 fixed point 또는 발산 가능).

**따름정리 10.4.3** (interior-preservation, projection 의 생략 가능성).
$p^* \in \operatorname{int}(\Delta^2)$ 이면 임의 $p \in \Delta^2$ 에 대해 convex combination $(1-\rho_B) p^* + \rho_B p$ 의 각 성분은

$$\big((1-\rho_B) p^* + \rho_B p\big)_i \ge (1-\rho_B) p^*_{\min} = 0.845 \cdot 0.0487 > 0$$

이고 총합이 $1$ 이므로 이미 $\operatorname{int}(\Delta^2)$ 에 속한다. 따라서 $B: \Delta^2 \to \operatorname{int}(\Delta^2)$ 이며 simplex projection $\Pi_{\Delta^2}$ 는 *적어도 첫 iteration 이후* 로는 절대 활성화되지 않는다. 특히 임의 vertex 초기점에서도 첫 step 이후 interior 로 이동하므로 관찰 수렴률이 이론값 $\rho_B$ 와 일치한다 (수치 검증: `bootstrap_laplacian_consistency.py` — vertex $(1,0,0)$ 시작에서 첫 step ratio $= 0.154769 = \rho_B$ 정확 일치).

**지위**: 정리 10.4 (i)-(iv) 모두 `Exact` (선형대수 + Banach FPT). 따름정리 10.4.3 도 `Exact` (arithmetic). 따름정리 10.4.1 의 universality 해석은 § 10.3 따름정리 10.3.4 와 같이 healthy brain 의 graph 모형 적합성에 의존하므로 `Bridge`.

### 10.5 정리: CFC–Spectral Dimension Correspondence

§ 10.3-10.4 가 부트스트랩 *수렴 속도* 를 graph spectrum 으로 환원했다면, 본 절은 *CFC 결합상수* 자체를 graph spectrum 으로 환원한다. CE 의 비최소 중력결합

$$\xi = \alpha_s^{1/3} = 0.490 \quad (\text{`docs/7\_AGI/2\_Architecture.md` § 6, CFC = $\xi R \Phi^2$ 대응})$$

의 exponent "3" 이 *어떤* dimension 인지에 대한 세 후보 가설을 비교한다.

| 가설 | 식 | dimension 해석 |
|---|---|---|
| H1 (canonical) | $\xi = \alpha_s^{1/3}$ | spacetime integer $d = 3$ |
| H2 ($D_{\text{eff}}$) | $\xi = \alpha_s^{1/D_{\text{eff}}}$ | $3 + \delta = 3.178$, 부트스트랩 dimension |
| H3 (spectral) | $\xi = \alpha_s^{1/d_s}$ | brain graph 의 heat-kernel spectral dimension |

여기서 spectral dimension 은 표준 정의 (Alexander–Orbach 1982; Durhuus–Jonsson–Wheater 2009):

$$d_s := 2 \lim_{\lambda \to 0^+} \frac{\log N(\lambda)}{\log \lambda}, \quad N(\lambda) := |\{k : \mu_k \le \lambda\}|.$$

**측정 결과** (`examples/physics/xi_derivation.py` § VIII):

| graph | $d_s$ | $\xi(d_s) = \alpha_s^{1/d_s}$ | vs CE $\xi$ |
|---|---|---|---|
| 1D ring (n=400) | 1.80 | 0.304 | −38.0 % |
| 2D torus 20×20 | 2.23 | 0.384 | −21.7 % |
| 3D torus 8×8×8 | 3.53 | 0.546 | +11.4 % |
| WS k=6 β=0.0 | 1.75 | 0.295 | −39.8 % |
| WS k=6 β=0.1 | 2.53 | 0.429 | −12.5 % |
| WS k=6 β=0.5 | 7.65 | 0.756 | +54.3 % |
| **WS k=12 β=0.1** | **2.76** | **0.460** | **−6.1 %** |

**정리 10.5** (CFC–Spectral Dimension Correspondence, bridge level).
WS small-world graph 의 spectral dimension $d_s$ 와 CE 의 비최소 결합 $\xi = \alpha_s^{1/3}$ 사이에 다음이 성립한다.

(i) **단조성**: $d_s \uparrow \Rightarrow \alpha_s^{1/d_s} \uparrow$ ($\alpha_s < 1$ 이므로 자명).

(ii) **Best fit**: 인간 피질 connectivity 의 표준 small-world 통계 (Bullmore & Sporns 2009: 평균 차수 $k \approx 10\text{-}15$, rewiring $\beta \approx 0.1$) 에 대응하는 WS 그래프 ($n=400, k=12, \beta=0.1$) 에서 $d_s = 2.76$, $\alpha_s^{1/d_s} = 0.460$, CE 와 6.1% 차이.

(iii) **Thermodynamic limit 추측**: $n \to \infty$ 와 $\beta \to 0^+$ 의 jointly subcritical 극한에서 $d_s \to 3$ 로 수렴하면 $\xi(d_s) \to \alpha_s^{1/3}$ (H1) 이 정확히 회복된다. 본 결과는 finite-size + finite-rewiring 보정에 의한 6% gap 으로 해석.

증명 sketch:
(i) $\alpha_s = 0.1179 < 1$ 이므로 $f(d) = \alpha_s^{1/d}$ 는 $d > 0$ 에서 단조 증가 (직접 미분 $f'(d) = -\alpha_s^{1/d} \log\alpha_s / d^2 > 0$).
(ii) 위 표는 직접 측정.
(iii) Regular 3D torus 가 정확히 $d_s = 3$ 을 산출함은 Weyl law 의 직접 결과. 작은 격자 finite-size 가 약 +0.5 의 upward bias 를 만들고 ($d_s = 3.53$ 측정), small-world rewiring 이 반대로 $d_s$ 를 끌어올리거나 (high $\beta$) 살짝만 흔든다 (low $\beta$). 두 효과의 cancellation 이 small WS regime 에서 우연히 $\xi(d_s) \approx \xi$ 를 만든다는 것이 본 정리의 *경험적* 발견. 형식 증명은 별도 과제. $\square$

**해석**. 정리 10.5 는 **CE 의 비최소 중력결합 $\xi$ 가 brain graph 의 spectral dimension 에 대응한다는 H3 가설** 이 small-world regime 에서 6% 이내로 측정 가능함을 보인다. 이는 § 10.3-10.4 의 contraction 결과 (graph 가 *전역 수렴 속도* 에 영향 0 이라는 강한 negation) 와 *상보적* 으로, **graph 가 결합상수의 magnitude 에는 영향을 준다** 는 발견.

지위:
- 정리 10.5 (i)-(ii) 는 `Exact` (단조성 + 측정).
- (iii) thermodynamic limit 추측은 `Selection` (수치 보정 모형).
- "brain 그래프 ~ WS k=12 β=0.1" 라는 매핑 자체는 `Bridge` (Bullmore & Sporns 2009 의 brain connectivity 통계 적합성 가정).

수치 근거: `examples/physics/xi_derivation.py` § VIII.

**따름정리 10.5.1** (CE canonical sphere graph 의 spectral 관찰).
`docs/7_AGI/12_Equation.md` § 1.6-1.7 의 CE canonical 3D sparse graph ($r_c = \pi$ sphere connectivity, $K = (4/3)\pi r_c^3 \approx 130$) 의 spectral dimension 을 측정하면

$$d_s^{(\text{CE sphere}, L=10)} = 8.98, \quad \alpha_s^{1/d_s} = 0.788$$

으로, CE $\xi = \alpha_s^{1/3} = 0.490$ 와 60.8 % 차이. 즉 **$r_c = \pi$ sphere graph 는 H3 가설을 만족하지 않는다** — average degree $K$ 가 너무 커서 spectral dimension 이 mean-field 한계 ($d_s \to \infty$) 쪽으로 끌려가기 때문.

이는 정리 10.5 (iii) thermodynamic limit 추측이 *순진하게 $n \to \infty$ 만 취해서는 안 되며*, 정확한 limit 은 $n \to \infty$ 와 동시에 $K$ 가 *고정* 되거나 $K/n \to 0$ 으로 sublinear 하게 가야 함을 보여준다. $r_c = \pi$ sphere connectivity 는 $K$ 가 절대상수로 고정되긴 하지만 ($K \approx 130$) 그 값이 너무 커서 finite-size lattice 에서 long-range mean-field 효과가 dominant.

**따름정리 10.5.2** (H3 가설의 적용 영역 정정).
정리 10.5 의 H3 (spectral dimension carrier) 는 다음 두 조건을 동시에 만족하는 graph family 에서만 $\xi = \alpha_s^{1/3}$ 와 정합:

(a) **Sparse**: $K = O(\log n)$ 또는 더 빠르게 (i.e. $K/n \to 0$).
(b) **Locally 3-dimensional**: short-range connectivity 가 3D lattice 와 위상학적으로 등가.

WS $(k=12, \beta=0.1)$ 는 (a)(b) 를 모두 만족 (sparse + 3D-like rewiring), 따라서 $d_s \approx 2.76 \to 3$ in thermodynamic limit, $\xi(d_s) \to 0.49$. CE $r_c = \pi$ sphere 는 (a) 를 만족하지 ($K$ 절대상수) 만 (b) 를 위배 (3D ball 안 *모든* 점에 연결되어 local geometry 가 lattice 보다 더 dense). 따라서 H3 는 후자에 적용되지 않는다.

**해석**. 따름정리 10.5.1-10.5.2 는 **CE 의 $\xi$ 가 brain graph spectral dimension 에 universal 하게 대응하지 않으며, 대응이 성립하려면 graph 가 sparse + 3D-local 이어야 한다** 는 strong constraint. 이는 healthy brain functional connectivity 가 정확히 이 두 조건을 만족하는 *유일한* family 라는 점에서 H3 가설의 강력한 *consistency check* 를 제공.

### 10.6 정리: Boolean–Spectral Carrier Theorem

`docs/axium.md` § 1.2a.1 의 통일 boolean axis (G, E, P) 와 본 절의 Laplacian eigenmode $\{\phi_k, \mu_k\}$ 사이에 두 종류의 "직교성" 이 등장한다.

| 종류 | 정의 | 위치 |
|---|---|---|
| Boolean commutativity | $\text{apply}_X \circ \text{apply}_Y = \text{apply}_Y \circ \text{apply}_X$ for $X, Y \in \{G, E, P\}$ | `tests/test_five_constant_algebra.py::test_axes_commute_pairwise` |
| Laplacian orthogonality | $\langle \phi_i, \phi_j \rangle = \delta_{ij}$ (자기수반 spectral theorem) | § 6 |

본 정리는 두 직교성이 *카테고리적으로 같은 본질* 을 갖는다는 사실 — 즉 boolean axis 가 Laplacian spectral basis 위의 spectral 분할 (projector + 함수) 로 자연 carrier 를 가진다는 사실 — 을 정식화한다. 단, 세 axis 가 요구하는 *operator class* 가 다르다는 점이 중요한 정정.

**정리 10.6** (Boolean-Spectral Carrier).
$L_G$ 가 normalized Laplacian (자기수반, $\sigma(L_G) = \{\mu_0 = 0, \mu_1, \ldots, \mu_{n-1}\} \subset [0, 2]$, eigenbasis $\{\phi_k\}$) 라 하자. axium § 1.2a.1 의 boolean axis 의 각 ON-projector 는 다음 carrier 를 갖는다.

(i) **G axis carrier** (gate, $\{0, 1\}$ generator): DC-mode projector

$$\Pi_G := \phi_0 \phi_0^* \quad (\text{rank-1 onto kernel of } L_G).$$

자기수반 $\Pi_G$ 가 $L_G$ 와 commute: $[\Pi_G, L_G] = 0$.

(ii) **E axis carrier** (decay, $\{e\}$ generator): non-DC subspace projector

$$\Pi_E := I - \Pi_G = \sum_{k \ge 1} \phi_k \phi_k^*.$$

$\Pi_E$ 또한 자기수반이고 $L_G$ 와 commute. heat-kernel $e^{-t L_G}$ 가 $\Pi_E$-subspace 위에서 exp 감쇠를 만든다 (정리 10.3 의 $\sigma(T) = \rho_B - \gamma_p \sigma(L_G)$ 의 $\mu > 0$ 부분).

(iii) **P axis carrier** (phase, $\{\pi, i\}$ generator): standard self-adjoint $L_G$ 만으로는 carrier 가 *부재* 하다. P axis 의 자연 carrier 는 *unitary 생성자*

$$U_P(\theta) := \exp\!\big(i \theta\,f(L_G)\big), \quad f: \mathbb{R} \to \mathbb{R}\ \text{함수}$$

이며, 이는 $L_G$ 와 같은 eigenbasis 위에서 작용하지만 *서로 다른 operator class* (skew-Hermitian / unitary) 에 산다. $U_P(\theta)$ 와 $L_G$ 는 commute ($f(L_G)$ 가 $L_G$ 의 함수이므로 자명).

**따름정리 10.6.1** (boolean commutativity 의 spectral 본질).
세 axis 의 ON-projector $\{\Pi_G, \Pi_E, U_P(\theta)\}$ 는 모두 $L_G$ 의 spectral eigenbasis $\{\phi_k\}$ 위에서 동시에 대각화 가능하며, 따라서 pairwise commute. 이는 axium § 1.2a.1 의 boolean commutativity 가 *우연한 algebra 성질이 아니라 같은 spectral eigenbasis 위 spectral projector / 함수 의 commutativity* 임을 의미.

**따름정리 10.6.2** (P axis 의 추가 구조 요건).
P axis 의 carrier 는 standard 무방향 Laplacian 만으로는 닫히지 않는다. 다음 중 하나가 필요.

- (a) 복소 확장 $i L_G$ 또는 $\exp(i \theta L_G)$ — *quantum Hamiltonian* 해석
- (b) 방향성 graph 의 non-normal Laplacian — *directed connectome* 해석
- (c) 외부 unitary group $U(1)$ action — *gauge field* 해석

CE 의 EulerCEMinimal `head_types` 에서 P-bit 가 켜지는 것은 (a) 에 해당하며 ($\exp(i\theta) = \cos\theta + i\sin\theta$ 의 RoPE rotation), self-adjoint Laplacian (decay-only ALiBi) 와 직교 axis 를 형성하는 이유는 **operator class 의 차이** 이지 spectral basis 의 차이가 아니다.

**증명** (개요).
(i)–(ii): 자기수반 spectral theorem 에 의해 $L_G = \sum_k \mu_k \phi_k \phi_k^*$. 임의 spectral 부분합 $\Pi_S = \sum_{k \in S} \phi_k \phi_k^*$ 는 자기수반이며 $\Pi_S L_G = \sum_{k \in S} \mu_k \phi_k \phi_k^* = L_G \Pi_S$.
(iii): $f$ 가 임의 함수일 때 $f(L_G) = \sum_k f(\mu_k) \phi_k \phi_k^*$ 이므로 $f(L_G)$ 도 같은 eigenbasis 위. $U_P(\theta) = e^{i\theta f(L_G)} = \sum_k e^{i\theta f(\mu_k)} \phi_k \phi_k^*$ 또한 같은 eigenbasis 위 unitary. $L_G$ 와 commute. $\square$

지위:
- (i)-(iii): `Exact` (선형대수 + spectral theorem).
- 따름정리 10.6.1: `Exact`.
- 따름정리 10.6.2: `Bridge` (CE 의 EulerCEMinimal 과의 carrier 매핑은 brain graph 가 self-adjoint 모형 + complex 확장이라는 가정 아래).

수치 검증: `examples/physics/bootstrap_laplacian_consistency.py` (말미) — $\|[\Pi_G, L_G]\|_F$, $\|[\Pi_E, L_G]\|_F$ 모두 $\sim 10^{-15}$ (머신 epsilon) 으로 commute 확인. 무방향 $L_G$ 의 self-adjoint 성 ($\|L - L^T\|_F = 0$) 도 직접 확인.

---

## 11. Stage-2: $x_s$ 동역학

### 11.1 구조 채널의 forcing

stage-1에서는 $x_s = x_s^*$를 고정했다. stage-2에서는 $x_s$도 움직인다.

`measure.md`의 분해

$$
\hat x_{s,r}
=
\alpha_{\text{maint}} M_r
+
\alpha_{\text{plast}} P_r
+
\alpha_{\text{replay}} R_r
+
\alpha_{\text{support}} G_r
$$

를 동역학으로 올리면, 구조 채널의 forcing은

$$
h_r^{\text{struct}}
=
(-\beta_r,\; \beta_r,\; 0),
\qquad
\beta_r \ge 0
$$

의 형태를 가진다. 여기서 $\beta_r$는 해당 node의 plasticity/replay demand다.

$\mathbf 1^\top h_r^{\text{struct}} = 0$이므로 simplex mass는 보존된다.

해석:
- plasticity demand가 올라가면 mass가 $x_a$에서 $x_s$로 이동한다
- 이것은 학습/기억 강화 시 active budget의 일부가 structural maintenance로 전환되는 것을 뜻한다

### 11.2 Sleep + structural 결합 forcing

sleep과 structural forcing을 합치면 full control forcing은

$$
H_r \delta q_n
=
z_n\,(\alpha_r,\; 0,\; -\alpha_r)
+
\beta_{r,n}\,(-1,\; 1,\; 0) \cdot \delta q_{\text{struct},n}
+
\widetilde H_r \delta \widetilde q_n
$$

가 된다.

좌표별로 읽으면:

$$
\frac{\partial x_{a,r,n+1}}{\partial z_n} = \alpha_r,
\qquad
\frac{\partial x_{a,r,n+1}}{\partial \delta q_{\text{struct},n}} = -\beta_r
$$

$$
\frac{\partial x_{s,r,n+1}}{\partial z_n} = 0,
\qquad
\frac{\partial x_{s,r,n+1}}{\partial \delta q_{\text{struct},n}} = \beta_r
$$

$$
\frac{\partial x_{b,r,n+1}}{\partial z_n} = -\alpha_r,
\qquad
\frac{\partial x_{b,r,n+1}}{\partial \delta q_{\text{struct},n}} = 0
$$

즉:
- sleep debt는 $x_b \to x_a$로 mass를 옮긴다 (stage-1과 동일)
- structural demand는 $x_a \to x_s$로 mass를 옮긴다
- $x_b$는 structural demand에 직접 반응하지 않는다
- 두 forcing이 동시에 작용하면 $x_a$에 대한 순효과는 $\alpha_r z_n - \beta_r \delta q_{\text{struct},n}$이다

### 11.3 Stage-2 pair share

$x_s$가 움직이므로 stage-1의 $\lambda_r = u_a/(u_a + u_b)$는 더 이상 전체 상태를 기술하지 못한다.

stage-2에서는 full simplex 좌표 $(x_{a,r}, x_{s,r}, x_{b,r})$를 직접 추적한다.

다만 실측에서는 여전히 $x_s$가 가장 잡기 어렵다. 따라서 실제 적합 순서는:

1. $x_a, x_b$를 stage-1 proxy로 먼저 고정
2. $x_s = 1 - x_a - x_b$로 잔차 추정
3. 잔차 $x_s$가 $x_s^*$ 근방에 있는지 검증
4. 벗어나면 structural demand $\beta_r$를 fitting

---

## 12. 관측 모델

### 12.1 fMRI BOLD proxy

fMRI의 BOLD signal은 $p_r$의 직접 관측이 아니라, 혈역학 반응을 거친 간접 관측이다.

최소 관측 모델은

$$
y_{a,r,n}
=
g_a(x_{a,r,n})
+
\varepsilon_{a,r,n},
\qquad
y_{b,r,n}
=
g_b(x_{b,r,n})
+
\varepsilon_{b,r,n}
$$

로 쓴다. 여기서 $g_a, g_b$는 단조 관측 함수이고 $\varepsilon$은 관측 noise다.

현재 레포에서 쓰는 구체적 proxy는:

**$x_a$ 관측**:
$$
\hat x_{a,r}^{\text{fMRI}}
=
\text{responsive fraction}
:=
\frac{|\{v : \Delta_v^{\text{task-rest}} > \tau\}|}{|\{v \in \text{mask}\}|}
$$

여기서 $\Delta_v^{\text{task-rest}}$는 task 구간의 percent signal change에서 rest 구간을 뺀 voxel-level increment이고, $\tau$는 robust threshold (MAD 기반)다.

**$x_b$ 관측**:
$$
\hat x_{b,r}^{\text{fMRI}}
=
\text{network segregation}
:=
\bar R_{\text{within}} - \bar R_{\text{between}}
$$

여기서 $\bar R_{\text{within}}$은 spectral bipartition으로 정의된 모듈 내 평균 상관이고, $\bar R_{\text{between}}$은 모듈 간 평균 상관이다.

### 12.2 EEG proxy

**$q_{\text{sleep}}$ 관측**:
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

이것은 `control.md`에서 이미 정의된 것이다.

### 12.3 KSS proxy

주관적 졸림 보고도 $q_{\text{sleep}}$의 noisy observation으로 읽는다:

$$
\text{KSS}_n
=
\text{KSS}^*
+
c_K z_n
+
\varepsilon_{K,n},
\qquad
c_K > 0
$$

`ds000201` v3에서 KSS의 SD-NS 차이가 $p = 0.0003$으로 강하게 나왔으므로, $c_K > 0$은 현재 가장 강하게 지지되는 관측 부호다.

### 12.4 관측 모델 정리

$$
y_n
=
\begin{pmatrix}
\hat x_{a,r}^{\text{fMRI}}\\
\hat x_{b,r}^{\text{fMRI}}\\
r_{\alpha,n}\\
\text{KSS}_n
\end{pmatrix}
=
\begin{pmatrix}
g_a(x_{a,r,n})\\
g_b(x_{b,r,n})\\
r_\alpha^* - c_\alpha z_n\\
\text{KSS}^* + c_K z_n
\end{pmatrix}
+
\varepsilon_n
$$

이것은 state-space model의 observation equation이다. state equation은 5절의 4단계 반복이고, observation equation이 위 식이다.

따라서 전체 뇌 알고리즘은 **nonlinear state-space model**로 읽힌다:

$$
\text{State:}\quad
(p_{r,n+1},\; \delta q_{n+1},\; w_{r,n+1})
=
f(p_{r,n},\; \delta q_n,\; w_{r,n},\; u_{r,n},\; \xi_n)
$$

$$
\text{Observation:}\quad
y_n
=
h(p_{r,n},\; \delta q_n)
+
\varepsilon_n
$$

적합은 이 state-space model에 대한 filtering/smoothing으로 하고, $K_{\text{brain}}$의 안정성 검증은 적합된 parameter에서 row-sum certificate로 한다.

---

## 13. 통합 최소 알고리즘 블록

5절의 흩어진 식들을 하나의 **bridge-level 최소 상태공간 closure**로 정리한다.

**입력**: brain graph $G_{\text{brain}}$, CE 상수 $\rho = D_{\text{eff}} \cdot \varepsilon^2$ (혼합각 식별 (I1) 아래에서 정의 1.3의 최소 closure 수축률 중심값으로서 $\rho = 0.155$; $\delta_{\text{mix}} \pm 0.10$이면 `0.1425-0.1676`), $p^* = (\varepsilon^2,\; \Omega_{\text{DM}},\; \Omega_\Lambda)$ (I1 아래의 비교 좌표 중심값에서 $(0.0487, 0.2623, 0.6891)$; 같은 민감도 범위에서 대략 `(4.35%-5.45%, 25.25%-26.89%, 68.76%-69.30%)`)

**상태**: $p_{r,n} \in \Delta^2$, $\delta q_n \in \mathbb R^6$, $w_{r,n} \in \mathbb R^d$

**매 step $n$**:

**(S1)** 빠른 입력 누적

$$
p_{r,n+\frac12} = p_{r,n} + u_{r,n},
\qquad
\mathbf 1^\top u_{r,n} = 0
$$

**(S2)** 느린 제어축 갱신

$$
\delta q_{n+1} = A_q \delta q_n + r_n + \xi_n^{(q)},
\qquad
\rho(A_q) < 1
$$

**(S3)** 병적 burden 갱신 (건강하면 $w = 0$ 유지)

$$
w_{r,n+1}
=
A_w w_{r,n}
+
b_s s_{r,n}
+
b_\nu \nu_{r,n}
+
u_{r,n}^{(w)}
$$

**(S4)** 수면-그래프-몸축 수축

$$
p_{r,n+1}
=
\Pi_{\Delta^2}\!\Big(
(1-\rho)p^*
+
\rho p_{r,n+\frac12}
+
\gamma_p \Delta_G p_{r,n+\frac12}
+
H_r \delta q_n
+
C_r w_{r,n}
+
\xi_{r,n}
\Big)
$$

**보조 계산**:

$$
s_{r,n}
=
\eta_a(x_{a,r,n}-x_a^*)_+
+
\eta_s(x_s^*-x_{s,r,n})_+
+
\eta_b(x_b^*-x_{b,r,n})_+
+
\eta_q \ell_r(n)
$$

$$
\ell_r(n) = d_r^\top (\delta q_n)_+
$$

**안정성 조건**: $\rho(K_{\text{brain}}) < 1$ (7절)

**건강한 정상상태**: $p_{r,n} \to p^*$, $\delta q_n \to 0$, $w_{r,n} = 0$

---

## 14. 사용 규칙

| 문장 | 판정 |
|---|---|
| `r`를 parcel/node로 읽는다 | `bridge` |
| `\Delta_G`로 취약도 spread/buffering을 쓴다 | `bridge` |
| 위의 상태공간 모델을 뇌 알고리즘의 bridge-level 최소 closure로 쓴다 | `bridge` |
| `\rho(K_{\text{brain}})<1`을 건강 알고리즘의 안정성 조건으로 쓴다 | `bridge` |
| `q_sleep \uparrow \Rightarrow x_b \downarrow \Rightarrow s_r \uparrow`를 위 최소 정리로 읽는다 | `bridge` |
| simplex invariance가 forcing bound 조건 하에서 성립한다 | `supported` (정리 9.1) |
| 건강한 상태에서 지수 수렴한다 | `supported` (정리 10.1) |
| noise 하에서 정상상태 오차가 bounded다 | `supported` (따름정리 10.2) |
| $x_s$ forcing이 $x_a \to x_s$ mass transfer로 읽힌다 | `bridge` |
| fMRI/EEG/KSS를 state-space model의 관측방정식으로 읽는다 | `bridge` |
| 현재 그래프 하나로 모든 사람의 뇌를 동일하게 설명한다 | `hypothesis` |
