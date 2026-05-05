## Mouse IBL/OpenAlyx region-interaction effective-connectivity proxy gate

Flat-unit gate 다음에는 weighted interaction proxy를 걸었다. 단, 여기서 "effective connectivity"라는 말은 causal connectivity가 아니다. 같은 trial window 안에서 channel-rescued region rates의 pairwise product가 additive region bins보다 더 설명력이 있는지 보는 약한 proxy다.

Hybrid region vector를 \(R_i^{\mathrm{hybrid}}=(r_{i1},\ldots,r_{id})\)라고 두면 interaction feature는 다음이다.

$$
z_{iab}
=
r_{ia}r_{ib},
\qquad
a<b.
$$

따라서 interaction-augmented region vector는

$$
R_i^{\mathrm{int}}
=
\left[
R_i^{\mathrm{hybrid}},
Z_i
\right],
\qquad
Z_i=\{z_{iab}\}_{a<b}.
$$

비교한 증분은 두 개다.

$$
\Delta_{\mathrm{int-region}}
=
\mathrm{BA}(R^{\mathrm{int}})
-
\mathrm{BA}(R^{\mathrm{hybrid}}),
$$

$$
\Delta_{\mathrm{task+int}}
=
\mathrm{BA}
\left(
\left[X^{\mathrm{task}},R^{\mathrm{int}}\right]
\right)
-
\mathrm{BA}
\left(
\left[X^{\mathrm{task}},R^{\mathrm{hybrid}}\right]
\right).
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_effective_connectivity_proxy_gate.py
```

Target replication:

| target | int beats region | int beats unit | task+int beats task+region | mean region BA | mean int BA | mean unit BA | mean int-region delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 3/5 | 0/5 | 1/5 | 0.675560 | 0.668513 | 0.744883 | -0.007047 |
| first movement speed | 3/5 | 1/5 | 3/5 | 0.742104 | 0.735488 | 0.756299 | -0.006616 |
| wheel action direction | 2/5 | 0/5 | 0/5 | 0.716240 | 0.700149 | 0.777766 | -0.016091 |

Candidate deltas:

| candidate | choice int-region | speed int-region | wheel int-region |
|---|---:|---:|---:|
| `witten29_thalamic_visual_reference` | -0.019573 | -0.019190 | -0.019955 |
| `nyu30_motor_striatal_multi_probe` | 0.004200 | 0.021640 | -0.021340 |
| `dy014_striatal_septal_probe` | 0.008361 | 0.022109 | 0.005625 |
| `dy011_motor_cortex_probe` | -0.032431 | 0.000113 | -0.067641 |
| `dy008_cp_somatosensory_thalamic_probe` | 0.004207 | -0.057753 | 0.022855 |

판정:

$$
\boxed{
\mathrm{same\text{-}window\ region\ interaction\ proxy}
\quad
\mathrm{does\ not\ pass.}
}
$$

이유는 count만 보면 choice와 speed에서 interaction이 3/5 후보에서 additive region을 이기지만, mean delta는 세 target 모두 음수이기 때문이다.

$$
\overline{\Delta}_{\mathrm{int-region}}
=
\begin{cases}
-0.007047,&\mathrm{choice},\\
-0.006616,&\mathrm{speed},\\
-0.016091,&\mathrm{wheel}.
\end{cases}
$$

또한 interaction은 top-unit을 거의 이기지 못한다.

$$
\mathrm{int>unit}
=
\mathrm{choice}\ 0/5,
\mathrm{speed}\ 1/5,
\mathrm{wheel}\ 0/5.
$$

따라서 \(K_{\mathrm{region}}(R\otimes R)\) 항은 현재 식에 승격하지 않는다.

$$
\boxed{
K_{\mathrm{same\ window}}^{(s)}
\left(
R_t^{(s,\mathrm{hybrid})}
\otimes
R_t^{(s,\mathrm{hybrid})}
\right)
\quad
\mathrm{not\ promoted.}
}
$$

이 음성 결과가 weighted interaction 전체를 부정하지는 않는다. 같은 window 안의 곱 \(r_a(t)r_b(t)\)는 방향성과 지연을 잃는다. 그래서 다음 gate는 아래처럼 source window와 target window를 분리한 lagged coupling으로 진행했다.

$$
z_{iab}^{(\delta)}
=
r_{ia}(t-\delta)r_{ib}(t),
\qquad
\delta>0.
$$

### Mouse IBL/OpenAlyx lagged region-coupling proxy gate

Lagged proxy에서는 source window와 target window를 분리했다. Choice와 first-movement speed는 `stimOn -300 ms`부터 `stimOn`까지를 source로 두고, target은 기존 stimulus window `stimOn +20 ms`부터 `+320 ms`까지로 둔다. Wheel action direction은 `firstMovement -400 ms`부터 `-100 ms`까지를 source로 두고, target은 기존 movement window `firstMovement -100 ms`부터 `+200 ms`까지로 둔다.

식은 다음이다.

$$
z_{iab}^{\mathrm{lag}}
=
r_{ia}^{\mathrm{source}}
r_{ib}^{\mathrm{target}}.
$$

Target-window region과 결합한 lagged model은

$$
R_i^{\mathrm{lag}}
=
\left[
R_i^{\mathrm{target}},
Z_i^{\mathrm{lag}}
\right].
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_lagged_coupling_proxy_gate.py
```

Target replication:

| target | lagged beats region | lagged beats unit | task+lagged beats task+region | mean source BA | mean region BA | mean lagged BA | mean unit BA | mean lagged-region delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 0/5 | 0/5 | 0/5 | 0.531141 | 0.675560 | 0.637484 | 0.744883 | -0.038076 |
| first movement speed | 1/5 | 1/5 | 0/5 | 0.619563 | 0.742104 | 0.706063 | 0.756299 | -0.036041 |
| wheel action direction | 0/5 | 0/5 | 0/5 | 0.534773 | 0.716240 | 0.674517 | 0.777766 | -0.041723 |

Candidate deltas:

| candidate | choice lag-region | speed lag-region | wheel lag-region |
|---|---:|---:|---:|
| `witten29_thalamic_visual_reference` | -0.061322 | -0.079969 | -0.077407 |
| `nyu30_motor_striatal_multi_probe` | -0.046398 | -0.004770 | -0.041413 |
| `dy014_striatal_septal_probe` | -0.018035 | -0.023810 | -0.023171 |
| `dy011_motor_cortex_probe` | -0.028084 | 0.008091 | -0.049638 |
| `dy008_cp_somatosensory_thalamic_probe` | -0.036539 | -0.079746 | -0.016987 |

판정:

$$
\boxed{
\mathrm{lagged\ region\ coupling\ proxy}
\quad
\mathrm{does\ not\ pass.}
}
$$

Same-window interaction보다도 결과가 더 명확한 음성이다.

$$
\overline{\Delta}_{\mathrm{lag-region}}
=
\begin{cases}
-0.038076,&\mathrm{choice},\\
-0.036041,&\mathrm{speed},\\
-0.041723,&\mathrm{wheel}.
\end{cases}
$$

그리고 top-unit도 이기지 못했다.

$$
\mathrm{lag>unit}
=
\mathrm{choice}\ 0/5,
\mathrm{speed}\ 1/5,
\mathrm{wheel}\ 0/5.
$$

따라서 이번 공개 IBL 5-candidate panel에서 승격되는 것은 \(K(R_{t-\delta}\otimes R_t)\)가 아니다.

$$
\boxed{
K_{\mathrm{lagged\ region}}^{(s)}
\left(
R_{t-\delta}^{(s,\mathrm{hybrid})}
\otimes
R_t^{(s,\mathrm{hybrid})}
\right)
\quad
\mathrm{not\ promoted.}
}
$$

이제 mouse 단계에서 남는 결론은 더 좁다.

$$
\boxed{
\mathrm{IBL\ mouse\ panel:}
\quad
\mathrm{task/history}
+
\mathrm{unit\ detail}
+
\mathrm{region\ compression},
\qquad
\mathrm{not\ yet\ weighted\ region\ coupling.}
}
$$

이 음성 결과 뒤에는 region-rate product가 아니라 all-unit nested regularization을 먼저 검사했다. 이유는 flat-unit gate가 이미 top-unit readout의 강함을 보였기 때문이다. 만약 \(U_t\)가 \(X_t\)와 \(R_t\) 뒤에서도 독립적으로 남으면, mouse 단계 방정식에는 explicit unit-detail residual을 승격해야 한다.

### Mouse IBL/OpenAlyx all-unit nested-regularization gate

Nested gate의 질문은 단순히 unit decoder가 region decoder보다 높은가가 아니다. 더 좁은 질문은 다음이다.

1. Task/history \(X_i\)가 이미 들어간 뒤 region \(R_i\)가 남는가.
2. Task/history와 region \(R_i\)가 이미 들어간 뒤 unit identity \(U_i\)가 남는가.
3. 반대로 task/history와 unit \(U_i\)가 이미 들어간 뒤 region compression \(R_i\)가 남는가.

여기서 \(X_i\)는 current stimulus, previous choice/reward, trial timing history로 구성한 task/history covariate이고, \(R_i\)는 channel-rescued hybrid region bin, \(U_i\)는 probe별 high-spike unit identity bin이다. 이번 실행에서는 `min_unit_spikes=1000`, `max_units_per_probe=192`를 사용했다. 이 cap은 계산량을 제한하기 위한 guard이며, 이전 top-unit 96개보다 넓은 unit panel이다.

비교한 nested model은 다음이다.

$$
M_X:
y_i\sim X_i,
\qquad
M_{XR}:
y_i\sim [X_i,R_i],
$$

$$
M_{XU}:
y_i\sim [X_i,U_i],
\qquad
M_{XRU}:
y_i\sim [X_i,R_i,U_i].
$$

각 model은 같은 deterministic stratified fold 위에서 ridge linear decoder로 평가한다.

$$
\hat\beta_m^{(-k)}
=
\arg\min_{\beta}
\left\|
y_{\mathrm{train}(k)}
-
Z_{m,\mathrm{train}(k)}\beta
\right\|_2^2
+
\lambda\|\beta\|_2^2,
\qquad
\lambda=1.
$$

Test fold \(k\)에서 얻은 score를 합쳐 balanced accuracy를 계산한다.

$$
\mathrm{BA}(M_m)
=
\frac12
\left(
\frac{TP_m}{P}
+
\frac{TN_m}{N}
\right).
$$

핵심 residual은 두 개다.

$$
\Delta_{U\mid X,R}
=
\mathrm{BA}(M_{XRU})
-
\mathrm{BA}(M_{XR}),
$$

$$
\Delta_{R\mid X,U}
=
\mathrm{BA}(M_{XRU})
-
\mathrm{BA}(M_{XU}).
$$

\(\Delta_{U\mid X,R}>0\)가 후보 session들에서 반복되면, unit identity는 task와 anatomical region을 넘어 독립적으로 살아남는다. 반대로 \(\Delta_{R\mid X,U}>0\)가 반복되면, high-dimensional unit readout 위에서도 region compression이 작지만 안정적인 정규화 항으로 남는다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_all_unit_nested_regularization_gate.py --permutations 10 --max-units-per-probe 192
```

Target replication:

| target | unit residual count | region residual count | unit>region after task | mean task BA | mean task+region BA | mean task+unit BA | mean task+region+unit BA | mean unit residual | mean region residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 0/5 | 3/5 | 0/5 | 0.844886 | 0.845811 | 0.800778 | 0.802420 | -0.043390 | 0.001643 |
| first movement speed | 1/5 | 3/5 | 1/5 | 0.689940 | 0.750006 | 0.707508 | 0.711956 | -0.038051 | 0.004448 |
| wheel action direction | 0/5 | 4/5 | 0/5 | 0.822181 | 0.837202 | 0.800453 | 0.803640 | -0.033562 | 0.003187 |

Candidate residuals:

| candidate | unit residuals | region residuals | choice \(U\mid X,R\) | speed \(U\mid X,R\) | wheel \(U\mid X,R\) | choice \(R\mid X,U\) | speed \(R\mid X,U\) | wheel \(R\mid X,U\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 0 | 2 | -0.051956 | -0.019195 | -0.036050 | 0.003912 | -0.008008 | 0.010234 |
| `nyu30_motor_striatal_multi_probe` | 0 | 2 | -0.004495 | -0.035997 | -0.003564 | 0.000827 | -0.010814 | 0.005592 |
| `dy014_striatal_septal_probe` | 1 | 1 | -0.017299 | 0.030612 | -0.044043 | -0.015933 | 0.003401 | -0.014457 |
| `dy011_motor_cortex_probe` | 0 | 3 | -0.068480 | -0.058257 | -0.054654 | 0.031650 | 0.021206 | 0.003795 |
| `dy008_cp_somatosensory_thalamic_probe` | 0 | 2 | -0.074722 | -0.107416 | -0.029497 | -0.012242 | 0.016453 | 0.010773 |

판정:

$$
\boxed{
\Delta_{U\mid X,R}
<
0
\quad
\mathrm{on\ mean\ for\ choice,\ speed,\ wheel.}
}
$$

수치로는 다음과 같다.

$$
\overline{\Delta}_{U\mid X,R}
=
\begin{cases}
-0.043390,&\mathrm{choice},\\
-0.038051,&\mathrm{speed},\\
-0.033562,&\mathrm{wheel}.
\end{cases}
$$

반대로 region residual은 작지만 반복된다.

$$
\mathrm{count}
\left(
\Delta_{R\mid X,U}>0
\right)
=
\mathrm{choice}\ 3/5,
\mathrm{speed}\ 3/5,
\mathrm{wheel}\ 4/5.
$$

따라서 이번 all-unit nested gate의 결론은 다음이다.

$$
\boxed{
B_{\mathrm{unit}}U_t
\quad
\mathrm{is\ not\ promoted\ as\ an\ independent\ additive\ residual\ after}
\quad
[X_t,R_t].
}
$$

그리고 더 중요한 정리:

$$
\boxed{
R_t^{(\mathrm{hybrid})}
\quad
\mathrm{survives\ as\ a\ compact\ anatomical\ regularizer\ over}
\quad
[X_t,U_t].
}
$$

이 결과는 flat-unit gate와 모순이 아니다. Flat-unit gate는 \(U_t\)와 \(R_t\)를 단독 또는 task와 각각 결합해 비교했다. 그 결과 top-unit은 choice/wheel에서 region보다 강했다. 하지만 nested gate는 \(R_t\)를 이미 포함한 뒤 \(U_t\)가 더해지는지를 본다. 고차원 unit bin은 candidate panel 평균에서 cross-validated generalization을 오히려 낮췄다. 즉 mouse 단계의 현재 식은 "unit identity가 중요하지 않다"가 아니라, "공개 IBL 5-candidate panel의 ridge nested decoder에서는 region-compressed state가 더 안정적인 중간 표현이다"로 읽어야 한다.

이 common-ridge 음성 판정은 여기서 멈추지 않는다. 바로 다음 gate에서 unit block의 regularization mismatch를 검사했다.

$$
\lambda_X\|\beta_X\|_2^2
+
\lambda_R\|\beta_R\|_2^2
+
\lambda_U\|\beta_U\|_2^2,
\qquad
\lambda_U
\gg
\lambda_R
\quad
\mathrm{or}
\quad
y_t\sim f(U_{t-\ell},R_t,X_t).
$$

### Mouse IBL/OpenAlyx block-regularized unit residual gate

Common-ridge nested gate의 가장 직접적인 반례는 penalty mismatch다. Unit block \(U_i\)는 high-dimensional이고, region block \(R_i\)는 compact하다. 같은 ridge \(\lambda=1\)을 쓰면 \(U_i\)는 fold마다 불안정해질 수 있다. 그래서 block-regularized gate는 \(\lambda_X=\lambda_R=1\)로 고정하고, \(\lambda_U\)만 outer train split 내부의 inner CV로 고른다.

$$
\hat\beta_{\lambda_U}^{(-k)}
=
\arg\min_\beta
\left\|y-Z\beta\right\|_2^2
+
\lambda_X\|\beta_X\|_2^2
+
\lambda_R\|\beta_R\|_2^2
+
\lambda_U\|\beta_U\|_2^2.
$$

$$
\lambda_U^{*(-k)}
=
\arg\max_{\lambda_U\in\{1,3,10,30,100,300,1000\}}
\mathrm{BA}_{\mathrm{inner}}(\lambda_U).
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_block_regularized_unit_gate.py
```

Target replication:

| target | \(U_{\mathrm{block}}\mid X,R\) count | \(R\mid X,U_{\mathrm{block}}\) count | unit>region after task | mean task BA | mean task+region BA | mean block task+unit BA | mean block task+region+unit BA | mean \(U_{\mathrm{block}}\mid X,R\) | mean \(R\mid X,U_{\mathrm{block}}\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 4/5 | 2/5 | 4/5 | 0.844886 | 0.845811 | 0.866224 | 0.862736 | 0.016925 | -0.003489 |
| first movement speed | 5/5 | 0/5 | 5/5 | 0.689940 | 0.750006 | 0.798966 | 0.784605 | 0.034599 | -0.014360 |
| wheel action direction | 5/5 | 4/5 | 3/5 | 0.822181 | 0.837202 | 0.855839 | 0.865689 | 0.028487 | 0.009850 |

판정:

$$
\boxed{
\Delta_{U\mid X,R}^{\mathrm{block}}
>
0
\quad
\mathrm{replicates\ for\ choice,\ speed,\ and\ wheel.}
}
$$

수치로는 다음이다.

$$
\overline{\Delta}_{U\mid X,R}^{\mathrm{block}}
=
\begin{cases}
0.016925,&\mathrm{choice},\\
0.034599,&\mathrm{speed},\\
0.028487,&\mathrm{wheel}.
\end{cases}
$$

따라서 common-ridge all-unit 음성 결과는 unit-detail이 없다는 결론이 아니라, high-dimensional unit block을 compact region block과 같은 penalty로 다룬 artifact로 해석한다. Mouse 단계 방정식에는 explicit unit-detail residual을 조건부 항으로 다시 올린다.

$$
\boxed{
B_{\mathrm{unit\ detail}}^{(s)}
U_t^{(s)}
\quad
\mathrm{survives\ after}
\quad
\left[
X_t^{(s,\mathrm{task})},
R_t^{(s,\mathrm{hybrid})}
\right]
\quad
\mathrm{under\ block\ regularization.}
}
$$

동시에 region residual도 wheel에서는 \(4/5\)로 남으므로, 결론은 unit-only가 아니다. 현재 mouse 항은 task/history, block-regularized unit detail, hybrid anatomical compression이 서로 다른 bias-variance 위치에서 살아남는 mixed readout이다.

### Mouse IBL/OpenAlyx temporal GLM coupling gate

Block-regularized gate가 살린 것은 같은 target window 안의 static unit detail이다. 따라서 다음 반례는 더 엄격하다. 같은 unit detail이 단순히 현재 window의 identity readout일 뿐인지, 아니면 조금 전 source window의 unit activity \(U_{t-\ell}\)가 task와 region, 그리고 현재 unit detail \(U_t\) 뒤에서도 행동 target을 더 설명하는지 보았다.

비교식은 다음 네 개다.

$$
\begin{aligned}
M_{XR} &: y_t \sim [X_t, R_t],\\
M_{XRU0} &: y_t \sim [X_t, R_t, U_t],\\
M_{XRUL} &: y_t \sim [X_t, R_t, U_{t-\ell}],\\
M_{XRU0UL} &: y_t \sim [X_t, R_t, U_t, U_{t-\ell}].
\end{aligned}
$$

여기서 \(X_t\)는 stimulus/history/task table, \(R_t\)는 target window의 hybrid region bin, \(U_t\)는 target window의 unit block, \(U_{t-\ell}\)는 source window의 lagged unit block이다. Choice와 speed는 stimulus 이전 window를 source로, wheel direction은 movement 이전 window를 source로 둔다. Unit block penalty는 outer train fold 내부의 inner CV에서 고른다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_temporal_glm_coupling_gate.py
```

5-candidate panel 결과:

| target | lag \(U\mid X,R\) | lag \(U\mid X,R,U_0\) | mean \(XR\) BA | mean \(XR+U_0\) BA | mean \(XR+U_L\) BA | mean \(XR+U_0+U_L\) BA | mean \(\Delta U_L\mid X,R\) | mean \(\Delta U_L\mid X,R,U_0\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 1/5 | 0/5 | 0.845811 | 0.862736 | 0.839904 | 0.852269 | -0.005907 | -0.010466 |
| first movement speed | 5/5 | 2/5 | 0.750006 | 0.784605 | 0.774394 | 0.788782 | 0.024388 | 0.004177 |
| wheel action direction | 2/5 | 2/5 | 0.837202 | 0.865689 | 0.839519 | 0.860067 | 0.002318 | -0.005622 |

판정:

$$
\boxed{
\mathrm{temporal\ GLM\ coupling\ gate}
=
\mathrm{failed\ under\ strict}\ U_L\mid X,R,U_0.
}
$$

해석은 중요하다. \(U_L\mid X,R\)만 보면 speed에서 5/5로 신호가 있다. 그러나 현재 window의 unit detail \(U_0\)를 같이 넣으면 choice는 0/5, speed는 2/5, wheel은 2/5로 내려간다. 즉 lagged unit signal은 일부 target에서 보이지만, 현재 unit detail을 이기는 독립 지연 결합항으로 승격되지는 않는다.

따라서 현 단계에서 올릴 수 있는 항은 다음이다.

$$
\boxed{
U_t\mid X_t,R_t
\quad
\mathrm{survives\ as\ block\ regularized\ unit\ detail,}
\qquad
U_{t-\ell}\mid X_t,R_t,U_t
\quad
\mathrm{does\ not\ yet\ survive.}
}
$$

결론적으로 mouse 항은 static unit identity와 anatomical compression의 혼합 readout까지는 닫혔지만, directed/effective temporal coupling claim은 아직 올리지 않는다. 다음 비교는 더 큰 registered panel과 더 명시적인 directed coupling 또는 state-space model에서 다시 해야 한다.

### Mouse IBL/OpenAlyx low-rank unit transition gate

Temporal GLM 실패 뒤에는 식을 다시 세웠다. 앞의 gate는 \(U_{t-\ell}\)가 행동 \(y_t\)를 현재 unit \(U_t\) 뒤에서도 더 설명하는지를 물었다. 하지만 진짜 시간 결합은 보통 다음처럼 먼저 neural state transition으로 나타난다.

$$
U_{t-\ell}
\rightarrow
U_t
\rightarrow
y_t.
$$

따라서 다음 후보식은 행동식이 아니라 transition 식이다.

$$
\boxed{
H_t
=
A_sH_{t-\ell}
+B_sX_t
+C_sR_t
+\epsilon_t
}
$$

여기서 \(H_t\)는 unit matrix \(U_t\)의 train-fold PCA low-rank population state다. 개별 unit 전체를 그대로 맞히는 all-unit transition scout에서는 pooled \(R^2\)는 좋아졌지만 feature 평균 \(R^2\)가 나빠졌다. 즉 temporal signal은 모든 unit feature에 균일하게 퍼진 것이 아니라, 몇 개 population axis에 몰려 있었다. 그래서 \(U_t\)가 아니라 \(H_t\)를 target으로 다시 검사했다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_low_rank_unit_transition_gate.py --components 12 --max-units-per-probe 64
```

5-candidate panel 결과:

| transition | positive after \(X\) | positive after \(X,R_0\) | mean \(\Delta R^2\mid X\) | mean \(\Delta R^2\mid X,R_0\) |
|---|---:|---:|---:|---:|
| pre-stimulus to stimulus latent | 5/5 | 5/5 | 0.195029 | 0.106973 |
| pre-movement to movement latent | 5/5 | 5/5 | 0.218151 | 0.127870 |

판정:

$$
\boxed{
H_{t-\ell}\mid X_t,R_t
\quad
\mathrm{predicts}
\quad
H_t
\quad
\mathrm{in\ 5/5\ candidates.}
}
$$

따라서 앞의 음성 결과는 "temporal coupling이 없다"가 아니다. 더 정확한 판정은 다음이다.

$$
\boxed{
U_{t-\ell}\not\Rightarrow y_t
\ \mathrm{after}\ U_t,
\qquad
H_{t-\ell}\Rightarrow H_t
\ \mathrm{after}\ X_t,R_t.
}
$$

즉 mouse temporal 항은 행동 decoder의 직접 잔차가 아니라, low-rank population state transition으로 다시 세운다. 다음 행동식은 \(H_t\)와 innovation \(\epsilon_t\)가 행동 target을 나누어 설명하는지로 넘어가야 한다.

### Mouse IBL/OpenAlyx innovation-to-behavior gate

Low-rank transition이 통과했으므로 다음 질문은 행동이 어디에 붙는가다. Transition 식은 현재 population state를 두 부분으로 나눈다.

$$
H_t
=
\hat H_t
+\epsilon_t,
\qquad
\hat H_t
=
A_sH_{t-\ell}
+B_sX_t
+C_sR_t.
$$

\(\hat H_t\)는 과거 population state, task/history, region compression으로 예측되는 궤적이다. \(\epsilon_t\)는 그 궤적으로 설명되지 않는 현재 population innovation이다. 행동식 후보는 다음처럼 다시 쓴다.

$$
\boxed{
y_t
=
g_s
\left(
X_t,
R_t,
\hat H_t,
\epsilon_t
\right)
}
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_innovation_behavior_gate.py
```

5-candidate panel 결과:

| target | \(\hat H_t\) positive | \(\epsilon_t\) positive | \(\epsilon_t\mid \hat H_t\) positive | mean \(\Delta \hat H_t\) | mean \(\Delta \epsilon_t\) | mean \(\Delta \epsilon_t\mid \hat H_t\) |
|---|---:|---:|---:|---:|---:|---:|
| choice sign | 1/5 | 4/5 | 4/5 | -0.001622 | 0.014292 | 0.010658 |
| first movement speed | 5/5 | 4/5 | 4/5 | 0.018635 | 0.020376 | 0.020288 |
| wheel action direction | 0/5 | 3/5 | 4/5 | -0.006206 | 0.008116 | 0.013981 |

판정:

$$
\boxed{
\epsilon_t
\mid
X_t,R_t,\hat H_t
\quad
\mathrm{predicts\ behavior}
\quad
\mathrm{in\ 4/5\ candidates\ for\ all\ three\ targets.}
}
$$

따라서 mouse temporal 항은 다음처럼 정리된다.

$$
\boxed{
H_{t-\ell}
\rightarrow
\hat H_t,
\qquad
\epsilon_t
\rightarrow
y_t.
}
$$

예측 가능한 population trajectory \(\hat H_t\)는 speed에서 강하게 행동을 담지만, choice와 wheel에서는 약하다. 반면 innovation \(\epsilon_t\)는 choice, speed, wheel 모두에서 \(\hat H_t\) 뒤에도 4/5로 남는다. 즉 seeded panel에서는 행동이 단순히 과거 상태가 밀고 온 smooth trajectory가 아니라, 그 trajectory에서 벗어난 현재 population innovation에 더 반복적으로 붙는다.

12-session registered panel 실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_innovation_behavior_gate.py --candidates-json examples/physics/evolution/mouse_ibl_channel_fallback_registered_panel_ranker_results.json --output-stem mouse_ibl_innovation_behavior_12panel
```

12-panel 결과:

| target | \(\hat H_t\) positive | \(\epsilon_t\) positive | \(\epsilon_t\mid \hat H_t\) positive | mean \(\Delta \hat H_t\) | mean \(\Delta \epsilon_t\) | mean \(\Delta \epsilon_t\mid \hat H_t\) |
|---|---:|---:|---:|---:|---:|---:|
| choice sign | 7/12 | 8/12 | 3/12 | 0.002320 | 0.008033 | 0.004624 |
| first movement speed | 9/12 | 10/12 | 9/12 | 0.009861 | 0.022655 | 0.016741 |
| wheel action direction | 5/12 | 8/12 | 7/12 | 0.001959 | 0.026489 | 0.029900 |

판정:

$$
\boxed{
\epsilon_t
\mid
X_t,R_t,\hat H_t
\quad
\mathrm{survives\ for\ action,\ not\ for\ choice,\ in\ the\ 12\ panel.}
}
$$

즉 5-candidate 결론은 “innovation이 세 target 모두에서 남는다”였지만, 12-panel 결론은 더 좁다. Speed와 wheel 같은 action target에서는 innovation readout이 유지된다. Choice는 task/history와 trial policy에 더 붙어 있어서, \(\epsilon_t\)가 \(\hat H_t\) 뒤에 반복적으로 남지 않는다. 다음 식은 choice readout과 action readout을 분리해야 한다.

### Mouse IBL/OpenAlyx choice/action innovation split

12-panel 결과를 같은 세션 단위로 다시 읽어, action target의 평균 innovation increment가 choice innovation increment보다 큰지 확인했다.

실행:

```bash
python3 examples/physics/evolution/mouse_ibl_choice_action_split_from_innovation.py
```

식 후보:

$$
\boxed{
y_{\mathrm{choice},t}
=
g_c
\left(
X_t,
R_t,
\hat H_t
\right),
\qquad
y_{\mathrm{action},t}
=
g_a
\left(
X_t,
R_t,
\hat H_t,
\epsilon_t
\right)
}
$$

결과:

| metric | value |
|---|---:|
| mean choice \(\epsilon_t\mid\hat H_t\) | 0.004624 |
| mean action \(\epsilon_t\mid\hat H_t\) | 0.023320 |
| mean action - choice | 0.018696 |
| median action - choice | 0.015874 |
| 95% bootstrap CI low | 0.006277 |
| 95% bootstrap CI high | 0.031995 |
| sign-flip p | 0.017090 |
| split supported | 9/12 |

판정:

$$
\boxed{
\Delta_{\mathrm{action}}
-
\Delta_{\mathrm{choice}}
>
0
\quad
\mathrm{in\ 9/12\ sessions.}
}
$$

따라서 12-panel 실패의 해석은 “innovation 식이 틀렸다”가 아니라 “choice와 action을 같은 \(y_t\)로 묶은 것이 너무 거칠었다”에 가깝다. Choice는 task/history와 policy 항에 더 붙고, speed/wheel action은 transition에서 남은 current population innovation에 더 붙는다.

### Mouse IBL/OpenAlyx directed latent-axis split gate

다음 반례는 \(\epsilon_t\) 전체 벡터가 너무 거칠다는 것이다. Full-vector readout에서는 choice가 약했지만, 개별 innovation axis 중 하나만 행동 decoder에 추가하면 choice가 다시 살아날 수 있다. 따라서 각 target마다 \(\epsilon_{t,1},\ldots,\epsilon_{t,12}\)를 하나씩 넣고, \(X_t,R_t,\hat H_t\) 뒤에서 best single axis increment를 측정했다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_directed_latent_axis_split_gate.py
```

식 후보:

$$
\boxed{
y_{j,t}
=
g_j
\left(
X_t,
R_t,
\hat H_t,
\epsilon_{t,k_j}
\right)
}
$$

12-panel 결과:

| target | best axis supported | mean best dBA | median best dBA | mean positive axes |
|---|---:|---:|---:|---:|
| choice sign | 11/12 | 0.009646 | 0.005360 | 3.917 |
| first movement speed | 11/12 | 0.020485 | 0.018147 | 4.750 |
| wheel action direction | 12/12 | 0.021819 | 0.018257 | 4.417 |

Choice/action best-axis split:

| metric | value |
|---|---:|
| mean action - choice best-axis dBA | 0.011506 |
| median action - choice best-axis dBA | 0.011937 |
| split supported | 8/12 |

판정:

$$
\boxed{
\exists k_j:
\epsilon_{t,k_j}
\mid
X_t,R_t,\hat H_t
\quad
\mathrm{predicts\ each\ target.}
}
$$

이 결과는 앞선 12-panel full-vector 판정을 수정한다. Choice에 innovation이 아예 없는 것이 아니다. Full-vector \(\epsilon_t\)를 한꺼번에 넣으면 choice는 불안정하지만, target별 single latent axis를 고르면 choice도 반복된다. 따라서 다음 병목은 innovation 존재 여부가 아니라, best axis가 세션마다 안정적인 subspace로 묶이는지다.

### Mouse IBL/OpenAlyx latent-axis stability

Directed-axis gate의 위험은 사후 선택이다. 12개 축 중 가장 좋은 축을 target마다 고르면 당연히 좋아질 수 있다. 그래서 best axis 번호가 세션 사이에서 반복되는지, 또는 적어도 top3 축에 충분히 몰리는지 확인했다.

실행:

```bash
python3 examples/physics/evolution/mouse_ibl_latent_axis_stability_from_directed.py
```

결과:

| target | top axis | top1 | top3 | entropy | top1 null p | stable identity | concentrated subspace |
|---|---:|---:|---:|---:|---:|---|---|
| choice sign | 6 | 3/12 | 7/12 | 0.750000 | 0.687350 | `False` | `False` |
| first movement speed | 4 | 2/12 | 6/12 | 0.860529 | 0.999950 | `False` | `False` |
| wheel action direction | 4 | 4/12 | 8/12 | 0.703510 | 0.167700 | `False` | `True` |

Within-session sharing도 약하다. Choice-speed same axis는 \(2/12\), choice-wheel은 \(4/12\), speed-wheel은 \(0/12\), 세 target이 모두 같은 axis인 경우는 \(0/12\)다.

판정:

$$
\boxed{
\exists k_j\ \mathrm{informative}
\quad
\not\Rightarrow
\quad
k_j\ \mathrm{stable}.
}
$$

따라서 directed single-axis는 행동 정보를 가진다. 하지만 stable named axis나 shared subspace로는 아직 승격하지 못한다. 다음 gate는 best axis를 test 결과로 고르지 않고, train fold 안에서 미리 고른 low-dimensional subspace가 outer test behavior를 설명하는지 확인해야 한다.

### Mouse IBL/OpenAlyx nested innovation-subspace gate

Latent-axis stability gate가 막은 것은 사후 선택 문제다. 12개 innovation axis 중 test fold에서 제일 잘 맞는 축을 고르면 choice, speed, wheel 모두 좋아진다. 하지만 논문식 주장으로 올리려면 test 결과를 보기 전에 axis 또는 subspace가 정해져야 한다.

그래서 nested gate는 outer fold의 test set을 완전히 숨긴 상태에서, outer train 안의 inner CV로 target별 top-3 innovation axis를 고른다. 그 뒤에만 outer test에서 \(X_t,R_t,\hat H_t\) 뒤에 그 subspace가 남는지 본다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_nested_innovation_subspace_gate.py
```

식 후보:

$$
\boxed{
S_j^\star
=
\mathrm{TopK}_{k=3}
\left[
\Delta_{\mathrm{inner}}
\left(
\epsilon_{t,k}
\mid
X_t,R_t,\hat H_t
\right)
\right],
\qquad
y_{j,t}
=
g_j
\left(
X_t,
R_t,
\hat H_t,
\epsilon_{t,S_j^\star}
\right)
}
$$

12-panel 결과:

| target | subspace supported | mean dBA | median dBA |
|---|---:|---:|---:|
| choice sign | 5/12 | 0.006472 | -0.000203 |
| first movement speed | 9/12 | 0.013697 | 0.011018 |
| wheel action direction | 8/12 | 0.020350 | 0.017363 |

Choice/action split:

| metric | value |
|---|---:|
| mean action - choice subspace dBA | 0.010552 |
| median action - choice subspace dBA | 0.007308 |
| split supported | 8/12 |

판정:

$$
\boxed{
\epsilon_{t,S_{\mathrm{train}}}
\mid
X_t,R_t,\hat H_t
\quad
\mathrm{survives\ for\ action,\ not\ for\ choice.}
}
$$

따라서 사후 best-axis 선택이라는 반례는 action에 대해서는 한 단계 약해졌다. 안정된 이름 붙은 축은 아직 없지만, train fold 안에서 미리 고른 작은 innovation subspace가 speed와 wheel의 outer test 행동을 반복적으로 설명한다. 반대로 choice는 nested subspace에서도 \(5/12\), median \(-0.000203\)에 그쳐 승격하지 않는다.

현재 mouse readout은 그래서 다음처럼 갈라진다.

$$
\boxed{
y_{\mathrm{choice},t}
=
g_c
\left(
X_t,
R_t,
\hat H_t,
P_t^{\mathrm{policy/history?}}
\right),
\qquad
y_{\mathrm{action},t}
=
g_a
\left(
X_t,
R_t,
\hat H_t,
\epsilon_{t,S_{\mathrm{train}}}
\right)
}
$$

여기서 \(P_t^{\mathrm{policy/history?}}\)는 아직 승격된 항이 아니라 다음 병목이다. Choice는 innovation이 전혀 없어서 실패한 것이 아니라, 공개 IBL task 구조에서는 stimulus, prior block, previous choice/reward, policy history와 더 강하게 겹쳐 있을 가능성이 크다. 다음 식 후보는 action subspace를 더 키우는 것이 아니라 choice-specific policy/history latent를 분리하는 쪽으로 가야 한다.

### Mouse IBL/OpenAlyx choice policy/history gate

Nested subspace 뒤의 다음 질문은 choice가 왜 약한지다. Action에서는 train-selected innovation subspace가 남았지만, choice에서는 \(5/12\)였다. 따라서 choice에 같은 innovation 항을 억지로 넣기보다, task/history 또는 policy-like covariate가 choice를 얼마나 먼저 설명하는지 확인했다.

비교한 model은 다음 순서다.

$$
M_X:
y_{\mathrm{choice}}\sim X_{\mathrm{policy/history}},
$$

$$
M_{XR}:
y_{\mathrm{choice}}\sim [X_{\mathrm{policy/history}},R],
$$

$$
M_{XRH}:
y_{\mathrm{choice}}\sim [X_{\mathrm{policy/history}},R,\hat H],
$$

$$
M_{XRH\epsilon}:
y_{\mathrm{choice}}\sim
[X_{\mathrm{policy/history}},R,\hat H,\epsilon_{S_{\mathrm{train}}}].
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_choice_policy_history_gate.py
```

12-panel model summary:

| model | mean BA | median BA |
|---|---:|---:|
| policy/history | 0.836522 | 0.843374 |
| policy/history + region | 0.839474 | 0.844699 |
| policy/history + region + predicted latent | 0.841794 | 0.849048 |
| policy/history + region + predicted latent + nested eps | 0.848266 | 0.843158 |

증분:

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| region after policy/history | 4/12 | 0.002952 | -0.000510 | `False` |
| predicted latent after policy/history + region | 7/12 | 0.002320 | 0.002501 | `True` |
| nested eps after policy/history + region + predicted latent | 5/12 | 0.006472 | -0.000203 | `False` |
| all neural after policy/history | 5/12 | 0.011744 | -0.000156 | `False` |

판정:

$$
\boxed{
\epsilon_{S_{\mathrm{train}}}
\ \mathrm{does\ not\ become\ a\ replicated\ choice\ term.}
}
$$

그러나 이것을 "choice는 policy만으로 완전히 닫혔다"라고 읽으면 과하다. Policy/history baseline 자체가 mean BA \(0.836522\)로 매우 강하고, nested eps는 반복 기준에 못 닿지만 mean dBA는 양수다. 또한 \(\hat H\)는 \(7/12\), mean \(0.002320\)으로 매우 작은 양성이다. 따라서 다음 식 후보는 policy/history를 더 세밀한 latent로 만들고, 그 뒤에 남는 neural residual을 다시 보는 형태가 되어야 한다.

현재 choice 후보식은 다음이다.

$$
\boxed{
y_{\mathrm{choice},t}
=
g_c
\left(
P_t^{\mathrm{policy/history}},
R_t,
\hat H_t
\right)
\eta_t,
\qquad
\epsilon_{t,S_{\mathrm{train}}}
\ \mathrm{not\ promoted\ for\ choice.}
}
$$

### Mouse IBL/OpenAlyx richer choice policy/history gate

다음에는 hand-built richer policy/history block을 만들었다. 기존 \(X_{\mathrm{policy/history}}\)는 current stimulus/probability와 previous trial history의 선형 묶음이었다. Richer block은 여기에 multi-lag choice/outcome, win-stay/lose-switch, stimulus-prior interaction, previous choice interaction, decayed choice/reward/feedback trace를 추가했다.

질문은 두 개다.

1. 이 richer policy block이 기존 linear policy/history를 반복적으로 이기는가.
2. richer policy 뒤에도 neural residual이 남는가.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_richer_choice_policy_gate.py
```

12-panel model summary:

| model | mean BA | median BA |
|---|---:|---:|
| linear policy/history | 0.836522 | 0.843374 |
| richer policy/history | 0.837102 | 0.850950 |
| richer policy/history + region | 0.839692 | 0.846948 |
| richer policy/history + region + predicted latent | 0.840482 | 0.851776 |
| richer policy/history + region + predicted latent + nested eps | 0.846516 | 0.853543 |

증분:

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| richer policy after linear policy | 5/12 | 0.000580 | -0.000956 | `False` |
| region after richer policy | 5/12 | 0.002590 | 0.001219 | `False` |
| predicted latent after richer policy + region | 5/12 | 0.000791 | 0.000021 | `False` |
| nested eps after richer policy + region + predicted latent | 5/12 | 0.006033 | 0.001085 | `False` |
| all neural after richer policy | 7/12 | 0.009414 | 0.002078 | `True` |

판정:

$$
\boxed{
P_{\mathrm{rich}}
\not\gg
X_{\mathrm{linear}},
\qquad
[R,\hat H,\epsilon_{S_{\mathrm{train}}}]
\mid
P_{\mathrm{rich}}
\ \mathrm{survives.}
}
$$

즉 손으로 만든 richer policy feature는 기존 linear policy/history를 반복적으로 이기지 못했다. 반대로 \(R\), \(\hat H\), \(\epsilon_{S_{\mathrm{train}}}\) 각각은 단독 증분으로는 약하지만, 셋을 모두 합친 all-neural residual은 \(7/12\), mean \(0.009414\)로 살아났다.

따라서 choice 병목은 "policy feature를 더 많이 넣으면 닫힌다"도 아니고 "action-style innovation subspace 하나를 넣으면 닫힌다"도 아니다. 다음 식 후보는 block synergy를 직접 검사해야 한다.

$$
\boxed{
y_{\mathrm{choice},t}
=
g_c
\left(
X_t,
\Phi_{\mathrm{policy}}(h_{<t}),
R_t,
\hat H_t,
\epsilon_{t,S_{\mathrm{train}}}
\right),
\qquad
\Delta_{\mathrm{all\ neural}\mid P_{\mathrm{rich}}}>0.
}
$$

다음 병목은 \(R\), \(\hat H\), \(\epsilon\) 중 어느 하나의 단독 승격이 아니라, \(P_{\mathrm{rich}}\) 뒤에서 세 neural block이 함께 남는 이유를 ablation/Shapley식으로 분해하는 것이다.

### Mouse IBL/OpenAlyx choice neural-block synergy gate

Synergy gate는 같은 outer fold 안에서 \(P_{\mathrm{rich}}\) 뒤에 붙는 neural block 조합을 모두 비교했다.

$$
P,\quad
P+R,\quad
P+\hat H,\quad
P+\epsilon,\quad
P+R+\hat H,\quad
P+R+\epsilon,\quad
P+\hat H+\epsilon,\quad
P+R+\hat H+\epsilon.
$$

여기서 \(\epsilon\)은 outer train 내부에서 고른 top-3 innovation subspace다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_choice_neural_synergy_gate.py
```

12-panel model summary:

| model | mean BA | median BA |
|---|---:|---:|
| \(P\) | 0.837102 | 0.850950 |
| \(P+\epsilon\) | 0.842266 | 0.852734 |
| \(P+\hat H\) | 0.840806 | 0.854294 |
| \(P+R\) | 0.839692 | 0.846948 |
| \(P+\hat H+\epsilon\) | 0.848622 | 0.859746 |
| \(P+R+\epsilon\) | 0.844350 | 0.850032 |
| \(P+R+\hat H\) | 0.840482 | 0.851776 |
| \(P+R+\hat H+\epsilon\) | 0.846516 | 0.853543 |

증분:

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| full after policy | 7/12 | 0.009414 | 0.002078 | `True` |
| full minus best single | 3/12 | -0.000933 | -0.000822 | `False` |
| full minus best pair | 1/12 | -0.003688 | -0.004323 | `False` |
| best single after policy | 8/12 | 0.010347 | 0.002667 | `True` |
| best pair after policy | 5/12 | 0.013102 | 0.001518 | `False` |

각 fixed model을 \(P\)와 직접 비교하면:

| model | positive | mean dBA | median dBA |
|---|---:|---:|---:|
| \(P+R\) | 5/12 | 0.002590 | 0.001219 |
| \(P+\hat H\) | 5/12 | 0.003704 | 0.000532 |
| \(P+\epsilon\) | 7/12 | 0.005164 | 0.002361 |
| \(P+R+\hat H\) | 3/12 | 0.003381 | 0.000869 |
| \(P+R+\epsilon\) | 3/12 | 0.007248 | -0.000623 |
| \(P+\hat H+\epsilon\) | 5/12 | 0.011520 | -0.000603 |
| \(P+R+\hat H+\epsilon\) | 7/12 | 0.009414 | 0.002078 |

판정:

$$
\boxed{
\mathrm{three\text{-}way\ synergy}
\quad
\mathrm{does\ not\ pass.}
}
$$

Full neural block은 \(P\) 뒤에서 살아나지만, best single이나 best pair를 반복적으로 이기지 못한다. 따라서 choice residual을 "세 neural block의 삼중 결합"으로 승격하면 안 된다.

대신 더 좁은 신호가 보인다.

$$
\boxed{
P_{\mathrm{rich}}+\epsilon_{S_{\mathrm{train}}}
\quad
\mathrm{survives\ for\ choice}
}
$$

\(P+\epsilon\)는 \(7/12\), mean \(0.005164\), median \(0.002361\)로 통과한다. 단, \(\epsilon\) subspace는 train fold 안에서 \(P,R,\hat H\) 조건으로 고른 축이므로 완전히 독립적인 choice-only axis는 아니다. 현재 해석은 "choice에는 innovation이 없다"가 아니라, "policy 뒤에 남는 choice innovation은 \(R,\hat H\)를 함께 partial out하면 불안정해지고, \(P+\epsilon\) 형태에서는 반복된다"이다.

따라서 choice 식은 다시 좁힌다.

$$
\boxed{
y_{\mathrm{choice},t}
=
g_c
\left(
P_t^{\mathrm{rich}},
\epsilon_{t,S_{\mathrm{train}}}
\right)
\eta_t,
\qquad
R_t,\hat H_t
\ \mathrm{are\ auxiliary/unstable\ for\ choice.}
}
$$
