## Mouse IBL/OpenAlyx task-baseline comparison gate

Channel-region rescue 다음에는 다른 반례를 검사했다. Region decoder가 실제 neural state를 읽는 것이 아니라, 단순 trial clock, stimulus table, previous-trial history만 복원하는 경우다. 이 반례를 다음처럼 분리했다.

$$
X_i^{\mathrm{timing}}
=
\left[
\mathrm{trial\ index},
\mathrm{session\ time},
\mathrm{stim\ delay},
\mathrm{go\ cue\ delay}
\right]_i
$$

$$
X_i^{\mathrm{task}}
=
\left[
X_i^{\mathrm{timing}},
\mathrm{signed\ contrast}_i,
|\mathrm{contrast}_i|,
\mathrm{stimulus\ side}_i,
\mathrm{probabilityLeft}_i,
H_{i-1}
\right].
$$

여기서 \(H_{i-1}\)는 previous choice, previous feedback, previous reward, previous signed contrast, previous reaction latency, previous response latency다. Current choice, current first movement, current response, current feedback은 baseline feature에서 제외했다. 이유는 이 값들이 이번 target 자체이거나 target 직후 outcome이라서 leakage가 되기 때문이다.

비교한 모델은 다음이다.

| model | feature | leakage rule |
|---|---|---|
| `timing_only` | trial clock and pre-response timing | current behavior/outcome excluded |
| `task_history` | timing, current stimulus, previous-trial history | current choice/movement/outcome excluded |
| `hybrid_region_by_probe` | channel-rescued probe-region spike rates | neural window feature |
| `task_history_plus_hybrid_region` | task history plus hybrid region | neural increment over task table |
| `global_rate` | one scalar total hybrid firing rate | flat firing-rate baseline |

두 차이를 명시했다.

$$
\Delta_{\mathrm{timing}}
=
\mathrm{BA}(R^{\mathrm{hybrid}})
-
\mathrm{BA}(X^{\mathrm{timing}}),
$$

$$
\Delta_{\mathrm{task}}
=
\mathrm{BA}
\left(
\left[X^{\mathrm{task}},R^{\mathrm{hybrid}}\right]
\right)
-
\mathrm{BA}(X^{\mathrm{task}}).
$$

\(\Delta_{\mathrm{timing}}>0\)이면 단순 session drift/trial clock 반례가 약해진다. \(\Delta_{\mathrm{task}}>0\)이면 current stimulus와 previous-trial history만으로 설명되지 않는 neural increment가 남는다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_task_baseline_comparison_gate.py
```

Target replication:

| target | hybrid beats timing | task+hybrid beats task | mean timing BA | mean task BA | mean hybrid BA | mean task+hybrid BA |
|---|---:|---:|---:|---:|---:|---:|
| choice sign | 5/5 | 2/5 | 0.544584 | 0.844886 | 0.675560 | 0.845811 |
| first movement speed | 5/5 | 5/5 | 0.616631 | 0.689940 | 0.742104 | 0.750006 |
| wheel action direction | 5/5 | 4/5 | 0.530258 | 0.822181 | 0.716240 | 0.837202 |

Mean deltas:

| target | mean \(\Delta_{\mathrm{timing}}\) | mean \(\Delta_{\mathrm{task}}\) |
|---|---:|---:|
| choice sign | 0.130975 | 0.000924 |
| first movement speed | 0.125473 | 0.060066 |
| wheel action direction | 0.185982 | 0.015020 |

Candidate deltas:

| candidate | choice \(\Delta_t\) | choice \(\Delta_x\) | speed \(\Delta_t\) | speed \(\Delta_x\) | wheel \(\Delta_t\) | wheel \(\Delta_x\) |
|---|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 0.134807 | 0.000000 | 0.217524 | 0.113459 | 0.233831 | -0.000211 |
| `nyu30_motor_striatal_multi_probe` | 0.300062 | 0.000526 | 0.075917 | 0.044396 | 0.359294 | 0.029426 |
| `dy014_striatal_septal_probe` | 0.070300 | -0.017691 | 0.049320 | 0.027211 | 0.063511 | 0.013514 |
| `dy011_motor_cortex_probe` | 0.094482 | -0.002186 | 0.066264 | 0.010610 | 0.182963 | 0.007380 |
| `dy008_cp_somatosensory_thalamic_probe` | 0.055226 | 0.023973 | 0.218338 | 0.104654 | 0.090310 | 0.024993 |

판정은 일부 통과다.

$$
\boxed{
\mathrm{timing\text{-}only\ counterexample}
\quad
\mathrm{rejected}
\quad
(5/5,\ 5/5,\ 5/5).
}
$$

그러나 task-history increment는 완전 통과가 아니다.

$$
\boxed{
\Delta_{\mathrm{task}}
>
0
\quad
\mathrm{for}\quad
\mathrm{speed}\ 5/5,
\mathrm{wheel}\ 4/5,
\mathrm{choice}\ 2/5.
}
$$

이 음성 결과는 중요하다. IBL visual decision task에서 choice는 current stimulus와 previous-trial history만으로 이미 매우 강하게 예측된다. 실제로 task-history mean BA는 choice에서 0.844886이고, hybrid-only mean BA 0.675560보다 높다. 따라서 choice에 대해 "region code가 task table보다 우월하다"는 말은 현재 자료에서 성립하지 않는다.

반대로 first-movement speed와 wheel action direction에서는 hybrid를 task-history에 더했을 때 평균 BA가 각각 0.060066, 0.015020 올라간다. 그러므로 mouse 항은 다음처럼 더 세밀하게 분리해야 한다.

$$
\Phi_{\mathrm{mammal}}^{(s)}(t)
=
B_{\mathrm{stim/history}}^{(s)}X_t^{(s,\mathrm{task})}
+
B_{\mathrm{probe,hybrid\ region}}^{(s)}
R_t^{(s,\mathrm{hybrid})}
+
B_{\mathrm{action\ increment}}^{(s)}
\left[
X_t^{(s,\mathrm{task})},
R_t^{(s,\mathrm{hybrid})}
\right]
+
\epsilon_s.
$$

여기서 \(B_{\mathrm{stim/history}}\)는 특히 choice를 강하게 설명하는 task-table term이고, \(B_{\mathrm{action\ increment}}\)는 speed와 wheel action direction에서 남는 neural increment다. 따라서 mouse 단계의 현재 결론은 다음이다.

$$
\boxed{
\mathrm{region/probe\ code}
\not\equiv
\mathrm{trial\ timing},
\qquad
\mathrm{but}
\qquad
\mathrm{choice}
\ \mathrm{is\ strongly\ task\text{-}covariate\ dominated.}
}
$$

이제 남은 강한 반례는 flat-neuron baseline과 effective-connectivity baseline이다. Timing-only 반례는 닫혔고, task-history 반례는 action 쪽에서는 약해졌지만 choice 쪽에서는 아직 남아 있다.

### Mouse IBL/OpenAlyx flat-unit versus hybrid-region comparison gate

Task-baseline gate 다음에는 representational counterexample을 검사했다. 같은 neural window 안에서 anatomical region/probe bin을 쓰지 않고, 개별 cluster identity를 flat feature로 쓰면 더 잘 되는가다. 이 반례가 중요하다. Region decoder가 global-rate와 timing을 이기더라도, flat unit이 훨씬 잘 되면 region 항은 "최대 예측기"가 아니라 "해부학적 압축 표현"으로 내려가야 한다.

Probe \(p\)에서 spike count 상위 \(K=96\) cluster를 골라 unit map을 만든다.

$$
U_p(c)
=
\begin{cases}
c,&c\in \mathrm{TopK}_p,\\
\mathrm{other\ units},&\mathrm{otherwise}.
\end{cases}
$$

Flat-unit feature는 다음이다.

$$
u_{ipu}
=
\frac{1}{b_i-a_i}
\sum_k
\mathbf 1[t_{pk}\in[a_i,b_i]]
\mathbf 1[U_p(c_{pk})=u].
$$

Hybrid region feature \(r_{ipg}\)와 직접 비교하기 위해 두 차이를 정의했다.

$$
\Delta_{\mathrm{unit-region}}
=
\mathrm{BA}(U^{\mathrm{top}})
-
\mathrm{BA}(R^{\mathrm{hybrid}}),
$$

$$
\Delta_{\mathrm{task+unit}}
=
\mathrm{BA}
\left(
\left[X^{\mathrm{task}},U^{\mathrm{top}}\right]
\right)
-
\mathrm{BA}
\left(
\left[X^{\mathrm{task}},R^{\mathrm{hybrid}}\right]
\right).
$$

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_flat_unit_region_comparison_gate.py
```

Target replication:

| target | region beats unit | unit beats region | mean region BA | mean unit BA | mean unit-region delta |
|---|---:|---:|---:|---:|---:|
| choice sign | 0/5 | 5/5 | 0.675560 | 0.744883 | 0.069323 |
| first movement speed | 3/5 | 2/5 | 0.742104 | 0.756299 | 0.014195 |
| wheel action direction | 0/5 | 5/5 | 0.716240 | 0.777766 | 0.061526 |

Task-history augmented comparison:

| target | task+region beats task+unit | task+unit beats task+region | mean task+region BA | mean task+unit BA |
|---|---:|---:|---:|---:|
| choice sign | 2/5 | 3/5 | 0.845811 | 0.849174 |
| first movement speed | 3/5 | 2/5 | 0.750006 | 0.754919 |
| wheel action direction | 3/5 | 2/5 | 0.837202 | 0.835943 |

Candidate deltas:

| candidate | choice unit-region | speed unit-region | wheel unit-region |
|---|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 0.106680 | -0.004797 | 0.059337 |
| `nyu30_motor_striatal_multi_probe` | 0.062357 | 0.031224 | 0.039216 |
| `dy014_striatal_septal_probe` | 0.046439 | 0.090136 | 0.078062 |
| `dy011_motor_cortex_probe` | 0.042530 | -0.023655 | 0.094797 |
| `dy008_cp_somatosensory_thalamic_probe` | 0.088610 | -0.021932 | 0.036220 |

판정:

$$
\boxed{
\mathrm{top\ unit\ readout}
>
\mathrm{hybrid\ region\ readout}
\quad
\mathrm{for}\quad
\mathrm{choice}\ 5/5,
\mathrm{wheel}\ 5/5.
}
$$

Speed는 혼합이다.

$$
\boxed{
\mathrm{speed:}\quad
\mathrm{region}\ 3/5
\quad\mathrm{vs}\quad
\mathrm{unit}\ 2/5,
\qquad
\overline{\Delta}_{\mathrm{unit-region}}=0.014195.
}
$$

따라서 flat-unit gate는 통과가 아니라 음성 결과다.

$$
\boxed{
\mathrm{region/probe\ code}
\neq
\mathrm{maximal\ neural\ decoder}.
}
$$

그러나 이것이 region 항의 폐기를 뜻하지는 않는다. Region/probe code는 top-unit보다 낮은 차원의 anatomical compression이다. 현재 mouse 단계에서 승격할 수 있는 항은 "unit-level 정보를 모두 이기는 region decoder"가 아니라, 다음과 같은 계층 분해다.

$$
N_t^{(s)}
=
C_{\mathrm{unit}\rightarrow\mathrm{region}}^{(s)}
U_t^{(s)}
+
\eta_t^{(s)},
\qquad
R_t^{(s,\mathrm{hybrid})}
=
C_{\mathrm{unit}\rightarrow\mathrm{region}}^{(s)}
U_t^{(s)}.
$$

여기서 \(U_t^{(s)}\)는 top-unit/cluster-level activity, \(R_t^{(s,\mathrm{hybrid})}\)는 channel-rescued anatomical compression, \(\eta_t^{(s)}\)는 region binning으로 잃는 unit-specific residual이다. Flat-unit gate의 의의는 \(\eta_t^{(s)}\)가 choice와 wheel에서 무시할 수 없다는 점을 밝혔다는 것이다.

따라서 mouse 항은 다음처럼 내려 적는다.

$$
\Phi_{\mathrm{mammal}}^{(s)}(t)
=
B_{\mathrm{stim/history}}^{(s)}X_t^{(s,\mathrm{task})}
+
B_{\mathrm{unit}}^{(s)}U_t^{(s)}
+
B_{\mathrm{region\ compression}}^{(s)}R_t^{(s,\mathrm{hybrid})}
+
B_{\mathrm{action\ increment}}^{(s)}
\left[
X_t^{(s,\mathrm{task})},
R_t^{(s,\mathrm{hybrid})}
\right]
+
\epsilon_s.
$$

즉 mouse에서 새로 얻은 결론은 region-only closure가 아니라 unit-detail과 anatomical compression의 분리다. 다음 강한 단계는 effective-connectivity다. 단순 unit identity \(U_t\)가 아니라 unit/region 사이의 weighted interaction \(K_{ij}\)가 action을 더 잘 설명하는지 봐야, C. elegans에서 시작한 weighted routing 명제를 포유류까지 연결할 수 있다. 이 질문을 아래 proxy gate로 바로 검사했다.
