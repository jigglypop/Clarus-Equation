## 단계 4: Mouse Neuropixels/IBL

Zebrafish 다음 단계는 mouse Neuropixels/IBL이다. 이유는 다음이다.

1. C. elegans에서 weighted routing을 닫았다.
2. Drosophila adult FlyWire에서 celltype/action/memory co-differentiation을 닫았다.
3. Zebrafish에서 activity state, perturbation behavior, discrete activity-behavior association을 닫았다.
4. 다음에는 포유류의 cortex, thalamus, striatum, midbrain, hippocampus가 decision/action/internal state를 어떻게 나누는지 봐야 한다.

후보식:

$$
P_{n+1}
=
\Pi
\left[
\rho P_n
+(1-\rho)P_*
+\gamma\Delta_G P_n
+U_{\mathrm{task},n}
+H(Q_n-Q^*)
+M_{\mathrm{internal},n}
+W_{\mathrm{workspace},n}
\right]
$$

Mouse/IBL 첫 gate:

| gate | 목적 | 성공 조건 |
|---|---|---|
| trial table audit | choice, feedback, stimulus, wheel timing이 한 세션에 있는지 | `_ibl_trials.table.pqt`, wheel datasets present |
| spike-region audit | spikes, clusters, channels, atlas region이 한 세션에 있는지 | spike times and region labels present |
| region activity closure | region-binned firing이 trial state를 예측하는지 | heldout decoding beats baseline |
| decision/action bridge | visual, thalamus, striatum, motor 계열이 choice/action timing을 분리하는지 | region-class model beats flat model |
| graph/flat comparison | 영역 묶음 모델이 flat neuron model보다 나은지 | \(\mathcal L_{\mathrm{region}}/\mathcal L_{\mathrm{flat}}<1\) |

### Mouse IBL/OpenAlyx metadata bridge audit

첫 실행은 전량 다운로드가 아니라 metadata bridge audit로 제한했다. 목표는 region-level decision/action gate에 필요한 최소 파일 묶음이 한 공개 세션 안에 함께 존재하는지 확인하는 것이다. Spike array 자체는 내려받지 않았다.

필요 조건:

| tier | required dataset types | 의미 |
|---|---|---|
| core | `trials.table`, `wheel.timestamps`, `spikes.times`, `spikes.clusters` | trial, wheel, spike time/cluster bridge |
| strict region | `clusters.brainLocationAcronyms_ccf_2017` | cluster-level CCF acronym labels |
| channel fallback | `clusters.channels`, `channels.brainLocationIds_ccf_2017` | channel CCF ids plus cluster-channel map |

실행:

```bash
uv run --no-project --with ONE-api python examples/physics/evolution/mouse_ibl_neuropixels_audit.py --max-candidates 5
```

결과:

| query | sessions found |
|---|---:|
| core + strict cluster region | 29 |
| core + channel-region fallback | 483 |

first strict candidates:

| eid | session | probes | strict ready | channel fallback ready |
|---|---|---|---|---|
| `d2832a38-27f6-452d-91d6-af72d794136c` | `wittenlab/Subjects/ibl_witten_29/2021-06-08/001` | `alf/probe00/pykilosort` | True | True |
| `dc21e80d-97d7-44ca-a729-a8e3f9b14305` | `wittenlab/Subjects/ibl_witten_26/2021-01-31/001` | `alf/probe00/pykilosort` | True | True |
| `8c2f7f4d-7346-42a4-a715-4d37a5208535` | `wittenlab/Subjects/ibl_witten_26/2021-01-29/001` | `alf/probe01/pykilosort` | True | True |
| `952870e5-f2a7-4518-9e6d-71585460f6fe` | `wittenlab/Subjects/ibl_witten_27/2021-01-19/001` | `alf/probe00/pykilosort`, `alf/probe01/pykilosort` | True | True |
| `c728f6fd-58e2-448d-aefb-a72c637b604c` | `wittenlab/Subjects/ibl_witten_27/2021-01-16/003` | `alf/probe00/pykilosort`, `alf/probe01/pykilosort` | True | True |

판정:

$$
\boxed{
\mathrm{Mouse/IBL\ metadata\ bridge}
\quad\text{is ready for}\quad
\mathrm{region\text{-}binned\ decision/action\ decoding.}
}
$$

이 결과는 mouse 단계의 포유류 항을 닫은 것이 아니다. 다만 다음 gate를 열 수 있는 최소 관측 bridge가 공개 OpenAlyx metadata 안에 있음을 확인한다. 이 metadata audit 다음에는 아래의 첫 strict-session region decision/action gate로 이어졌다.

### Mouse IBL/OpenAlyx first strict-session decision/action gate

첫 strict candidate인 `d2832a38-27f6-452d-91d6-af72d794136c`를 실제로 내려받아 trial, wheel, spike, cluster, CCF acronym을 연결했다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_region_decision_action_gate.py
```

loaded arrays:

| item | value |
|---|---:|
| session | `wittenlab/Subjects/ibl_witten_29/2021-06-08/001` |
| probe collection | `alf/probe00/pykilosort` |
| trials | 663 |
| wheel samples | 1126509 |
| spikes | 34822347 |
| observed cluster slots | 700 |
| strict acronym rows | 648 |
| spikes with strict acronym row | 33789015 |
| spikes assigned `unknown` | 1033332 |
| unknown spike fraction | 0.029674 |

여기서 중요한 점은 cluster acronym file이 648 row인데 observed spike cluster id는 0부터 699까지라는 것이다. 따라서 이 세션에서 cluster-region bridge는 완전히 깨진 것이 아니라 약 \(97.03\%\) spike에 대해 strict acronym label을 주고, 나머지 \(2.97\%\) spike는 `unknown` bin으로 보존하는 형태다. 즉 삭제가 아니라 censored region bin이다.

trial \(i\), spike \(k\), spike time \(t_k\), cluster id \(c_k\), region/group map \(G(c_k)\)를 두면 region-binned feature는 다음처럼 정의했다.

$$
x_{ig}
=\frac{1}{b_i-a_i}
\sum_k
\mathbf 1[t_k\in[a_i,b_i]]
\mathbf 1[G(c_k)=g].
$$

이 식은 단순 spike count가 아니라 duration-normalized rate다. 만약 \(b_i-a_i\)가 trial마다 달라지면 단순 count는 response duration을 그대로 leak한다. 그래서 이 gate에서는 다음 두 고정 window를 사용했다.

| window | interval | target |
|---|---|---|
| stimulus window | `stimOn_times + 20 ms` to `stimOn_times + 320 ms` | choice sign, first-movement speed |
| movement window | `firstMovement_times - 100 ms` to `firstMovement_times + 200 ms` | wheel action direction |

비교한 feature model은 세 개다.

| model | definition | 의미 |
|---|---|---|
| `region_family` | visual cortex, visual thalamus, somatosensory thalamus, hippocampus, cingulate, other, unknown | 넓은 anatomical family |
| `acronym_region` | high-spike CCF acronyms plus unknown | 더 세분한 region acronym model |
| `global_rate` | \(\sum_g x_{ig}\) one scalar | 영역 구조를 지운 flat firing-rate baseline |

decoder는 z-scored ridge linear classifier다.

$$
\hat w
=
\arg\min_w
\|y-Xw\|_2^2
+\lambda\|w\|_2^2.
$$

각 fold에서 train set 평균과 표준편차로 \(X\)를 표준화하고, held-out fold의 score sign으로 binary label을 예측했다. 판정 지표는 class imbalance에 덜 민감한 balanced accuracy다.

$$
\mathrm{BA}
=
\frac12
\left(
\frac{TP}{P}
+
\frac{TN}{N}
\right).
$$

null은 label permutation이다. 즉 관측 feature \(X\)는 그대로 두고 \(y\)만 섞어 같은 cross-validation을 반복한다.

$$
p
=
\frac{
1+\#\{\mathrm{BA}_{\mathrm{perm}}\ge \mathrm{BA}_{\mathrm{obs}}\}
}{
1+N_{\mathrm{perm}}
}.
$$

결과:

| target | window | model | n | BA | AUC | permutation BA mean | p(BA>=obs) | delta vs global |
|---|---|---|---:|---:|---:|---:|---:|---:|
| choice sign | stimulus | `region_family` | 663 | 0.582259 | 0.611570 | 0.500529 | 0.004975 | 0.078072 |
| choice sign | stimulus | `acronym_region` | 663 | 0.632548 | 0.687778 | 0.501580 | 0.004975 | 0.128361 |
| choice sign | stimulus | `global_rate` | 663 | 0.504187 | 0.531745 | 0.500748 | 0.149254 | 0.000000 |
| first movement speed | stimulus | `region_family` | 625 | 0.748756 | 0.830804 | 0.500294 | 0.004975 | 0.067195 |
| first movement speed | stimulus | `acronym_region` | 625 | 0.847971 | 0.916452 | 0.502408 | 0.004975 | 0.166411 |
| first movement speed | stimulus | `global_rate` | 625 | 0.681561 | 0.754680 | 0.500018 | 0.004975 | 0.000000 |
| wheel action direction | movement | `region_family` | 661 | 0.672194 | 0.733726 | 0.500501 | 0.004975 | 0.078659 |
| wheel action direction | movement | `acronym_region` | 661 | 0.709179 | 0.760170 | 0.500026 | 0.004975 | 0.115644 |
| wheel action direction | movement | `global_rate` | 661 | 0.593535 | 0.614640 | 0.500477 | 0.004975 | 0.000000 |

판정:

$$
\boxed{
\mathrm{region/acronym\ activity}
\;>\;
\mathrm{global\ firing\ rate}
\;>\;
\mathrm{label\ permutation}
}
$$

따라서 첫 strict IBL session에서는 mouse region-binned activity가 choice, action direction, action timing class를 실제로 지지한다.

의의는 세 가지다.

첫째, Drosophila/Zebrafish에서 얻은 \(P_t\rightarrow b_t\) 항이 포유류에서는 단일 neural population이 아니라 region-indexed population vector로 올라간다.

$$
R_t
=
\left[
r_{\mathrm{VIS}}(t),
r_{\mathrm{TH}}(t),
r_{\mathrm{HIP}}(t),
r_{\mathrm{CG}}(t),
\ldots
\right].
$$

둘째, behavior variable은 더 이상 단일 \(b_t\)가 아니라 choice, wheel direction, first-movement latency처럼 분해된다.

$$
y_t
=
\left[
y_t^{\mathrm{choice}},
y_t^{\mathrm{wheel}},
y_t^{\mathrm{speed}}
\right],
\qquad
\hat y_t
=
h(BR_t).
$$

셋째, global firing rate가 유의한 target도 있지만, region/acronym model이 consistently 더 높다. 그러므로 mouse 단계에서 새로 승격되는 항은 "많이 발화한다"가 아니라 "어느 영역 묶음이 어느 행동 변수를 나누는가"다.

$$
\boxed{
\Phi_{\mathrm{mammal}}(t)
=
B_{\mathrm{region}}R_t
+
B_{\mathrm{action}}R_{t-\tau:t}
}
$$

단, 이 gate는 mouse 전체를 닫지 않는다. 첫 probe의 강한 영역은 thalamus, visual cortex, hippocampus 쪽이고 motor/striatal loop가 충분히 포함되지 않았다. 따라서 다음 mouse gate는 다중 probe 또는 motor/striatum 포함 세션에서 같은 decoder를 반복해, \(B_{\mathrm{region}}\)이 특정 probe 위치의 우연한 readout이 아니라 일반적인 region/action bridge인지 확인해야 한다. 이 병목을 아래 audit와 multi-probe gate로 바로 이어서 검사했다.

### Mouse IBL/OpenAlyx motor-striatum candidate audit

첫 strict-session gate의 한계를 줄이기 위해, spike array를 새로 내려받기 전에 strict 29 sessions의 작은 cluster-level 파일만 훑었다. 사용한 파일은 `clusters.brainLocationAcronyms_ccf_2017.npy`와 `clusters.metrics.pqt`다. 목표는 motor cortex, striatal complex, septal/subpallial, basal-ganglia-output family가 실제 spike-count support를 갖는 probe를 찾는 것이다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_motor_striatum_audit.py --max-candidates 8
```

candidate probes:

| eid | session | collection | target clusters | target spikes from metrics | target family spike counts |
|---|---|---|---:|---:|---|
| `4720c98a-a305-4fba-affb-bbfa00a724a4` | `danlab/Subjects/DY_014/2020-07-14/001` | `alf/probe01/pykilosort` | 449 | 16530280 | septal 14090847, striatal 2439433 |
| `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `alf/probe01/pykilosort` | 209 | 8137407 | septal 3638363, striatal 4499044 |
| `cf43dbb1-6992-40ec-a5f9-e8e838d0f643` | `danlab/Subjects/DY_011/2020-02-08/001` | `alf/probe00/pykilosort` | 98 | 4211707 | motor 4211707 |
| `ee13c19e-2790-4418-97ca-48f02e8013bb` | `danlab/Subjects/DY_008/2020-03-04/001` | `alf/probe00/pykilosort` | 74 | 3713674 | striatal 3713674 |
| `5ec72172-3901-4771-8777-6e9490ca51fc` | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `alf/probe00/pykilosort` | 78 | 2649647 | motor 2649647 |

판정:

$$
\boxed{
\mathrm{NYU\text{-}30}
=
\mathrm{probe00}_{\mathrm{motor}}
+
\mathrm{probe01}_{\mathrm{striatal/septal}}
}
$$

NYU-30 session은 같은 behavioral trial table에서 motor cortex probe와 striatal/septal probe를 동시에 걸 수 있는 유일한 high-support multi-probe 후보였다. 그래서 다음 gate는 이 session의 두 probe를 결합했다.

### Mouse IBL/OpenAlyx multi-probe motor-striatal region gate

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_multi_probe_region_gate.py
```

loaded arrays:

| item | value |
|---|---:|
| eid | `5ec72172-3901-4771-8777-6e9490ca51fc` |
| session | `angelakilab/Subjects/NYU-30/2020-10-22/001` |
| trials | 933 |
| wheel samples | 1309392 |
| probe00 spikes | 24114519 |
| probe01 spikes | 31802063 |
| probe00 target-family spikes | 2649647 |
| probe01 target-family spikes | 8137407 |
| probe00 unknown spike fraction | 0.191488 |
| probe01 unknown spike fraction | 0.744123 |

여기서 `unknown` fraction은 중요한 제한이다. 특히 probe01은 strict acronym rows가 209인데 observed cluster slots가 561이라, 많은 spike가 CCF acronym 밖의 `unknown` bin으로 간다. 따라서 이 결과는 "모든 striatal unit이 완전하게 해부학적으로 라벨링되었다"는 뜻이 아니다. 더 정확한 표현은 다음이다.

$$
\boxed{
\mathrm{multi\text{-}probe\ decoding\ passed}
\quad\text{under a large preserved unknown bin.}
}
$$

multi-probe feature는 probe index \(p\)를 명시해서 만들었다.

$$
x_{ipg}
=
\frac{1}{b_i-a_i}
\sum_k
\mathbf 1[t_{pk}\in[a_i,b_i]]
\mathbf 1[G_p(c_{pk})=g].
$$

따라서 single-probe region vector \(R_i\)는 다음 multi-probe vector로 올라간다.

$$
R_i^{\mathrm{multi}}
=
\left[
x_{i,\mathrm{probe00},:},
x_{i,\mathrm{probe01},:}
\right].
$$

비교 모델:

| model | definition | 의미 |
|---|---|---|
| `family_collapsed` | probe를 합쳐 family별 rate만 남김 | 영역 family는 보존, probe identity 제거 |
| `family_by_probe` | probe-family block을 모두 보존 | 영역과 probe identity 모두 보존 |
| `motor_striatal_family_by_probe` | motor, striatal, septal, BG-output family만 사용 | target loop 최소 모델 |
| `acronym_by_probe` | probe별 high-spike CCF acronym bins | 가장 세분한 region/probe model |
| `global_rate` | 전체 probe/family rate 합 | flat firing-rate baseline |

결과:

| target | model | n | BA | AUC | permutation BA mean | p(BA>=obs) | delta vs global |
|---|---|---:|---:|---:|---:|---:|---:|
| choice sign | `family_collapsed` | 930 | 0.712854 | 0.791844 | 0.497251 | 0.004975 | 0.213534 |
| choice sign | `family_by_probe` | 930 | 0.775737 | 0.851160 | 0.496264 | 0.004975 | 0.276417 |
| choice sign | `motor_striatal_family_by_probe` | 930 | 0.624427 | 0.650034 | 0.497894 | 0.004975 | 0.125107 |
| choice sign | `acronym_by_probe` | 930 | 0.836748 | 0.910556 | 0.495400 | 0.004975 | 0.337428 |
| choice sign | `global_rate` | 930 | 0.499320 | 0.518565 | 0.500412 | 0.691542 | 0.000000 |
| first movement speed | `family_collapsed` | 833 | 0.637411 | 0.691760 | 0.499162 | 0.004975 | 0.031189 |
| first movement speed | `family_by_probe` | 833 | 0.661424 | 0.711285 | 0.499547 | 0.004975 | 0.055202 |
| first movement speed | `motor_striatal_family_by_probe` | 833 | 0.570213 | 0.615759 | 0.494481 | 0.004975 | -0.036009 |
| first movement speed | `acronym_by_probe` | 833 | 0.673460 | 0.713608 | 0.499607 | 0.004975 | 0.067239 |
| first movement speed | `global_rate` | 833 | 0.606222 | 0.625300 | 0.501346 | 0.004975 | 0.000000 |
| wheel action direction | `family_collapsed` | 930 | 0.708404 | 0.789737 | 0.501398 | 0.004975 | 0.195642 |
| wheel action direction | `family_by_probe` | 930 | 0.760339 | 0.850429 | 0.502493 | 0.004975 | 0.247577 |
| wheel action direction | `motor_striatal_family_by_probe` | 930 | 0.640382 | 0.693815 | 0.500447 | 0.004975 | 0.127620 |
| wheel action direction | `acronym_by_probe` | 930 | 0.851462 | 0.920076 | 0.504272 | 0.004975 | 0.338700 |
| wheel action direction | `global_rate` | 930 | 0.512762 | 0.578558 | 0.499815 | 0.009950 | 0.000000 |

판정:

$$
\boxed{
\mathrm{probe\text{-}indexed\ region/acronym\ code}
\gg
\mathrm{global\ firing\ rate}
}
$$

choice와 wheel action direction에서는 `motor_striatal_family_by_probe` 자체도 global-rate baseline보다 높고 permutation gate를 통과한다. first-movement speed에서는 motor/striatal 최소 모델이 global baseline보다 낮으므로, speed class는 motor/striatal loop만으로 닫지 않고 broader family/acronym context가 필요하다.

따라서 mouse 단계의 방정식 항은 다음처럼 한 단계 더 구체화된다.

$$
\boxed{
\Phi_{\mathrm{mammal}}(t)
=
B_{\mathrm{probe,region}}
R_t^{\mathrm{multi}}
+
B_{\mathrm{loop}}
R_{t-\tau:t}^{\mathrm{motor/striatal}}
}
$$

여기서 \(B_{\mathrm{probe,region}}\)은 probe identity와 region identity를 함께 보존하는 readout이고, \(B_{\mathrm{loop}}\)는 motor/striatal family만으로 설명 가능한 action component다. 다만 이 결론은 아직 single session이다. 다음 mouse closure는 이 gate를 여러 sessions에 반복해, \(\Phi_{\mathrm{mammal}}\)이 NYU-30 특이 readout인지 일반적인 포유류 region loop인지 분리해야 한다.

### Mouse IBL/OpenAlyx cross-session region generalization gate

NYU-30 multi-probe gate 다음 병목은 단일 session 특이성이다. 이를 위해 첫 thalamic/visual reference, NYU-30 multi-probe, 그리고 motor-striatum audit에서 고른 세 single-probe candidates를 같은 protocol로 반복했다.

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_cross_session_region_generalization_gate.py
```

candidate panel:

| candidate | kind | session | collection | reason |
|---|---|---|---|---|
| `witten29_thalamic_visual_reference` | single | `wittenlab/Subjects/ibl_witten_29/2021-06-08/001` | `alf/probe00/pykilosort` | first strict reference |
| `nyu30_motor_striatal_multi_probe` | multi | `angelakilab/Subjects/NYU-30/2020-10-22/001` | `probe00`, `probe01` | motor cortex plus striatal/septal bridge |
| `dy014_striatal_septal_probe` | single | `danlab/Subjects/DY_014/2020-07-14/001` | `alf/probe01/pykilosort` | highest target-family spike support |
| `dy011_motor_cortex_probe` | single | `danlab/Subjects/DY_011/2020-02-08/001` | `alf/probe00/pykilosort` | motor cortex candidate |
| `dy008_cp_somatosensory_thalamic_probe` | single | `danlab/Subjects/DY_008/2020-03-04/001` | `alf/probe00/pykilosort` | CP plus somatosensory cortex/thalamus |

replication summary:

| target | candidates | passed | mean best BA | mean delta vs global |
|---|---:|---:|---:|---:|
| choice sign | 5 | 4 | 0.670933 | 0.146035 |
| first movement speed | 5 | 5 | 0.751153 | 0.087768 |
| wheel action direction | 5 | 4 | 0.703612 | 0.170495 |

candidate summary:

| candidate | trials | passed targets | choice BA | speed BA | wheel BA |
|---|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 3 | 0.632548 | 0.847971 | 0.709179 |
| `nyu30_motor_striatal_multi_probe` | 933 | 3 | 0.836748 | 0.673460 | 0.851462 |
| `dy014_striatal_septal_probe` | 608 | 1 | 0.564950 | 0.627551 | 0.560469 |
| `dy011_motor_cortex_probe` | 402 | 3 | 0.623139 | 0.761131 | 0.665209 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 3 | 0.697282 | 0.845653 | 0.731743 |

판정:

$$
\boxed{
\mathrm{cross\text{-}session\ candidate\ panel}
\quad
4/5\ \mathrm{candidates\ pass\ all\ three\ targets}
}
$$

이 결과는 NYU-30 한 세션의 우연한 readout이라는 설명을 약하게 만든다. 특히 choice와 wheel action direction은 5개 후보 중 4개에서 global-rate baseline과 majority baseline을 넘고 permutation gate를 통과한다. first-movement speed는 5개 모두에서 통과했다.

단, DY_014 striatal/septal 단독 probe는 speed만 통과했다. choice와 wheel에서는 p-value는 낮지만 class imbalance에 대한 majority baseline을 넘지 못해 실패로 처리했다. 따라서 결론은 "striatal/septal 단독 probe가 충분하다"가 아니다. 더 정확한 결론은 다음이다.

$$
\boxed{
\mathrm{probe/region\ indexed\ readout}
\ \mathrm{generalizes\ across\ candidate\ sessions,}
\qquad
\mathrm{but\ loop\ sufficiency\ is\ target\ and\ coverage\ dependent.}
}
$$

이제 mouse 항은 session index \(s\)를 포함해 쓸 수 있다.

$$
\Phi_{\mathrm{mammal}}^{(s)}(t)
=
B_{\mathrm{probe,region}}^{(s)}
R_{t}^{(s)}
+
B_{\mathrm{loop}}^{(s)}
R_{t-\tau:t}^{(s,\mathrm{motor/striatal})}
+\epsilon_s.
$$

여기서 \(\epsilon_s\)는 probe placement, unknown-bin fraction, lab/session differences를 흡수하는 session residual이다. Cross-session gate의 의의는 \(\epsilon_s\)가 0이라는 것이 아니라, \(B_{\mathrm{probe,region}}^{(s)}R_t^{(s)}\) 항이 여러 후보 세션에서 반복된다는 점이다.

### Mouse IBL/OpenAlyx channel-region rescue gate

Cross-session gate를 통과한 뒤에도 가장 큰 약점은 `unknown` bin이었다. 특히 NYU-30 multi-probe gate에서 probe01은 strict acronym rows가 209인데 observed cluster slots가 561이라, spike의 74.4123%가 strict acronym 밖으로 빠졌다. 이 상태에서 decoder가 통과해도 결론은 "region code가 충분하다"가 아니라 "큰 censored bin을 보존한 상태에서도 decoding이 된다"에 머문다.

그래서 다음 gate는 strict cluster acronym만 쓰던 region map을 channel-level CCF id fallback으로 보강했다. IBL/OpenAlyx에는 다음 두 파일이 함께 있다.

| file | 의미 |
|---|---|
| `clusters.channels.npy` | cluster \(c\)가 놓인 recording channel \(\chi(c)\) |
| `channels.brainLocationIds_ccf_2017.npy` | channel \(\chi\)의 Allen CCF 2017 numeric region id \(I(\chi)\) |

핵심은 strict acronym label을 버리는 것이 아니라, strict acronym이 없는 cluster에만 fallback을 적용하는 것이다. Session \(s\), probe \(p\), cluster \(c\)에 대해 strict acronym을 \(A_{sp}(c)\), cluster-channel map을 \(\chi_{sp}(c)\), channel CCF id를 \(I_{sp}(\chi)\)라고 두면 hybrid map은 다음처럼 정의한다.

$$
G_{\mathrm{hybrid},sp}(c)
=
\begin{cases}
A_{sp}(c),
&A_{sp}(c)\neq\varnothing,\\
I_{sp}(\chi_{sp}(c)),
&A_{sp}(c)=\varnothing\ \mathrm{and}\ I_{sp}(\chi_{sp}(c))>0,\\
\varnothing,
&\mathrm{otherwise}.
\end{cases}
$$

여기서 \(A_{sp}(c)\)는 CCF acronym이고, \(I_{sp}(\chi_{sp}(c))\)는 numeric CCF id다. 둘은 같은 종류의 label이 아니므로 생물학적 의미를 동일하게 읽으면 안 된다. 이 gate에서 hybrid bin의 역할은 "정확한 region acronym 복원"이 아니라, strict acronym row 부족 때문에 censored 되던 cluster를 channel-registered anatomical id bin으로 되살리는 것이다.

Feature 식은 기존 region-window rate와 같은 형태를 유지한다.

$$
x_{ipg}^{\mathrm{hybrid}}
=
\frac{1}{b_i-a_i}
\sum_k
\mathbf 1[t_{pk}\in[a_i,b_i]]
\mathbf 1[G_{\mathrm{hybrid},sp}(c_{pk})=g].
$$

따라서 비교한 model은 다음 세 해부학 표현과 하나의 flat baseline이다.

| model | definition | 목적 |
|---|---|---|
| `strict_acronym_by_probe` | strict CCF acronym만 사용, 나머지는 `unknown` | 이전 cross-session gate와 같은 censored reference |
| `channel_region_id_by_probe` | cluster-channel에서 얻은 numeric CCF id만 사용 | acronym 없이 channel registration만으로 되는지 확인 |
| `hybrid_acronym_channel_id_by_probe` | acronym이 있으면 acronym, 없으면 channel CCF id | unknown-bin rescue model |
| `global_rate` | hybrid feature 전체 rate 합 | region/probe 구조를 지운 flat firing-rate baseline |

실행:

```bash
uv run --no-project --with ONE-api --with pandas --with pyarrow python examples/physics/evolution/mouse_ibl_channel_region_rescue_gate.py
```

Coverage 결과:

| item | value |
|---|---:|
| candidates | 5 |
| total spikes | 147796874 |
| strict unknown spikes | 39861227 |
| hybrid unknown spikes | 0 |
| rescued unknown spikes | 39861227 |
| strict unknown fraction | 0.269703 |
| hybrid unknown fraction | 0.000000 |
| rescued fraction of strict unknown | 1.000000 |

즉 이번 5-candidate panel 안에서는 strict acronym만 사용할 때 spike의 26.9703%가 `unknown`으로 들어갔지만, channel CCF id fallback을 적용하면 모든 strict-unknown spike가 anatomical id bin으로 복구된다.

Target replication:

| target | candidates | hybrid passed | mean strict BA | mean channel BA | mean hybrid BA | mean hybrid delta strict | mean hybrid delta global |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice sign | 5 | 4 | 0.652550 | 0.677024 | 0.675560 | 0.023010 | 0.150661 |
| first movement speed | 5 | 5 | 0.748035 | 0.741201 | 0.742104 | -0.005931 | 0.078719 |
| wheel action direction | 5 | 4 | 0.691198 | 0.707498 | 0.716240 | 0.025042 | 0.183122 |

Candidate summary:

| candidate | trials | strict unknown | hybrid unknown | rescued strict unknown | hybrid passed targets | choice hybrid BA | speed hybrid BA | wheel hybrid BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0.029674 | 0.000000 | 1.000000 | 3 | 0.636460 | 0.844756 | 0.725472 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0.505794 | 0.000000 | 1.000000 | 3 | 0.838249 | 0.655437 | 0.860886 |
| `dy014_striatal_septal_probe` | 608 | 0.084599 | 0.000000 | 1.000000 | 1 | 0.570300 | 0.627551 | 0.563511 |
| `dy011_motor_cortex_probe` | 402 | 0.232669 | 0.000000 | 1.000000 | 3 | 0.612129 | 0.737152 | 0.693547 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 0.232660 | 0.000000 | 1.000000 | 3 | 0.720661 | 0.845623 | 0.737784 |

판정:

$$
\boxed{
\mathrm{strict\ unknown\ caveat}
\quad
\xrightarrow{\mathrm{channel\ CCF\ id\ fallback}}
\quad
\mathrm{registered\ hybrid\ region\ code}
}
$$

그리고 decoding 반복성은 유지된다.

$$
\boxed{
\mathrm{hybrid\ region\ code}
>
\mathrm{global\ firing\ rate}
\quad
\mathrm{in}\quad
\mathrm{choice}\ 4/5,
\mathrm{speed}\ 5/5,
\mathrm{wheel}\ 4/5.
}
$$

DY_014는 여전히 speed만 통과한다. 따라서 이 gate는 "모든 probe에서 모든 행동 변수가 닫혔다"가 아니라, "unknown-bin이 큰 artifact라는 약점을 줄여도 cross-session region/action decoding은 유지된다"는 결론이다.

이 결과를 식으로 쓰면, 이전의 residual \(\epsilon_s\)를 더 잘게 나눌 수 있다.

$$
\epsilon_s
=
\epsilon_{\mathrm{placement},s}
+\epsilon_{\mathrm{lab},s}
+\epsilon_{\mathrm{registration},s}
+\epsilon_{\mathrm{unregistered},s}.
$$

Strict acronym gate에서는 \(\epsilon_{\mathrm{registration},s}\)와 \(\epsilon_{\mathrm{unregistered},s}\)가 모두 `unknown` bin에 섞여 있었다. Channel-region rescue 뒤에는 관측 가능한 channel CCF id가 \(R_t^{\mathrm{hybrid}}\) 안으로 들어오므로, 남는 residual은 더 작고 더 엄격한 의미의 미등록 잔차다.

$$
R_t^{(s,\mathrm{hybrid})}
=
\left[
r_{p,g}^{(s)}(t)
\right]_{p\in\mathcal P_s,\ g\in\mathcal G_{\mathrm{acronym}\cup\mathrm{ccf\ id}}}.
$$

따라서 현재 mouse 항은 다음처럼 갱신한다.

$$
\boxed{
\Phi_{\mathrm{mammal}}^{(s)}(t)
=
B_{\mathrm{probe,hybrid\ region}}^{(s)}
R_t^{(s,\mathrm{hybrid})}
+
B_{\mathrm{loop}}^{(s)}
R_{t-\tau:t}^{(s,\mathrm{motor/striatal})}
+
\epsilon_{\mathrm{placement/lab},s}.
}
$$

여기서 중요한 변화는 \(B_{\mathrm{probe,region}}\)이 strict acronym-only readout에서 hybrid registered readout으로 승격된 점이다. Mouse 단계의 남은 병목은 이제 "cluster acronym row 부족 때문에 생긴 unknown artifact"가 아니라, 더 큰 panel에서 같은 결과가 유지되는지, 그리고 timing-only/flat-neuron/effective-connectivity baseline을 얼마나 더 깰 수 있는지다.

