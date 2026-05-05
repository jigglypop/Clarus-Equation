# Mouse IBL/OpenAlyx channel-region rescue gate

Cross-session gate의 큰 제한이던 strict acronym `unknown` bin을 줄이기 위해 channel-level CCF id fallback을 추가했다.
Strict acronym이 있는 cluster는 그대로 두고, strict acronym이 없는 cluster만 `clusters.channels -> channels.brainLocationIds_ccf_2017`로 복구한다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- min label spikes: 100000
- coverage passed: `True`
- decoding passed: `True`
- rescue gate passed: `True`

## coverage rescue

| item | value |
|---|---:|
| total spikes | 147796874 |
| strict unknown spikes | 39861227 |
| hybrid unknown spikes | 0 |
| rescued unknown spikes | 39861227 |
| strict unknown fraction | 0.269703 |
| hybrid unknown fraction | 0.000000 |
| rescued fraction of strict unknown | 1.000000 |

The hybrid cluster map is

$$
G_{\mathrm{hybrid}}(c)=\begin{cases}
A(c),&A(c)\neq\varnothing,\\
I(\chi(c)),&A(c)=\varnothing\ \mathrm{and}\ I(\chi(c))>0,\\
\varnothing,&\mathrm{otherwise}.
\end{cases}
$$

Here \(A(c)\) is the strict cluster acronym, \(\chi(c)\) is the channel assigned to cluster \(c\), and \(I(\chi(c))\) is the channel CCF region id.
The rescued feature keeps the same duration-normalized trial-window form:

$$
x_{ipg}^{\mathrm{hybrid}}=\frac{1}{b_i-a_i}\sum_k \mathbf 1[t_{pk}\in[a_i,b_i]]\mathbf 1[G_{\mathrm{hybrid},p}(c_{pk})=g].
$$

## target replication

| target | candidates | hybrid passed | mean strict BA | mean channel BA | mean hybrid BA | mean hybrid delta strict | mean hybrid delta global |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 4 | 0.652550 | 0.677024 | 0.675560 | 0.023010 | 0.150661 |
| `first_movement_speed` | 5 | 5 | 0.748035 | 0.741201 | 0.742104 | -0.005931 | 0.078719 |
| `wheel_action_direction` | 5 | 4 | 0.691198 | 0.707498 | 0.716240 | 0.025042 | 0.183122 |

## candidate summaries

| candidate | trials | strict unknown | hybrid unknown | rescued strict unknown | hybrid passed targets | choice hybrid BA | speed hybrid BA | wheel hybrid BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0.029674 | 0.000000 | 1.000000 | 3 | 0.636460 | 0.844756 | 0.725472 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0.505794 | 0.000000 | 1.000000 | 3 | 0.838249 | 0.655437 | 0.860886 |
| `dy014_striatal_septal_probe` | 608 | 0.084599 | 0.000000 | 1.000000 | 1 | 0.570300 | 0.627551 | 0.563511 |
| `dy011_motor_cortex_probe` | 402 | 0.232669 | 0.000000 | 1.000000 | 3 | 0.612129 | 0.737152 | 0.693547 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 0.232660 | 0.000000 | 1.000000 | 3 | 0.720661 | 0.845623 | 0.737784 |

## per-probe coverage

### witten29_thalamic_visual_reference

- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- reason: first strict-session thalamic/visual/hippocampal reference

| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alf/probe00/pykilosort` | 34822347 | 648 | 700 | 384 | 0.029674 | 0.000000 | 1.000000 | 17 |

### nyu30_motor_striatal_multi_probe

- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- reason: same-session motor cortex plus striatal/septal multi-probe bridge

| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alf/probe00/pykilosort` | 24114519 | 643 | 908 | 384 | 0.191488 | 0.000000 | 1.000000 | 16 |
| `alf/probe01/pykilosort` | 31802063 | 209 | 561 | 384 | 0.744123 | 0.000000 | 1.000000 | 13 |

### dy014_striatal_septal_probe

- eid: `4720c98a-a305-4fba-affb-bbfa00a724a4`
- session: `danlab/Subjects/DY_014/2020-07-14/001`
- reason: highest target-family spike support in motor-striatum audit

| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alf/probe01/pykilosort` | 18435624 | 464 | 678 | 384 | 0.084599 | 0.000000 | 1.000000 | 16 |

### dy011_motor_cortex_probe

- eid: `cf43dbb1-6992-40ec-a5f9-e8e838d0f643`
- session: `danlab/Subjects/DY_011/2020-02-08/001`
- reason: single-probe motor cortex candidate

| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alf/probe00/pykilosort` | 9543943 | 205 | 329 | 384 | 0.232669 | 0.000000 | 1.000000 | 15 |

### dy008_cp_somatosensory_thalamic_probe

- eid: `ee13c19e-2790-4418-97ca-48f02e8013bb`
- session: `danlab/Subjects/DY_008/2020-03-04/001`
- reason: CP plus somatosensory cortex/thalamus candidate

| probe | spikes | strict rows | cluster-channel rows | CCF id rows | strict unknown | hybrid unknown | rescued strict unknown | hybrid features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alf/probe00/pykilosort` | 29078378 | 419 | 685 | 384 | 0.232660 | 0.000000 | 1.000000 | 26 |

## verdict

- coverage passed: `True`
- decoding passed: `True`
- rescue gate passed: `True`

해석:

- Channel fallback은 strict acronym 밖으로 빠진 spike를 버리지 않고 CCF id bin으로 되살린다.
- 따라서 `unknown`은 더 이상 cluster acronym row 부족만 의미하지 않고, channel CCF id까지 없는 진짜 미등록 잔차가 된다.
- Hybrid code가 여러 target에서 global-rate baseline을 계속 넘으면, mouse 항의 session residual \(\epsilon_s\) 중 `unknown` 성분은 관측 가능 registration error로 한 단계 분해된다.
