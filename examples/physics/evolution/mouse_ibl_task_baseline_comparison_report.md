# Mouse IBL/OpenAlyx task-baseline comparison gate

Channel-region rescue 뒤의 다음 반례는 region decoder가 단순 trial timing, stimulus table, previous-trial history만 읽는 경우다.
이 gate는 current choice, current first movement, current response, current feedback을 baseline feature에서 제외하고 hybrid region feature와 비교한다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- min delta for baseline win: 0.0
- timing counterexample rejected: `True`
- task increment supported: `False`
- baseline gate passed: `False`

## models

| model | feature definition | leakage rule |
|---|---|---|
| `timing_only` | trial index, session time, stim delay, go-cue delay | no choice/movement/outcome |
| `task_history` | timing + current stimulus/probability + previous trial choice/reward/latency | no current choice/movement/outcome |
| `hybrid_region_by_probe` | channel-rescued probe-region spike rates | neural window feature |
| `task_history_plus_hybrid_region` | task history baseline plus hybrid region feature | incremental neural contribution |
| `global_rate` | one scalar total hybrid firing rate | flat firing-rate baseline |

The main comparison is

$$
\Delta_{\mathrm{timing}}=\mathrm{BA}(R^{\mathrm{hybrid}})-\mathrm{BA}(X^{\mathrm{timing}}),
\qquad
\Delta_{\mathrm{task}}=\mathrm{BA}([X^{\mathrm{task}},R^{\mathrm{hybrid}}])-\mathrm{BA}(X^{\mathrm{task}}).
$$

## target replication

| target | candidates | hybrid beats timing | task+hybrid beats task | mean timing BA | mean task BA | mean hybrid BA | mean task+hybrid BA | mean delta timing | mean delta task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 5 | 2 | 0.544584 | 0.844886 | 0.675560 | 0.845811 | 0.130975 | 0.000924 |
| `first_movement_speed` | 5 | 5 | 5 | 0.616631 | 0.689940 | 0.742104 | 0.750006 | 0.125473 | 0.060066 |
| `wheel_action_direction` | 5 | 5 | 4 | 0.530258 | 0.822181 | 0.716240 | 0.837202 | 0.185982 | 0.015020 |

## candidate summaries

| candidate | trials | hybrid beats timing | task+hybrid beats task | choice delta timing | choice delta task | speed delta timing | speed delta task | wheel delta timing | wheel delta task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 3 | 1 | 0.134807 | 0.000000 | 0.217524 | 0.113459 | 0.233831 | -0.000211 |
| `nyu30_motor_striatal_multi_probe` | 933 | 3 | 3 | 0.300062 | 0.000526 | 0.075917 | 0.044396 | 0.359294 | 0.029426 |
| `dy014_striatal_septal_probe` | 608 | 3 | 2 | 0.070300 | -0.017691 | 0.049320 | 0.027211 | 0.063511 | 0.013514 |
| `dy011_motor_cortex_probe` | 402 | 3 | 2 | 0.094482 | -0.002186 | 0.066264 | 0.010610 | 0.182963 | 0.007380 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 3 | 3 | 0.055226 | 0.023973 | 0.218338 | 0.104654 | 0.090310 | 0.024993 |

## per-candidate target details

### witten29_thalamic_visual_reference

- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- reason: first strict-session thalamic/visual/hippocampal reference

| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 663 | 0.501653 | 0.888223 | 0.636460 | 0.888223 | 0.134807 | -0.251763 | 0.000000 |
| `first_movement_speed` | 625 | 0.627232 | 0.734527 | 0.844756 | 0.847987 | 0.217524 | 0.110229 | 0.113459 |
| `wheel_action_direction` | 661 | 0.491641 | 0.859095 | 0.725472 | 0.858884 | 0.233831 | -0.133623 | -0.000211 |

### nyu30_motor_striatal_multi_probe

- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- reason: same-session motor cortex plus striatal/septal multi-probe bridge

| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 930 | 0.538187 | 0.895508 | 0.838249 | 0.896034 | 0.300062 | -0.057259 | 0.000526 |
| `first_movement_speed` | 833 | 0.579520 | 0.618235 | 0.655437 | 0.662631 | 0.075917 | 0.037202 | 0.044396 |
| `wheel_action_direction` | 930 | 0.501592 | 0.845588 | 0.860886 | 0.875014 | 0.359294 | 0.015297 | 0.029426 |

### dy014_striatal_septal_probe

- eid: `4720c98a-a305-4fba-affb-bbfa00a724a4`
- session: `danlab/Subjects/DY_014/2020-07-14/001`
- reason: highest target-family spike support in motor-striatum audit

| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 605 | 0.500000 | 0.836293 | 0.570300 | 0.818602 | 0.070300 | -0.265992 | -0.017691 |
| `first_movement_speed` | 588 | 0.578231 | 0.634354 | 0.627551 | 0.661565 | 0.049320 | -0.006803 | 0.027211 |
| `wheel_action_direction` | 604 | 0.500000 | 0.815704 | 0.563511 | 0.829218 | 0.063511 | -0.252193 | 0.013514 |

### dy011_motor_cortex_probe

- eid: `cf43dbb1-6992-40ec-a5f9-e8e838d0f643`
- session: `danlab/Subjects/DY_011/2020-02-08/001`
- reason: single-probe motor cortex candidate

| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 396 | 0.517647 | 0.813717 | 0.612129 | 0.811530 | 0.094482 | -0.201588 | -0.002186 |
| `first_movement_speed` | 377 | 0.670888 | 0.716031 | 0.737152 | 0.726641 | 0.066264 | 0.021122 | 0.010610 |
| `wheel_action_direction` | 395 | 0.510584 | 0.802093 | 0.693547 | 0.809473 | 0.182963 | -0.108546 | 0.007380 |

### dy008_cp_somatosensory_thalamic_probe

- eid: `ee13c19e-2790-4418-97ca-48f02e8013bb`
- session: `danlab/Subjects/DY_008/2020-03-04/001`
- reason: CP plus somatosensory cortex/thalamus candidate

| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 400 | 0.665435 | 0.790691 | 0.720661 | 0.814664 | 0.055226 | -0.070030 | 0.023973 |
| `first_movement_speed` | 363 | 0.627284 | 0.746555 | 0.845623 | 0.851208 | 0.218338 | 0.099068 | 0.104654 |
| `wheel_action_direction` | 403 | 0.647474 | 0.788426 | 0.737784 | 0.813419 | 0.090310 | -0.050642 | 0.024993 |

## verdict

- timing counterexample rejected: `True`
- task increment supported: `False`
- baseline gate passed: `False`

해석:

- Hybrid region feature가 timing-only baseline을 반복적으로 넘으면, 단순 session drift 또는 trial clock 반례가 약해진다.
- `task_history_plus_hybrid_region`이 `task_history`를 넘으면, current stimulus와 previous-trial history만으로는 남는 neural increment가 있다는 뜻이다.
- 이 gate는 causal proof가 아니다. 다만 mouse region/probe readout을 task-table artifact보다 더 강한 형태로 걸러낸다.
