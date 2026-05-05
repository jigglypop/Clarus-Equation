# Mouse IBL/OpenAlyx flat-unit versus hybrid-region comparison gate

Task-baseline gate 다음 반례는 anatomical region/probe bin이 아니라 개별 unit identity만으로 충분한 경우다.
이 gate는 probe별 spike count 상위 unit을 flat feature로 만들고, channel-rescued hybrid region feature와 같은 target에서 비교한다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- max units per probe: 96
- top unit dominates region: `True`
- region survives flat-unit comparison: `True`
- flat-unit gate passed: `False`

## models

| model | feature definition |
|---|---|
| `hybrid_region_by_probe` | channel-rescued anatomical acronym/CCF-id bins by probe |
| `top_unit_by_probe` | highest-spike clusters by probe plus `other_units` bin |
| `task_history_plus_hybrid_region` | task-history baseline plus hybrid region bins |
| `task_history_plus_top_unit` | task-history baseline plus top-unit bins |
| `global_rate` | one scalar total hybrid firing rate |

The comparison is

$$
\Delta_{\mathrm{unit-region}}=\mathrm{BA}(U^{\mathrm{top}})-\mathrm{BA}(R^{\mathrm{hybrid}}),
\qquad
\Delta_{\mathrm{task+unit}}=\mathrm{BA}([X^{\mathrm{task}},U^{\mathrm{top}}])-\mathrm{BA}([X^{\mathrm{task}},R^{\mathrm{hybrid}}]).
$$

## target replication

| target | candidates | region beats unit | unit beats region | task+region beats task+unit | task+unit beats task+region | mean region BA | mean unit BA | mean task+region BA | mean task+unit BA | mean unit-region delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 0 | 5 | 2 | 3 | 0.675560 | 0.744883 | 0.845811 | 0.849174 | 0.069323 |
| `first_movement_speed` | 5 | 3 | 2 | 3 | 2 | 0.742104 | 0.756299 | 0.750006 | 0.754919 | 0.014195 |
| `wheel_action_direction` | 5 | 0 | 5 | 3 | 2 | 0.716240 | 0.777766 | 0.837202 | 0.835943 | 0.061526 |

## candidate summaries

| candidate | trials | region>unit | unit>region | task+region>task+unit | task+unit>task+region | choice unit-region | speed unit-region | wheel unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 1 | 2 | 3 | 0 | 0.106680 | -0.004797 | 0.059337 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0 | 3 | 0 | 3 | 0.062357 | 0.031224 | 0.039216 |
| `dy014_striatal_septal_probe` | 608 | 0 | 3 | 1 | 2 | 0.046439 | 0.090136 | 0.078062 |
| `dy011_motor_cortex_probe` | 402 | 1 | 2 | 2 | 1 | 0.042530 | -0.023655 | 0.094797 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 1 | 2 | 2 | 1 | 0.088610 | -0.021932 | 0.036220 |

## per-candidate details

### witten29_thalamic_visual_reference

- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- reason: first strict-session thalamic/visual/hippocampal reference

| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 663 | 0.636460 | 0.743140 | 0.888223 | 0.879311 | 0.106680 | -0.008912 |
| `first_movement_speed` | 625 | 0.844756 | 0.839959 | 0.847987 | 0.835151 | -0.004797 | -0.012836 |
| `wheel_action_direction` | 661 | 0.725472 | 0.784808 | 0.858884 | 0.838508 | 0.059337 | -0.020376 |

### nyu30_motor_striatal_multi_probe

- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- reason: same-session motor cortex plus striatal/septal multi-probe bridge

| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 930 | 0.838249 | 0.900606 | 0.896034 | 0.928941 | 0.062357 | 0.032907 |
| `first_movement_speed` | 833 | 0.655437 | 0.686661 | 0.662631 | 0.697467 | 0.031224 | 0.034836 |
| `wheel_action_direction` | 930 | 0.860886 | 0.900101 | 0.875014 | 0.908300 | 0.039216 | 0.033285 |

### dy014_striatal_septal_probe

- eid: `4720c98a-a305-4fba-affb-bbfa00a724a4`
- session: `danlab/Subjects/DY_014/2020-07-14/001`
- reason: highest target-family spike support in motor-striatum audit

| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 605 | 0.570300 | 0.616740 | 0.818602 | 0.835824 | 0.046439 | 0.017222 |
| `first_movement_speed` | 588 | 0.627551 | 0.717687 | 0.661565 | 0.729592 | 0.090136 | 0.068027 |
| `wheel_action_direction` | 604 | 0.563511 | 0.641574 | 0.829218 | 0.807273 | 0.078062 | -0.021945 |

### dy011_motor_cortex_probe

- eid: `cf43dbb1-6992-40ec-a5f9-e8e838d0f643`
- session: `danlab/Subjects/DY_011/2020-02-08/001`
- reason: single-probe motor cortex candidate

| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 396 | 0.612129 | 0.654659 | 0.811530 | 0.772488 | 0.042530 | -0.039042 |
| `first_movement_speed` | 377 | 0.737152 | 0.713498 | 0.726641 | 0.716214 | -0.023655 | -0.010427 |
| `wheel_action_direction` | 395 | 0.693547 | 0.788344 | 0.809473 | 0.840004 | 0.094797 | 0.030531 |

### dy008_cp_somatosensory_thalamic_probe

- eid: `ee13c19e-2790-4418-97ca-48f02e8013bb`
- session: `danlab/Subjects/DY_008/2020-03-04/001`
- reason: CP plus somatosensory cortex/thalamus candidate

| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 400 | 0.720661 | 0.809271 | 0.814664 | 0.829306 | 0.088610 | 0.014642 |
| `first_movement_speed` | 363 | 0.845623 | 0.823690 | 0.851208 | 0.796172 | -0.021932 | -0.055036 |
| `wheel_action_direction` | 403 | 0.737784 | 0.774004 | 0.813419 | 0.785632 | 0.036220 | -0.027787 |

## verdict

- top unit dominates region: `True`
- region survives flat-unit comparison: `True`
- flat-unit gate passed: `False`

해석:

- Top-unit readout이 hybrid region을 반복적으로 이기면, region/probe 항은 maximal decoder가 아니라 compressed anatomical readout으로 내려간다.
- Hybrid region이 일부 target에서 top-unit과 같거나 더 좋으면, anatomical binning이 단순 정보 손실만은 아니라는 뜻이다.
- 이 gate는 all-unit decoder가 아니라 top-unit bounded decoder다. 따라서 flat-neuron 반례의 강한 버전은 아직 더 큰 unit set 또는 nested regularization으로 남는다.
