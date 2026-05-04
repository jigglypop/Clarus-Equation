# Mouse IBL/OpenAlyx cross-session region generalization gate

목표는 mouse region/action 항이 NYU-30 한 세션의 특이 readout인지, 후보 세션 패널에서 반복되는지 확인하는 것이다.
각 후보는 같은 fixed-window region/acronym decoder와 global-rate baseline, label permutation null을 사용한다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- generalization passed: `True`

## target replication

| target | candidates | passed | mean best BA | mean delta global |
|---|---:|---:|---:|---:|
| `choice_sign` | 5 | 4 | 0.670933 | 0.146035 |
| `first_movement_speed` | 5 | 5 | 0.751153 | 0.087768 |
| `wheel_action_direction` | 5 | 4 | 0.703612 | 0.170495 |

## candidate summaries

| candidate | kind | collections | trials | passed targets | choice BA | speed BA | wheel BA |
|---|---|---|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | single | `alf/probe00/pykilosort` | 663 | 3 | 0.632548 | 0.847971 | 0.709179 |
| `nyu30_motor_striatal_multi_probe` | multi | `alf/probe00/pykilosort`, `alf/probe01/pykilosort` | 933 | 3 | 0.836748 | 0.673460 | 0.851462 |
| `dy014_striatal_septal_probe` | single | `alf/probe01/pykilosort` | 608 | 1 | 0.564950 | 0.627551 | 0.560469 |
| `dy011_motor_cortex_probe` | single | `alf/probe00/pykilosort` | 402 | 3 | 0.623139 | 0.761131 | 0.665209 |
| `dy008_cp_somatosensory_thalamic_probe` | single | `alf/probe00/pykilosort` | 409 | 3 | 0.697282 | 0.845653 | 0.731743 |

## per-candidate details

### witten29_thalamic_visual_reference

- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- reason: first strict-session thalamic/visual/hippocampal reference

| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | `acronym_region` | 663 | `{"0": 300, "1": 363}` | 0.632548 | 0.687778 | 0.004975 | 0.504187 | 0.128361 | True |
| `first_movement_speed` | `acronym_region` | 625 | `{"0": 312, "1": 313}` | 0.847971 | 0.916452 | 0.004975 | 0.681561 | 0.166411 | True |
| `wheel_action_direction` | `acronym_region` | 661 | `{"0": 342, "1": 319}` | 0.709179 | 0.760170 | 0.004975 | 0.593535 | 0.115644 | True |

### nyu30_motor_striatal_multi_probe

- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- reason: same-session motor cortex plus striatal/septal multi-probe bridge

| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | `acronym_by_probe` | 930 | `{"0": 417, "1": 513}` | 0.836748 | 0.910556 | 0.004975 | 0.499320 | 0.337428 | True |
| `first_movement_speed` | `acronym_by_probe` | 833 | `{"0": 416, "1": 417}` | 0.673460 | 0.713608 | 0.004975 | 0.606222 | 0.067239 | True |
| `wheel_action_direction` | `acronym_by_probe` | 930 | `{"0": 522, "1": 408}` | 0.851462 | 0.920076 | 0.004975 | 0.512762 | 0.338700 | True |

### dy014_striatal_septal_probe

- eid: `4720c98a-a305-4fba-affb-bbfa00a724a4`
- session: `danlab/Subjects/DY_014/2020-07-14/001`
- reason: highest target-family spike support in motor-striatum audit

| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | `acronym_region` | 605 | `{"0": 388, "1": 217}` | 0.564950 | 0.617642 | 0.004975 | 0.500000 | 0.064950 | False |
| `first_movement_speed` | `acronym_region` | 588 | `{"0": 294, "1": 294}` | 0.627551 | 0.667372 | 0.004975 | 0.561224 | 0.066327 | True |
| `wheel_action_direction` | `acronym_region` | 604 | `{"0": 222, "1": 382}` | 0.560469 | 0.655512 | 0.004975 | 0.498691 | 0.061778 | False |

### dy011_motor_cortex_probe

- eid: `cf43dbb1-6992-40ec-a5f9-e8e838d0f643`
- session: `danlab/Subjects/DY_011/2020-02-08/001`
- reason: single-probe motor cortex candidate

| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | `acronym_region` | 396 | `{"0": 226, "1": 170}` | 0.623139 | 0.668116 | 0.004975 | 0.516762 | 0.106377 | True |
| `first_movement_speed` | `region_family` | 377 | `{"0": 188, "1": 189}` | 0.761131 | 0.828493 | 0.004975 | 0.718662 | 0.042469 | True |
| `wheel_action_direction` | `acronym_region` | 395 | `{"0": 167, "1": 228}` | 0.665209 | 0.735608 | 0.004975 | 0.528456 | 0.136753 | True |

### dy008_cp_somatosensory_thalamic_probe

- eid: `ee13c19e-2790-4418-97ca-48f02e8013bb`
- session: `danlab/Subjects/DY_008/2020-03-04/001`
- reason: CP plus somatosensory cortex/thalamus candidate

| target | best model | n | class counts | BA | AUC | p(BA>=obs) | global BA | delta global | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | `region_family` | 400 | `{"0": 146, "1": 254}` | 0.697282 | 0.764454 | 0.004975 | 0.604223 | 0.093059 | True |
| `first_movement_speed` | `acronym_region` | 363 | `{"0": 181, "1": 182}` | 0.845653 | 0.933186 | 0.004975 | 0.749256 | 0.096397 | True |
| `wheel_action_direction` | `acronym_region` | 403 | `{"0": 258, "1": 145}` | 0.731743 | 0.844106 | 0.004975 | 0.532144 | 0.199599 | True |

## verdict

- candidates passing all three targets: 4 / 5
- generalization passed: `True`

이 gate는 mouse 항을 완전히 닫는 최종 다기관 검정은 아니지만, region/probe-indexed action readout이 단일 NYU-30 세션에만 묶인 현상은 아니라는 중간 결론을 준다.
