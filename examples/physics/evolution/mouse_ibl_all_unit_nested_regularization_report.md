# Mouse IBL/OpenAlyx all-unit nested-regularization gate

Flat-unit gate 다음 반례는 단순 unit-vs-region 비교가 아니라 nested residual이다.
이 gate는 task/history와 channel-rescued region을 먼저 넣은 뒤, high-spike unit identity가 cross-validated BA를 추가로 올리는지 검사한다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 10
- min unit spikes: 1000
- max units per probe: 192
- unit residual after task+region supported: `False`
- region residual after task+unit supported: `True`
- all-unit nested regularization passed: `False`

## nested equations

$$
M_X:y_i\sim X_i,\quad
M_{XR}:y_i\sim[X_i,R_i],\quad
M_{XU}:y_i\sim[X_i,U_i],\quad
M_{XRU}:y_i\sim[X_i,R_i,U_i].
$$

The main residuals are

$$
\Delta_{U\mid X,R}=\mathrm{BA}(M_{XRU})-\mathrm{BA}(M_{XR}),
\qquad
\Delta_{R\mid X,U}=\mathrm{BA}(M_{XRU})-\mathrm{BA}(M_{XU}).
$$

`U_i`는 probe별 high-spike unit identity다. `max_units_per_probe<=0`이면 threshold를 넘는 모든 unit을 쓰고, 양수이면 computational guard로 상위 unit만 남긴다.

## target replication

| target | candidates | unit residual count | region residual count | unit>region after task | mean task BA | mean task+region BA | mean task+unit BA | mean task+region+unit BA | mean unit residual | mean region residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 0 | 3 | 0 | 0.844886 | 0.845811 | 0.800778 | 0.802420 | -0.043390 | 0.001643 |
| `first_movement_speed` | 5 | 1 | 3 | 1 | 0.689940 | 0.750006 | 0.707508 | 0.711956 | -0.038051 | 0.004448 |
| `wheel_action_direction` | 5 | 0 | 4 | 0 | 0.822181 | 0.837202 | 0.800453 | 0.803640 | -0.033562 | 0.003187 |

## candidate summaries

| candidate | trials | unit residuals | region residuals | choice U_given_XR | speed U_given_XR | wheel U_given_XR | choice R_given_XU | speed R_given_XU | wheel R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0 | 2 | -0.051956 | -0.019195 | -0.036050 | 0.003912 | -0.008008 | 0.010234 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0 | 2 | -0.004495 | -0.035997 | -0.003564 | 0.000827 | -0.010814 | 0.005592 |
| `dy014_striatal_septal_probe` | 608 | 1 | 1 | -0.017299 | 0.030612 | -0.044043 | -0.015933 | 0.003401 | -0.014457 |
| `dy011_motor_cortex_probe` | 402 | 0 | 3 | -0.068480 | -0.058257 | -0.054654 | 0.031650 | 0.021206 | 0.003795 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 0 | 2 | -0.074722 | -0.107416 | -0.029497 | -0.012242 | 0.016453 | 0.010773 |

## per-candidate details

### witten29_thalamic_visual_reference

- eid: `d2832a38-27f6-452d-91d6-af72d794136c`
- session: `wittenlab/Subjects/ibl_witten_29/2021-06-08/001`
- reason: first strict-session thalamic/visual/hippocampal reference

| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 663 | 0.888223 | 0.888223 | 0.832355 | 0.836267 | -0.051956 | 0.003912 |
| `first_movement_speed` | 625 | 0.734527 | 0.847987 | 0.836800 | 0.828792 | -0.019195 | -0.008008 |
| `wheel_action_direction` | 661 | 0.859095 | 0.858884 | 0.812600 | 0.822834 | -0.036050 | 0.010234 |

### nyu30_motor_striatal_multi_probe

- eid: `5ec72172-3901-4771-8777-6e9490ca51fc`
- session: `angelakilab/Subjects/NYU-30/2020-10-22/001`
- reason: same-session motor cortex plus striatal/septal multi-probe bridge

| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 930 | 0.895508 | 0.896034 | 0.890712 | 0.891539 | -0.004495 | 0.000827 |
| `first_movement_speed` | 833 | 0.618235 | 0.662631 | 0.637449 | 0.626634 | -0.035997 | -0.010814 |
| `wheel_action_direction` | 930 | 0.845588 | 0.875014 | 0.865858 | 0.871450 | -0.003564 | 0.005592 |

### dy014_striatal_septal_probe

- eid: `4720c98a-a305-4fba-affb-bbfa00a724a4`
- session: `danlab/Subjects/DY_014/2020-07-14/001`
- reason: highest target-family spike support in motor-striatum audit

| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 605 | 0.836293 | 0.818602 | 0.817236 | 0.801303 | -0.017299 | -0.015933 |
| `first_movement_speed` | 588 | 0.634354 | 0.661565 | 0.688776 | 0.692177 | 0.030612 | 0.003401 |
| `wheel_action_direction` | 604 | 0.815704 | 0.829218 | 0.799632 | 0.785175 | -0.044043 | -0.014457 |

### dy011_motor_cortex_probe

- eid: `cf43dbb1-6992-40ec-a5f9-e8e838d0f643`
- session: `danlab/Subjects/DY_011/2020-02-08/001`
- reason: single-probe motor cortex candidate

| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 396 | 0.813717 | 0.811530 | 0.711400 | 0.743050 | -0.068480 | 0.031650 |
| `first_movement_speed` | 377 | 0.716031 | 0.726641 | 0.647177 | 0.668383 | -0.058257 | 0.021206 |
| `wheel_action_direction` | 395 | 0.802093 | 0.809473 | 0.751024 | 0.754819 | -0.054654 | 0.003795 |

### dy008_cp_somatosensory_thalamic_probe

- eid: `ee13c19e-2790-4418-97ca-48f02e8013bb`
- session: `danlab/Subjects/DY_008/2020-03-04/001`
- reason: CP plus somatosensory cortex/thalamus candidate

| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 400 | 0.790691 | 0.814664 | 0.752184 | 0.739942 | -0.074722 | -0.012242 |
| `first_movement_speed` | 363 | 0.746555 | 0.851208 | 0.727339 | 0.743792 | -0.107416 | 0.016453 |
| `wheel_action_direction` | 403 | 0.788426 | 0.813419 | 0.773149 | 0.783921 | -0.029497 | 0.010773 |

## verdict

- unit residual after task+region supported: `False`
- region residual after task+unit supported: `True`
- all-unit nested regularization passed: `False`

해석:

- \(\Delta_{U\mid X,R}>0\)가 반복되면 mouse 단계 방정식에 explicit unit-detail residual을 남겨야 한다.
- \(\Delta_{R\mid X,U}>0\)가 반복되면 anatomical compression도 unit decoder 위에서 독립 항으로 남는다.
- 이 gate는 coupling이 아니라 nested decoding이다. 다음 강한 버전은 unit-to-unit GLM coupling 또는 trial-split lag selection이다.
