# Mouse IBL/OpenAlyx temporal GLM coupling gate

This gate asks whether lagged unit activity survives after task/history, target-window hybrid region bins, and current-window unit detail.

## setup

- candidates: 5
- folds: 5
- inner folds: 3
- task penalty: 1.0
- region penalty: 1.0
- unit penalties: `[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]`
- min unit spikes: 1000
- max units per probe: 192
- lagged unit after task+region supported: `True`
- lagged unit after task+region+current-unit supported: `False`
- temporal GLM coupling gate passed: `False`

## nested comparison

$$
M_{XR}: y_i\sim[X_i,R_i],
\qquad
M_{XRU_0}: y_i\sim[X_i,R_i,U_{0,i}],
$$

$$
M_{XRU_L}: y_i\sim[X_i,R_i,U_{L,i}],
\qquad
M_{XRU_0U_L}: y_i\sim[X_i,R_i,U_{0,i},U_{L,i}].
$$

The strict residual is

$$
\Delta_{U_L\mid X,R,U_0}
=
\mathrm{BA}(M_{XRU_0U_L})
-
\mathrm{BA}(M_{XRU_0}).
$$

## target replication

| target | candidates | lag U after XR | lag U after XR+U0 | mean XR BA | mean XR+U0 BA | mean XR+UL BA | mean XR+U0+UL BA | mean UL|XR | mean UL|XR,U0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 1 | 0 | 0.845811 | 0.862736 | 0.839904 | 0.852269 | -0.005907 | -0.010466 |
| `first_movement_speed` | 5 | 5 | 2 | 0.750006 | 0.784605 | 0.774394 | 0.788782 | 0.024388 | 0.004177 |
| `wheel_action_direction` | 5 | 2 | 2 | 0.837202 | 0.865689 | 0.839519 | 0.860067 | 0.002318 | -0.005622 |

## candidate summaries

| candidate | trials | lag U after XR | lag U after XR+U0 | choice UL|XR,U0 | speed UL|XR,U0 | wheel UL|XR,U0 |
|---|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 2 | 2 | -0.001667 | 0.012810 | 0.000105 |
| `nyu30_motor_striatal_multi_probe` | 933 | 2 | 1 | -0.007797 | 0.020375 | -0.013790 |
| `dy014_striatal_septal_probe` | 608 | 1 | 0 | -0.028196 | -0.001701 | -0.004870 |
| `dy011_motor_cortex_probe` | 402 | 1 | 1 | -0.002941 | -0.010554 | 0.003585 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 2 | 0 | -0.011730 | -0.000046 | -0.013138 |

## verdict

- lagged unit after task+region supported: `True`
- lagged unit after task+region+current-unit supported: `False`
- temporal GLM coupling gate passed: `False`

해석:

- 양성이면 static unit identity 뒤에도 lagged unit activity가 남으므로 temporal coupling 후보로 승격한다.
- 음성이면 block-regularized unit detail은 주로 target-window static readout으로 남고, causal/effective coupling claim은 보류한다.
