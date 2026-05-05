# Mouse IBL/OpenAlyx unit transition dynamics gate

This gate asks a neural transition question before returning to behavior.

$$
U_t = A U_{t-\ell}+B X_t+C R_t+\epsilon_t
$$

The tested residual is:

$$
\Delta_{\mathrm{transition}\mid X,R_0}
=
R^2[U_t\mid X_t,R_t,U_{t-\ell}]
-
R^2[U_t\mid X_t,R_t].
$$

## setup

- candidates: 1
- folds: 5
- task penalty: 1.0
- region penalty: 1.0
- lag unit penalty: 100.0
- min pooled delta: 0.002
- min mean-feature delta: 0.0005
- unit transition dynamics gate passed: `False`

## transition replication

| transition | candidates | pos after X | pos after X,R0 | mean dR2 after X | mean dR2 after X,R0 | supported |
|---|---:|---:|---:|---:|---:|---|
| `pre_stimulus_to_stimulus_unit` | 1 | 0 | 0 | 0.246408 | 0.187064 | `False` |
| `pre_movement_to_movement_unit` | 1 | 0 | 0 | 0.050350 | 0.008384 | `False` |

## candidate summaries

| candidate | trials | pre-stim dR2 after X | pre-stim dR2 after X,R0 | pre-move dR2 after X | pre-move dR2 after X,R0 |
|---|---:|---:|---:|---:|---:|
| `nyu30_motor_striatal_multi_probe` | 933 | 0.246408 | 0.187064 | 0.050350 | 0.008384 |

## verdict

- Positive result means lagged unit activity predicts the next unit state after task/history and region compression.
- Negative result means the previous behavioral temporal-GLM failure is not rescued by a simple linear unit-transition model.
