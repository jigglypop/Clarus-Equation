# Mouse IBL/OpenAlyx low-rank unit transition gate

$$
H_t=A H_{t-\ell}+B X_t+C R_t+\epsilon_t
$$

## setup

- candidates: 5
- components: 12
- folds: 5
- low-rank unit transition gate passed: `True`

## replication

| transition | candidates | pos after X | pos after X,R0 | mean dR2 after X | mean dR2 after X,R0 | supported |
|---|---:|---:|---:|---:|---:|---|
| `pre_stimulus_to_stimulus_latent` | 5 | 5 | 5 | 0.195029 | 0.106973 | `True` |
| `pre_movement_to_movement_latent` | 5 | 5 | 5 | 0.218151 | 0.127870 | `True` |

## candidate summaries

| candidate | trials | pre-stim dR2 XR | pre-move dR2 XR |
|---|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0.097887 | 0.082994 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0.131070 | 0.072784 |
| `dy014_striatal_septal_probe` | 608 | 0.063714 | 0.058707 |
| `dy011_motor_cortex_probe` | 402 | 0.177502 | 0.282063 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 0.064691 | 0.142801 |
