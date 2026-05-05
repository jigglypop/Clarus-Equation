# Mouse IBL/OpenAlyx innovation-to-behavior gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_t)
$$

## setup

- candidates: 5
- components: 12
- folds: 5
- innovation behavior gate passed: `True`

## target replication

| target | candidates | Hhat positive | eps positive | eps after Hhat positive | mean dHhat | mean deps | mean deps after Hhat | innovation supported |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `choice_sign` | 5 | 1 | 4 | 4 | -0.001622 | 0.014292 | 0.010658 | `True` |
| `first_movement_speed` | 5 | 5 | 4 | 4 | 0.018635 | 0.020376 | 0.020288 | `True` |
| `wheel_action_direction` | 5 | 0 | 3 | 4 | -0.006206 | 0.008116 | 0.013981 | `True` |

## verdict

- `Hhat` asks whether the predictable latent trajectory carries behavior.
- `eps after Hhat` asks whether the unpredictable latent innovation carries extra behavior.
