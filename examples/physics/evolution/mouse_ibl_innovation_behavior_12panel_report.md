# Mouse IBL/OpenAlyx innovation-to-behavior gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_t)
$$

## setup

- candidates: 12
- components: 12
- folds: 5
- innovation behavior gate passed: `True`

## target replication

| target | candidates | Hhat positive | eps positive | eps after Hhat positive | mean dHhat | mean deps | mean deps after Hhat | innovation supported |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `choice_sign` | 12 | 7 | 8 | 3 | 0.002320 | 0.008033 | 0.004624 | `False` |
| `first_movement_speed` | 12 | 9 | 10 | 9 | 0.009861 | 0.022655 | 0.016741 | `True` |
| `wheel_action_direction` | 12 | 5 | 8 | 7 | 0.001959 | 0.026489 | 0.029900 | `True` |

## verdict

- `Hhat` asks whether the predictable latent trajectory carries behavior.
- `eps after Hhat` asks whether the unpredictable latent innovation carries extra behavior.
