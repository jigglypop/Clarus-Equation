# Mouse IBL/OpenAlyx innovation-to-behavior gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_t)
$$

## setup

- candidates: 3
- components: 12
- folds: 5
- innovation behavior gate passed: `False`

## target replication

| target | candidates | Hhat positive | eps positive | eps after Hhat positive | mean dHhat | mean deps | mean deps after Hhat | innovation supported |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `choice_sign` | 3 | 1 | 2 | 2 | -0.000803 | 0.020499 | 0.018358 | `False` |
| `first_movement_speed` | 3 | 2 | 3 | 3 | 0.010310 | 0.027922 | 0.019631 | `False` |
| `wheel_action_direction` | 3 | 2 | 3 | 3 | 0.004218 | 0.048108 | 0.048475 | `False` |

## verdict

- `Hhat` asks whether the predictable latent trajectory carries behavior.
- `eps after Hhat` asks whether the unpredictable latent innovation carries extra behavior.
